"""1D-CNN temporal encoder for FSQ phenological tokenisation.

Architecture
------------
Input  : (batch, obs_per_window, n_bands)

1. Transpose to (batch, n_bands, obs)  — Conv1d channel convention.
2. Stack of Conv1d → Norm → GELU → Dropout blocks (configurable depth).
3. Global temporal pooling — collapses all observations within the window
   into a single vector, making the encoder invariant to the number of
   cloud-free observations.
4. Linear(d_hidden, n_dims)  — project to FSQ latent dimensionality.
5. tanh × half  — scale output to [-half, half] for FSQ quantisation.

The straight-through gradient flows from FSQ back through step 4 and beyond.

Variable-length / masked windows (opt-in, off by default)
-----------------------------------------------------------
By default (``obs_mask=None`` in ``forward``, ``norm_type="batch"`` in
``__init__``), this encoder behaves exactly as it always has: every position
in the ``obs`` axis is assumed real, pooling is a plain global average
(``AdaptiveAvgPool1d``), and normalisation is ``BatchNorm1d``. This is
CropGlobe's path — fixed-cadence windows, never any padding — and it is
untouched by anything below. Crucially, ``self.conv`` is still built as a
single flat ``nn.Sequential`` exactly as before (same layer order, same
auto-generated state_dict keys: ``conv.0``, ``conv.1``, ...), so existing
CropGlobe checkpoints load into this class unchanged.

Some datasets (EuroCropsML) have a variable, sometimes small number of real
observations per calendar window, batched together with zero-padding up to a
fixed ``W_MAX``. Passing an ``obs_mask: (batch, obs)`` bool tensor (True =
real, False = padding) activates two changes, both needed together:

  - Pooling becomes a masked mean (sum over real positions / count of real
    positions) instead of a plain average — otherwise the padded zeros would
    dilute the pooled result.
  - If ``norm_type="layer"``, ``BatchNorm1d`` is replaced with a per-timestep
    ``LayerNorm`` (normalises across channels only, independently at each
    timestep). This matters because ``BatchNorm1d`` pools statistics over the
    batch AND time axes jointly — with a large padded fraction (measured at
    ~50-65% for a sensible ``W_MAX`` on EuroCropsML), that would materially
    bias the learned running statistics towards zero. LayerNorm never
    aggregates across timesteps, so padding in one position can't
    contaminate another's statistics.

Padded positions are also re-zeroed after every conv block (not just left to
the initial input masking), to bound how much a padded zero can "bleed" into
neighbouring real timesteps through the conv kernel's receptive field
compounding across repeated layers. This is done by iterating ``self.conv``'s
children directly and re-masking every 4th layer (each block is exactly
conv → norm → act → dropout), rather than restructuring the module, so the
default path's parameter layout never changes.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor


class _ChannelLayerNorm(nn.Module):
    """LayerNorm over the channel dimension of a ``(B, C, T)`` tensor,
    applied independently at each timestep — never aggregates across ``T``
    or the batch (see module docstring for why that matters with padding)."""

    def __init__(self, num_channels: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(num_channels)

    def forward(self, x: Tensor) -> Tensor:
        return self.norm(x.transpose(1, 2)).transpose(1, 2)


class TemporalCNNEncoder(nn.Module):
    """Temporal 1D-CNN encoder.

    Parameters
    ----------
    n_bands     : Number of spectral bands (10 for Sentinel-2).
    channels    : Hidden channel sizes for each Conv1d block, e.g. [32, 64, 128].
    kernel_size : Convolution kernel size (default 3).
    n_dims      : FSQ latent dimensionality (output of projection head).
    n_levels    : FSQ levels per dimension (used to compute tanh scaling).
    dropout     : Dropout probability applied after each conv block.
    norm_type   : ``"batch"`` (default, ``BatchNorm1d`` — CropGlobe's
                  fixed-window path, always fully real data) or ``"layer"``
                  (per-timestep ``LayerNorm`` — required whenever windows can
                  contain padding; see module docstring).
    """

    _LAYERS_PER_BLOCK = 4  # conv, norm, act, dropout

    def __init__(
        self,
        n_bands: int = 10,
        channels: list[int] | None = None,
        kernel_size: int = 3,
        n_dims: int = 4,
        n_levels: int = 5,
        dropout: float = 0.1,
        norm_type: str = "batch",
    ) -> None:
        super().__init__()
        if norm_type not in ("batch", "layer"):
            raise ValueError(f"norm_type must be 'batch' or 'layer', got {norm_type!r}")

        channels = channels or [32, 64, 128]
        self._half = n_levels // 2
        self.norm_type = norm_type

        # Flat Sequential, same layer order as always -- for norm_type="batch"
        # this produces byte-identical state_dict keys to the original
        # implementation, so existing CropGlobe checkpoints still load.
        layers: list[nn.Module] = []
        in_ch = n_bands
        pad = kernel_size // 2          # same-padding to preserve length
        for out_ch in channels:
            norm: nn.Module = nn.BatchNorm1d(out_ch) if norm_type == "batch" else _ChannelLayerNorm(out_ch)
            layers += [
                nn.Conv1d(in_ch, out_ch, kernel_size, padding=pad),
                norm,
                nn.GELU(),
                nn.Dropout(dropout),
            ]
            in_ch = out_ch

        self.conv = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)                   # → (B, C_last, 1) -- unmasked path only
        self.proj = nn.Linear(channels[-1], n_dims)           # → (B, n_dims)

    def forward(self, x: Tensor, obs_mask: Optional[Tensor] = None) -> Tensor:
        """Encode a batch of windows.

        Parameters
        ----------
        x        : (batch, obs_per_window, n_bands)
        obs_mask : (batch, obs_per_window) bool, optional — True at real
                   (non-padded) observations. ``None`` (default) preserves
                   the original, always-real-data behaviour exactly
                   (CropGlobe). See module docstring for what passing a mask
                   changes and why.

        Returns
        -------
        z : (batch, n_dims)  — values in [-half, half], ready for FSQ.
        """
        x = x.transpose(1, 2)                           # (B, n_bands, obs)

        if obs_mask is None:
            h = self.conv(x)                             # (B, C_last, obs)
            h = self.pool(h).squeeze(-1)                 # (B, C_last)
        else:
            mask = obs_mask.unsqueeze(1).to(x.dtype)     # (B, 1, obs)
            h = x * mask
            for i, layer in enumerate(self.conv):
                h = layer(h)
                if (i + 1) % self._LAYERS_PER_BLOCK == 0:
                    h = h * mask                         # re-zero padding after every conv block
            denom = mask.sum(dim=2).clamp(min=1e-6)      # (B, 1) -- real-observation count
            h = h.sum(dim=2) / denom                     # masked mean -> (B, C_last)

        z = self.proj(h)                                 # (B, n_dims)
        z = torch.tanh(z) * self._half                   # scale to [-half, half]
        return z
