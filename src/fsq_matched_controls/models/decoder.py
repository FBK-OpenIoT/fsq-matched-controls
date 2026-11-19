"""Symmetric 1D-CNN decoder for FSQ phenological tokenisation.

Architecture (mirror of encoder)
---------------------------------
Input  : (batch, n_dims)  — quantised FSQ code

1. Linear(n_dims, d_hidden) — inverse of encoder projection.
2. Unsqueeze + expand to (batch, d_hidden, obs_per_window).
   Broadcasting simulates the inverse of GlobalAvgPool:
   the same hidden vector is repeated for every timestep.
3. Stack of Conv1d → BatchNorm1d → GELU blocks (channels reversed).
4. Linear 1×1 convolution: d_final → n_bands, output shape (batch, n_bands, obs).
5. Transpose back to (batch, obs, n_bands).

Loss minimisation forces the code to retain enough spectral information
to reconstruct, on average, the per-band reflectance pattern within the
calendar window — which is exactly what phenological tokenisation requires.
"""

from __future__ import annotations

import torch.nn as nn
from torch import Tensor


class TemporalCNNDecoder(nn.Module):
    """Symmetric temporal 1D-CNN decoder.

    Parameters
    ----------
    n_dims          : FSQ latent dimensionality (match encoder).
    channels        : Encoder channel sizes [32, 64, 128].
                      The decoder reverses them: [128, 64, 32].
    kernel_size     : Convolution kernel size (match encoder).
    n_bands         : Number of spectral bands in output (10 for Sentinel-2).
    obs_per_window  : Number of timesteps to reconstruct per window.
    dropout         : Dropout probability.
    """

    def __init__(
        self,
        n_dims: int = 4,
        channels: list[int] | None = None,
        kernel_size: int = 3,
        n_bands: int = 10,
        obs_per_window: int = 6,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        channels = channels or [32, 64, 128]
        rev = list(reversed(channels))                       # [128, 64, 32]
        self.obs_per_window = obs_per_window

        # Project from FSQ code → first decoder hidden dim
        self.proj = nn.Linear(n_dims, rev[0])

        # Conv blocks over the time axis
        layers: list[nn.Module] = []
        pad = kernel_size // 2
        in_ch = rev[0]
        for out_ch in rev[1:]:
            layers += [
                nn.Conv1d(in_ch, out_ch, kernel_size, padding=pad),
                nn.BatchNorm1d(out_ch),
                nn.GELU(),
                nn.Dropout(dropout),
            ]
            in_ch = out_ch

        self.conv = nn.Sequential(*layers)

        # 1×1 conv: last hidden → n_bands
        self.out_conv = nn.Conv1d(rev[-1], n_bands, kernel_size=1)

    def forward(self, z_q: Tensor) -> Tensor:
        """Decode quantised code back to spectral windows.

        Parameters
        ----------
        z_q : (batch, n_dims)

        Returns
        -------
        x_hat : (batch, obs_per_window, n_bands)
        """
        h = self.proj(z_q)                                   # (B, rev[0])
        # Broadcast to all timesteps: (B, rev[0], 1) → (B, rev[0], obs)
        h = h.unsqueeze(-1).expand(-1, -1, self.obs_per_window)
        h = self.conv(h)                                     # (B, rev[-1], obs)
        x_hat = self.out_conv(h)                             # (B, n_bands, obs)
        return x_hat.transpose(1, 2)                         # (B, obs, n_bands)

