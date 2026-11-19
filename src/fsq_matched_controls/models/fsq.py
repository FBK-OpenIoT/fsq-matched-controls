"""Finite Scalar Quantization (FSQ).

Reference: Mentzer et al. "Finite Scalar Quantization: VQ-VAE Made Simple"
           ICLR 2024.  https://arxiv.org/abs/2309.15505

Key properties
--------------
- No codebook EMA, no commitment loss — only reconstruction loss.
- Codebook collapse is impossible by construction.
- Straight-through gradient estimation.
- Token IDs via mixed-radix encoding of quantised integer tuples.

Usage
-----
    fsq = FSQ(n_dims=4, n_levels=5)          # 5^4 = 625 codes
    z_pre = encoder(x)                        # (B, n_dims) — any real values
    z_q, token_ids = fsq(z_pre)              # quantised + token IDs
    x_hat = decoder(z_q)
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class FSQ(nn.Module):
    """Finite Scalar Quantizer.

    Parameters
    ----------
    n_dims   : Number of independent scalar dimensions to quantise.
    n_levels : Number of discrete levels per dimension.
                Codebook size = n_levels ** n_dims.
                Recommended configurations (from paper §3):
                  (3, 5) → 125  |  (4, 5) → 625   |  (5, 5) → 3125
                  (4, 4) → 256  |  (5, 4) → 1024  |  (6, 4) → 4096

    Input assumption
    ---------------
    The FSQ layer expects inputs already scaled to the quantisation range.
    Typical pattern in the encoder::

        z = linear_proj(h)              # (B, n_dims)
        z = torch.tanh(z) * half        # scale to [-half, half]
        z_q, ids = fsq(z)
    """

    def __init__(self, n_dims: int, n_levels: int) -> None:
        super().__init__()
        self.n_dims = n_dims
        self.n_levels = n_levels
        self._half = n_levels // 2          # quantisation range: [-half, half]

        # Pre-compute mixed-radix powers for token-ID encoding
        # token_id = sum_d( z_idx_d * n_levels^d )
        powers = n_levels ** torch.arange(n_dims, dtype=torch.long)
        self.register_buffer("_powers", powers)

    # ── properties ───────────────────────────────────────────────────────────

    @property
    def codebook_size(self) -> int:
        return self.n_levels ** self.n_dims

    # ── core quantisation ────────────────────────────────────────────────────

    def _quantize(self, z: Tensor) -> Tensor:
        """Round each scalar to the nearest integer in {-half, …, half}.

        Straight-through: gradients pass through the rounding operation.
        """
        z_clamped = z.clamp(-self._half, self._half)
        z_rounded = z_clamped.round()
        # Straight-through estimator: forward uses rounded, backward uses original
        return z + (z_rounded - z).detach()

    def _to_token_ids(self, z_q: Tensor) -> Tensor:
        """Convert quantised tensor to integer token IDs.

        Parameters
        ----------
        z_q : (batch, n_dims)  — values in {-half, …, half}

        Returns
        -------
        ids : (batch,)  long tensor, values in [0, codebook_size)
        """
        z_idx = (z_q + self._half).long()          # shift to [0, n_levels)
        z_idx = z_idx.clamp(0, self.n_levels - 1)  # safety clamp
        ids = (z_idx * self._powers).sum(dim=-1)
        return ids

    def decode_token_ids(self, ids: Tensor) -> Tensor:
        """Convert token IDs back to quantised integer tuples.

        Parameters
        ----------
        ids : (batch,) long tensor

        Returns
        -------
        z_q : (batch, n_dims)  float tensor in {-half, …, half}
        """
        digits = []
        remaining = ids.clone()
        for _ in range(self.n_dims):
            digits.append(remaining % self.n_levels)
            remaining = remaining // self.n_levels
        z_idx = torch.stack(digits, dim=-1).float()    # (batch, n_dims)
        return z_idx - self._half                       # shift back

    # ── forward ──────────────────────────────────────────────────────────────

    def forward(self, z: Tensor) -> tuple[Tensor, Tensor]:
        """Quantise ``z`` and return quantised values + token IDs.

        Parameters
        ----------
        z : (batch, n_dims)  — encoder output, should be pre-scaled to
                               roughly [-half, half] (e.g., via tanh * half).

        Returns
        -------
        z_q      : (batch, n_dims)  straight-through quantised tensor
        token_ids: (batch,)         integer IDs in [0, codebook_size)
        """
        z_q = self._quantize(z)
        token_ids = self._to_token_ids(z_q)
        return z_q, token_ids

