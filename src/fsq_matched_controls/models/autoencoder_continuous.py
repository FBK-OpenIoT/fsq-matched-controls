"""Twin `twin_noquant`: the letter's reconstruction autoencoder without the quantizer in the training path (Table II, supplement S5.3).

Same encoder (the CNN window encoder with its bounded output tanh x 2, d = 4), same decoder and same initial weights as ``FSQAutoencoder`` for a given seed; the decoder reads the continuous bounded latent z
instead of the rounded z_q, so no rounding (and no straight-through estimator) takes part in training. The rounding grid of the letter (FSQ, five levels per dimension) is applied only after training,
by ``self.fsq`` (post-hoc rounding); in the forward pass it is computed under ``no_grad`` for code-use logging only. ``quantize_in_forward = True`` restores the parent's forward pass (used by the gate that
compares the two classes on the same weights).
"""
from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor

from fsq_matched_controls.models.autoencoder import FSQAutoencoder


class ContinuousAutoencoder(FSQAutoencoder):
    quantize_in_forward: bool = False

    def forward(self, x: Tensor, obs_mask: Optional[Tensor] = None) -> tuple[Tensor, Tensor, Tensor]:
        z = self.encoder(x, obs_mask=obs_mask)
        if self.quantize_in_forward:
            z_q, token_ids = self.fsq(z)
            return self.decoder(z_q), z_q, token_ids
        with torch.no_grad():
            _, token_ids = self.fsq(z)          # post-hoc rounding: code-use statistics only, not used by the decoder, no gradient
        return self.decoder(z), z, token_ids

    def encode(self, x: Tensor, obs_mask: Optional[Tensor] = None) -> tuple[Tensor, Tensor]:
        """continuous latent z and the post-hoc rounded token ids"""
        z = self.encoder(x, obs_mask=obs_mask)
        _, token_ids = self.fsq(z)
        return z, token_ids
