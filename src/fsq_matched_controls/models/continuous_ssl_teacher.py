"""Twin `twin_teacher` (slow-teacher target; Table II, supplement S5.3): masked prediction without FSQ whose targets are the continuous latent of a slow (EMA) teacher.

``ContinuousPredictiveSSL`` (twin `twin_bands`: the letter's CNN window encoder with the bounded output tanh x 2, d = 4, the letter's context transformer, the letter's masking and recipe) with two changes: the head regresses
the teacher's d-dimensional latent of the unmasked window of the masked month (mean squared error, stop-gradient: the teacher never receives a gradient and runs in evaluation mode), and the teacher is a copy of the
student encoder updated as ``teacher = m * teacher + (1 - m) * student`` after every optimizer step (parameters and floating-point batch-norm buffers), m = 0.999 at the first step rising linearly to 0.9999 at the end of epoch 10.
There is no quantizer in the training path. Collapse is logged per epoch for train and validation: the mean per-dimension variance of the student and teacher latents over the batch's months, and their effective rank
(exp of the entropy of the normalised squared singular values of the centred latent matrix, between 1 and d). If the latent collapses, that is the result.

The teacher's weights are part of the state dict and therefore of every checkpoint.
"""
from __future__ import annotations

import copy
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor

from fsq_matched_controls.models.continuous_ssl import ContinuousPredictiveSSL, ContinuousSSLConfig
from fsq_matched_controls.models.predictive_ssl import ContextTransformer, FSQPredictiveSSL


def effective_rank(z: Tensor) -> Tensor:
    """z (n, d) -> exp(entropy of the normalised squared singular values of the centred matrix), in [1, d]"""
    zc = z.float() - z.float().mean(0, keepdim=True)
    s = torch.linalg.svdvals(zc) ** 2
    p = s / s.sum().clamp(min=1e-12)
    return torch.exp(-(p * torch.log(p.clamp(min=1e-12))).sum())


class ContinuousTeacherSSL(ContinuousPredictiveSSL):
    M_START, M_END, RAMP_EPOCHS = 0.999, 0.9999, 10

    def __init__(self, config: ContinuousSSLConfig | dict) -> None:
        super().__init__(config)
        c = self.config
        self.n_target = c.n_dims
        self.context_transformer = ContextTransformer(
            n_dims=c.n_dims, codebook_size=c.n_dims, d_model=c.d_model, n_heads=c.n_heads, n_layers=c.n_layers, dim_feedforward=c.dim_feedforward,
            dropout=c.transformer_dropout, max_seq_len=c.max_seq_len, factorized_head=False, n_levels=c.n_levels, head_mode="joint")     # the joint head regresses the d-dimensional latent
        self.teacher_encoder = copy.deepcopy(self.encoder)
        for p in self.teacher_encoder.parameters():
            p.requires_grad_(False)
        self.teacher_encoder.eval()
        self._last_m = self.M_START
        self._last_z: Optional[tuple[Tensor, Tensor]] = None

    def train(self, mode: bool = True) -> "ContinuousTeacherSSL":
        super().train(mode)
        self.teacher_encoder.eval()
        return self

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> tuple[Tensor, Tensor, Tensor]:
        B, N, obs, bands = x.shape
        z = self.encoder(x.reshape(B * N, obs, bands)).view(B, N, -1)
        if mask is None:
            mask = FSQPredictiveSSL._sample_mask(self, B, N, device=x.device, valid_mask=None)
        pred = self.context_transformer(z, mask)                                   # (B, N, d)
        with torch.no_grad():
            target = self.teacher_encoder(x.reshape(B * N, obs, bands)).view(B, N, -1)
        self._last_z = (z.detach(), target)
        return pred, target, mask

    def _step(self, batch: tuple, stage: str) -> Tensor:
        loss = super()._step(batch, stage)
        zs, zt = self._last_z
        zs, zt = zs.reshape(-1, zs.shape[-1]), zt.reshape(-1, zt.shape[-1])
        on_step = False
        self.log(f"{stage}_student_var", zs.var(0, unbiased=False).mean(), on_epoch=True, on_step=on_step, sync_dist=True)
        self.log(f"{stage}_teacher_var", zt.var(0, unbiased=False).mean(), on_epoch=True, on_step=on_step, sync_dist=True)
        self.log(f"{stage}_student_eff_rank", effective_rank(zs), on_epoch=True, on_step=on_step, sync_dist=True)
        self.log(f"{stage}_teacher_eff_rank", effective_rank(zt), on_epoch=True, on_step=on_step, sync_dist=True)
        return loss

    def momentum(self) -> float:
        n = int(self.trainer.global_step)
        T = self.RAMP_EPOCHS * int(self.trainer.num_training_batches)
        frac = 1.0 if T <= 0 else min(1.0, max(0.0, (n - 1) / T))
        return self.M_START + (self.M_END - self.M_START) * frac

    @torch.no_grad()
    def update_teacher(self, m: float) -> None:
        for pt, ps in zip(self.teacher_encoder.parameters(), self.encoder.parameters()):
            pt.mul_(m).add_(ps.detach(), alpha=1.0 - m)
        for bt, bs in zip(self.teacher_encoder.buffers(), self.encoder.buffers()):
            if bt.dtype.is_floating_point:
                bt.mul_(m).add_(bs, alpha=1.0 - m)
            else:
                bt.copy_(bs)
        self._last_m = m

    def on_train_batch_end(self, outputs, batch, batch_idx) -> None:
        self.update_teacher(self.momentum())
