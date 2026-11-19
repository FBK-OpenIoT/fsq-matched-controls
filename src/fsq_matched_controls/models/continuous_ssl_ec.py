"""Twins `twin_bands` and twin_teacher on EuroCropsML (Table II, supplement S5.3): the continuous (no quantizer) masked predictors for
datasets with missing months and padded windows, as the letter's EuroCropsML predictive model reads them.

``ContinuousPredictiveSSLEC`` (twin_bands): the letter's encoder (layer norm, observation mask), context transformer and masking, the quantizer out of the training path; the head regresses the masked month's own window values
at its real observations (mean squared error over real observations only, as the letter's reconstruction loss). ``ContinuousTeacherSSLEC`` (twin_teacher): the head regresses the d-dimensional latent of an EMA teacher
(stop-gradient, evaluation mode, momentum 0.999 rising to 0.9999 over ten epochs) for the same window; collapse (latent variance, effective rank) logged per epoch.
Batches: ``(windows, labels, regions, valid_mask, obs_mask)``. Masked months are drawn among valid months (``FSQPredictiveSSL._sample_mask``); ``input_mask = mask | ~valid_mask`` hides invalid and masked months from the
transformer; ``supervise_mask = mask & valid_mask``.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor

from fsq_matched_controls.models.continuous_ssl import ContinuousPredictiveSSL, ContinuousSSLConfig
from fsq_matched_controls.models.continuous_ssl_teacher import ContinuousTeacherSSL, effective_rank
from fsq_matched_controls.models.predictive_ssl import FSQPredictiveSSL


def _unpack(batch: tuple):
    if len(batch) == 5:
        windows, _, _, valid_mask, obs_mask = batch
    elif len(batch) == 4:
        windows, _, _, valid_mask = batch; obs_mask = None
    else:
        windows = batch[0]; valid_mask = None; obs_mask = None
    return windows, valid_mask, obs_mask


def _masks(model, B: int, N: int, device, valid_mask: Optional[Tensor]):
    mask = FSQPredictiveSSL._sample_mask(model, B, N, device=device, valid_mask=valid_mask)
    if valid_mask is None:
        return mask, mask, mask
    return mask, mask | ~valid_mask, mask & valid_mask


class ContinuousPredictiveSSLEC(ContinuousPredictiveSSL):
    def forward(self, x: Tensor, mask: Optional[Tensor] = None, valid_mask: Optional[Tensor] = None, obs_mask: Optional[Tensor] = None):
        B, N, obs, bands = x.shape
        om = obs_mask.reshape(B * N, obs) if obs_mask is not None else None
        z = self.encoder(x.reshape(B * N, obs, bands), obs_mask=om).view(B, N, -1)
        if mask is None:
            mask, input_mask, supervise = _masks(self, B, N, x.device, valid_mask)
        else:
            input_mask, supervise = (mask, mask) if valid_mask is None else (mask | ~valid_mask, mask & valid_mask)
        pred = self.context_transformer(z, input_mask)                              # (B, N, obs * bands)
        target = x.reshape(B, N, obs * bands).detach()
        weight = (obs_mask.unsqueeze(-1).expand(B, N, obs, bands).reshape(B, N, obs * bands).to(x.dtype) if obs_mask is not None else torch.ones_like(target))
        return pred, target, weight, supervise

    def _step(self, batch: tuple, stage: str) -> Tensor:
        windows, valid_mask, obs_mask = _unpack(batch)
        pred, target, weight, supervise = self(windows, valid_mask=valid_mask, obs_mask=obs_mask)
        if int(supervise.sum()) == 0:
            loss = pred.sum() * 0.0; r2 = torch.zeros((), device=windows.device)
        else:
            p, t, w = pred[supervise], target[supervise], weight[supervise]
            loss = (((p - t) ** 2) * w).sum() / w.sum().clamp(min=1.0)
            with torch.no_grad():
                tm = (t * w).sum() / w.sum().clamp(min=1.0); var = (((t - tm) ** 2) * w).sum() / w.sum().clamp(min=1.0)
                r2 = 1.0 - loss / var.clamp(min=1e-12)
        self.log(f"{stage}_loss", loss, prog_bar=True, on_epoch=True, on_step=(stage == "train"), sync_dist=True)
        self.log(f"{stage}_r2", r2, prog_bar=False, on_epoch=True, on_step=False, sync_dist=True)
        return loss


class ContinuousTeacherSSLEC(ContinuousTeacherSSL):
    def forward(self, x: Tensor, mask: Optional[Tensor] = None, valid_mask: Optional[Tensor] = None, obs_mask: Optional[Tensor] = None):
        B, N, obs, bands = x.shape
        om = obs_mask.reshape(B * N, obs) if obs_mask is not None else None
        z = self.encoder(x.reshape(B * N, obs, bands), obs_mask=om).view(B, N, -1)
        if mask is None:
            mask, input_mask, supervise = _masks(self, B, N, x.device, valid_mask)
        else:
            input_mask, supervise = (mask, mask) if valid_mask is None else (mask | ~valid_mask, mask & valid_mask)
        pred = self.context_transformer(z, input_mask)                              # (B, N, d)
        with torch.no_grad():
            target = self.teacher_encoder(x.reshape(B * N, obs, bands), obs_mask=om).view(B, N, -1)
        self._last_z = (z.detach(), target, supervise)
        return pred, target, supervise

    def _step(self, batch: tuple, stage: str) -> Tensor:
        windows, valid_mask, obs_mask = _unpack(batch)
        pred, target, supervise = self(windows, valid_mask=valid_mask, obs_mask=obs_mask)
        if int(supervise.sum()) == 0:
            loss = pred.sum() * 0.0; r2 = torch.zeros((), device=windows.device)
        else:
            p, t = pred[supervise], target[supervise]; loss = F.mse_loss(p, t)
            with torch.no_grad():
                r2 = 1.0 - loss / t.var(unbiased=False).clamp(min=1e-12)
        self.log(f"{stage}_loss", loss, prog_bar=True, on_epoch=True, on_step=(stage == "train"), sync_dist=True)
        self.log(f"{stage}_r2", r2, prog_bar=False, on_epoch=True, on_step=False, sync_dist=True)
        zs, zt, sup = self._last_z
        valid = sup.new_ones(zs.shape[:2]) if valid_mask is None else valid_mask
        zs_, zt_ = zs[valid.bool()], zt[valid.bool()]                               # collapse statistics over the valid months only
        if len(zs_) > 1:
            self.log(f"{stage}_student_var", zs_.var(0, unbiased=False).mean(), on_epoch=True, on_step=False, sync_dist=True)
            self.log(f"{stage}_teacher_var", zt_.var(0, unbiased=False).mean(), on_epoch=True, on_step=False, sync_dist=True)
            self.log(f"{stage}_student_eff_rank", effective_rank(zs_), on_epoch=True, on_step=False, sync_dist=True)
            self.log(f"{stage}_teacher_eff_rank", effective_rank(zt_), on_epoch=True, on_step=False, sync_dist=True)
        return loss
