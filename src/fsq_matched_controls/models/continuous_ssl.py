"""Twin `twin_bands` (band-value target): continuous-latent masked-input prediction (Table II, supplement S5.3).

The letter's predictive model without a quantizer: the letter's CNN window encoder (output tanh x 2, a bounded continuous latent z in
[-2, 2]^d, no FSQ), the letter's context transformer, and a head of the same shape that regresses the masked month's own input
window (obs_per_window x n_bands values) by mean squared error. The target is the input (exogenous, detached), so the
self-referential collapse path of token prediction does not exist.

``ContextTransformer`` and ``FSQPredictiveSSL._sample_mask`` are reused as they are.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor
from lightning.pytorch import LightningModule

from fsq_matched_controls.models.encoder import TemporalCNNEncoder
from fsq_matched_controls.models.predictive_ssl import ContextTransformer, FSQPredictiveSSL


@dataclass
class ContinuousSSLConfig:
    n_bands: int = 10
    obs_per_window: int = 6
    n_tokens: int = 7
    encoder_channels: list[int] = field(default_factory=lambda: [32, 64, 128])
    kernel_size: int = 3
    dropout: float = 0.1
    norm_type: str = "batch"
    n_dims: int = 4                 # latent width d (the letter uses 4)
    n_levels: int = 5               # only sets the tanh scale of the encoder output (half = 2); there is no quantizer
    mask_ratio: float = 0.15
    min_masks: int = 1
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 3
    dim_feedforward: int = 256
    transformer_dropout: float = 0.1
    max_seq_len: int = 100
    lr: float = 1e-3
    lr_scheduler: str = "plateau"
    weight_decay: float = 1e-4
    max_epochs: int = 100
    plateau_factor: float = 0.5
    plateau_patience: int = 10

    def to_dict(self) -> dict:
        return asdict(self)


class ContinuousPredictiveSSL(LightningModule):
    """Masked input prediction through a bounded continuous latent.

    Training input  : (B, N, obs_per_window, n_bands) parcel sequences (the letter's data module).
    Feature access  : ``encode_z`` (single windows -> z), ``encode_sequence`` (parcel sequences -> z and the transformer's state h with
                      no month masked).
    """

    def __init__(self, config: ContinuousSSLConfig | dict) -> None:
        super().__init__()
        if isinstance(config, dict):
            config = ContinuousSSLConfig(**config)
        self.config = config
        self.save_hyperparameters({"config": config.to_dict()})
        self.encoder = TemporalCNNEncoder(
            n_bands=config.n_bands, channels=config.encoder_channels, kernel_size=config.kernel_size,
            n_dims=config.n_dims, n_levels=config.n_levels, dropout=config.dropout, norm_type=config.norm_type,
        )
        self.n_target = config.obs_per_window * config.n_bands
        # ContextTransformer's joint head is Linear-GELU-LayerNorm-Linear with ``codebook_size`` outputs: here the window values
        self.context_transformer = ContextTransformer(
            n_dims=config.n_dims, codebook_size=self.n_target, d_model=config.d_model, n_heads=config.n_heads,
            n_layers=config.n_layers, dim_feedforward=config.dim_feedforward, dropout=config.transformer_dropout,
            max_seq_len=config.max_seq_len, factorized_head=False, n_levels=config.n_levels, head_mode="joint",
        )

    # ---- features ----------------------------------------------------------------------------------------------------------------
    def encode_z(self, x: Tensor) -> Tensor:
        """x (B, obs, bands) -> z (B, n_dims), values in [-2, 2]."""
        return self.encoder(x)

    def encode_sequence(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """x (B, N, obs, bands) -> (z (B, N, n_dims), h (B, N, d_model)); h is the transformer output with every month visible."""
        B, N, obs, bands = x.shape
        z = self.encoder(x.reshape(B * N, obs, bands)).view(B, N, -1)
        emb = self.context_transformer.input_proj(z)
        emb = emb + self.context_transformer.pos_emb(torch.arange(N, device=x.device))
        return z, self.context_transformer.transformer(emb)

    # ---- training ----------------------------------------------------------------------------------------------------------------
    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> tuple[Tensor, Tensor, Tensor]:
        B, N, obs, bands = x.shape
        z = self.encoder(x.reshape(B * N, obs, bands)).view(B, N, -1)
        if mask is None:
            mask = FSQPredictiveSSL._sample_mask(self, B, N, device=x.device, valid_mask=None)
        pred = self.context_transformer(z, mask)                      # (B, N, obs * bands)
        target = x.reshape(B, N, obs * bands).detach()
        return pred, target, mask

    def _step(self, batch: tuple, stage: str) -> Tensor:
        windows = batch[0]
        pred, target, mask = self(windows)
        if int(mask.sum()) == 0:
            loss = pred.sum() * 0.0
            r2 = torch.zeros((), device=windows.device)
        else:
            p, t = pred[mask], target[mask]
            loss = F.mse_loss(p, t)
            with torch.no_grad():
                r2 = 1.0 - loss / t.var(unbiased=False).clamp(min=1e-12)       # share of the masked windows' variance explained
        on_step = stage == "train"
        self.log(f"{stage}_loss", loss, prog_bar=True, on_epoch=True, on_step=on_step, sync_dist=True)
        self.log(f"{stage}_r2", r2, prog_bar=False, on_epoch=True, on_step=False, sync_dist=True)
        return loss

    def training_step(self, batch: tuple, batch_idx: int) -> Tensor:
        return self._step(batch, "train")

    def validation_step(self, batch: tuple, batch_idx: int) -> None:
        self._step(batch, "val")

    def test_step(self, batch: tuple, batch_idx: int) -> None:
        self._step(batch, "test")

    def configure_optimizers(self) -> dict:
        trainable = [p for p in self.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(trainable, lr=self.config.lr, weight_decay=self.config.weight_decay)
        if self.config.lr_scheduler == "plateau":
            sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=self.config.plateau_factor, patience=self.config.plateau_patience)
            return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "monitor": "val_loss"}}
        return {"optimizer": opt}

    @classmethod
    def load(cls, checkpoint_path: Path | str) -> "ContinuousPredictiveSSL":
        ckpt = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
        flat = ckpt.get("hyper_parameters", {}).get("config", {})
        return cls.load_from_checkpoint(str(checkpoint_path), config=ContinuousSSLConfig(**flat))

    @classmethod
    def from_init_state(cls, config: ContinuousSSLConfig | dict, init_state_path: Path | str) -> "ContinuousPredictiveSSL":
        """The untrained control of a run: the weights it had before its first optimisation step."""
        model = cls(config)
        model.load_state_dict(torch.load(str(init_state_path), map_location="cpu"))
        return model
