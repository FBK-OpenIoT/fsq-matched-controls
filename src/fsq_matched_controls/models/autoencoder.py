"""FSQ Autoencoder — PyTorch Lightning Module.

Full pipeline per forward pass:
    x  →  Encoder  →  Linear+tanh  →  FSQ  →  Decoder  →  x̂
    loss = MSE(x, x̂)    (pure reconstruction, no auxiliary losses)

Logged metrics (MLflow + optional W&B)
---------------------------------------
- train_loss / val_loss / test_loss   : MSE reconstruction loss
- codebook_utilisation                : % of codes seen in each epoch
- per_band_mse_*                      : per-band reconstruction quality
- val_r2                              : average R² across all bands

Saved artefacts (in experiment_dir)
--------------------------------------
- best.ckpt                   : best val_loss checkpoint
- last.ckpt                   : last epoch checkpoint
- norm_stats.json             : normalization stats (saved by DataModule)
- hparams.yaml                : all hyperparameters (auto-saved by Lightning)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from lightning.pytorch import LightningModule

from fsq_matched_controls.models.encoder import TemporalCNNEncoder
from fsq_matched_controls.models.decoder import TemporalCNNDecoder
from fsq_matched_controls.models.fsq import FSQ

log = logging.getLogger(__name__)


# ── Config dataclass ──────────────────────────────────────────────────────────

@dataclass
class FSQAutoencoderConfig:
    """All hyperparameters for the FSQ autoencoder.

    Designed to be serialisable and trackable by MLflow / W&B.
    """
    # Data shape
    n_bands: int = 10
    obs_per_window: int = 6

    # Encoder / Decoder
    encoder_channels: list[int] = field(default_factory=lambda: [32, 64, 128])
    kernel_size: int = 3
    dropout: float = 0.1
    # "batch" (default, BatchNorm1d) for fixed-cadence windows with no padding
    # (CropGlobe); "layer" (per-timestep LayerNorm) whenever windows can
    # contain padding (EuroCropsML's variable-observation windows) -- see
    # training/models/encoder.py's module docstring for why that matters.
    norm_type: str = "batch"

    # FSQ
    n_dims: int = 4
    n_levels: int = 5          # codebook_size = n_levels ** n_dims

    # Optimisation
    lr: float = 1e-3
    lr_scheduler: str = "plateau"     # "plateau" | "cosine" | "none"
    weight_decay: float = 1e-4
    max_epochs: int = 100             # used by CosineAnnealingLR (plateau ignores this)
    plateau_factor: float = 0.5       # ReduceLROnPlateau: LR multiplier on stagnation
    plateau_patience: int = 10        # ReduceLROnPlateau: epochs before LR reduction

    @property
    def codebook_size(self) -> int:
        return self.n_levels ** self.n_dims

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["codebook_size"] = self.codebook_size
        return d


# ── Lightning Module ──────────────────────────────────────────────────────────

class FSQAutoencoder(LightningModule):
    """Self-supervised FSQ autoencoder for phenological tokenisation.

    Parameters
    ----------
    config : ``FSQAutoencoderConfig`` instance or dict (auto-converted).
    """

    def __init__(self, config: FSQAutoencoderConfig | dict) -> None:
        super().__init__()

        if isinstance(config, dict):
            config = FSQAutoencoderConfig(**{k: v for k, v in config.items()
                                             if k != "codebook_size"})
        self.config = config
        # Save as {"config": {...}} so Lightning restores via __init__(config={...})
        self.save_hyperparameters({"config": config.to_dict()})

        # Model components
        self.encoder = TemporalCNNEncoder(
            n_bands=config.n_bands,
            channels=config.encoder_channels,
            kernel_size=config.kernel_size,
            n_dims=config.n_dims,
            n_levels=config.n_levels,
            dropout=config.dropout,
            norm_type=config.norm_type,
        )
        self.fsq = FSQ(n_dims=config.n_dims, n_levels=config.n_levels)
        self.decoder = TemporalCNNDecoder(
            n_dims=config.n_dims,
            channels=config.encoder_channels,   # decoder reverses internally
            kernel_size=config.kernel_size,
            n_bands=config.n_bands,
            obs_per_window=config.obs_per_window,
            dropout=config.dropout,
        )

        # Codebook utilisation tracking (accumulated per epoch)
        self._seen_codes: set[int] = set()

    # ── forward ──────────────────────────────────────────────────────────────

    def forward(self, x: Tensor, obs_mask: Optional[Tensor] = None) -> tuple[Tensor, Tensor, Tensor]:
        """Full autoencoder forward pass.

        Parameters
        ----------
        x        : (batch, obs_per_window, n_bands)
        obs_mask : (batch, obs_per_window) bool, optional -- True at real
                   (non-padded) observations. ``None`` (default) preserves
                   the original always-real-data behaviour (CropGlobe); see
                   ``TemporalCNNEncoder``'s docstring for what passing a mask
                   changes (only meaningful together with ``norm_type="layer"``).

        Returns
        -------
        x_hat      : (batch, obs_per_window, n_bands)  reconstructed window
        z_q        : (batch, n_dims)                    quantised code
        token_ids  : (batch,)                           integer token IDs
        """
        z = self.encoder(x, obs_mask=obs_mask)
        z_q, token_ids = self.fsq(z)
        x_hat = self.decoder(z_q)
        return x_hat, z_q, token_ids

    def encode(self, x: Tensor, obs_mask: Optional[Tensor] = None) -> tuple[Tensor, Tensor]:
        """Encode and quantise only (no decoding) — used for tokenisation.

        Returns
        -------
        z_q       : (batch, n_dims)
        token_ids : (batch,)
        """
        z = self.encoder(x, obs_mask=obs_mask)
        z_q, token_ids = self.fsq(z)
        return z_q, token_ids

    # ── shared step ──────────────────────────────────────────────────────────

    @staticmethod
    def _unpack_batch(batch: tuple) -> tuple[Tensor, Optional[Tensor]]:
        # (x, labels) for fixed-window datasets (CropGlobe); (x, labels,
        # obs_mask) for datasets with variable-observation windows (EuroCropsML).
        if len(batch) == 3:
            x, _, obs_mask = batch
            return x, obs_mask
        x, _ = batch                                        # labels unused in pretraining
        return x, None

    @staticmethod
    def _masked_mse(x_hat: Tensor, x: Tensor, obs_mask: Optional[Tensor]) -> tuple[Tensor, Tensor]:
        """MSE reconstruction loss + per-band MSE.

        With ``obs_mask`` given, the decoder has no way to know which of its
        fixed-shape output positions are supposed to be padding, so scoring
        it against the raw (zero-filled) padding would pressure it to
        reconstruct arbitrary padding zeros instead of real signal -- the
        loss is therefore only computed at real (non-padded) positions.
        """
        if obs_mask is None:
            loss = F.mse_loss(x_hat, x)
            per_band_mse = ((x_hat - x) ** 2).mean(dim=(0, 1))
            return loss, per_band_mse

        m = obs_mask.unsqueeze(-1).to(x.dtype)               # (B, obs, 1)
        sq_err = (x_hat - x) ** 2 * m                         # (B, obs, n_bands)
        n_real = m.sum().clamp(min=1e-6)
        loss = sq_err.sum() / (n_real * x.shape[-1])
        per_band_mse = sq_err.sum(dim=(0, 1)) / n_real
        return loss, per_band_mse

    def _step(self, batch: tuple, stage: str) -> Tensor:
        x, obs_mask = self._unpack_batch(batch)
        x_hat, _, token_ids = self(x, obs_mask=obs_mask)

        loss, per_band_mse = self._masked_mse(x_hat, x, obs_mask)
        with torch.no_grad():
            per_band_mse = per_band_mse.detach()

        # Log scalars
        self.log(f"{stage}_loss", loss, prog_bar=True, on_epoch=True, on_step=(stage == "train"))
        for b_idx, b_mse in enumerate(per_band_mse):
            self.log(f"{stage}_band{b_idx}_mse", b_mse, on_epoch=True, on_step=False)

        # Track codebook utilisation (only during training)
        if stage == "train":
            self._seen_codes.update(token_ids.detach().cpu().tolist())

        return loss

    def training_step(self, batch: tuple, batch_idx: int) -> Tensor:
        return self._step(batch, "train")

    def validation_step(self, batch: tuple, batch_idx: int) -> None:
        x, obs_mask = self._unpack_batch(batch)
        x_hat, _, _ = self(x, obs_mask=obs_mask)

        loss, _ = self._masked_mse(x_hat, x, obs_mask)
        self.log("val_loss", loss, prog_bar=True, on_epoch=True, on_step=False)

        # R² per band (batch-level accumulation is approximate but fast).
        # With obs_mask, both the residual and total sums of squares are
        # restricted to real positions for the same reason the loss is.
        with torch.no_grad():
            if obs_mask is None:
                ss_res = ((x - x_hat) ** 2).sum(dim=(0, 1))
                ss_tot = ((x - x.mean(dim=(0, 1), keepdim=True)) ** 2).sum(dim=(0, 1))
            else:
                m = obs_mask.unsqueeze(-1).to(x.dtype)
                n_real = m.sum(dim=(0, 1)).clamp(min=1e-6)
                x_mean = (x * m).sum(dim=(0, 1)) / n_real     # (n_bands,) -- mean over real positions only
                ss_res = (((x - x_hat) ** 2) * m).sum(dim=(0, 1))
                ss_tot = (((x - x_mean) ** 2) * m).sum(dim=(0, 1))
            r2 = 1 - ss_res / (ss_tot + 1e-8)
            self.log("val_r2", r2.mean(), on_epoch=True, on_step=False)

    def test_step(self, batch: tuple[Tensor, Tensor], batch_idx: int) -> None:
        self._step(batch, "test")

    def on_train_epoch_end(self) -> None:
        """Log codebook utilisation at the end of each epoch."""
        utilisation = len(self._seen_codes) / self.config.codebook_size
        self.log("codebook_utilisation", utilisation, on_epoch=True)
        self._seen_codes.clear()

    # ── optimiser ────────────────────────────────────────────────────────────

    def configure_optimizers(self) -> dict:
        opt = torch.optim.AdamW(
            self.parameters(),
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
        )
        if self.config.lr_scheduler == "plateau":
            sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
                opt, mode="min",
                factor=self.config.plateau_factor,
                patience=self.config.plateau_patience,
            )
            return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "monitor": "val_loss"}}
        elif self.config.lr_scheduler == "cosine":
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt, T_max=self.config.max_epochs, eta_min=self.config.lr * 0.01
            )
            return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": "epoch"}}
        return {"optimizer": opt}

    # ── convenience ──────────────────────────────────────────────────────────

    @classmethod
    def load(cls, checkpoint_path: Path | str) -> "FSQAutoencoder":
        """Load from a Lightning checkpoint.

        Handles both old flat hparams format ``{n_bands: 10, …}`` and the
        current nested format ``{config: {n_bands: 10, …}}``.
        Config is always passed explicitly so the stored hparams are ignored.
        """
        ckpt = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
        hparams = ckpt.get("hyper_parameters", {})

        # Normalise to a flat dict of FSQAutoencoderConfig fields
        if "config" in hparams:
            flat = hparams["config"]          # new nested format
        else:
            flat = hparams                    # old flat format

        config = FSQAutoencoderConfig(
            **{k: v for k, v in flat.items() if k != "codebook_size"}
        )
        return cls.load_from_checkpoint(str(checkpoint_path), config=config)

