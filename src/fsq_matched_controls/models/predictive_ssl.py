"""FSQ Predictive SSL — Masked Token Prediction over Phenological Sequences.

Motivation
----------
The reconstruction SSL (autoencoder) minimises MSE between the reconstructed
and original spectral window.  This forces the code to retain atmospheric /
soil-brightness noise that is irrelevant to phenology and *not* shared across
regions — the exact opposite of what cross-region invariance requires.

Predictive SSL (this module) instead learns to predict *missing tokens* from
*surrounding tokens*.  Because phenological grammar (green-up → peak → senescence)
is approximately the same for the same crop regardless of country, the model
is implicitly incentivised to map the same phenological state to the same token
in all regions.

⚠️  Collapse risk & the BEiT-style fix
---------------------------------------
In the default end-to-end mode the same encoder produces BOTH:
  (a) the ground-truth token IDs used as CE targets, and
  (b) the z_q embeddings fed into the context transformer via straight-through.

This gives the network a trivial collapse solution: encode every window to the
same token → perfect CE loss → zero information in the representation.
The same failure mode required BYOL/SimSiam/DINO to use momentum encoders and
data2vec / HuBERT to use fully offline target generators.

**Safe alternative — BEiT-style frozen tokenizer (recommended):**

  Stage 1 (already done): train reconstruction FSQ autoencoder (train.py).
  Stage 2 (this script): freeze encoder + FSQ completely; train ONLY the
  context transformer on masked token prediction.

  The frozen tokenizer acts exactly like BPE in BERT — fixed vocabulary
  provider, zero gradient leakage.  The CE loss shapes only the context
  transformer's representations.  The invariance argument is fully preserved.

  Activate via ``use_frozen_tokenizer=True`` in the config, together with
  a ``tokenizer_ckpt`` path pointing to the reconstruction SSL checkpoint.
  Training script: pass ``--tokenizer-ckpt <path>`` on the CLI.

Modes
-----
* ``use_frozen_tokenizer=False`` (default):
    End-to-end trainable.  Fast to set up, but collapse is possible.
    Monitor ``codebook_utilisation``; if it drops below ~10 % within the
    first 10 epochs, collapse has occurred — switch to frozen mode.

* ``use_frozen_tokenizer=True``:
    BEiT-style.  Encoder + FSQ loaded from a reconstruction checkpoint and
    frozen.  Only the context transformer is optimised.  Collapse-proof.

Pipeline
--------
[W_1 … W_N]  →  CNN encoder  →  [z_1 … z_N]  →  FSQ  →  [t_1 … t_N]
                                                                 ↓
                                           mask 15 % of tokens  →  [t1, t2, MASK, t4, …]
                                                                 ↓
                                           small context Transformer
                                                                 ↓
                                           predict masked token IDs
Loss: CrossEntropy(predicted_token_id, true_token_id)

Gradient path (end-to-end):  CE → transformer → unmasked z_q → straight-through → encoder
Gradient path (frozen):       CE → transformer only  (encoder/FSQ receive zero gradient)

Compatibility
-------------
The ``encode(x)`` method accepts the same single-window input as
``FSQAutoencoder.encode()``, so the existing ``tokenize_parcel_dataset``
evaluation function works without modification.

Logged metrics
--------------
- train_loss / val_loss / test_loss : CrossEntropy on masked positions
- train_acc  / val_acc  / test_acc  : Top-1 accuracy on masked positions
- codebook_utilisation              : % of codes seen per epoch (true tokens)
- mask_ratio                        : logged as constant
- tokenizer_frozen                  : 1.0 if BEiT-mode, 0.0 if end-to-end

Artefacts (experiment_dir)
--------------------------
- best.ckpt   : best val_loss checkpoint
- last.ckpt   : last epoch checkpoint
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from lightning.pytorch import LightningModule

from fsq_matched_controls.models.encoder import TemporalCNNEncoder
from fsq_matched_controls.models.fsq import FSQ

log = logging.getLogger(__name__)

#: Largest joint code space for which exact joint metrics (distinct-code count,
#: joint top-5 accuracy) are still cheap to materialise.  Above this the
#: factorized head logs per-dimension entropy only.  625 (4d x 5L) and 3125
#: (5d x 5L) are under it; 5^8 = 390625 is not.
JOINT_METRIC_MAX_CODES: int = 4096


# ── Config dataclass ──────────────────────────────────────────────────────────

@dataclass
class FSQPredictiveSSLConfig:
    """All hyperparameters for the FSQ predictive SSL model.

    Designed to be serialisable and trackable by MLflow / W&B.
    Keep all fields JSON-serialisable (no nested objects).
    """
    # ── Data shape ────────────────────────────────────────────────────────────
    n_bands: int = 10
    obs_per_window: int = 6       # observations per calendar-month window
    n_tokens: int = 12            # sequence length (months per parcel); set from data

    # ── Encoder (same CNN backbone as autoencoder) ────────────────────────────
    encoder_channels: list[int] = field(default_factory=lambda: [32, 64, 128])
    kernel_size: int = 3
    dropout: float = 0.1
    # "batch" (default, BatchNorm1d) for fixed-cadence windows with no padding
    # (CropGlobe); "layer" (per-timestep LayerNorm) whenever windows can
    # contain padding (EuroCropsML's variable-observation windows) -- see
    # training/models/encoder.py's module docstring for why that matters.
    norm_type: str = "batch"

    # ── FSQ vocabulary ──────────────────────────────────���─────────────────────
    n_dims: int = 4               # latent dimensionality
    n_levels: int = 5             # levels per dim → 5^4 = 625 codes

    # ── Prediction head parameterisation ──────────────────────────────────────
    # False (default): single joint softmax over all ``codebook_size`` codes.
    #   Exact, but the output layer grows as n_levels ** n_dims, so it cannot
    #   scale past ~5-6 dims (5^8 = 390k logits; 5^16 = 1.5e11).
    # True (factorized): one softmax of size ``n_levels`` per latent dimension,
    #   loss = sum of per-dimension cross-entropies. The output layer then grows
    #   as n_dims * n_levels, i.e. linearly in width. This models the joint code
    #   distribution as a product of independent per-dimension distributions --
    #   strictly less expressive than the joint softmax, which is why the
    #   validation gate must confirm it costs nothing at 4 dims before any
    #   width sweep is run off it.
    factorized_head: bool = False

    # "joint" (default) | "factorized" | "autoregressive".
    # Takes precedence over the legacy ``factorized_head`` flag when set to a
    # non-default value; ``factorized_head: true`` maps to "factorized" so
    # existing configs and checkpoints keep working.
    head_mode: str = "joint"

    # ── Masking ───────────────────────────────────────────────────────────────
    mask_ratio: float = 0.15      # fraction of tokens to mask per sequence
    min_masks: int = 1            # guarantee at least this many masked per sequence

    # ── Context Transformer ───────────────────────────────────────────────────
    d_model: int = 128            # transformer hidden dimension
    n_heads: int = 4              # attention heads (d_model must be divisible)
    n_layers: int = 3             # number of TransformerEncoderLayer blocks
    dim_feedforward: int = 256    # feed-forward inner dimension
    transformer_dropout: float = 0.1
    max_seq_len: int = 100        # max supported sequence length (for pos embedding)

    # ── Optimisation ──────────────────────────────────────────────────────────
    lr: float = 1e-3
    lr_scheduler: str = "plateau"  # "plateau" | "cosine" | "none"
    weight_decay: float = 1e-4
    max_epochs: int = 100
    plateau_factor: float = 0.5
    plateau_patience: int = 10

    # ── Collapse prevention ────────────────────────────────────────────────────
    # False (default): end-to-end trainable encoder+FSQ — fast but collapse-prone.
    # True  (BEiT):   encoder+FSQ frozen from a reconstruction checkpoint;
    #                  only the context transformer is optimised.  Collapse-proof.
    #                  Requires the model to be built via
    #                  FSQPredictiveSSL.from_pretrained_tokenizer().
    use_frozen_tokenizer: bool = False

    @property
    def resolved_head_mode(self) -> str:
        if self.head_mode != "joint":
            return self.head_mode
        return "factorized" if self.factorized_head else "joint"

    @property
    def codebook_size(self) -> int:
        return self.n_levels ** self.n_dims

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["codebook_size"] = self.codebook_size
        return d


# ── Context Transformer ───────────────────────────────────────────────────────

class ContextTransformer(nn.Module):
    """Small transformer that predicts masked FSQ token IDs from context.

    Architecture
    ------------
    1. ``input_proj``:  Linear(n_dims, d_model) — project FSQ z_q to model dim.
    2. ``mask_token``:  Learned embedding vector replacing masked positions.
    3. ``pos_emb``:     Learned positional embedding (up to max_seq_len positions).
    4. Transformer encoder (Pre-LN, batch_first) — contextual mixing.
    5. ``head``:        Linear(d_model, codebook_size) — token ID logits.

    Forward pass
    ------------
    z_q  : (B, N, n_dims)   — quantised FSQ embeddings for all positions
    mask : (B, N)           — True at positions to be predicted (masked)

    Returns logits : (B, N, codebook_size)
    """

    def __init__(
        self,
        n_dims: int,
        codebook_size: int,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 3,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        max_seq_len: int = 100,
        factorized_head: bool = False,
        n_levels: int = 5,
        head_mode: str = "joint",
    ) -> None:
        super().__init__()

        self.d_model = d_model
        self.n_dims = n_dims
        self.n_levels = n_levels
        self.head_mode = head_mode
        self.factorized_head = factorized_head
        self.input_proj = nn.Linear(n_dims, d_model)

        if head_mode == "autoregressive":
            # Chain-rule head: p(level_d | context, TRUE levels of dims < d).
            # Unlike the factorized head this makes no independence assumption —
            # the chain rule is exact — while the output layer still grows as
            # n_dims * n_levels rather than n_levels ** n_dims.
            #
            # Computed in ONE PARALLEL PASS, not a sequential loop: the prefix
            # encoding for every dimension is an exclusive cumulative sum over
            # per-(dim, level) embeddings of the true levels, so all n_dims
            # conditionals are produced by a single batched MLP call.
            self.level_emb = nn.Embedding(n_dims * n_levels, d_model)
            self.dim_emb = nn.Embedding(n_dims, d_model)

        # Learnable MASK token embedding (replaces masked z_q → input_proj)
        self.mask_token = nn.Parameter(torch.zeros(d_model))
        nn.init.normal_(self.mask_token, std=0.02)

        # Learned positional embedding
        self.pos_emb = nn.Embedding(max_seq_len, d_model)

        # Pre-LayerNorm transformer encoder (more stable, common in modern BERT variants)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,   # Pre-LN
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Prediction head.  Joint mode: d_model → codebook_size logits.
        # Factorized mode: d_model → n_dims * n_levels, reshaped in forward()
        # to (B, N, n_dims, n_levels) -- one independent softmax per dimension.
        # Everything before the final Linear is identical in both modes, so the
        # two heads differ only in output parameterisation.
        # joint          : one softmax over the whole code space
        # factorized     : n_dims independent softmaxes, emitted in one vector
        # autoregressive : ONE softmax of size n_levels, applied per dimension to
        #                  a (context + prefix + dim) input, so dimensions are
        #                  distinguished by ``dim_emb`` rather than by separate
        #                  output blocks
        if head_mode == "autoregressive":
            out_features = n_levels
        elif factorized_head or head_mode == "factorized":
            out_features = n_dims * n_levels
        else:
            out_features = codebook_size
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
            nn.Linear(d_model, out_features),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.trunc_normal_(m.weight, std=0.02)

    def forward(self, z_q: Tensor, mask: Tensor,
                true_levels: Optional[Tensor] = None) -> Tensor:
        """
        Parameters
        ----------
        z_q  : (B, N, n_dims)  — quantised FSQ embeddings (all positions)
        mask : (B, N)          — True at positions that should be masked / predicted
        true_levels : (B, N, n_dims) long — required only by the autoregressive
                      head, which conditions each dimension on the true levels
                      of earlier dimensions (teacher forcing).

        Returns
        -------
        logits : (B, N, codebook_size)            if joint head
                 (B, N, n_dims, n_levels)         if factorized head
        """
        B, N, _ = z_q.shape

        # Project all positions to d_model
        emb_encoded = self.input_proj(z_q)   # (B, N, d_model)

        # Replace masked positions with the learned MASK token
        # torch.where ensures correct gradient routing:
        #   masked positions  → gradient to mask_token parameter
        #   unmasked positions → gradient to encoder / z_q
        mask_emb = self.mask_token.view(1, 1, self.d_model).expand(B, N, self.d_model)
        mask_3d = mask.unsqueeze(-1).expand(B, N, self.d_model)
        emb = torch.where(mask_3d, mask_emb, emb_encoded)          # (B, N, d_model)

        # Add positional embedding
        positions = torch.arange(N, device=z_q.device)              # (N,)
        emb = emb + self.pos_emb(positions)                          # (B, N, d_model)

        # Contextual mixing
        ctx = self.transformer(emb)                                  # (B, N, d_model)

        if self.head_mode == "autoregressive":
            return self._ar_logits(ctx, true_levels)

        # Predict token logits at every position (loss only applied at masked ones)
        logits = self.head(ctx)
        if self.factorized_head or self.head_mode == "factorized":
            logits = logits.view(B, N, self.n_dims, self.n_levels)
        return logits

    def _ar_logits(self, ctx: Tensor, true_levels: Tensor) -> Tensor:
        """Chain-rule conditionals for every dimension, in one parallel pass.

        Parameters
        ----------
        ctx         : (B, N, d_model)         — contextual representation
        true_levels : (B, N, n_dims) long     — teacher-forced true level indices

        Returns
        -------
        logits : (B, N, n_dims, n_levels) where ``logits[..., d, :]`` are the
                 logits of p(level_d | ctx, true levels of dims 0..d-1).

        The prefix for dimension d must contain dims strictly before d.  That is
        an *exclusive* cumulative sum, obtained by cumsum-then-shift, so all
        dimensions are computed simultaneously.  Dimension 0 receives a zero
        prefix, i.e. it is conditioned on context alone.
        """
        B, N, _ = ctx.shape
        D, L = self.n_dims, self.n_levels
        # per-(dim, level) embedding of each true level
        offs = torch.arange(D, device=ctx.device) * L                 # (D,)
        emb = self.level_emb(true_levels + offs)                      # (B, N, D, d_model)
        # exclusive cumulative sum over the dimension axis
        incl = emb.cumsum(dim=2)
        prefix = torch.zeros_like(incl)
        prefix[:, :, 1:] = incl[:, :, :-1]                            # (B, N, D, d_model)

        dims = self.dim_emb(torch.arange(D, device=ctx.device))       # (D, d_model)
        h = ctx.unsqueeze(2) + prefix + dims                          # (B, N, D, d_model)
        return self.head(h)                                           # (B, N, D, L)


# ── Lightning Module ──────────────────────────────────────────────────────────

class FSQPredictiveSSL(LightningModule):
    """Self-supervised FSQ model with masked token prediction (BERT-style).

    Input shape (training)  : (batch_parcels, n_tokens, obs_per_window, n_bands)
    Input shape (evaluation): (batch_windows, obs_per_window, n_bands)  [via encode()]

    The model can be used as a drop-in replacement for ``FSQAutoencoder`` in
    the evaluation pipeline: ``tokenize_parcel_dataset`` works unchanged because
    ``encode()`` accepts the same single-window signature.

    Parameters
    ----------
    config : ``FSQPredictiveSSLConfig`` instance or dict (auto-converted).
    """

    def __init__(self, config: FSQPredictiveSSLConfig | dict) -> None:
        super().__init__()
        if isinstance(config, dict):
            config = FSQPredictiveSSLConfig(**{
                k: v for k, v in config.items() if k != "codebook_size"
            })
        self.config = config
        self.save_hyperparameters({"config": config.to_dict()})

        # ── CNN encoder (identical architecture to FSQAutoencoder) ────────────
        self.encoder = TemporalCNNEncoder(
            n_bands=config.n_bands,
            channels=config.encoder_channels,
            kernel_size=config.kernel_size,
            n_dims=config.n_dims,
            n_levels=config.n_levels,
            dropout=config.dropout,
            norm_type=config.norm_type,
        )

        # ── FSQ quantiser ──────────────────────────────────────────────────────
        self.fsq = FSQ(n_dims=config.n_dims, n_levels=config.n_levels)

        # ── Context transformer for masked token prediction ───────────────────
        self.context_transformer = ContextTransformer(
            n_dims=config.n_dims,
            codebook_size=config.codebook_size,
            d_model=config.d_model,
            n_heads=config.n_heads,
            n_layers=config.n_layers,
            dim_feedforward=config.dim_feedforward,
            dropout=config.transformer_dropout,
            max_seq_len=config.max_seq_len,
            factorized_head=config.factorized_head,
            n_levels=config.n_levels,
            head_mode=config.resolved_head_mode,
        )

        # ── Utilisation tracking ──────────────────────────────────────────────
        # Per-dimension level histogram (n_dims, n_levels), accumulated over an
        # epoch.  This is the width-independent collapse diagnostic: a dimension
        # that has collapsed to a constant has zero entropy regardless of how
        # many dimensions there are.  The old joint ``codebook_utilisation``
        # ratio is meaningless past ~6 dims (at 16 dims the denominator is
        # 1.5e11 while the dataset can only ever expose ~1e6 distinct codes),
        # so it is only logged when the code space is small enough to count.
        #
        # NOTE: deliberately a plain attribute, not a registered buffer --
        # registering it would add a state_dict key and break strict loading of
        # the existing joint-head checkpoints (seeds 1904/2024/4242).
        self._level_counts = torch.zeros(config.n_dims, config.n_levels, dtype=torch.long)
        # Distinct-code tracking as a boolean presence vector rather than a
        # Python set, so nothing leaves the GPU mid-epoch.  Only allocated when
        # the code space is small enough to be worth counting at all.
        self._seen_mask = torch.zeros(
            config.codebook_size if config.codebook_size <= JOINT_METRIC_MAX_CODES else 0,
            dtype=torch.bool,
        )

    # ── Tokenizer freezing (BEiT-style collapse prevention) ──────────────────

    @property
    def tokenizer_frozen(self) -> bool:
        """True when encoder + FSQ are frozen (BEiT-style mode)."""
        return not next(self.encoder.parameters()).requires_grad

    def freeze_tokenizer(self) -> None:
        """Freeze encoder and FSQ in-place, forcing them into eval mode.

        After calling this, only the context transformer (input_proj, mask_token,
        pos_emb, transformer, head) is updated by the optimiser.  Call this
        before ``trainer.fit()`` to activate BEiT-style training.

        Gradient isolation is automatic: frozen parameters have
        ``requires_grad=False``, so PyTorch never builds the computation graph
        through them.  No explicit ``torch.no_grad()`` context manager is needed
        in ``forward()``.

        ``requires_grad_(False)`` alone is not enough: BatchNorm/Dropout behaviour
        is gated by ``module.training``, not ``requires_grad``.  Without also
        forcing ``eval()``, the "frozen" encoder would still normalise using
        live mini-batch statistics (instead of the Stage-1 running stats) and
        still apply stochastic dropout, making the supposedly-fixed tokenizer
        non-deterministic.  ``train()`` below re-asserts eval mode on every
        Lightning-triggered epoch-start ``.train()`` call so this doesn't get
        silently undone.
        """
        for p in self.encoder.parameters():
            p.requires_grad_(False)
        for p in self.fsq.parameters():
            p.requires_grad_(False)
        self.encoder.eval()
        self.fsq.eval()
        log.info(
            "Tokenizer frozen (BEiT-style) and set to eval mode.  "
            "Only context transformer will be optimised."
        )

    def train(self, mode: bool = True) -> "FSQPredictiveSSL":
        """Keep a frozen tokenizer in eval mode across Lightning's epoch loop.

        ``Trainer`` calls ``pl_module.train()`` at the start of every training
        epoch, which would otherwise flip the frozen encoder/FSQ back into
        train mode (live BatchNorm stats, active dropout) even though their
        parameters are frozen.  See ``freeze_tokenizer`` for why that matters.
        """
        super().train(mode)
        if mode and self.tokenizer_frozen:
            self.encoder.eval()
            self.fsq.eval()
        return self

    @classmethod
    def from_pretrained_tokenizer(
        cls,
        config: FSQPredictiveSSLConfig | dict,
        tokenizer_ckpt: str | Path,
    ) -> "FSQPredictiveSSL":
        """Build FSQPredictiveSSL with encoder + FSQ loaded from a reconstruction
        checkpoint and immediately frozen (BEiT-style, collapse-proof).

        The reconstruction checkpoint can come from ``train.py``
        (``FSQAutoencoder``) or ``train_backbone.py`` (``FSQBackboneAutoencoder``
        with ``d_pre == n_dims``).  Only encoder and FSQ weights are transferred;
        the decoder is discarded.

        Parameters
        ----------
        config          : ``FSQPredictiveSSLConfig`` or equivalent dict.
        tokenizer_ckpt  : Path to the reconstruction SSL ``.ckpt`` file.

        Returns
        -------
        model : ``FSQPredictiveSSL`` with frozen tokenizer and randomly
                initialised context transformer.

        Example
        -------
        ::

            model = FSQPredictiveSSL.from_pretrained_tokenizer(
                config=my_cfg,
                tokenizer_ckpt="experiments/recon/checkpoints/best.ckpt",
            )
            trainer.fit(model, train_dataloaders=..., val_dataloaders=...)
        """
        from fsq_matched_controls.models.autoencoder import FSQAutoencoder

        if isinstance(config, dict):
            config = FSQPredictiveSSLConfig(**{
                k: v for k, v in config.items() if k != "codebook_size"
            })

        # Force the flag in config so it is saved in the checkpoint hparams
        config.use_frozen_tokenizer = True

        model = cls(config)

        log.info("Loading tokenizer weights from: %s", tokenizer_ckpt)
        ae = FSQAutoencoder.load(tokenizer_ckpt)

        # Validate architecture compatibility
        if ae.config.n_dims != config.n_dims:
            raise ValueError(
                f"n_dims mismatch: tokenizer has {ae.config.n_dims}, "
                f"config has {config.n_dims}."
            )
        if ae.config.n_levels != config.n_levels:
            raise ValueError(
                f"n_levels mismatch: tokenizer has {ae.config.n_levels}, "
                f"config has {config.n_levels}."
            )

        model.encoder.load_state_dict(ae.encoder.state_dict())
        model.fsq.load_state_dict(ae.fsq.state_dict())
        model.freeze_tokenizer()

        log.info(
            "Tokenizer loaded and frozen.  "
            "Trainable params: context transformer only (%d params).",
            sum(p.numel() for p in model.parameters() if p.requires_grad),
        )
        return model

    # ── Core forward pass ───────────────────────────���────────────────────────

    def forward(
        self,
        x: Tensor,
        mask: Optional[Tensor] = None,
        valid_mask: Optional[Tensor] = None,
        obs_mask: Optional[Tensor] = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Full predictive SSL forward pass over a parcel sequence.

        Parameters
        ----------
        x          : (B, N, obs_per_window, n_bands)  — full parcel sequence
        mask       : (B, N) bool, optional  — True at positions to mask.
                     If None, random masking is sampled at ``config.mask_ratio``
                     (restricted to ``valid_mask`` positions if given).
        valid_mask : (B, N) bool, optional  — True where the position (month)
                     holds real data.  ``None`` means every position is real
                     (CropGlobe: a full calendar grid, no missing months).
                     Datasets with structurally missing positions (e.g.
                     EuroCropsML months with zero cloud-free observations)
                     pass this so those positions are never sampled as
                     supervision targets, yet are still always hidden from
                     the transformer as masked context (never fed in as if
                     they were real content).
        obs_mask   : (B, N, obs_per_window) bool, optional — True at real
                     (non-padded) observations *within* each window. ``None``
                     (default) preserves the original always-real-data
                     behaviour (CropGlobe). Distinct from ``valid_mask``,
                     which operates at the whole-window (month) level; this
                     one operates within a window, for datasets where a
                     window itself is a variable-length, padded sequence of
                     observations (EuroCropsML) — see
                     ``TemporalCNNEncoder``'s docstring.

        Returns
        -------
        logits         : (B, N, codebook_size)  — token ID prediction logits
        token_ids      : (B, N)  — true quantised token IDs (meaningless at
                         invalid positions — see ``valid_mask``)
        input_mask     : (B, N)  — positions replaced by the MASK token in
                         the transformer input (random mask ∪ invalid positions)
        supervise_mask : (B, N)  — positions the CE loss/accuracy are
                         computed over (random mask ∩ valid positions)
        """
        B, N, obs, bands = x.shape

        # ── 1. Encode all windows independently ──────────────────────────────
        # Reshape to (B*N, obs, bands) so the window-level CNN processes in one pass
        x_flat = x.reshape(B * N, obs, bands)
        obs_mask_flat = obs_mask.reshape(B * N, obs) if obs_mask is not None else None
        z_flat = self.encoder(x_flat, obs_mask=obs_mask_flat)  # (B*N, n_dims)
        z_q_flat, ids_flat = self.fsq(z_flat)               # (B*N, n_dims), (B*N,)

        z_q = z_q_flat.view(B, N, -1)                       # (B, N, n_dims)
        token_ids = ids_flat.view(B, N)                     # (B, N)

        # ── 2. Random masking (if not provided) ───────────────────────────────
        if mask is None:
            mask = self._sample_mask(B, N, device=x.device, valid_mask=valid_mask)  # (B, N) bool

        if valid_mask is None:
            input_mask = mask
            supervise_mask = mask
        else:
            # Missing positions are always hidden from the transformer (like a
            # training-time mask) but never supervised on (no ground truth).
            input_mask = mask | ~valid_mask
            supervise_mask = mask & valid_mask

        # ── 3. Context transformer: predict token IDs at masked positions ─────
        # The autoregressive head needs the true per-dimension levels for
        # teacher forcing.  They are derived from the same token_ids the joint
        # head trains against, so all three heads share identical supervision.
        true_levels = None
        if self.config.resolved_head_mode == "autoregressive":
            true_levels = (self.fsq.decode_token_ids(token_ids.reshape(-1))
                           + self.fsq._half).long().view(B, N, -1)
        logits = self.context_transformer(z_q, input_mask, true_levels)

        return logits, token_ids, input_mask, supervise_mask

    def encode(self, x: Tensor, obs_mask: Optional[Tensor] = None) -> tuple[Tensor, Tensor]:
        """Encode a batch of single windows — compatible with FSQAutoencoder.encode().

        This method is called by ``tokenize_parcel_dataset`` during evaluation.

        Parameters
        ----------
        x        : (batch, obs_per_window, n_bands)
        obs_mask : (batch, obs_per_window) bool, optional -- see ``forward``.

        Returns
        -------
        z_q       : (batch, n_dims)
        token_ids : (batch,)
        """
        z = self.encoder(x, obs_mask=obs_mask)
        z_q, token_ids = self.fsq(z)
        return z_q, token_ids

    def encode_sequence(self, x: Tensor, obs_mask: Optional[Tensor] = None) -> Tensor:
        """Context transformer hidden states for a full parcel sequence (no masking).

        All token positions are visible — no MASK tokens are inserted.
        Returns the contextually-enriched representation at every position,
        which is what distinguishes PRED features from plain EMB features.

        Use inside ``torch.no_grad()`` with the model in ``eval()`` mode.

        Parameters
        ----------
        x        : (B, N, obs_per_window, n_bands)
        obs_mask : (B, N, obs_per_window) bool, optional -- see ``forward``.

        Returns
        -------
        ctx : (B, N, d_model) — context-aware representations for all positions

        Example
        -------
        ::

            model.eval()
            with torch.no_grad():
                ctx = model.encode_sequence(windows)   # (B, N, d_model)
            feats = ctx.cpu().numpy().reshape(B, -1)   # (B, N * d_model)
        """
        B, N, obs, bands = x.shape
        x_flat = x.reshape(B * N, obs, bands)
        obs_mask_flat = obs_mask.reshape(B * N, obs) if obs_mask is not None else None

        z_flat = self.encoder(x_flat, obs_mask=obs_mask_flat)  # (B*N, n_dims)
        z_q_flat, _ = self.fsq(z_flat)          # (B*N, n_dims)
        z_q = z_q_flat.view(B, N, -1)           # (B, N, n_dims)

        # Project to d_model
        emb = self.context_transformer.input_proj(z_q)           # (B, N, d_model)

        # Add positional embedding
        positions = torch.arange(N, device=x.device)
        emb = emb + self.context_transformer.pos_emb(positions)  # (B, N, d_model)

        # Context mixing — all positions visible, no masking
        ctx = self.context_transformer.transformer(emb)          # (B, N, d_model)
        return ctx

    # ── Masking helpers ───────────────────────────────────────────────────────

    def _sample_mask(
        self,
        B: int,
        N: int,
        device: torch.device,
        valid_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Sample a random binary mask with at least ``min_masks`` True entries.

        Parameters
        ----------
        B : batch size
        N : sequence length (n_tokens)
        valid_mask : (B, N) bool, optional — True where real data exists.
            The random mask is drawn only from valid positions per row
            (masking a position with no ground-truth token would be
            pointless).  ``None`` means every position is valid.

        Returns
        -------
        mask : (B, N) bool  — True at positions to be masked

        Notes
        -----
        Vectorised across the batch.  The previous implementation looped over
        rows with a ``torch.randperm`` per row, which cost ~11 ms per step at
        batch 64 — 28 % of the whole training step, and growing *linearly* with
        batch size (164 ms at batch 1024), which is what made large batches
        pointless for this model.  This version is 0.31 ms at any batch size.
        The sampled distribution is unchanged: each row still gets exactly
        ``n_mask`` positions drawn uniformly without replacement from its valid
        positions.  Only the RNG stream differs, which is seed-level variation.
        """
        valid = (
            torch.ones(B, N, dtype=torch.bool, device=device)
            if valid_mask is None else valid_mask
        )
        n_valid = valid.sum(dim=1)                                    # (B,)
        n_mask = torch.clamp(
            torch.ceil(n_valid * self.config.mask_ratio).long(),
            min=self.config.min_masks,
        )
        # keep >=1 valid unmasked position for context; rows with <=1 valid
        # position get 0 masks (nothing usable to supervise)
        n_mask = torch.minimum(n_mask, (n_valid - 1).clamp(min=0))

        # Rank valid positions by a random score; invalid positions sort last
        # (+inf) so they can never be selected.
        scores = torch.rand(B, N, device=device).masked_fill(~valid, float("inf"))
        order = scores.argsort(dim=1)
        ranks = torch.empty_like(order)
        ranks.scatter_(1, order, torch.arange(N, device=device).expand(B, N))
        return ranks < n_mask.unsqueeze(1)

    # ── Shared step ──────────────────────────────────────────────────────────

    def _step(self, batch: tuple, stage: str) -> Tensor:
        """Compute CE loss and accuracy at masked positions.

        Accepts parcel-level batches: ``(windows, labels, countries)`` where
        ``windows`` is ``(B, N, obs, n_bands)``; ``(windows, labels,
        countries, valid_mask)`` for datasets with structurally missing
        whole-window positions, ``valid_mask`` ``(B, N)`` bool; or
        ``(windows, labels, countries, valid_mask, obs_mask)`` for datasets
        where a window itself is a variable-length padded sequence of
        observations, ``obs_mask`` ``(B, N, obs)`` bool (EuroCropsML).
        """
        obs_mask = None
        if len(batch) == 5:
            windows, _, _, valid_mask, obs_mask = batch       # labels & countries unused
        elif len(batch) == 4:
            windows, _, _, valid_mask = batch                # labels & countries unused
        else:
            windows, _, _ = batch                             # labels & countries unused
            valid_mask = None

        logits, token_ids, _input_mask, supervise_mask = self(
            windows, valid_mask=valid_mask, obs_mask=obs_mask
        )
        # logits         : (B, N, codebook_size)
        # token_ids      : (B, N)
        # supervise_mask : (B, N) bool — CE/accuracy computed here only

        # ── Loss: CE at supervised positions only (random mask ∩ valid data) ──
        n_masked = supervise_mask.sum()
        factorized = self.config.resolved_head_mode in ("factorized", "autoregressive")
        dim_acc = None
        if n_masked == 0:
            # Edge case: no supervisable tokens (e.g. a row with ≤1 valid position)
            loss = logits.sum() * 0.0   # differentiable zero
            acc = torch.zeros(1, device=windows.device)[0]
            top5_acc = torch.zeros(1, device=windows.device)[0]
        elif not factorized:
            masked_logits  = logits[supervise_mask]           # (M, codebook_size)
            masked_targets = token_ids[supervise_mask]        # (M,)
            loss = F.cross_entropy(masked_logits, masked_targets)
            with torch.no_grad():
                acc = (masked_logits.argmax(-1) == masked_targets).float().mean()
                # ── Top-5 Accuracy Calculation ────────────────────────────────
                # Get the indices of the top 5 highest logits
                _, top5_indices = masked_logits.topk(5, dim=-1) # (M, 5)
                # Check if the target is anywhere along the top 5 dimension
                top5_correct = (top5_indices == masked_targets.unsqueeze(-1)).any(dim=-1) # (M,)
                top5_acc = top5_correct.float().mean()
        else:
            # ── Factorized head ───────────────────────────────────────────────
            masked_logits  = logits[supervise_mask]           # (M, n_dims, n_levels)
            masked_targets = token_ids[supervise_mask]        # (M,)
            M, D, L = masked_logits.shape

            # Per-dimension level targets, recovered from the SAME token IDs the
            # joint head is trained against.  The mixed-radix encoding in
            # FSQ._to_token_ids is bijective, so decode_token_ids round-trips
            # exactly; deriving targets this way (rather than re-deriving them
            # from z_q independently) guarantees the two heads are trained on
            # identical supervision and differ only in parameterisation.
            level_targets = (
                self.fsq.decode_token_ids(masked_targets) + self.fsq._half
            ).long()                                           # (M, n_dims)

            # loss = sum over dims of (mean over masked positions) CE.
            # cross_entropy(reduction="mean") averages over M*D, so scaling by D
            # recovers the per-dimension sum -- i.e. the negative log-likelihood
            # of the factorized joint, on the same scale as the joint head's CE.
            loss = F.cross_entropy(
                masked_logits.reshape(M * D, L), level_targets.reshape(M * D)
            ) * D

            with torch.no_grad():
                correct_per_dim = masked_logits.argmax(-1) == level_targets  # (M, D)
                dim_acc = correct_per_dim.float().mean()
                # Joint exact-match: ALL dimensions correct at that position.
                # This is the quantity comparable to the joint head's ``acc``
                # (top-1 over joint codes) -- per-dimension accuracy is NOT,
                # and substituting it would silently invalidate the gate.
                acc = correct_per_dim.all(dim=-1).float().mean()

                if (self.config.resolved_head_mode == "autoregressive"
                        or self.config.codebook_size > JOINT_METRIC_MAX_CODES):
                    # Joint top-5 is not computed for the autoregressive head.
                    # Its joint log-prob for an arbitrary code needs the
                    # conditionals under that code's own prefix, so materialising
                    # all codes would require one head evaluation per distinct
                    # prefix ((L^D - 1)/(L - 1) of them) at every step. The
                    # summed-CE loss is already the exact joint NLL, so
                    # ``val_loss`` stays directly comparable to the joint head
                    # without this diagnostic.
                    top5_acc = torch.zeros(1, device=windows.device)[0]
                else:
                    # Factorized head only: materialise exact joint log-probs
                    # from the per-dimension log-probs, so joint top-5 stays
                    # comparable with the joint head at small widths.
                    logp = F.log_softmax(masked_logits, dim=-1)   # (M, D, L)
                    codes = torch.arange(
                        self.config.codebook_size, device=masked_logits.device
                    )
                    # digits[c, d] = level index of dim d in joint code c
                    digits = (
                        codes.unsqueeze(-1) // self.fsq._powers
                    ) % self.config.n_levels                      # (C, D)
                    # joint_logp[m, c] = sum_d logp[m, d, digits[c, d]]
                    joint_logp = sum(
                        logp[:, d, digits[:, d]] for d in range(D)
                    )                                             # (M, C)
                    _, top5_indices = joint_logp.topk(5, dim=-1)
                    top5_acc = (
                        top5_indices == masked_targets.unsqueeze(-1)
                    ).any(dim=-1).float().mean()

        # ── Logging ────────────────────────────────────────────────────────���──
        on_step = (stage == "train")
        self.log(f"{stage}_loss", loss, prog_bar=True, on_epoch=True, on_step=on_step,
                 sync_dist=True)
        self.log(f"{stage}_acc",  acc,  prog_bar=True, on_epoch=True, on_step=False,
                 sync_dist=True)
        self.log(f"{stage}_top5_acc", top5_acc, prog_bar=False, on_epoch=True, on_step=False,
                 sync_dist=True)
        if dim_acc is not None:
            # Per-(position, dimension) top-1.  Always >= the joint exact-match
            # ``acc`` above; logged separately so the two are never confused.
            self.log(f"{stage}_dim_acc", dim_acc, prog_bar=False, on_epoch=True,
                     on_step=False, sync_dist=True)
            # Mean per-dimension CE, i.e. loss / n_dims.  ``loss`` itself sums
            # over dimensions, so it grows with width by construction; this
            # normalised version is what can be compared across widths.
            self.log(f"{stage}_loss_per_dim", loss.detach() / self.config.n_dims,
                     prog_bar=False, on_epoch=True, on_step=False, sync_dist=True)

        # Track true token usage for utilisation / collapse diagnostics —
        # exclude invalid positions (whatever a zero-filled placeholder
        # quantizes to is meaningless and would pollute the stats).
        if stage == "train":
            ids_for_util = token_ids if valid_mask is None else token_ids[valid_mask]
            ids_for_util = ids_for_util.detach().reshape(-1)
            # Per-dimension level histogram — the width-independent diagnostic.
            # Accumulated ON DEVICE and only pulled to CPU at epoch end: doing
            # the .cpu()/.tolist() round-trip per step forces a sync every
            # iteration and stalls the pipeline for a metric nobody reads until
            # the epoch boundary.
            D, L = self.config.n_dims, self.config.n_levels
            levels = (self.fsq.decode_token_ids(ids_for_util) + self.fsq._half).long()
            if self._level_counts.device != levels.device:
                self._level_counts = self._level_counts.to(levels.device)
            # one bincount for all dims: offset dim d into its own [d*L, (d+1)*L) slot
            offsets = torch.arange(D, device=levels.device) * L
            self._level_counts += torch.bincount(
                (levels + offsets).reshape(-1), minlength=D * L
            ).view(D, L)
            if self.config.codebook_size <= JOINT_METRIC_MAX_CODES:
                if self._seen_mask.device != ids_for_util.device:
                    self._seen_mask = self._seen_mask.to(ids_for_util.device)
                self._seen_mask[ids_for_util] = True

        return loss

    def training_step(self, batch: tuple, batch_idx: int) -> Tensor:
        return self._step(batch, "train")

    def validation_step(self, batch: tuple, batch_idx: int) -> None:
        self._step(batch, "val")

    def test_step(self, batch: tuple, batch_idx: int) -> None:
        self._step(batch, "test")

    def on_train_epoch_end(self) -> None:
        """Log utilisation / collapse diagnostics at the end of each epoch.

        Per-dimension level entropy is the primary metric: it is bounded by
        log2(n_levels) regardless of width, so it stays interpretable across
        the whole sweep, and a collapsed dimension shows up as entropy → 0 even
        when the other dimensions are healthy.  ``dim_entropy_min`` is the
        sharpest single number to watch.

        The joint ``codebook_utilisation`` ratio is only logged when the code
        space is small enough to be countable — past ~6 dims the denominator
        exceeds what the dataset could ever expose, so the ratio would read
        ≈0 for a perfectly healthy model.
        """
        counts = self._level_counts.float().cpu()              # (n_dims, n_levels)
        total = counts.sum(dim=1, keepdim=True).clamp(min=1.0)
        probs = counts / total
        # 0 * log2(0) := 0
        entropy = -(probs * torch.log2(probs.clamp(min=1e-12))).sum(dim=1)  # (n_dims,)

        self.log("dim_entropy_mean", entropy.mean(), on_epoch=True)
        self.log("dim_entropy_min", entropy.min(), on_epoch=True)
        # Summed per-dimension entropy = effective bits per token actually used,
        # versus the nominal n_dims * log2(n_levels) the code space allows.
        self.log("effective_bits_per_token", entropy.sum(), on_epoch=True)
        self.log("nominal_bits_per_token",
                 self.config.n_dims * math.log2(self.config.n_levels), on_epoch=True)
        for d in range(self.config.n_dims):
            self.log(f"dim_entropy_{d}", entropy[d], on_epoch=True)

        if self.config.codebook_size <= JOINT_METRIC_MAX_CODES:
            n_seen = int(self._seen_mask.sum().item())
            self.log("n_distinct_codes", float(n_seen), on_epoch=True)
            self.log("codebook_utilisation",
                     n_seen / self.config.codebook_size, on_epoch=True)

        self.log("mask_ratio", self.config.mask_ratio, on_epoch=True)
        self.log("tokenizer_frozen", float(self.tokenizer_frozen), on_epoch=True)
        self._seen_mask.zero_()
        self._level_counts.zero_()

    # ── Optimiser ────────────────────────────────────────────────────────────

    def configure_optimizers(self) -> dict:
        # Only pass parameters that require gradients.
        # In frozen-tokenizer mode this excludes encoder + FSQ,
        # saving memory on optimizer states for ~80 % of parameters.
        trainable = [p for p in self.parameters() if p.requires_grad]
        if not trainable:
            raise RuntimeError(
                "No trainable parameters found.  "
                "If using use_frozen_tokenizer=True, make sure "
                "freeze_tokenizer() was called after __init__."
            )
        opt = torch.optim.AdamW(
            trainable,
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
        )
        if self.config.lr_scheduler == "plateau":
            sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
                opt, mode="min",
                factor=self.config.plateau_factor,
                patience=self.config.plateau_patience,
            )
            return {
                "optimizer": opt,
                "lr_scheduler": {"scheduler": sched, "monitor": "val_loss"},
            }
        elif self.config.lr_scheduler == "cosine":
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt, T_max=self.config.max_epochs, eta_min=self.config.lr * 0.01,
            )
            return {
                "optimizer": opt,
                "lr_scheduler": {"scheduler": sched, "interval": "epoch"},
            }
        return {"optimizer": opt}

    # ── Persistence ──────────────────────────────────────────────────────────

    @classmethod
    def load(cls, checkpoint_path: Path | str) -> "FSQPredictiveSSL":
        """Load from a Lightning checkpoint.

        Handles both the nested ``{config: {...}}`` format and a flat dict.
        """
        ckpt = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
        hparams = ckpt.get("hyper_parameters", {})
        flat = hparams.get("config", hparams)
        config = FSQPredictiveSSLConfig(
            **{k: v for k, v in flat.items() if k != "codebook_size"}
        )
        return cls.load_from_checkpoint(str(checkpoint_path), config=config)






