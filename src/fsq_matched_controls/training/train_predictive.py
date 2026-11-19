"""FSQ Predictive SSL training entry point.

Predictive SSL replaces the reconstruction objective with masked token prediction:
instead of MSE(W, W_reconstructed), it minimises CrossEntropy(predicted_token_id,
true_token_id) — the context transformer must guess each masked monthly token from
its phenological neighbours.

The letter's runs train end to end (no ``--tokenizer-ckpt``): the encoder produces both
the targets and the context, with no anti-collapse term; ``codebook_utilisation`` is logged
per epoch. ``--tokenizer-ckpt`` (frozen tokenizer) exists but is not used by the letter.

Usage
-----
    python -m fsq_matched_controls.training.train_predictive --config configs/letter/<run>.yaml
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import torch
import numpy as np
import mlflow
import mlflow.pytorch
from lightning.pytorch import seed_everything

from fsq_matched_controls.training._train_utils import (
    load_config,
    build_datamodule,
    build_mlflow_logger,
    build_trainer,
    make_exp_dir,
    hold_tag,
    exp_name,
)
from fsq_matched_controls.models.predictive_ssl import FSQPredictiveSSL, FSQPredictiveSSLConfig
from fsq_matched_controls.analysis.tokenize import tokenize_parcel_dataset, TokenisationResult
from fsq_matched_controls.analysis.token_analysis import (
    codebook_utilisation,
    compute_tsne,
    plot_2d_embedding,
    jsd_matrix,
    plot_jsd_matrix,
)
from fsq_matched_controls.probe.linear_probe import build_bow_features
from fsq_matched_controls.data.fold_definitions import CROPGLOBE_LABEL_NAMES

log = logging.getLogger(__name__)


# ── Config helpers ────────────────────────────────────────────────────────────

def merge_cli(cfg: dict, args: argparse.Namespace) -> dict:
    """Override config values with explicit CLI arguments."""
    if args.hold_out:
        cfg["hold_out_countries"] = args.hold_out
    if args.max_epochs is not None:
        cfg["max_epochs"] = args.max_epochs
    if args.batch_size is not None:
        cfg["parcel_batch_size"] = args.batch_size
    # --tokenizer-ckpt overrides the yaml value (null by default)
    if args.tokenizer_ckpt is not None:
        cfg["tokenizer_ckpt"] = args.tokenizer_ckpt
    if args.max_parcels is not None:
        cfg["max_parcels"] = args.max_parcels
    if args.limit_train_batches is not None:
        cfg["limit_train_batches"] = args.limit_train_batches
    if args.limit_val_batches is not None:
        cfg["limit_val_batches"] = args.limit_val_batches
    return cfg


# ── Post-training evaluation ─────────────────────────────────────────────────

def run_evaluation(
    model: FSQPredictiveSSL,
    dm,
    experiment_dir: Path,
    run_id: str,
) -> None:
    """Tokenise held-out parcels and log evaluation artefacts to MLflow.

    ``FSQPredictiveSSL.encode()`` accepts the same window-level input as
    ``FSQAutoencoder.encode()``, so ``tokenize_parcel_dataset`` is reused
    without modification.
    """
    if not dm.hold_out_countries:
        log.info("No held-out countries — skipping cross-region evaluation.")
        return

    log.info("=== Post-training evaluation on held-out: %s ===", dm.hold_out_countries)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. Tokenise held-out parcels (window-by-window via encode())
    held_out_result: TokenisationResult = tokenize_parcel_dataset(
        model, dm.parcel_dataloader("held_out"), device=device
    )
    result_path = experiment_dir / "held_out_tokenisation.npz"
    held_out_result.save(result_path)
    mlflow.log_artifact(str(result_path))

    # 2. Codebook utilisation
    all_ids = np.concatenate(held_out_result.token_sequences)
    cb_stats = codebook_utilisation(all_ids, model.config.codebook_size)
    mlflow.log_metrics({f"eval_{k}": v for k, v in cb_stats.items() if isinstance(v, float)})
    log.info("Codebook utilisation: %.1f%%", cb_stats["utilisation"] * 100)

    # 3. t-SNE of quantised embeddings
    z_flat = held_out_result.z_q.reshape(held_out_result.n_parcels, -1)
    xy, idx = compute_tsne(z_flat, max_samples=5_000)
    labels_sub = held_out_result.labels[idx]

    fig = plot_2d_embedding(
        xy, labels_sub,
        label_names=CROPGLOBE_LABEL_NAMES,
        title=f"t-SNE (Predictive SSL) — Held-out: {', '.join(dm.hold_out_countries)}",
    )
    tsne_path = experiment_dir / "tsne_held_out.png"
    fig.savefig(tsne_path, bbox_inches="tight")
    mlflow.log_artifact(str(tsne_path))
    log.info("t-SNE plot saved → %s", tsne_path)

    # 4. BOW feature summary (full linear probe requires a separate script)
    bow_held = build_bow_features(held_out_result.token_sequences, model.config.codebook_size)
    log.info(
        "BOW features: %d parcels × %d dims",
        bow_held.shape[0], bow_held.shape[1],
    )
    log.info(
        "Full linear probe: run evaluation/run_linear_probe.py with this checkpoint."
    )


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Train FSQ Predictive SSL Tokenizer")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument("--hold-out", nargs="+", default=None,
                        help="Override hold_out_countries from config")
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Override parcel_batch_size from config")
    parser.add_argument(
        "--tokenizer-ckpt", default=None, metavar="PATH",
        help=(
            "Path to a reconstruction SSL checkpoint (FSQAutoencoder). "
            "When provided, the encoder + FSQ are loaded from this checkpoint "
            "and FROZEN for the entire training run (BEiT-style). "
            "Only the context transformer is optimised. "
            "RECOMMENDED to prevent codebook collapse. "
            "If omitted, trains end-to-end (default, collapse-prone)."
        ),
    )
    parser.add_argument("--fast-dev-run", action="store_true",
                        help="Run 1 train + 1 val batch (smoke test)")
    parser.add_argument("--max-parcels", type=int, default=None,
                        help="EuroCropsML only: truncate every split to the first N "
                             "parcels before loading (setup() is a single-threaded "
                             "np.load() per npz file with no DataLoader-level "
                             "parallelism, so --fast-dev-run alone does not make "
                             "setup() fast -- this does).")
    parser.add_argument("--no-eval", action="store_true",
                        help="Skip post-training evaluation step")
    parser.add_argument("--limit-train-batches", type=int, default=None,
                        help="Short run: train on this many batches per epoch (checkpoints are still saved)")
    parser.add_argument("--limit-val-batches", type=int, default=None,
                        help="Short run: validate on this many batches per epoch")
    args = parser.parse_args()


    #  to unlock the A100's Tensor Cores for massive training speedups:
    torch.set_float32_matmul_precision('high')

    # ── Load config ──────────────────────────────────────────────────────────
    cfg = load_config(args.config)
    cfg = merge_cli(cfg, args)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s - %(message)s",
    )
    seed_everything(cfg.get("random_seed", 1904), workers=True)

    # ── Experiment dir + data ─────────────────────────────────────────────────
    hold_out = cfg.get("hold_out_countries")
    if hold_out is None or hold_out == []:
        cfg["hold_out_countries"] = ["none"]
    elif isinstance(hold_out, str):
        # If it's a single string, wrap it in a list
        cfg["hold_out_countries"] = [hold_out]

    _hold_tag = hold_tag(cfg)
    _exp_name = exp_name(cfg)
    exp_dir = make_exp_dir(cfg)
    log.info("Experiment dir: %s", exp_dir)

    dm = build_datamodule(cfg, exp_dir)

    parcel_batch_size = cfg.get("parcel_batch_size", 64)

    log.info(
        "obs_per_window=%d | n_tokens=%d | n_months=%d | codebook (n_dims=%d, n_levels=%d) = %d codes",
        dm.obs_per_window, dm.n_tokens, dm.n_months,
        cfg.get("n_dims", 4), cfg.get("n_levels", 5),
        cfg.get("n_levels", 5) ** cfg.get("n_dims", 4),
    )
    log.info(
        "Parcel batch size: %d (each sample = %d monthly windows)",
        parcel_batch_size, dm.n_tokens,
    )

    # ── Model ────────────────────────────────────────────────────────────────
    model_cfg = FSQPredictiveSSLConfig(
        n_bands=cfg.get("n_bands", 10),
        obs_per_window=dm.obs_per_window,
        n_tokens=dm.n_tokens,
        encoder_channels=cfg.get("encoder_channels", [32, 64, 128]),
        kernel_size=cfg.get("kernel_size", 3),
        dropout=cfg.get("dropout", 0.1),
        norm_type=cfg.get("norm_type", "batch"),
        n_dims=cfg.get("n_dims", 4),
        n_levels=cfg.get("n_levels", 5),
        factorized_head=cfg.get("factorized_head", False),
        head_mode=cfg.get("head_mode", "joint"),
        mask_ratio=cfg.get("mask_ratio", 0.15),
        min_masks=cfg.get("min_masks", 1),
        d_model=cfg.get("d_model", 128),
        n_heads=cfg.get("n_heads", 4),
        n_layers=cfg.get("n_layers", 3),
        dim_feedforward=cfg.get("dim_feedforward", 256),
        transformer_dropout=cfg.get("transformer_dropout", 0.1),
        max_seq_len=cfg.get("max_seq_len", 100),
        lr=cfg.get("lr", 1e-3),
        lr_scheduler=cfg.get("lr_scheduler", "plateau"),
        weight_decay=cfg.get("weight_decay", 1e-4),
        max_epochs=cfg.get("max_epochs", 100),
        plateau_factor=cfg.get("plateau_factor", 0.5),
        plateau_patience=cfg.get("plateau_patience", 10),
    )

    tokenizer_ckpt: str | None = cfg.get("tokenizer_ckpt", None)
    log.info("HERE SHOULD BE THE LOGGER OF BEIT LOGGING")
    if tokenizer_ckpt:
        # ── BEiT-style (recommended, collapse-proof) ─────────────────────────
        log.info(
            "BEiT-style mode: loading frozen tokenizer from %s", tokenizer_ckpt
        )
        model_cfg.use_frozen_tokenizer = True
        model = FSQPredictiveSSL.from_pretrained_tokenizer(
            config=model_cfg,
            tokenizer_ckpt=tokenizer_ckpt,
        )
        training_mode = "beit_frozen_tokenizer"
    else:
        # ── End-to-end (default, collapse-prone) ─────────────────────────────
        log.warning(
            "End-to-end mode: encoder + FSQ are trainable alongside the "
            "context transformer.  Collapse is possible — monitor "
            "codebook_utilisation in MLflow.  "
            "Pass --tokenizer-ckpt to activate the safe BEiT-style mode."
        )
        model = FSQPredictiveSSL(config=model_cfg)
        training_mode = "end_to_end"

    n_params_total = sum(p.numel() for p in model.parameters())
    n_params_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(
        "FSQPredictiveSSL [%s]: %d total params, %d trainable",
        training_mode, n_params_total, n_params_trainable,
    )
    head_mode = "factorized" if model_cfg.factorized_head else "joint"
    log.info(
        "Prediction head: %s — n_dims=%d, n_levels=%d, output layer = %d units "
        "(joint head at this width would need %d)",
        head_mode, model_cfg.n_dims, model_cfg.n_levels,
        model_cfg.n_dims * model_cfg.n_levels if model_cfg.factorized_head
        else model_cfg.codebook_size,
        model_cfg.codebook_size,
    )

    # ── Loggers ───────────────────────────────────────────────────────────────
    mlflow_logger = build_mlflow_logger(
        cfg, run_name=_exp_name,
        tags={
            "model_type": "predictive_ssl",
            "training_mode": training_mode,
            "source": dm.source,
            "hold_out": _hold_tag,
            "n_dims": str(model_cfg.n_dims),
            "n_levels": str(model_cfg.n_levels),
            "head_mode": head_mode,
            "codebook_size": str(model_cfg.codebook_size),
            "mask_ratio": str(model_cfg.mask_ratio),
            "n_layers": str(model_cfg.n_layers),
            "d_model": str(model_cfg.d_model),
            "tokenizer_frozen": str(model_cfg.use_frozen_tokenizer),
        },
    )
    loggers = [mlflow_logger]

    if cfg.get("log_wandb", False):
        from lightning.pytorch.loggers import WandbLogger
        loggers.append(WandbLogger(
            project=cfg.get("wandb_project", "croplang-fsq"),
            entity=cfg.get("wandb_entity"),
            name=_exp_name,
        ))

    # ── Trainer ──────────────────────────────────────────────────────────────
    trainer, ckpt_callback = build_trainer(
        cfg, exp_dir,
        logger=loggers,
        ckpt_filename="predictive_best",
        fast_dev_run=args.fast_dev_run,
    )

    #  🔍 Look for existing checkpoints to avoid restarting from zero
    checkpoint_dir = Path(exp_dir) / "checkpoints"
    last_ckpt_path = checkpoint_dir / "last.ckpt"
    best_ckpt_path = checkpoint_dir / "predictive_best.ckpt"

    ckpt_path = None
    if last_ckpt_path.exists():
        ckpt_path = str(last_ckpt_path)
        log.info("🔄 Found 'last.ckpt'. Resuming training from previous state...")
    elif best_ckpt_path.exists():
        ckpt_path = str(best_ckpt_path)
        log.info("📈 Found 'predictive_best.ckpt'. Run is complete or loading weights...")
    else:
        log.info("🌱 No existing checkpoint found. Starting training from scratch.")

    # ── Log config as MLflow params ───────────────────────────────────────────
    with mlflow.start_run(run_id=mlflow_logger.run_id):
        mlflow.log_params({
            "model_type": "predictive_ssl",
            "training_mode": training_mode,
            "tokenizer_ckpt": str(tokenizer_ckpt) if tokenizer_ckpt else "none",
            "source": dm.source,
            "train_countries": ",".join(dm.train_countries),
            "hold_out_countries": ",".join(dm.hold_out_countries),
            "n_train_parcels": len(dm._train_parcel_ds),
            "n_val_parcels": len(dm._val_parcel_ds),
            "obs_per_window": dm.obs_per_window,
            "n_tokens": dm.n_tokens,
            "days_per_token": dm.days_per_token,
            "overlap_days": dm.overlap_days,
            "cadence_days": dm.cadence_days,
            "parcel_batch_size": parcel_batch_size,
            **{k: v for k, v in model_cfg.to_dict().items()
               if not isinstance(v, list) and k != "codebook_size"},
            "encoder_channels": str(model_cfg.encoder_channels),
            "codebook_size": model_cfg.codebook_size,
            "n_params_total": sum(p.numel() for p in model.parameters()),
            "n_params_trainable": sum(p.numel() for p in model.parameters() if p.requires_grad),
        })

        log.info("Starting predictive SSL training …")
        log.info(
            "Pipeline: [W_1…W_N] → CNN → FSQ → [t_1…t_N] → mask %.0f%% → "
            "context transformer → CE loss on masked tokens",
            model_cfg.mask_ratio * 100,
        )

        # Pass the identified ckpt_path directly into fit
        trainer.fit(
            model,
            train_dataloaders=dm.parcel_train_dataloader(batch_size=parcel_batch_size),
            val_dataloaders=dm.parcel_val_dataloader(batch_size=parcel_batch_size),
            ckpt_path=ckpt_path,
        )

        # Resolve the definitive best checkpoint path for testing and evaluation
        # (If trainer.fit skipped training, ckpt_callback.best_model_path might be empty,
        # so we fall back to our confirmed best_ckpt_path file)
        final_best_ckpt = ckpt_callback.best_model_path or (
            str(best_ckpt_path) if best_ckpt_path.exists() else "last"
        )

        if dm.hold_out_countries and not args.fast_dev_run:
            log.info("Testing on held-out countries (parcel-level) …")
            trainer.test(
                model,
                dataloaders=dm.parcel_dataloader(
                    "held_out", batch_size=parcel_batch_size
                ),
                ckpt_path=final_best_ckpt,
            )

        if not args.no_eval and not args.fast_dev_run:
            best_model = FSQPredictiveSSL.load(final_best_ckpt)
            run_evaluation(best_model, dm, exp_dir, mlflow_logger.run_id)

    log.info("Done. Experiment dir: %s", exp_dir)
    print(f"\nPREDICTIVE_SSL_CKPT={final_best_ckpt}")


if __name__ == "__main__":
    main()

