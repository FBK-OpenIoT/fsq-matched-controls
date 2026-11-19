"""FSQ Tokenizer training entry point (reconstruction objective).

Usage
-----
    python -m fsq_matched_controls.training.train --config configs/letter/<run>.yaml

    # Override hold-out country from CLI:
    python -m fsq_matched_controls.training.train --config configs/letter/<run>.yaml --hold-out AUS

What this script does
---------------------
1. Load and validate YAML config.
2. Build CropGlobeDataModule → setup() → log data stats to MLflow.
3. Build FSQAutoencoder with obs_per_window from data.
4. Attach MLflow logger (+ optional W&B logger).
5. Configure Lightning Trainer with ModelCheckpoint + EarlyStopping.
6. Run trainer.fit(model, datamodule).
7. Run trainer.test() on held-out countries.
8. Save evaluation outputs (codebook stats, t-SNE, linear probe) as artefacts.
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
from fsq_matched_controls.models.autoencoder import FSQAutoencoder, FSQAutoencoderConfig
from fsq_matched_controls.analysis.tokenize import tokenize_parcel_dataset, TokenisationResult
from fsq_matched_controls.analysis.token_analysis import (
    codebook_utilisation,
    compute_tsne,
    plot_2d_embedding,
    jsd_matrix,
    plot_jsd_matrix,
)
from fsq_matched_controls.probe.linear_probe import (
    build_bow_features,
    train_linear_probe,
    eval_linear_probe,
)
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
        cfg["batch_size"] = args.batch_size
    if args.max_parcels is not None:
        cfg["max_parcels"] = args.max_parcels
    if args.limit_train_batches is not None:
        cfg["limit_train_batches"] = args.limit_train_batches
    if args.limit_val_batches is not None:
        cfg["limit_val_batches"] = args.limit_val_batches
    return cfg


# ── Post-training evaluation ─────────────────────────────────────────────────

def run_evaluation(
    model: FSQAutoencoder,
    dm: CropGlobeDataModule,
    experiment_dir: Path,
    run_id: str,
) -> None:
    """Tokenise held-out parcels and log evaluation artefacts to MLflow."""
    if not dm.hold_out_countries:
        log.info("No held-out countries — skipping cross-region evaluation.")
        return

    log.info("=== Post-training evaluation on held-out: %s ===", dm.hold_out_countries)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. Tokenise held-out parcels
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

    # 3. t-SNE of quantised embeddings (flatten n_months × n_dims → feature)
    z_flat = held_out_result.z_q.reshape(held_out_result.n_parcels, -1)
    xy, idx = compute_tsne(z_flat, max_samples=5_000)
    labels_sub = held_out_result.labels[idx]

    fig = plot_2d_embedding(
        xy, labels_sub,
        label_names=CROPGLOBE_LABEL_NAMES,
        title=f"t-SNE — Held-out: {', '.join(dm.hold_out_countries)}",
    )
    tsne_path = experiment_dir / "tsne_held_out.png"
    fig.savefig(tsne_path, bbox_inches="tight")
    mlflow.log_artifact(str(tsne_path))
    log.info("t-SNE plot saved → %s", tsne_path)

    # 4. Linear probe on BOW features
    #    We need training parcel embeddings — for now we tokenise the validation
    #    set as a proxy (full train set tokenisation requires parcel_dataloader for train).
    #    This is a quick evaluation; for the full LOOCV probe, use a dedicated script.
    bow_held = build_bow_features(held_out_result.token_sequences, model.config.codebook_size)
    log.info(
        "BOW features: %d parcels × %d dims",
        bow_held.shape[0], bow_held.shape[1],
    )
    log.info("(Full linear probe requires training zone parcel tokenisation — "
             "run evaluation/run_linear_probe.py separately.)")


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Train FSQ Phenological Tokenizer")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument("--hold-out", nargs="+", default=None,
                        help="Override hold_out_countries from config")
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
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
    seed_everything(cfg.get("random_seed", 42), workers=True)

    # ── Experiment dir + data ─────────────────────────────────────────────────
    _hold_tag = hold_tag(cfg)
    _exp_name = exp_name(cfg)
    exp_dir = make_exp_dir(cfg)
    log.info("Experiment dir: %s", exp_dir)

    dm = build_datamodule(cfg, exp_dir)
    log.info(
        "obs_per_window=%d | n_months=%d | codebook (n_dims=%d, n_levels=%d) = %d codes",
        dm.obs_per_window, dm.n_months,
        cfg.get("n_dims", 4), cfg.get("n_levels", 5),
        cfg.get("n_levels", 5) ** cfg.get("n_dims", 4),
    )

    # ── Model ────────────────────────────────────────────────────────────────
    model_cfg = FSQAutoencoderConfig(
        n_bands=cfg.get("n_bands", 10),
        obs_per_window=dm.obs_per_window,
        encoder_channels=cfg.get("encoder_channels", [32, 64, 128]),
        kernel_size=cfg.get("kernel_size", 3),
        dropout=cfg.get("dropout", 0.1),
        norm_type=cfg.get("norm_type", "batch"),
        n_dims=cfg.get("n_dims", 4),
        n_levels=cfg.get("n_levels", 5),
        lr=cfg.get("lr", 1e-3),
        lr_scheduler=cfg.get("lr_scheduler", "plateau"),
        weight_decay=cfg.get("weight_decay", 1e-4),
        max_epochs=cfg.get("max_epochs", 100),
        plateau_factor=cfg.get("plateau_factor", 0.5),
        plateau_patience=cfg.get("plateau_patience", 10),
    )
    model = FSQAutoencoder(config=model_cfg)

    # ── Loggers ───────────────────────────────────────────────────────────────
    mlflow_logger = build_mlflow_logger(
        cfg, run_name=_exp_name,
        tags={
            "source": dm.source,
            "hold_out": _hold_tag,
            "n_dims": str(model_cfg.n_dims),
            "n_levels": str(model_cfg.n_levels),
            "codebook_size": str(model_cfg.codebook_size),
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
        ckpt_filename="best",
        fast_dev_run=args.fast_dev_run,
    )

    # ── Log config as MLflow params ───────────────────────────────────────────
    with mlflow.start_run(run_id=mlflow_logger.run_id):
        mlflow.log_params({
            "source": dm.source,
            "train_countries": ",".join(dm.train_countries),
            "hold_out_countries": ",".join(dm.hold_out_countries),
            "n_train_windows": len(dm._train_ds),
            "n_val_windows": len(dm._val_ds),
            "obs_per_window": dm.obs_per_window,
            "n_tokens": dm.n_tokens,
            "days_per_token": dm.days_per_token,
            "overlap_days": dm.overlap_days,
            "cadence_days": dm.cadence_days,
            "n_months": dm.n_months,
            **{k: v for k, v in model_cfg.to_dict().items() if not isinstance(v, list)},
            "encoder_channels": str(model_cfg.encoder_channels),
        })

        # Look for existing checkpoints to avoid restarting from zero (e.g. if a
        # previous run completed training but crashed during post-training eval).
        checkpoint_dir = Path(exp_dir) / "checkpoints"
        last_ckpt_path = checkpoint_dir / "last.ckpt"
        best_ckpt_path = checkpoint_dir / "best.ckpt"

        resume_ckpt_path = None
        if last_ckpt_path.exists():
            resume_ckpt_path = str(last_ckpt_path)
            log.info("Found 'last.ckpt'. Resuming training from previous state...")
        elif best_ckpt_path.exists():
            resume_ckpt_path = str(best_ckpt_path)
            log.info("Found 'best.ckpt'. Run is complete or loading weights...")
        else:
            log.info("No existing checkpoint found. Starting training from scratch.")

        log.info("Starting training …")
        trainer.fit(model, datamodule=dm, ckpt_path=resume_ckpt_path)

        final_best_ckpt = ckpt_callback.best_model_path or (
            str(best_ckpt_path) if best_ckpt_path.exists() else "last"
        )

        if dm.hold_out_countries and not args.fast_dev_run:
            log.info("Testing on held-out countries …")
            trainer.test(model, datamodule=dm, ckpt_path=final_best_ckpt)

        if not args.no_eval and not args.fast_dev_run:
            best_model = FSQAutoencoder.load(final_best_ckpt)
            run_evaluation(best_model, dm, exp_dir, mlflow_logger.run_id)

    log.info("Done. Experiment dir: %s", exp_dir)


if __name__ == "__main__":
    main()

