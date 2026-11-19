"""Training entry point of twins `twin_bands` and twin_teacher on EuroCropsML (datasets with missing months and padded windows: batches with valid_mask and obs_mask).
--variant ssl: twin_bands (band-value targets); --variant teacher: twin_teacher (EMA-teacher latent targets). The letter's data module, encoder, context transformer and recipe.

    python -m fsq_matched_controls.training.train_twins_eurocropsml --variant ssl --config configs/twins/eurocropml_continuous_ssl_d4_seed1904_vm.yaml

The initial weights are written to <experiment dir>/init_state.pt before the first optimisation step (same seed, no training step).
The best checkpoint is checkpoints/{CKPT_NAME}.ckpt.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path


import torch
import mlflow
from lightning.pytorch import seed_everything

from fsq_matched_controls.training._train_utils import load_config, build_datamodule, build_mlflow_logger, build_trainer, make_exp_dir, hold_tag, exp_name
from fsq_matched_controls.models.continuous_ssl import ContinuousSSLConfig
from fsq_matched_controls.models.continuous_ssl_ec import ContinuousPredictiveSSLEC, ContinuousTeacherSSLEC

log = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the continuous twins `twin_bands` / twin_teacher on EuroCropsML")
    parser.add_argument("--config", required=True)
    parser.add_argument("--variant", choices=["ssl", "teacher"], required=True)
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--fast-dev-run", action="store_true", help="1 train + 1 val batch (smoke test)")
    args = parser.parse_args()

    torch.set_float32_matmul_precision("high")
    cfg = load_config(args.config)
    if args.max_epochs is not None:
        cfg["max_epochs"] = args.max_epochs
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(name)s - %(message)s")
    seed_everything(cfg.get("random_seed", 1904), workers=True)
    hold_out = cfg.get("hold_out_countries")
    if hold_out is None or hold_out == []:
        cfg["hold_out_countries"] = ["none"]
    elif isinstance(hold_out, str):
        cfg["hold_out_countries"] = [hold_out]
    _exp_name = exp_name(cfg)
    exp_dir = make_exp_dir(cfg)
    log.info("Experiment dir: %s", exp_dir)
    dm = build_datamodule(cfg, exp_dir)
    parcel_batch_size = cfg.get("parcel_batch_size", 64)

    model_cfg = ContinuousSSLConfig(
        n_bands=cfg.get("n_bands", 10), obs_per_window=dm.obs_per_window, n_tokens=dm.n_tokens,
        encoder_channels=cfg.get("encoder_channels", [32, 64, 128]), kernel_size=cfg.get("kernel_size", 3),
        dropout=cfg.get("dropout", 0.1), norm_type=cfg.get("norm_type", "batch"), n_dims=cfg.get("n_dims", 4),
        n_levels=cfg.get("n_levels", 5), mask_ratio=cfg.get("mask_ratio", 0.15), min_masks=cfg.get("min_masks", 1),
        d_model=cfg.get("d_model", 128), n_heads=cfg.get("n_heads", 4), n_layers=cfg.get("n_layers", 3),
        dim_feedforward=cfg.get("dim_feedforward", 256), transformer_dropout=cfg.get("transformer_dropout", 0.1),
        max_seq_len=cfg.get("max_seq_len", 100), lr=cfg.get("lr", 1e-3), lr_scheduler=cfg.get("lr_scheduler", "plateau"),
        weight_decay=cfg.get("weight_decay", 1e-4), max_epochs=cfg.get("max_epochs", 100),
        plateau_factor=cfg.get("plateau_factor", 0.5), plateau_patience=cfg.get("plateau_patience", 10),
    )
    model = (ContinuousPredictiveSSLEC if args.variant == "ssl" else ContinuousTeacherSSLEC)(model_cfg)
    n_params = sum(p.numel() for p in model.parameters())
    log.info("EC continuous predictor (%s): d=%d, %d parameters", args.variant, model_cfg.n_dims, n_params)

    CKPT_NAME = "continuous_best" if args.variant == "ssl" else "teacher_best"
    checkpoint_dir = Path(exp_dir) / "checkpoints"
    last_ckpt_path, best_ckpt_path = checkpoint_dir / "last.ckpt", checkpoint_dir / f"{CKPT_NAME}.ckpt"
    init_path = Path(exp_dir) / "init_state.pt"
    if not init_path.exists() and not last_ckpt_path.exists():
        torch.save(model.state_dict(), init_path)             # the untrained control: weights before the first step
        log.info("Saved the initial weights to %s", init_path)

    mlflow_logger = build_mlflow_logger(cfg, run_name=_exp_name, tags={"model_type": "continuous_ssl_ec_" + args.variant, "n_dims": str(model_cfg.n_dims),
                                                                       "source": dm.source, "hold_out": hold_tag(cfg), "twin": "twins_eurocropsml"})
    trainer, ckpt_callback = build_trainer(cfg, exp_dir, logger=[mlflow_logger], ckpt_filename=CKPT_NAME, fast_dev_run=args.fast_dev_run)

    ckpt_path = None
    if last_ckpt_path.exists():
        ckpt_path = str(last_ckpt_path)
        log.info("Found last.ckpt: resuming")
    elif best_ckpt_path.exists():
        ckpt_path = str(best_ckpt_path)
        log.info("Found %s.ckpt: run is complete or loading weights", CKPT_NAME)

    with mlflow.start_run(run_id=mlflow_logger.run_id):
        mlflow.log_params({"model_type": "continuous_ssl_ec_" + args.variant, "n_dims": model_cfg.n_dims, "n_params_total": n_params, "source": dm.source,
                           "n_train_parcels": len(dm._train_parcel_ds), "n_val_parcels": len(dm._val_parcel_ds), "parcel_batch_size": parcel_batch_size,
                           "obs_per_window": dm.obs_per_window, "n_tokens": dm.n_tokens, "random_seed": cfg.get("random_seed", 1904)})
        trainer.fit(model, train_dataloaders=dm.parcel_train_dataloader(batch_size=parcel_batch_size),
                    val_dataloaders=dm.parcel_val_dataloader(batch_size=parcel_batch_size), ckpt_path=ckpt_path)
        final_best = ckpt_callback.best_model_path or (str(best_ckpt_path) if best_ckpt_path.exists() else "last")
    log.info("Done. Experiment dir: %s", exp_dir)
    print(f"\nEC_CONTINUOUS_CKPT={final_best}")


if __name__ == "__main__":
    main()
