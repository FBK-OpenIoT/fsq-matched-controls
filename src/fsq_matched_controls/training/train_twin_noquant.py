"""Training entry point of twin `twin_noquant`: the letter's reconstruction autoencoder trained without the quantizer in the training path.

    python -m fsq_matched_controls.training.train_twin_noquant --config configs/twins/cropglobe_fsq_ov0_625_reconstruction_nofsq_transductive_seed42_vm.yaml

Everything is train.py's: configuration loading, seeding, the stock datamodule, optimiser, scheduler, early stopping, best-checkpoint rule (val_loss), resume from last.ckpt; only the model class differs
(``ContinuousAutoencoder``: the decoder reads the continuous bounded latent). No post-training evaluation or test pass.
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
from fsq_matched_controls.models.autoencoder import FSQAutoencoderConfig
from fsq_matched_controls.models.autoencoder_continuous import ContinuousAutoencoder

log = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the reconstruction autoencoder without the quantizer (twin_noquant)")
    parser.add_argument("--config", required=True)
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--fast-dev-run", action="store_true")
    args = parser.parse_args()

    torch.set_float32_matmul_precision("high")
    cfg = load_config(args.config)
    if args.max_epochs is not None:
        cfg["max_epochs"] = args.max_epochs
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(name)s - %(message)s")
    seed_everything(cfg.get("random_seed", 42), workers=True)

    exp_dir = make_exp_dir(cfg)
    log.info("Experiment dir: %s", exp_dir)
    dm = build_datamodule(cfg, exp_dir)

    model_cfg = FSQAutoencoderConfig(
        n_bands=cfg.get("n_bands", 10), obs_per_window=dm.obs_per_window, encoder_channels=cfg.get("encoder_channels", [32, 64, 128]), kernel_size=cfg.get("kernel_size", 3),
        dropout=cfg.get("dropout", 0.1), norm_type=cfg.get("norm_type", "batch"), n_dims=cfg.get("n_dims", 4), n_levels=cfg.get("n_levels", 5), lr=cfg.get("lr", 1e-3),
        lr_scheduler=cfg.get("lr_scheduler", "plateau"), weight_decay=cfg.get("weight_decay", 1e-4), max_epochs=cfg.get("max_epochs", 100),
        plateau_factor=cfg.get("plateau_factor", 0.5), plateau_patience=cfg.get("plateau_patience", 10))
    model = ContinuousAutoencoder(config=model_cfg)

    mlflow_logger = build_mlflow_logger(cfg, run_name=exp_name(cfg), tags={"source": dm.source, "hold_out": hold_tag(cfg), "n_dims": str(model_cfg.n_dims), "variant": "no_quantizer_in_training"})
    trainer, ckpt_callback = build_trainer(cfg, exp_dir, logger=[mlflow_logger], ckpt_filename="best", fast_dev_run=args.fast_dev_run)

    with mlflow.start_run(run_id=mlflow_logger.run_id):
        mlflow.log_params({"source": dm.source, "n_train_windows": len(dm._train_ds), "n_val_windows": len(dm._val_ds), "obs_per_window": dm.obs_per_window,
                           **{k: v for k, v in model_cfg.to_dict().items() if not isinstance(v, list)}, "encoder_channels": str(model_cfg.encoder_channels)})
        ckdir = Path(exp_dir) / "checkpoints"
        resume = str(ckdir / "last.ckpt") if (ckdir / "last.ckpt").exists() else None
        log.info("Starting training (%s) ...", "resuming from last.ckpt" if resume else "from scratch")
        trainer.fit(model, datamodule=dm, ckpt_path=resume)
    log.info("Done. Experiment dir: %s", exp_dir)


if __name__ == "__main__":
    main()
