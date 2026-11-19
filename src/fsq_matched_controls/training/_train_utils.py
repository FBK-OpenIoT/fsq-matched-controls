"""Shared helpers used by train.py, train_backbone.py, and sweep_vocab.py.

All three training scripts share the same boilerplate for:
  • loading / merging YAML config
  • building CropGlobeDataModule
  • building a Lightning Trainer with standard callbacks
  • building an MLFlowLogger

Centralising them here means a single change propagates everywhere.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml
import mlflow
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from lightning.pytorch.loggers import MLFlowLogger

from fsq_matched_controls import paths
from fsq_matched_controls.data.cropglobe import CropGlobeDataModule
from fsq_matched_controls.data.eurocropml import (
    EuroCropsMLDataModule,
    DEFAULT_W_MAX as _EUROCROPML_DEFAULT_W_MAX,
    DEFAULT_W_MIN as _EUROCROPML_DEFAULT_W_MIN,
)

log = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────

def load_config(path: str | Path) -> dict:
    """Load a YAML config. The configs' default roots are resolved through ``fsq_matched_controls.paths``:
    ``data_root: "data"`` becomes ``paths.DATA_ROOT`` (env FSQ_DATA_ROOT) and ``experiment_root: "_experiments"``
    becomes ``paths.RUNS_DIR`` (env FSQ_RUNS_DIR). Any other value is used as written."""
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if cfg.get("data_root") == "data":
        cfg["data_root"] = str(paths.DATA_ROOT)
    if cfg.get("experiment_root") == "_experiments":
        cfg["experiment_root"] = str(paths.RUNS_DIR)
    return cfg


def _hold_out_tag_list(cfg: dict) -> list[str]:
    """Countries for the run-naming tag, ``["none"]`` if there are none.

    ``cfg.get("hold_out_countries", ["none"])`` only falls back to the default
    when the KEY IS MISSING -- a YAML ``hold_out_countries: null`` (used by the
    all-countries SSL configs, since self-supervised training needs no labels)
    leaves the key present with value ``None``, which ``.get`` happily returns
    unchanged, and ``"_".join(None)`` then raises ``TypeError: can only join an
    iterable``.  This treats both "missing" and "present but null/empty" the
    same way.
    """
    hold_out = cfg.get("hold_out_countries")
    return list(hold_out) if hold_out else ["none"]


def make_exp_dir(cfg: dict) -> Path:
    """Build and create the experiment directory from config."""
    hold_tag = "_".join(_hold_out_tag_list(cfg))
    exp_name = f"{cfg['experiment_name']}_holdout_{hold_tag}"
    exp_dir = Path(cfg["experiment_root"]) / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    return exp_dir


def hold_tag(cfg: dict) -> str:
    return "_".join(_hold_out_tag_list(cfg))

def exp_name(cfg: dict) -> str:
    return f"{cfg['experiment_name']}_holdout_{hold_tag(cfg)}"


# ── DataModule ────────────────────────────────────────────────────────────────

def build_datamodule(
    cfg: dict, exp_dir: Path | None = None
) -> CropGlobeDataModule | EuroCropsMLDataModule:
    """Construct and setup a DataModule from a config dict.

    ``cfg["dataset"]`` selects which one (default ``"cropglobe"``, so
    existing configs are unaffected). Both DataModules expose the same
    public interface (``.setup()``, ``.parcel_train_dataloader()``, etc.),
    so callers (train.py / train_predictive.py) don't need to branch.
    """
    dataset = cfg.get("dataset", "cropglobe")

    # ``"none"`` is a naming sentinel, not a country.  The training scripts
    # rewrite an empty/null ``hold_out_countries`` to ``["none"]`` so the
    # experiment directory and MLflow tag read ``..._holdout_none``.  Passing
    # that straight through makes ``dm.hold_out_countries`` truthy, which
    # defeats the "skip held-out evaluation" guards downstream and then fails
    # inside ``parcel_dataloader("held_out")`` on a country that does not
    # exist -- after training has completed and checkpoints are saved, so the
    # run is marked FAILED in MLflow despite having succeeded.  Normalise it
    # here so the sentinel never reaches a DataModule.
    _hold_out = [c for c in (cfg.get("hold_out_countries") or []) if c != "none"]

    if dataset == "cropglobe":
        raw_dir = Path(cfg["data_root"]) / "cropglobe" / "raw"
        dm = CropGlobeDataModule(
            raw_dir=raw_dir,
            source=cfg.get("source", "Median"),
            train_countries=cfg.get("train_countries"),
            hold_out_countries=_hold_out,
            val_fraction=cfg.get("val_fraction", 0.1),
            batch_size=cfg.get("batch_size", 512),
            num_workers=cfg.get("num_workers", 0),
            random_seed=cfg.get("random_seed", 42),
            experiment_dir=exp_dir,
            dn_scale=cfg.get("dn_scale", 10_000.0),
            days_per_token=cfg.get("days_per_token", 30),
            overlap_days=cfg.get("overlap_days", 0),
        )
    elif dataset == "eurocropml":
        dm = EuroCropsMLDataModule(
            data_root=cfg["data_root"],
            split_name=cfg.get("split_name", "latvia_portugal_vs_estonia"),
            finetune_variant=cfg.get("finetune_variant", "all"),
            val_fraction=cfg.get("val_fraction", 0.1),
            batch_size=cfg.get("batch_size", 512),
            num_workers=cfg.get("num_workers", 0),
            random_seed=cfg.get("random_seed", 42),
            experiment_dir=exp_dir,
            dn_scale=cfg.get("dn_scale", 10_000.0),
            norm_stats_path=cfg.get("norm_stats_path"),
            max_parcels=cfg.get("max_parcels"),
            w_max=cfg.get("w_max", _EUROCROPML_DEFAULT_W_MAX),
            w_min=cfg.get("w_min", _EUROCROPML_DEFAULT_W_MIN),
            allowed_hcat_codes=cfg.get("allowed_hcat_codes"),
            include_holdout_in_pretrain=cfg.get("include_holdout_in_pretrain", False),
        )
    else:
        raise ValueError(f"Unknown dataset '{dataset}'. Use 'cropglobe' or 'eurocropml'.")

    dm.setup()
    log.info(
        "DataModule: dataset=%s | source=%s | train=%s | hold_out=%s | "
        "obs_per_window=%d | n_tokens=%d",
        dataset, dm.source, dm.train_countries, dm.hold_out_countries,
        dm.obs_per_window, dm.n_tokens,
    )
    return dm


# ── MLflow ────────────────────────────────────────────────────────────────────

_DEFAULT_TRACKING_URI = "sqlite:///_mlruns/mlflow.db"
_DEFAULT_EXPERIMENT   = "croplang-fsq-cropglobe"


def build_mlflow_logger(
    cfg: dict,
    run_name: str,
    tags: dict[str, str] | None = None,
) -> MLFlowLogger:
    """Create an MLFlowLogger and set the global tracking URI."""
    tracking_uri = cfg.get("mlflow_tracking_uri", _DEFAULT_TRACKING_URI)
    experiment   = cfg.get("mlflow_experiment_name", _DEFAULT_EXPERIMENT)
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment)
    return MLFlowLogger(
        experiment_name=experiment,
        tracking_uri=tracking_uri,
        run_name=run_name,
        tags=tags or {},
    )


# ── Trainer ───────────────────────────────────────────────────────────────────

def build_trainer(
    cfg: dict,
    exp_dir: Path,
    logger: Any,
    ckpt_filename: str = "best",
    extra_callbacks: list | None = None,
    *,
    fast_dev_run: bool = False,
    precision: str | None = None,
    enable_progress_bar: bool = True,
    early_stop_patience_key: str = "early_stop_patience",
) -> tuple[Trainer, ModelCheckpoint]:
    """Build a Lightning Trainer with standard callbacks.

    Returns
    -------
    trainer, ckpt_callback
        The Trainer and the ModelCheckpoint callback (so callers can read
        ``ckpt_callback.best_model_path`` after training).
    """
    ckpt_cb = ModelCheckpoint(
        dirpath=str(exp_dir / "checkpoints"),
        filename=ckpt_filename,
        monitor="val_loss",
        mode="min",
        save_last=True,
        verbose=True,
    )
    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=cfg.get(early_stop_patience_key, 15),
        mode="min",
        verbose=True,
    )
    callbacks = [ckpt_cb, early_stop, LearningRateMonitor("epoch")]
    if extra_callbacks:
        callbacks.extend(extra_callbacks)

    trainer_kwargs: dict[str, Any] = dict(
        max_epochs=cfg.get("max_epochs", 100),
        accelerator="auto",
        devices=1,
        logger=logger,
        callbacks=callbacks,
        log_every_n_steps=10,
        fast_dev_run=fast_dev_run,
        deterministic=True,
        enable_progress_bar=enable_progress_bar,
    )
    if precision is not None:
        trainer_kwargs["precision"] = precision
    # Opt-in only (absent/0/null -> Lightning's default of no clipping, so no
    # existing config's behaviour changes). Added specifically for EuroCropsML
    # runs that showed a "converges cleanly, then drifts for many epochs"
    # instability pattern in both reconstruction and predictive-SSL training --
    # bounds how far any single noisy batch (this dataset's per-window real-
    # observation count varies 2-10, unlike CropGlobe's fixed 6) can push the
    # weights in one step.
    gradient_clip_val = cfg.get("gradient_clip_val")
    if gradient_clip_val:
        trainer_kwargs["gradient_clip_val"] = gradient_clip_val
    # Opt-in short runs (check_training): absent in every config, so the letter's runs are unaffected.
    for key in ("limit_train_batches", "limit_val_batches"):
        if cfg.get(key) is not None:
            trainer_kwargs[key] = cfg[key]

    return Trainer(**trainer_kwargs), ckpt_cb

