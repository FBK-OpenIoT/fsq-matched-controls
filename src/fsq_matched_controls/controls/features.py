"""Monthly band means: the input of both controls (and of the raw-band probe).

Each month of a series is represented by the mean of its observations in each of the ten bands, after the per-band normalisation
of the letter's runs (CropGlobe: the transductive runs' statistics, computed on the ten splits; EuroCropsML: the runs' statistics,
computed on the Latvia + Portugal pretraining data; both are stored in the shipped checkpoints). On EuroCropsML only real
observations enter the mean, and a month without any is zero (invalid).

    CropGlobe   : probe pool (the eight splits other than France and Australia), France, Australia: (N, 7, 10)
    EuroCropsML : probe pool (Latvia + Portugal pretraining parcels), Estonia: (N, 9, 10), with a validity mask (N, 9)

The arrays are built with the package's data modules from the datasets in FSQ_DATA_ROOT and cached in FSQ_FEATURES
(controls_monthly_<dataset>.npz). MONTHLY_SHA256 holds fingerprints of the letter's own arrays.
"""
from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

import numpy as np
import torch

from fsq_matched_controls import paths
from fsq_matched_controls.probe.linear_probe import build_raw_features

log = logging.getLogger(__name__)

PROBE_HOLDOUT = ["FRA", "AUS"]
# checkpoints whose norm_stats are the letter's normalisation of the monthly band means (identical in every run of that group)
NORM_SOURCE = {
    "cropglobe": "cropglobe_predictive_ssl_ov0_625_seed2024_vm_holdout_none",            # any transductive CropGlobe run
    "eurocropsml": "eurocropml_predictive_ssl_625_heldout_estonia_gradclip_seed2024_vm_holdout_Estonia",  # any EuroCropsML run
}
# SHA-256 of the letter's monthly band-mean arrays (float32, months x bands flattened), split -> digest
MONTHLY_SHA256: dict[str, dict[str, str]] = {
    "cropglobe": {"train": "433f52602e398f93a39b42a504430c47837c05dc4d4d1a70a9de9af0c9eab78f",
                  "FRA": "4a983ad69f501bff2280356d3b91b97b663bf0e7bb21d5aed8b2f618bc5b69eb",
                  "AUS": "58ab05a1410144e0b7a5bb0bc7a1b967108dd1b6fdcdda7abb13f6ebf10ae8b8"},
    "eurocropsml": {"train": "84796d6239c5667a1f98c88c8825b5f50dead31e17213f96a9613acab187d50c",
                    "estonia": "c6f88e8e4420a84df962b3714263f85de62c47dabfff1c33a74cdcfadaf95235"},
}


def sha256(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a, dtype=np.float32).reshape(len(a), -1).tobytes()).hexdigest()


def _norm_stats_file(dataset: str) -> Path:
    ck = torch.load(paths.CHECKPOINTS / f"{NORM_SOURCE[dataset]}.pt", map_location="cpu", weights_only=True)
    out = paths.FEATURES / f"norm_stats_{dataset}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(ck["norm_stats"]), encoding="utf-8")
    return out


def cropglobe_monthly(cache: bool = True) -> dict:
    """{"X": {split: (N, 7, 10)}, "y": {split: (N,)}, "V": {split: (N, 7) all True}} for splits train, FRA, AUS."""
    f = paths.FEATURES / "controls_monthly_cropglobe.npz"
    if cache and f.exists():
        z = np.load(f)
        X = {s: z[f"X_{s}"] for s in ("train", "FRA", "AUS")}; y = {s: z[f"y_{s}"] for s in X}
    else:
        from fsq_matched_controls.data.cropglobe import CropGlobeDataModule
        dm = CropGlobeDataModule(raw_dir=paths.DATA_ROOT / "cropglobe" / "raw", hold_out_countries=list(PROBE_HOLDOUT),
                                 norm_stats_path=str(_norm_stats_file("cropglobe")), source="Median", train_countries=None,
                                 batch_size=64, days_per_token=30, overlap_days=0, dn_scale=10_000.0)
        dm.setup()
        Xtr, ytr, _ = build_raw_features(dm.parcel_dataloader(split="all_train", batch_size=64))
        Xho, yho, cho = build_raw_features(dm.parcel_dataloader(split="held_out", batch_size=64))
        cho = np.array(cho)
        X = {"train": Xtr, "FRA": Xho[cho == "FRA"], "AUS": Xho[cho == "AUS"]}
        y = {"train": ytr, "FRA": yho[cho == "FRA"], "AUS": yho[cho == "AUS"]}
        X = {s: a.reshape(len(a), 7, 10) for s, a in X.items()}
        f.parent.mkdir(parents=True, exist_ok=True)
        np.savez(f, **{f"X_{s}": X[s] for s in X}, **{f"y_{s}": y[s] for s in y})
    V = {s: np.ones(X[s].shape[:2], bool) for s in X}
    return dict(X=X, y=y, V=V)


def eurocropsml_monthly(cache: bool = True) -> dict:
    """{"X": {split: (N, 9, 10)}, "y": {split: (N,)}, "V": {split: (N, 9)}} for splits train (Latvia + Portugal) and estonia."""
    f = paths.FEATURES / "controls_monthly_eurocropsml.npz"
    if cache and f.exists():
        z = np.load(f)
        X = {s: z[f"X_{s}"] for s in ("train", "estonia")}; y = {s: z[f"y_{s}"] for s in X}
    else:
        from fsq_matched_controls.data.eurocropml import EuroCropsMLDataModule
        dm = EuroCropsMLDataModule(data_root=str(paths.DATA_ROOT), split_name="latvia_portugal_vs_estonia", finetune_variant="all",
                                   batch_size=64, norm_stats_path=str(_norm_stats_file("eurocropsml")), w_max=10, w_min=2)
        dm.setup()
        Xtr, ytr, _ = build_raw_features(dm.parcel_dataloader(split="train", batch_size=64))
        Xho, yho, _ = build_raw_features(dm.parcel_dataloader(split="held_out", batch_size=64))
        X = {"train": Xtr.reshape(len(Xtr), 9, 10), "estonia": Xho.reshape(len(Xho), 9, 10)}
        y = {"train": ytr, "estonia": yho}
        f.parent.mkdir(parents=True, exist_ok=True)
        np.savez(f, **{f"X_{s}": X[s] for s in X}, **{f"y_{s}": y[s] for s in y})
    V = {s: ~(X[s] == 0).all(axis=2) for s in X}
    return dict(X=X, y=y, V=V)


def check_fingerprints(dataset: str, data: dict) -> bool | None:
    """True if every split equals the letter's array (SHA-256), None if no fingerprint is stored."""
    ref = MONTHLY_SHA256.get(dataset)
    if not ref:
        return None
    return all(sha256(data["X"][s]) == h for s, h in ref.items())
