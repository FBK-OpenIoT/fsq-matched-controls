"""CropGlobe PyTorch Dataset and PyTorch Lightning DataModule.

Data format in raw .npz files
-------------------------------
    fea : (n_parcels, 10_bands, n_timestamps)   float64  DN values [0, ~16000]
    lbl : (n_parcels,)                           int32    harmonised crop codes

After loading we transpose fea → (n_parcels, n_timestamps, 10_bands) and
apply a sliding window so each window covers a fixed number of calendar days.

Token window design
-------------------
Phenological changes are calendar-time phenomena.  We therefore parameterise
tokenisation in *calendar days*, not in observation counts:

    days_per_token  — calendar days each token covers (default 30 ≈ 1 month)
    overlap_days    — calendar-day overlap between consecutive tokens (default 0)

The number of observations per token and the stride (in observations) are
derived automatically from the source cadence:

    obs_per_token = days_per_token  // cadence_days
    stride_obs    = (days_per_token - overlap_days) // cadence_days   (≥ 1)
    n_tokens      = (T - obs_per_token) // stride_obs + 1

This keeps the semantics consistent across datasets with different acquisition
rates, and makes overlap a physically meaningful concept.

Sources
-------
    Median      Median/median_S2_{CC}_month5-11_day5.npz   7 months, cadence=5 days
    for_CropNet for_CropNet/median_S2_{CC}_month1-12_day5.npz  12 months, cadence=5 days

Available countries (CC codes used in filenames)
------------------------------------------------
    ARG AUS BEL CHN FRA GBR NLD USA USA_17 USA_20
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, random_split
from lightning.pytorch import LightningDataModule

from fsq_matched_controls.data.normalization import NormStats, compute_norm_stats

log = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────

#: All file-system country codes present in the dataset.
ALL_COUNTRIES: list[str] = [
    "ARG", "AUS", "BEL", "CHN", "FRA", "GBR", "NLD",
    "USA", "USA_17", "USA_20",
]

#: Best-effort harmonised label names drawn from the CropGlobe p=repo CropNet/util/label.py (Tong et al. 2025).
LABEL_NAMES: dict[int, str] = {
    0: "Corn",
    1: "Soybeans",
    2: "Rice",
    3: "Wheat",
    4: "Sugarcane",
    5: "Cotton",
    6: "Other",
}

#: Per-source configuration (folder, filename pattern, month window info).
SOURCE_CONFIG: dict[str, dict] = {
    "Median": {
        "dir": "Median",
        "pattern": "median_S2_{cc}_month5-11_day5.npz",
        "n_months": 7,
        "cadence_days": 5,            # "day5" = 5-day composites
        "month_labels": list(range(5, 12)),   # May–Nov
    },
    "for_CropNet": {
        "dir": "for_CropNet",
        "pattern": "median_S2_{cc}_month1-12_day5.npz",
        "n_months": 12,
        "cadence_days": 5,            # "day5" = 5-day composites
        "month_labels": list(range(1, 13)),   # Jan–Dec
    },
}

# ── Low-level helpers ─────────────────────────────────────────────────────────


def _npz_path(raw_dir: Path, country: str, source: str) -> Path:
    cfg = SOURCE_CONFIG[source]
    return raw_dir / cfg["dir"] / cfg["pattern"].format(cc=country)


def _load_country_raw(
    raw_dir: Path,
    country: str,
    source: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (fea, lbl) with fea shape (N, T, B) float32, lbl (N,) int64."""
    path = _npz_path(raw_dir, country, source)
    if not path.exists():
        raise FileNotFoundError(
            f"CropGlobe file not found: {path}\n"
            f"Download it first with the data pipeline."
        )
    data = np.load(path)
    fea = data["fea"].astype(np.float32)   # (N, B=10, T)
    lbl = data["lbl"].astype(np.int64)     # (N,)
    fea = fea.transpose(0, 2, 1)           # → (N, T, B)
    log.debug("  %s: %d parcels, %d timestamps, %d bands", country, *fea.shape)
    return fea, lbl


def _apply_sliding_window(
    fea: np.ndarray,
    obs_per_token: int,
    stride_obs: int,
) -> np.ndarray:
    """Sliding-window tokenisation over the time axis.

    Parameters
    ----------
    fea          : (N, T, B) — parcels × timestamps × bands
    obs_per_token: number of observations per token window
    stride_obs   : step between window starts (in observations);
                   ``stride_obs < obs_per_token`` produces overlapping windows

    Returns
    -------
    (N, n_tokens, obs_per_token, B)
    where ``n_tokens = (T - obs_per_token) // stride_obs + 1``
    """
    N, T, B = fea.shape
    starts = list(range(0, T - obs_per_token + 1, stride_obs))
    if not starts:
        raise ValueError(
            f"obs_per_token={obs_per_token} > T={T}. "
            "Reduce days_per_token or use a source with more timestamps."
        )
    # Stack along a new window axis — shape (N, n_tokens, obs_per_token, B)
    windows = np.stack([fea[:, s: s + obs_per_token, :] for s in starts], axis=1)
    return windows


def _apply_windowing(fea: np.ndarray, n_months: int) -> np.ndarray:
    """Legacy helper: non-overlapping monthly windows.

    Kept for backward compatibility with notebooks that import it directly.
    Prefer ``_apply_sliding_window`` for new code.
    """
    N, T, B = fea.shape
    obs = T // n_months
    return _apply_sliding_window(fea, obs_per_token=obs, stride_obs=obs)


# ── Window-level dataset (used for FSQ autoencoder training) ─────────────────


class CropGlobeWindowDataset(Dataset):
    """One sample = one calendar-month window from one parcel.

    Returned tensors
    ----------------
    window : float32 Tensor  (obs_per_window, n_bands)
    label  : int64 Tensor    scalar
    """

    def __init__(
        self,
        windows: np.ndarray,   # (n_total_windows, obs, bands)
        labels: np.ndarray,    # (n_total_windows,)
    ) -> None:
        self.windows = torch.from_numpy(windows).float()
        self.labels = torch.from_numpy(labels).long()

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        return self.windows[idx], self.labels[idx]


# ── Parcel-level dataset (used for tokenisation / evaluation) ─────────────────


class CropGlobeParcelDataset(Dataset):
    """One sample = all month windows for a single parcel.

    Returned tensors
    ----------------
    windows : float32 Tensor  (n_months, obs_per_window, n_bands)
    label   : int64 Tensor    scalar
    country : str
    """

    def __init__(
        self,
        windows: np.ndarray,      # (N, n_months, obs, bands)
        labels: np.ndarray,       # (N,)
        countries: list[str],     # (N,) — country code per parcel
    ) -> None:
        self.windows = torch.from_numpy(windows).float()
        self.labels = torch.from_numpy(labels).long()
        self.countries = countries

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor, str]:
        return self.windows[idx], self.labels[idx], self.countries[idx]


# ── PyTorch Lightning DataModule ────────────────────────────────────────────


class CropGlobeDataModule(LightningDataModule):
    """Lightning DataModule for CropGlobe FSQ autoencoder training.

    Splits
    ------
    Train  : 90 % of parcels from ``train_countries`` (windows)
    Val    : 10 % of parcels from ``train_countries`` (windows, in-dist monitoring)
    Test   : Parcels from ``hold_out_countries`` (windows, zero-shot eval)

    Normalization stats are computed from training parcels only, then applied
    to all splits.  Stats are saved to ``experiment_dir / norm_stats.json``.

    Token window design
    -------------------
    Windows are defined in **calendar days** so the semantics are consistent
    across datasets with different acquisition cadences:

        days_per_token  — calendar days each token covers (default 30 ≈ 1 month)
        overlap_days    — overlap between consecutive token windows in calendar
                          days (default 0 = non-overlapping)

    The actual observation counts (``obs_per_window``) and stride are derived
    automatically from the source cadence (``cadence_days`` in SOURCE_CONFIG).

    Parameters
    ----------
    raw_dir           : Path to ``<data_root>/cropglobe/raw``.
    source            : ``"Median"`` (month 5–11) or ``"for_CropNet"`` (month 1–12).
    train_countries   : List of country codes to use for training/val.
                        Defaults to ALL_COUNTRIES minus hold_out_countries.
    hold_out_countries: Countries withheld from all training (LOOCV test set).
    val_fraction      : Fraction of train parcels to use for validation.
    batch_size        : Window batch size for DataLoaders.
    num_workers       : DataLoader worker count. Use 0 on Windows.
    random_seed       : Seed for parcel-level train/val split.
    experiment_dir    : Where to save norm_stats.json and other artefacts.
    dn_scale          : Divisor to convert DN → [0, 1] range before z-scoring.
    days_per_token    : Calendar days covered by each token (default 30).
    overlap_days      : Calendar-day overlap between consecutive tokens (default 0).
    norm_stats_path   : If provided, load normalisation statistics from this
                        ``norm_stats.json`` file instead of recomputing them from
                        ``train_countries`` data.  **Always set this when running
                        probe / evaluation notebooks** so the feature distribution
                        seen by the probe matches exactly what the SSL model was
                        trained on.  The file is saved automatically during training
                        at ``<experiment_dir>/norm_stats.json``.
    stratify_val_by_country : If ``True``, the 10 % validation split is drawn
                        proportionally from **each country individually** before
                        merging, guaranteeing every training country appears in the
                        val set.  Default ``False`` (random shuffle of the merged
                        pool — fast and fine when all countries have many parcels).
    """

    def __init__(
        self,
        raw_dir: Path | str,
        source: str = "Median",
        train_countries: Optional[list[str]] = None,
        hold_out_countries: Optional[list[str]] = None,
        val_fraction: float = 0.1,
        batch_size: int = 512,
        num_workers: int = 0,
        random_seed: int = 1904,
        experiment_dir: Optional[Path | str] = None,
        dn_scale: float = 10_000.0,
        days_per_token: int = 30,
        overlap_days: int = 0,
        norm_stats_path: Optional[Path | str] = None,
        stratify_val_by_country: bool = False,
    ) -> None:
        super().__init__()
        self.raw_dir = Path(raw_dir)
        self.source = source
        self.hold_out_countries = hold_out_countries or []
        self.train_countries = train_countries or [
            c for c in ALL_COUNTRIES if c not in self.hold_out_countries
        ]
        self.val_fraction = val_fraction
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.random_seed = random_seed
        self.experiment_dir = Path(experiment_dir) if experiment_dir else None
        self.dn_scale = dn_scale
        self.days_per_token = days_per_token
        self.overlap_days = overlap_days
        self._norm_stats_path: Optional[Path] = (
            Path(norm_stats_path) if norm_stats_path else None
        )
        self.stratify_val_by_country = stratify_val_by_country

        # Derived from source config
        src_cfg = SOURCE_CONFIG[source]
        self.cadence_days: int = src_cfg["cadence_days"]
        self.n_months: int = src_cfg["n_months"]
        self.n_bands: int = 10

        # Derived from calendar-day params + cadence
        self.obs_per_window: int = days_per_token // self.cadence_days
        stride_days = days_per_token - overlap_days
        if stride_days <= 0:
            raise ValueError(
                f"overlap_days ({overlap_days}) must be < days_per_token ({days_per_token})."
            )
        self._stride_obs: int = max(1, stride_days // self.cadence_days)

        # Set after setup()
        self.norm_stats: Optional[NormStats] = None
        self.n_tokens: Optional[int] = None       # tokens per parcel (set in setup)

        self._train_ds: Optional[CropGlobeWindowDataset] = None
        self._val_ds: Optional[CropGlobeWindowDataset] = None
        self._test_ds: Optional[CropGlobeWindowDataset] = None
        self._test_parcel_ds: Optional[CropGlobeParcelDataset] = None
        self._train_parcel_ds: Optional[CropGlobeParcelDataset] = None
        self._val_parcel_ds: Optional[CropGlobeParcelDataset] = None
        # All train_countries parcels (no 80/10 split) — for probe training only
        self._all_train_parcel_ds: Optional[CropGlobeParcelDataset] = None

    # ── setup ────────────────────────────────────────────────────────────────

    def setup(self, stage: Optional[str] = None) -> None:
        """Load, window, normalise, and split all data."""
        if self._train_ds is not None:
            return  # already set up

        log.info("CropGlobeDataModule.setup — source=%s", self.source)

        # 1. Load training countries
        train_fea_list, train_lbl_list, train_cc_list = [], [], []
        for cc in self.train_countries:
            try:
                fea, lbl = _load_country_raw(self.raw_dir, cc, self.source)
                train_fea_list.append(fea)
                train_lbl_list.append(lbl)
                train_cc_list.extend([cc] * len(lbl))
                log.info("  [train] %s: %d parcels", cc, len(lbl))
            except FileNotFoundError as e:
                log.warning("  Skipping %s: %s", cc, e)

        if not train_fea_list:
            raise RuntimeError("No training data loaded. Check raw_dir and country codes.")

        train_fea = np.concatenate(train_fea_list, axis=0)   # (N_train, T, B)
        train_lbl = np.concatenate(train_lbl_list, axis=0)   # (N_train,)
        # train_cc_list already has per-parcel country codes (N_train,)

        # 2. Determine windowing parameters
        T = train_fea.shape[1]
        self.n_tokens = (T - self.obs_per_window) // self._stride_obs + 1
        log.info(
            "  Windowing: %d timestamps | cadence=%d days | days_per_token=%d "
            "→ obs_per_token=%d | overlap=%d days → stride=%d obs | n_tokens=%d",
            T, self.cadence_days, self.days_per_token,
            self.obs_per_window, self.overlap_days, self._stride_obs, self.n_tokens,
        )

        # 3. Window training data → (N, n_tokens, obs_per_token, B)
        train_win = _apply_sliding_window(train_fea, self.obs_per_window, self._stride_obs)

        # 4. Compute (or load) normalization stats — always from SSL training set
        if self._norm_stats_path is not None:
            self.norm_stats = NormStats.load(self._norm_stats_path)
            log.info("  Norm stats loaded ← %s", self._norm_stats_path)
            # Cross-check: warn if dn_scale mismatch
            if self.norm_stats.dn_scale != self.dn_scale:
                log.warning(
                    "  dn_scale mismatch: DataModule has %.1f but norm_stats.json "
                    "has %.1f.  Using the value from norm_stats.json.",
                    self.dn_scale, self.norm_stats.dn_scale,
                )
        else:
            self.norm_stats = compute_norm_stats(train_win, dn_scale=self.dn_scale)
            if self.experiment_dir:
                self.experiment_dir.mkdir(parents=True, exist_ok=True)
                self.norm_stats.save(self.experiment_dir / "norm_stats.json")
                log.info("  Norm stats saved → %s", self.experiment_dir / "norm_stats.json")

        # 5. Normalise training windows
        train_win_norm = self.norm_stats.apply(train_win)    # (N, W, obs, B)

        # 6. Parcel-level train/val split (split parcels, then flatten to windows)
        rng = np.random.default_rng(self.random_seed)
        n_parcels = len(train_win_norm)

        if self.stratify_val_by_country:
            # Stratified split: take val_fraction from each country individually
            # so every country is guaranteed to appear in the val set.
            val_idx_list, trn_idx_list = [], []
            offset = 0
            for cc in self.train_countries:
                n_cc = train_cc_list.count(cc)
                if n_cc == 0:
                    continue
                cc_local = rng.permutation(n_cc)          # indices within this country
                n_val_cc = max(1, int(n_cc * self.val_fraction))
                val_idx_list.append(cc_local[:n_val_cc] + offset)
                trn_idx_list.append(cc_local[n_val_cc:] + offset)
                offset += n_cc
            val_idx = np.concatenate(val_idx_list)
            trn_idx = np.concatenate(trn_idx_list)
            log.info(
                "  Stratified val split: %d train parcels | %d val parcels",
                len(trn_idx), len(val_idx),
            )
        else:
            # Default: random shuffle of the merged pool (fast, fine for large datasets)
            idx = rng.permutation(n_parcels)
            n_val = max(1, int(n_parcels * self.val_fraction))
            val_idx = idx[:n_val]
            trn_idx = idx[n_val:]

        def _to_window_ds(win: np.ndarray, lbl: np.ndarray) -> CropGlobeWindowDataset:
            # win: (N, W, obs, B) → flatten parcel × month → (N*W, obs, B)
            N, W, obs, B = win.shape
            flat_win = win.reshape(N * W, obs, B)
            flat_lbl = np.repeat(lbl, W)
            return CropGlobeWindowDataset(flat_win, flat_lbl)

        self._train_ds = _to_window_ds(train_win_norm[trn_idx], train_lbl[trn_idx])
        self._val_ds = _to_window_ds(train_win_norm[val_idx], train_lbl[val_idx])

        # Parcel-level train dataset (for tokenisation / predictive SSL training)
        trn_cc = [train_cc_list[i] for i in trn_idx]
        self._train_parcel_ds = CropGlobeParcelDataset(
            train_win_norm[trn_idx], train_lbl[trn_idx], trn_cc
        )

        # Parcel-level val dataset (for predictive SSL validation)
        val_cc = [train_cc_list[i] for i in val_idx]
        self._val_parcel_ds = CropGlobeParcelDataset(
            train_win_norm[val_idx], train_lbl[val_idx], val_cc
        )

        # ALL train_countries parcels — no 80/10 split.
        # Use parcel_dataloader("all_train") when training probes so no data is wasted.
        # NOTE: the SSL model never saw the 10% val during training — using all parcels
        # for the downstream probe is correct (the model is frozen, no leakage).
        self._all_train_parcel_ds = CropGlobeParcelDataset(
            train_win_norm, train_lbl, train_cc_list
        )

        log.info(
            "  Train windows: %d | Val windows: %d",
            len(self._train_ds), len(self._val_ds),
        )

        # 7. Load and normalise hold-out countries
        if self.hold_out_countries:
            ho_fea_list, ho_lbl_list, ho_cc_list = [], [], []
            for cc in self.hold_out_countries:
                try:
                    fea, lbl = _load_country_raw(self.raw_dir, cc, self.source)
                    ho_fea_list.append(fea)
                    ho_lbl_list.append(lbl)
                    ho_cc_list.extend([cc] * len(lbl))
                    log.info("  [held-out] %s: %d parcels", cc, len(lbl))
                except FileNotFoundError as e:
                    log.warning("  Skipping held-out %s: %s", cc, e)

            if ho_fea_list:
                ho_fea = np.concatenate(ho_fea_list, axis=0)
                ho_lbl = np.concatenate(ho_lbl_list, axis=0)
                ho_win = _apply_sliding_window(ho_fea, self.obs_per_window, self._stride_obs)
                ho_win_norm = self.norm_stats.apply(ho_win)

                # Window-level test dataset (for standard metrics)
                N, W, obs, B = ho_win_norm.shape
                flat_win = ho_win_norm.reshape(N * W, obs, B)
                flat_lbl = np.repeat(ho_lbl, W)
                self._test_ds = CropGlobeWindowDataset(flat_win, flat_lbl)

                # Parcel-level test dataset (for tokenisation + linear probe)
                self._test_parcel_ds = CropGlobeParcelDataset(
                    ho_win_norm, ho_lbl, ho_cc_list
                )
                log.info("  Test windows: %d", len(self._test_ds))

    # ── DataLoaders ──────────────────────────────────────────────────────────

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self._train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self._val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )

    def test_dataloader(self) -> DataLoader:
        if self._test_ds is None:
            raise RuntimeError("No hold-out countries configured.")
        return DataLoader(
            self._test_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )

    def parcel_dataloader(
        self,
        split: str = "held_out",
        batch_size: int = 64,
    ) -> DataLoader:
        """Parcel-level dataloader for tokenisation and probe evaluation.

        Parameters
        ----------
        split : ``"held_out"``   — held-out/test parcels (zero-shot countries).
                ``"train"``      — 80 % of train_countries parcels (SSL-training
                                   partition only; the 10 % val is excluded).
                ``"all_train"``  — **all** train_countries parcels (100 %, no
                                   90/10 split).  *Preferred for probe training*:
                                   the model is frozen so using the full set is
                                   correct and wastes no labelled data.
        """
        if split == "held_out":
            if self._test_parcel_ds is None:
                raise RuntimeError("No hold-out countries configured.")
            ds = self._test_parcel_ds
        elif split == "train":
            if self._train_parcel_ds is None:
                raise RuntimeError("Call setup() first.")
            ds = self._train_parcel_ds
        elif split == "all_train":
            if self._all_train_parcel_ds is None:
                raise RuntimeError("Call setup() first.")
            ds = self._all_train_parcel_ds
        else:
            raise ValueError(
                f"Unknown split '{split}'. Use 'held_out', 'train', or 'all_train'."
            )
        return DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,           # always single-process for eval
            collate_fn=_parcel_collate,
        )

    def parcel_train_dataloader(self, batch_size: int = 64) -> DataLoader:
        """Parcel-level training DataLoader for predictive SSL.

        Returns shuffled batches of full-year parcel sequences:
        ``(windows, labels, countries)`` where
        ``windows`` is ``(batch, n_tokens, obs_per_window, n_bands)``.

        Use this in place of ``train_dataloader()`` when training
        ``FSQPredictiveSSL`` — the model needs the full sequence of monthly
        tokens per parcel, not individual windows.
        """
        if self._train_parcel_ds is None:
            raise RuntimeError("Call setup() first.")
        return DataLoader(
            self._train_parcel_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
            collate_fn=_parcel_collate,
        )

    def parcel_val_dataloader(self, batch_size: int = 64) -> DataLoader:
        """Parcel-level validation DataLoader for predictive SSL.

        Returns un-shuffled batches of full-year parcel sequences.
        Mirrors ``parcel_train_dataloader()`` for the validation split.
        """
        if self._val_parcel_ds is None:
            raise RuntimeError("Call setup() first.")
        return DataLoader(
            self._val_parcel_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
            collate_fn=_parcel_collate,
        )

    # ── Properties ───────────────────────────────────────────────────────────

    @property
    def n_classes(self) -> int:
        """Number of unique crop labels in the training set."""
        if self._train_ds is None:
            raise RuntimeError("Call setup() first.")
        return int(self._train_ds.labels.max().item()) + 1


def _parcel_collate(
    batch: list[tuple[Tensor, Tensor, str]],
) -> tuple[Tensor, Tensor, list[str]]:
    """Collate function for parcel-level dataset."""
    windows = torch.stack([b[0] for b in batch])
    labels = torch.stack([b[1] for b in batch])
    countries = [b[2] for b in batch]
    return windows, labels, countries

