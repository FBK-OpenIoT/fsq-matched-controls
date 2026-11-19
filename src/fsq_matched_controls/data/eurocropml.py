"""EuroCropsML PyTorch Dataset and PyTorch Lightning DataModule.

Data format
-----------
``preprocess/<NUTS3>_<parcel_id>_<EC_hcat_c>.npz`` (already cloud-filtered),
each containing:
    data   : (T, 13) int64        — raw Sentinel-2 DN reflectance, T varies
                                     per parcel (irregular real acquisitions)
    dates  : (T,) datetime64[D]   — real, irregular acquisition dates
    center : (lon, lat)

``split/<split_name>/pretrain/region_split.json``:
    {"train": [...npz filenames...], "val": [...]}
``split/<split_name>/finetune/region_split_<variant>.json``  (variant in
    1/5/10/20/100/200/500/all):
    {"train": [...k-shot held-out-region...], "val": [...], "test": [...]}

Unlike CropGlobe (fixed 5-day cadence, uniform month grid, zero missing data),
EuroCropsML has a variable number of cloud-free daily observations per
parcel at irregular dates, with some calendar months structurally empty
(e.g. Baltic winter snow cover). This module keeps CropGlobe's "one token =
one calendar month" semantic, but — unlike an earlier version of this module
that median-composited each month into a single vector before the encoder
ever saw it — each token's real observations are now passed to the encoder
as a padded, variable-length sequence, and the encoder does its own learned
temporal pooling over them (exactly like CropGlobe's within-window
aggregation). Two masks travel alongside the data, at two different levels:

    valid_mask : (n_tokens,)          — whole month has >= w_min real
                 observations at all. See ``FSQPredictiveSSL``'s
                 ``valid_mask`` parameter for how missing months are handled
                 during masked-token training.
    obs_mask   : (n_tokens, w_max)    — which observations *within* a valid
                 month are real vs. zero-padding. See
                 ``TemporalCNNEncoder``'s module docstring (encoder.py) for
                 how this drives masked pooling + why ``norm_type="layer"``
                 matters once padding is involved.

``w_max`` and ``w_min`` were picked from real data, not guessed: a 500-parcel
survey of per-(parcel, month) observation counts (Mar-Nov, Latvia+Portugal)
found p90=9, p95=10, max=15, with ~49% mean padding fraction already at
w_max=10 -- so growing w_max further buys negligible extra coverage for a lot
more padding. w_max=10 keeps truncation to ~1.7% of month-observations.
w_min=2 (need at least 2 real looks for a month to count as observed, not
imputed) matches the "don't impute" principle behind valid_mask.

Month range and band mapping were empirically validated (see
``docs/research`` notes / survey script), not guessed:
    - Coverage across a 300-parcel sample of the Latvia+Portugal pretrain
      split: months Mar-Nov sit at 95.7-100% parcel coverage; Dec/Jan/Feb sit
      at 28-49% — excluded from the fixed token grid entirely rather than
      carried as near-always-missing dead weight positions.
    - EuroCropsML raw band order (13 bands, standard Sentinel-2 order):
      [B1,B2,B3,B4,B5,B6,B7,B8,B8A,B9,B10,B11,B12]. CropGlobe's order (10
      bands, used everywhere else in this codebase — encoder, FSQ, the token
      interpretability notebooks' hardcoded RED=2/NIR=3): [B2,B3,B4,B8,B5,B6,
      B7,B8A,B11,B12]. ``BAND_INDEX`` below selects+reorders EuroCropsML's 13
      bands into that same 10-band convention so RED/NIR keep landing on
      indices (2,3).

Labels
------
The crop code (``EC_hcat_c``) is embedded directly in each npz filename — no
parquet join needed for training. The label vocabulary is built from whatever
codes appear in the pretrain split (141 in the Latvia+Portugal pretrain/train
split, vs. 176 raw codes across all 3 countries per the exploration notebook —
the difference is finetune/Estonia-only codes). Codes seen only in the
held-out region's finetune split and not in the training vocabulary are
mapped to ``-1`` (genuinely unseen classes are a real part of a cross-region
few-shot benchmark, not a bug to hide).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from lightning.pytorch import LightningDataModule

from fsq_matched_controls.data.normalization import NormStats, compute_norm_stats

log = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────

#: Calendar months covered by each token — Mar-Nov, validated against actual
#: per-month observation coverage (see module docstring). 9 tokens.
MONTHS: list[int] = list(range(3, 12))
N_TOKENS: int = len(MONTHS)

#: Default max real observations kept per month window (padded/truncated to
#: this) and minimum real observations for a month to count as valid, not
#: imputed. See module docstring for how these were picked from real data.
DEFAULT_W_MAX: int = 10
DEFAULT_W_MIN: int = 2

#: Selects + reorders EuroCropsML's 13 raw bands into CropGlobe's 10-band
#: convention: [B2,B3,B4,B8,B5,B6,B7,B8A,B11,B12].
BAND_INDEX: list[int] = [1, 2, 3, 7, 4, 5, 6, 8, 11, 12]
N_BANDS: int = len(BAND_INDEX)

#: NUTS3-prefix -> country/region name, for logging + the per-parcel
#: "countries" field (interface parity with CropGlobeParcelDataset).
REGION_NAMES: dict[str, str] = {
    "EE": "Estonia",
    "LV": "Latvia",
    "PT": "Portugal",
}


# ── Low-level helpers ─────────────────────────────────────────────────────────


def _parse_ec_hcat_c(filename: str) -> str:
    """Extract the crop code from ``<NUTS3>_<parcel_id>_<EC_hcat_c>.npz``."""
    return filename[:-4].split("_")[-1]


def _region_of(filename: str) -> str:
    """Extract the 2-letter country prefix (e.g. 'LV') from a NUTS3-led filename."""
    return filename[:2]


def _load_split_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"EuroCropsML split file not found: {path}")
    return json.loads(path.read_text())


def _npz_to_padded_months(
    npz_path: Path,
    w_max: int,
    w_min: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load one parcel's ragged (T,13) array onto the fixed ``MONTHS`` grid,
    keeping real observations (padded/truncated to ``w_max``) instead of
    collapsing them to a median — the encoder does its own temporal pooling.

    Returns
    -------
    windows    : (N_TOKENS, w_max, N_BANDS) float32 — real DN reflectance,
                 zero-padded past each month's real observation count
    obs_mask   : (N_TOKENS, w_max) bool — True at real (non-padded)
                 observations within a month
    valid_mask : (N_TOKENS,) bool — True where the month has >= w_min real
                 observations (not just >=1 -- see module docstring)
    """
    z = np.load(npz_path, allow_pickle=True)
    data = z["data"][:, BAND_INDEX].astype(np.float32)   # (T, N_BANDS)
    dates = z["dates"]

    windows = np.zeros((N_TOKENS, w_max, N_BANDS), dtype=np.float32)
    obs_mask = np.zeros((N_TOKENS, w_max), dtype=bool)
    valid_mask = np.zeros(N_TOKENS, dtype=bool)

    if len(dates) == 0:
        return windows, obs_mask, valid_mask

    month_of_obs = np.array([int(str(d)[5:7]) for d in dates])
    for t_idx, month in enumerate(MONTHS):
        sel = np.where(month_of_obs == month)[0]
        n_real = len(sel)
        if n_real < w_min:
            continue
        n_keep = min(n_real, w_max)
        windows[t_idx, :n_keep] = data[sel[:n_keep]]
        obs_mask[t_idx, :n_keep] = True
        valid_mask[t_idx] = True

    return windows, obs_mask, valid_mask


def _load_npz_list(
    preprocess_dir: Path,
    filenames: list[str],
    w_max: int,
    w_min: int,
    max_parcels: Optional[int] = None,
    log_every: int = 5_000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], list[str]]:
    """Return (windows, obs_mask, valid_mask, ec_hcat_c_codes, regions) for a
    list of npz filenames.

    windows    : (N, N_TOKENS, w_max, N_BANDS) float32
    obs_mask   : (N, N_TOKENS, w_max) bool
    valid_mask : (N, N_TOKENS) bool

    Parameters
    ----------
    max_parcels : If given, only the first ``max_parcels`` filenames are
        loaded — for fast local smoke tests (``--fast-dev-run`` still pays
        the full ``setup()`` load cost otherwise, since this loop is what
        populates the Dataset, not the DataLoader/Trainer).
    log_every   : Log progress every N files — this loop is single-threaded
        and each ``np.load()`` can be slow on a OneDrive "Files On-Demand"
        folder (cloud-only files must be fetched first), so silent progress
        here is easy to mistake for a hang.
    """
    if max_parcels is not None:
        filenames = filenames[:max_parcels]

    total = len(filenames)
    log.info("  Loading %d npz files from %s ...", total, preprocess_dir)

    windows_list, obs_mask_list, valid_list, codes, regions = [], [], [], [], []
    n_missing = 0
    for i, fname in enumerate(filenames):
        if i and i % log_every == 0:
            log.info("    ... %d/%d npz files loaded", i, total)
        p = preprocess_dir / fname
        if not p.exists():
            n_missing += 1
            continue
        win, obs_mask, valid = _npz_to_padded_months(p, w_max=w_max, w_min=w_min)
        windows_list.append(win)
        obs_mask_list.append(obs_mask)
        valid_list.append(valid)
        codes.append(_parse_ec_hcat_c(fname))
        regions.append(_region_of(fname))

    if n_missing:
        log.warning("  %d/%d listed npz files were not found on disk", n_missing, total)
    log.info("  Done: %d/%d npz files loaded", len(windows_list), total)

    if windows_list:
        windows = np.stack(windows_list, axis=0)
        obs_mask_arr = np.stack(obs_mask_list, axis=0)
        valid_mask = np.stack(valid_list, axis=0)
    else:
        windows = np.zeros((0, N_TOKENS, w_max, N_BANDS), np.float32)
        obs_mask_arr = np.zeros((0, N_TOKENS, w_max), bool)
        valid_mask = np.zeros((0, N_TOKENS), bool)
    return windows, obs_mask_arr, valid_mask, codes, regions


def _load_npz_list_cached(
    preprocess_dir: Path,
    filenames: list[str],
    cache_path: Path,
    w_max: int,
    w_min: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], list[str]]:
    """Like ``_load_npz_list``, but backed by a one-time consolidated cache.

    First call for a given ``cache_path``: reads every listed npz file (slow --
    ~200ms/file observed on this OneDrive setup for cold files, no way around
    touching each one at least once) and writes ONE consolidated ``.npz`` with
    the raw (un-normalised, un-label-encoded) arrays.

    Every subsequent call for the same ``cache_path``: one bulk read instead of
    up to ~240k individual file opens -- same order of magnitude as CropGlobe's
    setup(), which reads a handful of pre-consolidated files.

    Deliberately caches BEFORE normalisation and BEFORE label-integer-encoding:
    normalisation is cheap/vectorised (never the bottleneck) and depends on
    which ``norm_stats`` gets applied (fresh-computed vs. loaded from a
    specific checkpoint); label encoding depends on the label vocabulary,
    which should stay stable regardless of ``max_parcels`` truncation. Caching
    the raw form keeps one cache file valid for every downstream use.

    ``cache_path`` is expected to already encode ``(w_max, w_min)`` in its
    directory name (see ``EuroCropsMLDataModule``) so a change to either
    can never silently load a cache built under a different windowing scheme.

    Always builds from the FULL ``filenames`` list (ignores ``max_parcels`` --
    that's applied by the caller after loading), so a cache built during a
    truncated smoke test is never partial.
    """
    if cache_path.exists():
        log.info("  Loading cached %s ...", cache_path)
        data = np.load(cache_path, allow_pickle=False)
        windows, obs_mask, valid_mask = data["windows"], data["obs_mask"], data["valid_mask"]
        codes = data["codes"].tolist()
        regions = data["regions"].tolist()
        log.info("  Loaded %d parcels from cache.", len(windows))
        return windows, obs_mask, valid_mask, codes, regions

    log.info("  No cache at %s -- building it now (one-time cost).", cache_path)
    windows, obs_mask, valid_mask, codes, regions = _load_npz_list(
        preprocess_dir, filenames, w_max=w_max, w_min=w_min
    )

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        windows=windows,
        obs_mask=obs_mask,
        valid_mask=valid_mask,
        codes=np.array(codes, dtype="<U20"),
        regions=np.array(regions, dtype="<U8"),
    )
    log.info("  Cached %d parcels -> %s", len(windows), cache_path)
    return windows, obs_mask, valid_mask, codes, regions


# ── Window-level dataset (used for FSQ autoencoder training, Stage 1) ────────


class EuroCropsMLWindowDataset(Dataset):
    """One sample = one valid calendar-month window from one parcel.

    Invalid (missing-month, i.e. fewer than ``w_min`` real observations)
    windows are filtered out before construction — Stage 1 reconstruction
    training has no between-month masking concept, so there is nothing
    useful to learn from a window with no real data at all.

    Returned tensors
    ----------------
    window   : float32 Tensor  (w_max, N_BANDS) — real observations, zero-padded
    label    : int64 Tensor    scalar            — class index (-1 if unseen)
    obs_mask : bool Tensor     (w_max,)          — True at real observations
    """

    def __init__(self, windows: np.ndarray, labels: np.ndarray, obs_mask: np.ndarray) -> None:
        self.windows = torch.from_numpy(windows).float()
        self.labels = torch.from_numpy(labels).long()
        self.obs_mask = torch.from_numpy(obs_mask).bool()

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor, Tensor]:
        return self.windows[idx], self.labels[idx], self.obs_mask[idx]


def _filter_by_codes(
    windows: np.ndarray,
    obs_mask: np.ndarray,
    valid_mask: np.ndarray,
    codes: list[str],
    regions: list[str],
    allowed: set[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], list[str]]:
    """Restrict a loaded split's parallel arrays to parcels whose HCAT code
    is in ``allowed``. See ``EuroCropsMLDataModule``'s ``allowed_hcat_codes``."""
    keep = np.array([c in allowed for c in codes], dtype=bool)
    codes_kept = [c for c, k in zip(codes, keep) if k]
    regions_kept = [r for r, k in zip(regions, keep) if k]
    return windows[keep], obs_mask[keep], valid_mask[keep], codes_kept, regions_kept


# ── Parcel-level dataset (used for tokenisation / predictive SSL / probe) ────


class EuroCropsMLParcelDataset(Dataset):
    """One sample = all month windows for a single parcel.

    Returned tensors
    ----------------
    windows    : float32 Tensor  (N_TOKENS, w_max, N_BANDS)
    label      : int64 Tensor    scalar            — class index (-1 if unseen)
    region     : str                               — 'Latvia' / 'Portugal' / 'Estonia'
    valid_mask : bool Tensor     (N_TOKENS,)       — True where the month is real
    obs_mask   : bool Tensor     (N_TOKENS, w_max) — True at real observations
                 within each month
    """

    def __init__(
        self,
        windows: np.ndarray,
        labels: np.ndarray,
        regions: list[str],
        valid_mask: np.ndarray,
        obs_mask: np.ndarray,
    ) -> None:
        self.windows = torch.from_numpy(windows).float()
        self.labels = torch.from_numpy(labels).long()
        self.regions = regions
        self.valid_mask = torch.from_numpy(valid_mask).bool()
        self.obs_mask = torch.from_numpy(obs_mask).bool()

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor, str, Tensor, Tensor]:
        return (
            self.windows[idx], self.labels[idx], self.regions[idx],
            self.valid_mask[idx], self.obs_mask[idx],
        )


def _parcel_collate(
    batch: list[tuple[Tensor, Tensor, str, Tensor, Tensor]],
) -> tuple[Tensor, Tensor, list[str], Tensor, Tensor]:
    """Collate function for EuroCropsMLParcelDataset — 5-tuple (adds valid_mask
    and obs_mask vs. CropGlobe's ``_parcel_collate`` in cropglobe.py)."""
    windows = torch.stack([b[0] for b in batch])
    labels = torch.stack([b[1] for b in batch])
    regions = [b[2] for b in batch]
    valid_mask = torch.stack([b[3] for b in batch])
    obs_mask = torch.stack([b[4] for b in batch])
    return windows, labels, regions, valid_mask, obs_mask


# ── PyTorch Lightning DataModule ────────────────────────────────────────────


class EuroCropsMLDataModule(LightningDataModule):
    """Lightning DataModule for EuroCropsML FSQ training.

    Mirrors ``CropGlobeDataModule``'s public interface (``.setup()``,
    ``.parcel_train_dataloader()``, ``.parcel_val_dataloader()``,
    ``.parcel_dataloader(split=...)``, ``.obs_per_window``, ``.n_tokens``,
    ``.source``, ``.train_countries``/``.hold_out_countries``) so
    ``train.py``/``train_predictive.py`` need no changes beyond the dataset
    dispatch in ``_train_utils.py::build_datamodule``.

    Splits
    ------
    Uses the dataset's own pre-built split JSON files directly (no re-deriving
    train/val/test from region lists like CropGlobe does):
        Train/Val : ``split/<split_name>/pretrain/region_split.json``
                    (Latvia+Portugal, for Stage 1 + Stage 2 SSL training)
        Held-out  : ``split/<split_name>/finetune/region_split_<variant>.json``
                    (Estonia few-shot train/val/test)

    Parameters
    ----------
    data_root        : Path to the directory containing ``eurocropml/``.
    split_name        : e.g. ``"latvia_portugal_vs_estonia"`` or
                        ``"latvia_vs_estonia"``.
    finetune_variant : Few-shot size for the held-out region
                        (``"1"``..``"500"`` or ``"all"``). Default ``"all"``.
    batch_size        : Window batch size for Stage-1 DataLoaders.
    num_workers       : DataLoader worker count. Use 0 on Windows.
    experiment_dir    : Where to save norm_stats.json and other artefacts.
    dn_scale          : Divisor to convert DN → [0, 1] range before z-scoring.
    norm_stats_path   : If provided, load normalisation statistics from this
                        file instead of recomputing from the pretrain split.
    max_parcels       : If given, truncate every split's file list to the
                        first ``max_parcels`` entries before loading. Loading
                        is a single-threaded ``np.load()`` per npz file with
                        no DataLoader-level parallelism, so ``--fast-dev-run``
                        alone does NOT make ``setup()`` fast — this does.
                        ``None`` (default) loads the full split, unchanged.
    w_max             : Max real observations kept per month window (padded
                        with zeros, or truncated, to this length). Default
                        picked from real per-month observation-count data
                        (see module docstring), not guessed.
    w_min             : Minimum real observations for a month to count as
                        valid at all (else marked invalid in ``valid_mask``,
                        never imputed). Default 2.
    allowed_hcat_codes : If given, restrict every split (pretrain train/val
                        AND the held-out finetune test set) to parcels whose
                        HCAT code is in this set, applied right after loading
                        (before the label vocabulary is built, so the vocab
                        itself only ever contains the allowed codes). Default
                        ``None`` — no filtering, every existing config is
                        unaffected. Added to let a tokenizer be pretrained on
                        only a handful of common crop classes, so its
                        codebook never has to spend capacity on distinctions
                        outside that restricted label space — mirrors how
                        CropGlobe's tokenizer only ever sees 7 classes,
                        rather than filtering evaluation labels post-hoc
                        against a tokenizer trained on the full label space.
    include_holdout_in_pretrain : If ``True``, the held-out region's raw
                        windows (the finetune "test" split, i.e. Estonia) are
                        folded into the SSL pretraining pool (Stage 1 window
                        dataset and Stage 2 parcel dataset) as UNLABELED data
                        -- their crop-type codes are never added to
                        ``label_vocab`` and never influence what the vocab
                        can predict; they exist purely so the encoder's
                        self-supervised objective (masked-token prediction,
                        no labels involved) sees Estonia's spectral
                        distribution during training. Normalisation stats are
                        computed over this same merged pool, matching what
                        the encoder actually trains on. The held-out finetune
                        split used for the DOWNSTREAM PROBE
                        (``_held_out_parcel_ds``, built at the end of
                        ``setup()``) is completely unaffected -- same file,
                        same labels, same zero-shot evaluation as always.
                        This is the unsupervised-domain-adaptation (UDA)
                        analogue of CropGlobe's ``hold_out_countries: null``:
                        target-domain data is used unlabeled during training,
                        never during the labeled evaluation. Default
                        ``False`` -- every existing config (strict domain
                        generalization, Estonia fully excluded from
                        pretraining) is unaffected.
    """

    def __init__(
        self,
        data_root: Path | str,
        split_name: str = "latvia_portugal_vs_estonia",
        finetune_variant: str = "all",
        val_fraction: float = 0.1,
        batch_size: int = 512,
        num_workers: int = 0,
        random_seed: int = 1904,
        experiment_dir: Optional[Path | str] = None,
        dn_scale: float = 10_000.0,
        norm_stats_path: Optional[Path | str] = None,
        max_parcels: Optional[int] = None,
        w_max: int = DEFAULT_W_MAX,
        w_min: int = DEFAULT_W_MIN,
        allowed_hcat_codes: Optional[list[str]] = None,
        include_holdout_in_pretrain: bool = False,
    ) -> None:
        super().__init__()
        if include_holdout_in_pretrain and allowed_hcat_codes is not None:
            # allowed_hcat_codes filters trn_* AFTER the UDA merge builds ssl_*
            # from the unfiltered trn_win/uda_win (see setup(), step 1a vs 1b) --
            # the two haven't been made to compose correctly. Combine them once
            # that ordering is fixed, not before.
            raise NotImplementedError(
                "include_holdout_in_pretrain + allowed_hcat_codes is not supported yet "
                "(the class-restriction filter runs after the UDA merge, so it would "
                "silently not apply to the held-out region's folded-in windows)."
            )
        self.include_holdout_in_pretrain = include_holdout_in_pretrain
        root = Path(data_root) / "eurocropml"
        self.preprocess_dir = root / "preprocess"
        self.split_dir = root / "split" / split_name
        self.split_name = split_name
        self.finetune_variant = finetune_variant
        self.val_fraction = val_fraction
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.random_seed = random_seed
        self.experiment_dir = Path(experiment_dir) if experiment_dir else None
        self.dn_scale = dn_scale
        self._norm_stats_path = Path(norm_stats_path) if norm_stats_path else None
        self.max_parcels = max_parcels
        self.w_max = w_max
        self.w_min = w_min
        self.allowed_hcat_codes: Optional[set[str]] = (
            set(allowed_hcat_codes) if allowed_hcat_codes else None
        )

        # ── Interface parity with CropGlobeDataModule (used for logging /
        # by train.py / train_predictive.py) ────────────────────────────────
        self.source = f"EuroCropsML:{split_name}"
        self.obs_per_window: int = w_max
        self.n_tokens: int = N_TOKENS
        self.n_months: int = N_TOKENS
        self.n_bands: int = N_BANDS
        self.cadence_days = None       # not applicable — calendar-month binning
        self.days_per_token = 30
        self.overlap_days = 0
        self.train_countries: list[str] = []   # filled in setup() from observed regions
        self.hold_out_countries: list[str] = []

        self.norm_stats: Optional[NormStats] = None
        self.label_vocab: dict[str, int] = {}

        self._train_ds: Optional[EuroCropsMLWindowDataset] = None
        self._val_ds: Optional[EuroCropsMLWindowDataset] = None
        self._test_ds: Optional[EuroCropsMLWindowDataset] = None
        self._train_parcel_ds: Optional[EuroCropsMLParcelDataset] = None
        self._val_parcel_ds: Optional[EuroCropsMLParcelDataset] = None
        self._held_out_parcel_ds: Optional[EuroCropsMLParcelDataset] = None

    # ── Label vocabulary ─────────────────────────────────────────────────────

    def _encode_labels(self, codes: list[str]) -> np.ndarray:
        return np.array([self.label_vocab.get(c, -1) for c in codes], dtype=np.int64)

    @property
    def n_classes(self) -> int:
        if not self.label_vocab:
            raise RuntimeError("Call setup() first.")
        return len(self.label_vocab)

    # ── setup ────────────────────────────────────────────────────────────────

    def setup(self, stage: Optional[str] = None) -> None:
        if self._train_parcel_ds is not None:
            return  # already set up

        log.info(
            "EuroCropsMLDataModule.setup — split=%s, w_max=%d, w_min=%d",
            self.split_name, self.w_max, self.w_min,
        )

        pretrain = _load_split_json(self.split_dir / "pretrain" / "region_split.json")
        finetune = _load_split_json(
            self.split_dir / "finetune" / f"region_split_{self.finetune_variant}.json"
        )

        # Cache path encodes (w_max, w_min) so a different windowing scheme
        # can never silently load a cache built under another one.
        cache_dir = (
            self.preprocess_dir.parent / "cache"
            / f"{self.split_name}_wmax{self.w_max}_wmin{self.w_min}"
        )

        # 1. Load pretrain train/val (Latvia+Portugal) -- from cache after the
        # first call (see _load_npz_list_cached). Full lists, NOT truncated by
        # max_parcels yet: the cache must always hold everything so a smoke
        # test never poisons it with a partial build, and the label vocabulary
        # below must stay stable regardless of max_parcels.
        trn_win, trn_obs_mask, trn_valid, trn_codes, trn_regions = _load_npz_list_cached(
            self.preprocess_dir, pretrain["train"], cache_dir / "pretrain_train.npz",
            w_max=self.w_max, w_min=self.w_min,
        )
        val_win, val_obs_mask, val_valid, val_codes, val_regions = _load_npz_list_cached(
            self.preprocess_dir, pretrain["val"], cache_dir / "pretrain_val.npz",
            w_max=self.w_max, w_min=self.w_min,
        )
        log.info(
            "  Pretrain (full): %d train parcels, %d val parcels", len(trn_win), len(val_win)
        )

        # 1a. UDA arm (opt-in): fold the held-out region's raw windows into the
        # SSL pretraining pool as UNLABELED data. Loaded here (ahead of its
        # normal step-6 spot) via the same cache key, so this costs nothing
        # extra once cached. Only ssl_* below is affected -- trn_win/trn_codes
        # stay Latvia+Portugal-only, so the label vocabulary (built from
        # trn_codes just below) and self.hold_out_countries / the downstream
        # probe's _held_out_parcel_ds (built at the end of this function from
        # the SAME file) are completely untouched by this branch.
        if self.include_holdout_in_pretrain:
            uda_win, uda_obs_mask, uda_valid, uda_codes, uda_regions = _load_npz_list_cached(
                self.preprocess_dir, finetune["test"], cache_dir / "finetune_test.npz",
                w_max=self.w_max, w_min=self.w_min,
            )
            log.info(
                "  UDA arm: folding %d held-out (%s) parcels into the SSL pretrain pool, "
                "unlabeled -- label vocab and the downstream probe's held-out split are unaffected.",
                len(uda_win), sorted({REGION_NAMES.get(r, r) for r in uda_regions}),
            )
            ssl_win = np.concatenate([trn_win, uda_win], axis=0)
            ssl_obs_mask = np.concatenate([trn_obs_mask, uda_obs_mask], axis=0)
            ssl_valid = np.concatenate([trn_valid, uda_valid], axis=0)
            ssl_codes = list(trn_codes) + list(uda_codes)
            ssl_regions = list(trn_regions) + list(uda_regions)
        else:
            ssl_win, ssl_obs_mask, ssl_valid = trn_win, trn_obs_mask, trn_valid
            ssl_codes, ssl_regions = trn_codes, trn_regions

        # 1b. Optional class-restriction filter — applied to the full loaded
        # set, before the label vocabulary is built, so the vocabulary and
        # every downstream step only ever see the allowed classes.
        if self.allowed_hcat_codes is not None:
            trn_win, trn_obs_mask, trn_valid, trn_codes, trn_regions = _filter_by_codes(
                trn_win, trn_obs_mask, trn_valid, trn_codes, trn_regions, self.allowed_hcat_codes
            )
            val_win, val_obs_mask, val_valid, val_codes, val_regions = _filter_by_codes(
                val_win, val_obs_mask, val_valid, val_codes, val_regions, self.allowed_hcat_codes
            )
            log.info(
                "  allowed_hcat_codes filter (%d codes): %d train parcels, %d val parcels remain",
                len(self.allowed_hcat_codes), len(trn_win), len(val_win),
            )

        # 2. Label vocabulary — built from the FULL pretrain train split,
        # before any max_parcels truncation, so it never changes with it.
        unique_codes = sorted(set(trn_codes))
        self.label_vocab = {code: i for i, code in enumerate(unique_codes)}
        log.info("  Label vocabulary: %d classes (from full pretrain train split)", len(self.label_vocab))

        # 2b. max_parcels truncation (dev/smoke-test knob) -- applied after
        # caching and after building the label vocabulary from the full set.
        # Truncates the ssl_* pool (== trn_* when the UDA arm is off, so this
        # is a no-op change for every existing config).
        if self.max_parcels is not None:
            ssl_win, ssl_obs_mask, ssl_valid = (
                ssl_win[: self.max_parcels], ssl_obs_mask[: self.max_parcels], ssl_valid[: self.max_parcels]
            )
            ssl_codes, ssl_regions = ssl_codes[: self.max_parcels], ssl_regions[: self.max_parcels]
            val_win, val_obs_mask, val_valid = (
                val_win[: self.max_parcels], val_obs_mask[: self.max_parcels], val_valid[: self.max_parcels]
            )
            val_codes, val_regions = val_codes[: self.max_parcels], val_regions[: self.max_parcels]
            log.info(
                "  max_parcels=%d -> using %d SSL-pool parcels, %d val parcels",
                self.max_parcels, len(ssl_win), len(val_win),
            )

        # train_countries reflects the SSL pool actually trained on (matches
        # CropGlobe's interface: with include_holdout_in_pretrain, this
        # includes the held-out region too, since its unlabeled spectra were
        # part of training -- self.hold_out_countries, set later from the
        # SAME finetune file, is what the downstream probe treats as unseen).
        self.train_countries = sorted({REGION_NAMES.get(r, r) for r in ssl_regions})

        ssl_lbl = self._encode_labels(ssl_codes)
        val_lbl = self._encode_labels(val_codes)

        # 3. Normalisation stats — computed from *real* (non-padded) SSL
        # training observations only (the UDA-merged pool when that arm is
        # on, else identical to the pretrain-train-only pool). Including
        # padded zero positions here would bias the learned mean/std towards
        # zero (same class of issue as BatchNorm1d over padding -- see
        # encoder.py's module docstring).
        if self._norm_stats_path is not None:
            self.norm_stats = NormStats.load(self._norm_stats_path)
            log.info("  Norm stats loaded ← %s", self._norm_stats_path)
        else:
            real_obs = ssl_win[ssl_obs_mask]   # (n_real_observations, N_BANDS)
            self.norm_stats = compute_norm_stats(real_obs, dn_scale=self.dn_scale)
            if self.experiment_dir:
                self.experiment_dir.mkdir(parents=True, exist_ok=True)
                self.norm_stats.save(self.experiment_dir / "norm_stats.json")
                log.info("  Norm stats saved → %s", self.experiment_dir / "norm_stats.json")

        ssl_win_norm = self.norm_stats.apply(ssl_win)
        val_win_norm = self.norm_stats.apply(val_win)

        # 4. Window-level datasets (Stage 1) — drop invalid (missing-month) windows.
        # Padding still present WITHIN a kept window; obs_mask travels with it
        # so FSQAutoencoder's masked pooling/loss can ignore it correctly.
        def _to_window_ds(win_norm, obs_mask, valid, lbl) -> EuroCropsMLWindowDataset:
            N, T, W, B = win_norm.shape
            flat_win = win_norm.reshape(N * T, W, B)
            flat_obs_mask = obs_mask.reshape(N * T, W)
            flat_valid = valid.reshape(N * T)
            flat_lbl = np.repeat(lbl, T)
            flat_win = flat_win[flat_valid]
            flat_obs_mask = flat_obs_mask[flat_valid]
            flat_lbl = flat_lbl[flat_valid]
            return EuroCropsMLWindowDataset(flat_win, flat_lbl, flat_obs_mask)

        self._train_ds = _to_window_ds(ssl_win_norm, ssl_obs_mask, ssl_valid, ssl_lbl)
        self._val_ds = _to_window_ds(val_win_norm, val_obs_mask, val_valid, val_lbl)
        log.info(
            "  Stage-1 windows (valid months only): train=%d, val=%d",
            len(self._train_ds), len(self._val_ds),
        )

        # 5. Parcel-level datasets (Stage 2 predictive SSL + probe)
        self._train_parcel_ds = EuroCropsMLParcelDataset(
            ssl_win_norm, ssl_lbl, ssl_regions, ssl_valid, ssl_obs_mask
        )
        self._val_parcel_ds = EuroCropsMLParcelDataset(
            val_win_norm, val_lbl, val_regions, val_valid, val_obs_mask
        )

        # 6. Held-out region (finetune split: few-shot train/val/test on Estonia).
        # "test" is shared across every finetune_variant (only "train" differs
        # by k-shot size), so this cache key doesn't need finetune_variant in it.
        ho_win, ho_obs_mask, ho_valid, ho_codes, ho_regions = _load_npz_list_cached(
            self.preprocess_dir, finetune["test"], cache_dir / "finetune_test.npz",
            w_max=self.w_max, w_min=self.w_min,
        )
        if self.allowed_hcat_codes is not None:
            ho_win, ho_obs_mask, ho_valid, ho_codes, ho_regions = _filter_by_codes(
                ho_win, ho_obs_mask, ho_valid, ho_codes, ho_regions, self.allowed_hcat_codes
            )
            log.info(
                "  allowed_hcat_codes filter: %d held-out (finetune test) parcels remain",
                len(ho_win),
            )
        if self.max_parcels is not None:
            ho_win, ho_obs_mask, ho_valid = (
                ho_win[: self.max_parcels], ho_obs_mask[: self.max_parcels], ho_valid[: self.max_parcels]
            )
            ho_codes, ho_regions = ho_codes[: self.max_parcels], ho_regions[: self.max_parcels]
        if len(ho_win):
            self.hold_out_countries = sorted({REGION_NAMES.get(r, r) for r in ho_regions})
            ho_lbl = self._encode_labels(ho_codes)
            ho_win_norm = self.norm_stats.apply(ho_win)
            self._held_out_parcel_ds = EuroCropsMLParcelDataset(
                ho_win_norm, ho_lbl, ho_regions, ho_valid, ho_obs_mask
            )
            # Window-level held-out test set (Stage 1 reconstruction test metrics) —
            # same construction as _train_ds/_val_ds, invalid months dropped.
            self._test_ds = _to_window_ds(ho_win_norm, ho_obs_mask, ho_valid, ho_lbl)
            n_unseen = int((ho_lbl == -1).sum())
            log.info(
                "  Held-out (%s): %d parcels (%d valid windows), %d with a class unseen in pretrain",
                self.hold_out_countries, len(ho_win), len(self._test_ds), n_unseen,
            )

    # ── DataLoaders ──────────────────────────────────────────────────────────

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self._train_ds, batch_size=self.batch_size, shuffle=True,
            num_workers=self.num_workers, pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self._val_ds, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers, pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )

    def test_dataloader(self) -> DataLoader:
        """Window-level held-out (Estonia) test set — required for
        ``trainer.test(model, datamodule=dm, ...)`` (Stage 1 reconstruction)."""
        if self._test_ds is None:
            raise RuntimeError("No held-out region data loaded.")
        return DataLoader(
            self._test_ds, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers, pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )

    def parcel_dataloader(self, split: str = "held_out", batch_size: int = 64) -> DataLoader:
        """Parcel-level dataloader for tokenisation and probe evaluation.

        Parameters
        ----------
        split : ``"held_out"`` — few-shot test parcels (held-out region).
                ``"train"``    — pretrain-train parcels (Latvia+Portugal).
        """
        if split == "held_out":
            if self._held_out_parcel_ds is None:
                raise RuntimeError("No held-out region data loaded.")
            ds = self._held_out_parcel_ds
        elif split in ("train", "all_train"):
            if self._train_parcel_ds is None:
                raise RuntimeError("Call setup() first.")
            ds = self._train_parcel_ds
        else:
            raise ValueError(f"Unknown split '{split}'. Use 'held_out' or 'train'.")
        return DataLoader(
            ds, batch_size=batch_size, shuffle=False,
            num_workers=0, collate_fn=_parcel_collate,
        )

    # ── Few-shot fine-tuning split (opt-in; setup() never touches this) ─────

    def load_finetune_split(
        self, variant: str, key: str = "train"
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], list[str]]:
        """Load one split of ``region_split_<variant>.json`` on demand.

        ``setup()`` only ever loads ``finetune["test"]`` (shared across every
        variant, used for the zero-shot cross-region evaluation). The
        official EuroCropsML benchmark also defines a k-shot
        ``finetune["train"]`` (1/5/10/20/100/200/500/all) meant for few-shot
        fine-tuning on the held-out region — this was never wired up before.
        Returns raw ``(windows, obs_mask, valid_mask, codes, regions)``: NOT
        normalised, NOT label-encoded. Callers decide both — in particular,
        re-encoding with ``self.label_vocab`` (pretrain-only) would silently
        fold every class never seen during pretraining into "Other", which
        defeats the point of a few-shot split explicitly designed to include
        such classes. Build an Estonia-local vocab from the codes returned
        here instead for genuine few-shot classification.
        """
        if self.norm_stats is None:
            raise RuntimeError("Call setup() first (need norm_stats).")
        finetune = _load_split_json(self.split_dir / "finetune" / f"region_split_{variant}.json")
        cache_dir = (
            self.preprocess_dir.parent / "cache"
            / f"{self.split_name}_wmax{self.w_max}_wmin{self.w_min}"
        )
        # "test" is identical across every variant (see setup()'s comment),
        # so it always shares the same cache file; "train"/"val" differ by
        # k-shot size and get their own variant-specific cache file.
        cache_name = "finetune_test.npz" if key == "test" else f"finetune_{key}_{variant}.npz"
        return _load_npz_list_cached(
            self.preprocess_dir, finetune[key], cache_dir / cache_name,
            w_max=self.w_max, w_min=self.w_min,
        )

    def finetune_train_dataloader(
        self, variant: str, batch_size: int = 64,
        label_vocab: Optional[dict[str, int]] = None,
    ) -> DataLoader:
        """Parcel-level DataLoader over the k-shot Estonia ``finetune/train``
        split. Pass an Estonia-local ``label_vocab`` (built from
        ``load_finetune_split(variant, "train")``'s and ``"test"``'s codes)
        for genuine few-shot learning; defaults to ``self.label_vocab``
        (pretrain-only — unseen-in-pretrain classes become -1), matching how
        every other parcel dataloader in this class encodes labels.
        """
        win, obs_mask, valid, codes, regions = self.load_finetune_split(variant, "train")
        vocab = label_vocab if label_vocab is not None else self.label_vocab
        lbl = np.array([vocab.get(c, -1) for c in codes], dtype=np.int64)
        win_norm = self.norm_stats.apply(win)
        ds = EuroCropsMLParcelDataset(win_norm, lbl, regions, valid, obs_mask)
        return DataLoader(
            ds, batch_size=batch_size, shuffle=False,
            num_workers=0, collate_fn=_parcel_collate,
        )

    def parcel_train_dataloader(self, batch_size: int = 64) -> DataLoader:
        """Parcel-level training DataLoader for predictive SSL.

        Returns shuffled ``(windows, labels, regions, valid_mask, obs_mask)``
        batches — see ``FSQPredictiveSSL.forward``'s ``valid_mask``/``obs_mask``
        parameters for how missing months / padding are handled.
        """
        if self._train_parcel_ds is None:
            raise RuntimeError("Call setup() first.")
        return DataLoader(
            self._train_parcel_ds, batch_size=batch_size, shuffle=True,
            num_workers=self.num_workers, pin_memory=True,
            persistent_workers=self.num_workers > 0, collate_fn=_parcel_collate,
        )

    def parcel_val_dataloader(self, batch_size: int = 64) -> DataLoader:
        if self._val_parcel_ds is None:
            raise RuntimeError("Call setup() first.")
        return DataLoader(
            self._val_parcel_ds, batch_size=batch_size, shuffle=False,
            num_workers=self.num_workers, pin_memory=True,
            persistent_workers=self.num_workers > 0, collate_fn=_parcel_collate,
        )
