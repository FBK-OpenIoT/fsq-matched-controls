"""Every file-system location the package reads or writes. Nothing else in the package hard-codes a path.

Defaults are relative to the repository root; each can be overridden with an environment variable:

    FSQ_RESULTS       per-run result tables and aggregates shipped with the repository   (default: <repo>/results)
    FSQ_OUTPUT        where regenerated tables and figures are written                    (default: <repo>/outputs)
    FSQ_RUNS_DIR      training outputs: one directory per run with checkpoints/ and norm_stats.json   (default: <repo>/runs)
    FSQ_DATA_ROOT     the public datasets as downloaded (CropGlobe, EuroCropsML)          (default: <repo>/data)
    FSQ_FEATURES      monthly band-mean arrays rebuilt from the datasets                  (default: <repo>/features)
"""
from __future__ import annotations

import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def _env(name: str, default: Path) -> Path:
    v = os.environ.get(name)
    return Path(v) if v else default


RESULTS = _env("FSQ_RESULTS", REPO / "results")
RUN_RESULTS = RESULTS / "runs"          # per-run probe rows of the 80 letter runs, control tables, probe diagnostics
TABLE2 = RESULTS / "table2_decomposition"          # Table II: twins (original and continued), step tests, continuous PCA
LETTER_NUMBERS = RESULTS / "letter_numbers"        # the analysis output tables behind the letter's numbers (Table I S7.1 records in B.csv)
SUPPLEMENT_S3 = RESULTS / "supplement_S3_controls" # k-means fits, bits per month
SUPPLEMENT_S4 = RESULTS / "supplement_S4_sensitivity"  # sensitivity analyses, few-shot Estonia, continued runs
SUPPLEMENT_S5 = RESULTS / "supplement_S5_runs"     # per-class F1, training stop epochs
SUPPLEMENT_S8 = RESULTS / "supplement_S8_training" # capped runs, parameter counts, training times
OUTPUT = _env("FSQ_OUTPUT", REPO / "outputs")
RUNS_DIR = _env("FSQ_RUNS_DIR", REPO / "runs")
DATA_ROOT = _env("FSQ_DATA_ROOT", REPO / "data")
FEATURES = _env("FSQ_FEATURES", REPO / "features")
CHECKPOINTS = REPO / "checkpoints"
CONFIGS = REPO / "configs"
MANUSCRIPT = REPO / "manuscript"
