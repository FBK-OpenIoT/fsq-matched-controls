"""Control check: the matched controls rebuilt from the datasets against the stored per-row results.

    python -m fsq_matched_controls.analysis.check_controls [--rows ...] [--from-csv outputs/controls/controls.csv]

Runs python -m fsq_matched_controls.controls (or reads its output with --from-csv) and compares every score with the stored one:
    rounded PCA          results/runs/diagnostic40_* (transductive), diagnostic44_* (strict), diagnostic62_* (Estonia)
    k-means vocabulary   results/runs/diagnostic72_k_readout.csv (variant K2), per row and fit seed
Rounded PCA: "identical" when equal; otherwise passes within PLATFORM_TOLERANCE (as in check_end_to_end).
k-means vocabulary: mini-batch k-means fits differ across platforms (on the letter's platform a refit of the same construction
reproduced every stored fit exactly, results/runs/diagnostic74_k2_switch.csv); each fit is listed, and the check applies to what
the letter uses, the mean of the five fits, within KMEANS_TOLERANCE (about twice the largest fit-to-fit sd, 0.0016, supplement S3).
Decisive check: Table I's verdicts recomputed with the rebuilt controls must equal the letter's.
Exit status 0 if all three hold.
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

from fsq_matched_controls import paths
from fsq_matched_controls.analysis import letter_runs as L
from fsq_matched_controls.analysis.tolerances import GENERAL as PLATFORM_TOLERANCE, KMEANS_MEAN as KMEANS_TOLERANCE

REGION = {"France, transductive": "FRA", "Australia, transductive": "AUS", "France, strict": "FRA", "Australia, strict": "AUS"}


def stored() -> dict:
    """(row, control, seed) -> (accuracy, macro_f1), at the stored precision."""
    out = {}
    d40 = pd.read_csv(paths.RUN_RESULTS / "diagnostic40_cropglobe_transductive_pca_kmeans.csv")
    d44 = pd.read_csv(paths.RUN_RESULTS / "diagnostic44_cropglobe_strict_pca_kmeans_refit.csv")
    for row, reg in REGION.items():
        d, rep = (d40, "PCA4_round5_transductive") if "transductive" in row else (d44, "PCA4_round5_strict")
        r = d[(d.representation == rep) & (d.probe == "logistic") & (d.held_out == reg)].iloc[0]
        out[(row, "rounded_pca", "")] = (float(r.test_acc), float(r.macro_f1))
    d62 = pd.read_csv(paths.RUN_RESULTS / "diagnostic62_pca_grid_variants.csv")
    r = d62[(d62.setting == "estonia") & (d62.representation == "PCA4_round5") & (d62.probe == "logistic")].iloc[0]
    out[("Estonia", "rounded_pca", "")] = (float(r.test_acc), float(r.macro_f1))
    d72 = pd.read_csv(paths.RUN_RESULTS / "diagnostic72_k_readout.csv")
    d72 = d72[d72["analysis"].isna() & (d72.variant == "K2")]
    for r in d72.itertuples():
        out[(r.row, "kmeans_vocabulary", str(int(float(r.seed))))] = (float(r.acc), float(r.f1))
    return out


def _verdict(da, df_, tol):
    if da == 0 and df_ == 0:
        return "identical", True
    if max(da, df_) <= tol:
        return f"within tolerance {tol:g} (|d| accuracy {da:.2e}, macro-F1 {df_:.2e})", True
    return f"DIFFERENT (|d| accuracy {da:.2e}, macro-F1 {df_:.2e})", False


def compare(df: pd.DataFrame) -> bool:
    ref = stored()
    ok = True
    df = df.copy(); df["seed"] = ["" if pd.isna(s) or s == "" else str(int(float(s))) for s in df.seed]
    rebuilt = {}
    for r in df[df.control == "rounded_pca"].itertuples():
        s = ref[(r.row, "rounded_pca", "")]
        v, good = _verdict(abs(r.accuracy - s[0]), abs(r.macro_f1 - s[1]), PLATFORM_TOLERANCE); ok &= good
        rebuilt.setdefault(r.row, {})["PCA"] = (r.accuracy, r.macro_f1)
        print(f"{r.row:24s} rounded PCA       {r.accuracy!r} / {r.macro_f1!r}  (stored {s[0]!r} / {s[1]!r})  {v}")
    km = df[df.control == "kmeans_vocabulary"]
    for row, g in km.groupby("row", sort=False):
        for r in g.itertuples():
            s = ref[(row, "kmeans_vocabulary", r.seed)]
            print(f"{row:24s} k-means fit {r.seed:<5s} {r.accuracy:.6f} / {r.macro_f1:.6f}  (stored {s[0]:.6f} / {s[1]:.6f}; |d| {abs(r.accuracy - s[0]):.1e} / {abs(r.macro_f1 - s[1]):.1e})")
        ma, mf = g.accuracy.mean(), g.macro_f1.mean()
        sa = np.mean([ref[(row, "kmeans_vocabulary", s)][0] for s in g.seed]); sf = np.mean([ref[(row, "kmeans_vocabulary", s)][1] for s in g.seed])
        v, good = _verdict(abs(ma - sa), abs(mf - sf), KMEANS_TOLERANCE); ok &= good
        rebuilt.setdefault(row, {})["kmeans"] = (ma, mf)
        print(f"{row:24s} k-means, mean of {len(g)} fits {ma:.6f} / {mf:.6f}  (stored {sa:.6f} / {sf:.6f})  {v}")
    if set(rebuilt) == set(L.ROWS) and all(len(v) == 2 for v in rebuilt.values()):
        ctl = {r: dict(L.CTL[r], **rebuilt[r]) for r in L.ROWS}
        _, v_new, _ = L.analyse(L.collect("all"), L.LOOKS, ctl=ctl)
        _, v_old, _ = L.analyse(L.collect("all"), L.LOOKS)
        same = v_new == v_old
        ok &= same
        print(f"\nTable I verdicts with the rebuilt controls equal the letter's: {same}"
              + ("" if same else "  " + "; ".join(f"{k}: {v_old[k]} -> {v_new[k]}" for k in v_old if v_old[k] != v_new[k])))
    return ok


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--rows", nargs="+")
    ap.add_argument("--from-csv", help="compare an existing output of python -m fsq_matched_controls.controls instead of running it")
    a = ap.parse_args(argv)
    if a.from_csv:
        df = pd.read_csv(a.from_csv, dtype={"seed": str}, keep_default_na=False)
    else:
        from fsq_matched_controls.controls.__main__ import run
        df = run(a.rows)
        print()
    ok = compare(df)
    print(f"\n{'PASS' if ok else 'FAIL'}: rounded PCA within {PLATFORM_TOLERANCE:g}, k-means five-fit means within {KMEANS_TOLERANCE:g}, "
          f"Table I verdicts with the rebuilt controls")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
