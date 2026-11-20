"""Supplement S3: the five fits of the k-means vocabulary per row (mean, sd, range of accuracy and macro-F1).

    python -m fsq_matched_controls.analysis.kmeans_fits [--compare]     -> outputs/kmeans_fits.csv, outputs/kmeans_fits_per_fit.csv

Reads the stored fits (results/runs/diagnostic72_k_readout.csv, variant K2, seeds 1904, 42, 101, 2024, 4242; beside them the refit
of the same construction in diagnostic74_k2_switch.csv, variant K2_refit). sd with n - 1. The gate column checks that the mean of
the five fits is Table I's k-means control. The rebuilt fits themselves are produced by python -m fsq_matched_controls.controls.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from fsq_matched_controls import paths
from fsq_matched_controls.analysis import letter_runs as L


def _stats(x):
    x = np.asarray(x, float)
    return dict(n=len(x), mean=x.mean(), sd=x.std(ddof=1), min=x.min(), max=x.max(), range=x.max() - x.min())


def tables():
    d72 = L._d72
    d74 = pd.read_csv(L.PS / "diagnostic74_k2_switch.csv")
    seeds = [str(s) for s in L.SEEDS5]
    rows, per_fit = [], []
    for r in L.ROWS:
        a = d72[(d72.row == r) & (d72.variant == "K2") & d72.seed.isin(seeds)].sort_values("seed")
        assert len(a) == 5, (r, len(a))
        for _, x in a.iterrows():
            per_fit.append(dict(row=r, source="diagnostic72 K2", seed=x.seed, acc=x.acc, f1=x.f1))
        sa, sf = _stats(a.acc), _stats(a.f1)
        ctl = L.CTL[r]["kmeans"]
        rows.append(dict(row=r, source="diagnostic72 K2", n=5, acc_mean=sa["mean"], acc_sd=sa["sd"], acc_min=sa["min"], acc_max=sa["max"],
                         acc_range=sa["range"], f1_mean=sf["mean"], f1_sd=sf["sd"], f1_min=sf["min"], f1_max=sf["max"], f1_range=sf["range"],
                         gate_equals_table_ii_control=bool(abs(sa["mean"] - ctl[0]) < 1e-9 and abs(sf["mean"] - ctl[1]) < 1e-9)))
        b = d74[(d74.row == r) & (d74.variant == "K2_refit")]
        if len(b):
            b = b.copy(); b["seed"] = b.seed.astype(int).astype(str)
            for _, x in b.iterrows():
                per_fit.append(dict(row=r, source="diagnostic74 K2_refit", seed=x.seed, acc=x.acc, f1=x.f1))
            ba, bf = _stats(b.acc), _stats(b.f1)
            rows.append(dict(row=r, source="diagnostic74 K2_refit", n=ba["n"], acc_mean=ba["mean"], acc_sd=ba["sd"], acc_min=ba["min"],
                             acc_max=ba["max"], acc_range=ba["range"], f1_mean=bf["mean"], f1_sd=bf["sd"], f1_min=bf["min"], f1_max=bf["max"],
                             f1_range=bf["range"], gate_equals_table_ii_control=None))
    return pd.DataFrame(rows), pd.DataFrame(per_fit)


def main():
    R, P = tables()
    paths.OUTPUT.mkdir(parents=True, exist_ok=True)
    R.to_csv(paths.OUTPUT / "kmeans_fits.csv", index=False)
    P.to_csv(paths.OUTPUT / "kmeans_fits_per_fit.csv", index=False)
    m = R[R.source == "diagnostic72 K2"]
    print(m[["row", "acc_mean", "acc_sd", "acc_range", "f1_mean", "f1_sd", "f1_range", "gate_equals_table_ii_control"]].to_string(index=False))
    print(f"-> {paths.OUTPUT / 'kmeans_fits.csv'}, {paths.OUTPUT / 'kmeans_fits_per_fit.csv'}")


def compare(outputs):
    """Regenerated tables against the stored ones (analysis.compare_tables)."""
    from fsq_matched_controls.analysis.compare_tables import report
    ok = report([(paths.OUTPUT / f"{name}.csv", folder / f"{name}.csv") for name, folder in outputs])
    print("PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    import sys
    main()
    if "--compare" in sys.argv:
        sys.exit(0 if compare([("kmeans_fits", paths.SUPPLEMENT_S3), ("kmeans_fits_per_fit", paths.SUPPLEMENT_S3)]) else 1)
