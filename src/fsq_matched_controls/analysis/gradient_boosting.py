"""Supplement S4.3: the gradient-boosting probe on the raw band means and on the ten-band k-means vocabulary (transductive CropGlobe).

    python -m fsq_matched_controls.analysis.gradient_boosting [--kmeans] [--nproc 4] [--compare]   -> outputs/gradient_boosting_raw.csv

HistGradientBoostingClassifier(max_iter=200, early_stopping=False) with random_state 1904, 42, 101, 2024, 4242, harmonized class
sets, France and Australia held out, accuracy and macro-F1:
    raw      the 70 monthly band means (seven months x ten bands);
    --kmeans the k-means vocabulary fitted on the ten band means of the ten-split pool (MiniBatchKMeans 625, batch 4,096, n_init 5,
             max_iter 300, fits with random_state 1904, 42, 101, 2024, 4242), each month read as its centroid's ten band values;
             every fit scored with every boosting state.
HistGradientBoosting bins continuous features from a random subsample when the pool has more than 200,000 rows (here 236,826), so
its scores on continuous features depend on the random state; the table gives the mean and the range over the five states and the
value at state 1904. The output has one line per fit (row_type fit) and per code, region and metric (row_type summary).
--compare: against results/supplement_S4_sensitivity/gradient_boosting_raw.csv (the rebuild on Windows) and gradient_boosting_raw_linux.csv
(the same rebuild on Linux); fits identical or within the platform tolerance.
The token scores of the masked-prediction runs are per_run_readouts --mode gb.
"""
from __future__ import annotations

import os

for _v in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import sys
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd

from fsq_matched_controls import paths
from fsq_matched_controls.analysis.tolerances import GENERAL as PLATFORM_TOLERANCE

STATES = (1904, 42, 101, 2024, 4242)
KSEEDS = (1904, 42, 101, 2024, 4242)


def _fits(job):
    from sklearn.cluster import MiniBatchKMeans
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import accuracy_score, f1_score
    from fsq_matched_controls.controls import features
    from fsq_matched_controls.probe.linear_probe import harmonize_label_sets
    code, ks = job
    d = features.cropglobe_monthly(); X, y = d["X"], d["y"]
    if code == "raw":
        F = {s: X[s].astype(np.float32).reshape(len(X[s]), -1) for s in X}
    else:
        pool10 = np.concatenate([X[s].reshape(-1, 10) for s in ("train", "FRA", "AUS")])
        km = MiniBatchKMeans(n_clusters=625, random_state=ks, batch_size=4096, n_init=5, max_iter=300).fit(pool10)
        F = {s: km.cluster_centers_[km.predict(X[s].reshape(-1, 10))].reshape(len(X[s]), -1) for s in X}
    rows = []
    for reg in ("FRA", "AUS"):
        ytr, yho, _ = harmonize_label_sets(y["train"], y[reg])
        for rs in STATES:
            t0 = time.time()
            clf = HistGradientBoostingClassifier(max_iter=200, random_state=rs, early_stopping=False).fit(F["train"], ytr)
            p = clf.predict(F[reg])
            rows.append(dict(code=code, kmeans_seed=ks if ks is not None else "", region=reg, random_state=rs, acc=float(accuracy_score(yho, p)),
                             f1=float(f1_score(yho, p, average="macro")), seconds=round(time.time() - t0, 1)))
    return rows


def run(kmeans=False, nproc=4) -> pd.DataFrame:
    jobs = [("raw", None)] + ([("kmeans625_tenband", s) for s in KSEEDS] if kmeans else [])
    rows = []
    with Pool(min(nproc, len(jobs))) as pool:
        for r in pool.imap_unordered(_fits, jobs):
            rows += r; print("done", r[0]["code"], r[0]["kmeans_seed"], flush=True)
    d = pd.DataFrame(rows); d["row_type"] = "fit"
    out = []
    for code, g in d.groupby("code"):
        for reg in ("FRA", "AUS"):
            for m in ("acc", "f1"):
                x = g[g.region == reg]; by = x.groupby("random_state")[m].mean()
                out.append(dict(row_type="summary", code=code, region=reg, metric=m, mean_over_states=float(by.mean()), min_over_states=float(by.min()),
                                max_over_states=float(by.max()), value_at_1904=float(by[1904]), n_fits_per_state=int(x.groupby("random_state").size().iloc[0])))
    df = pd.concat([d, pd.DataFrame(out)], ignore_index=True)
    paths.OUTPUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(paths.OUTPUT / "gradient_boosting_raw.csv", index=False)
    return df


def compare(df, ref_name="gradient_boosting_raw.csv") -> bool:
    ref = pd.read_csv(paths.SUPPLEMENT_S4 / ref_name)
    a = df[df.row_type == "fit"].copy(); b = ref[ref.row_type == "fit"].copy()
    for x in (a, b):
        x["kmeans_seed"] = x["kmeans_seed"].fillna("").astype(str).str.replace(".0", "", regex=False)
    m = a.merge(b, on=["code", "kmeans_seed", "region", "random_state"], suffixes=("", "_ref"))
    d = float(max((m.acc - m.acc_ref).abs().max(), (m.f1 - m.f1_ref).abs().max()))
    ok = len(m) == len(a) and d <= PLATFORM_TOLERANCE
    print(f"{ref_name}: {len(m)} of {len(a)} fits matched; max |d| {d:.2e} -> {'identical' if d == 0 else ('within tolerance' if ok else 'DIFFERENT')}")
    return ok


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--kmeans", action="store_true"); ap.add_argument("--nproc", type=int, default=4); ap.add_argument("--compare", action="store_true")
    a = ap.parse_args(argv)
    df = run(a.kmeans, a.nproc)
    print(df[df.row_type == "summary"].to_string(index=False))
    if a.compare:
        ok_w = compare(df, "gradient_boosting_raw.csv")          # the rebuild on Windows (the letter's platform)
        ok_l = compare(df, "gradient_boosting_raw_linux.csv")    # the same rebuild on Linux (GPU node, CPU)
        return 0 if (ok_w or ok_l) else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
