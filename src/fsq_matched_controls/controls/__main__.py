"""Build the two matched controls and run the letter's probe on them.

    python -m fsq_matched_controls.controls [--rows "France, strict" ...] [--seeds 1904 42 ...] [--no-cache]

For every row: rounded PCA (one fit) and the k-means vocabulary (five fits), each read by the letter's probe (StandardScaler +
logistic regression, lbfgs, C 1, at most 300 iterations, random state 1904) trained on the probe pool with labels harmonized
between pool and region. Writes outputs/controls/controls.csv (one line per row, control and seed) and prints the five-fit means.
The datasets are read from FSQ_DATA_ROOT; the monthly band means are cached in FSQ_FEATURES.
"""
from __future__ import annotations

import os

for _v in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")   # the letter's controls ran single-threaded

import argparse
import sys
import time

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler

from fsq_matched_controls import paths
from fsq_matched_controls.controls import features, kmeans_vocabulary as KV, rounded_pca as RP
from fsq_matched_controls.controls.rows import ROWS, STANDARDIZE
from fsq_matched_controls.probe.linear_probe import harmonize_label_sets

PROBE_SEED = 1904
OUT = paths.OUTPUT / "controls" / "controls.csv"


def probe(Xtr, ytr, Xev, yev):
    sc = StandardScaler()
    clf = LogisticRegression(max_iter=300, C=1.0, solver="lbfgs", random_state=PROBE_SEED)
    clf.fit(sc.fit_transform(Xtr), ytr)
    pred = clf.predict(sc.transform(Xev))
    return float(accuracy_score(yev, pred)), float(f1_score(yev, pred, average="macro"))


def load(dataset, cache=True):
    data = features.cropglobe_monthly(cache) if dataset == "cropglobe" else features.eurocropsml_monthly(cache)
    ok = features.check_fingerprints(dataset, data)
    print(f"monthly band means, {dataset}: " + ", ".join(f"{s} {data['X'][s].shape}" for s in data["X"])
          + ("" if ok is None else f"; identical to the letter's arrays (SHA-256): {ok}"), flush=True)
    return data


def run(rows=None, seeds=None, cache=True):
    rows = rows or list(ROWS)
    seeds = seeds or KV.SEEDS
    t0 = time.time()
    out, data, fits = [], {}, {}
    for row in rows:
        spec = ROWS[row]
        d = data.get(spec["dataset"]) or data.setdefault(spec["dataset"], load(spec["dataset"], cache))
        X, V, y = d["X"], d["V"], d["y"]
        reg = spec["region"]
        yt, yh, _ = (harmonize_label_sets(y["train"], y[reg], other_class=-1) if spec["dataset"] == "eurocropsml"
                     else harmonize_label_sets(y["train"], y[reg]))
        key = (spec["dataset"], tuple(spec["pool"]))
        if key not in fits:
            pool = np.concatenate([X[s].reshape(-1, X[s].shape[-1])[V[s].reshape(-1)] for s in spec["pool"]])
            fits[key] = dict(pool=pool, pca=RP.fit_pca(pool), vocab={})
        F = fits[key]
        # rounded PCA
        std = STANDARDIZE[row]
        if std is None:
            raise SystemExit(f"no standardization set for {row}")
        P = {s: RP.project(F["pca"], X[s]) for s in set(std) | {"train", reg}}
        scaler = RP.scaler_on([P[s] for s in std], [V[s] for s in std])
        a, f = probe(RP.rounded(scaler, P["train"], V["train"]), yt, RP.rounded(scaler, P[reg], V[reg]), yh)
        out.append(dict(row=row, control="rounded_pca", seed="", accuracy=a, macro_f1=f))
        print(f"{row:24s} rounded PCA          accuracy {a:.6f}  macro-F1 {f:.6f}  ({time.time() - t0:.0f} s)", flush=True)
        # k-means vocabulary
        if "projection" not in F:
            F["projection"] = KV.fit_pool_projection(F["pool"])
        pca, psc, pool_std = F["projection"]
        for seed in seeds:
            if seed not in F["vocab"]:
                F["vocab"][seed] = KV.fit_vocabulary(pca, psc, pool_std, seed)
            voc = F["vocab"][seed]
            a, f = probe(voc.represent(X["train"], V["train"]), yt, voc.represent(X[reg], V[reg]), yh)
            out.append(dict(row=row, control="kmeans_vocabulary", seed=str(seed), accuracy=a, macro_f1=f))
            print(f"{row:24s} k-means seed {seed:<6d}  accuracy {a:.6f}  macro-F1 {f:.6f}  ({time.time() - t0:.0f} s)", flush=True)
    df = pd.DataFrame(out)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    km = df[df.control == "kmeans_vocabulary"].groupby("row", sort=False)[["accuracy", "macro_f1"]].mean()
    print("\nk-means vocabulary, mean of the fits:\n" + km.to_string(float_format=lambda v: f"{v:.4f}"))
    print(f"-> {OUT}")
    return df


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--rows", nargs="+", choices=list(ROWS))
    ap.add_argument("--seeds", nargs="+", type=int)
    ap.add_argument("--no-cache", action="store_true", help="rebuild the monthly band means even if cached")
    a = ap.parse_args(argv)
    run(a.rows, a.seeds, cache=not a.no_cache)
    return 0


if __name__ == "__main__":
    sys.exit(main())
