"""Supplement S3 (and the letter's bits per month): plug-in entropy of each control's month codes on the evaluation region.

    python -m fsq_matched_controls.analysis.bits_per_month [--compare]   -> outputs/bits_per_month_per_fit.csv, bits_per_month_summary.csv

One estimator for every code: the plug-in Shannon entropy (bits) of the empirical distribution of the evaluation region's month
codes, over all months and over valid months only (different for Estonia only). Codes: rounded PCA (the five-level grid cell of
the four rounded scores; standardization on the probe pool plus the evaluated region), the k-means vocabulary (five fits) and,
for comparison, k-means fitted on the ten band means (five fits). No probe is involved: the values are deterministic given the
fits. --compare: rounded PCA and the ten-band k-means equal to the stored values at their four decimals; the letter's k-means
vocabulary (whose fits differ across platforms, see analysis.tolerances) by its five-fit means within KMEANS_BITS. The tokens' bits per month are in the per-run tables
(results/runs/n_probe_*.csv, column entropy_bits on the probe pool).
"""
from __future__ import annotations

import os

for _v in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import sys

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans

from fsq_matched_controls import paths
from fsq_matched_controls.controls import features, kmeans_vocabulary as KV, rounded_pca as RP
from fsq_matched_controls.analysis.tolerances import KMEANS_BITS
from fsq_matched_controls.controls.rows import ROWS

W5 = np.array([125, 25, 5, 1])


def ent(ids):
    c = np.bincount(np.asarray(ids).reshape(-1)).astype(float)
    p = c[c > 0] / c.sum()
    return float(-(p * np.log2(p)).sum())


def per_fit() -> pd.DataFrame:
    data = {"cropglobe": features.cropglobe_monthly(), "eurocropsml": features.eurocropsml_monthly()}
    out, fits = [], {}
    for row, spec in ROWS.items():
        d = data[spec["dataset"]]; X, V = d["X"], d["V"]; reg = spec["region"]
        key = (spec["dataset"], tuple(spec["pool"]))
        if key not in fits:
            pool = np.concatenate([X[s][V[s]] for s in spec["pool"]])
            pca, psc, pool_std = KV.fit_pool_projection(pool)
            fits[key] = dict(pool=pool, pca=pca, psc=psc, pool_std=pool_std,
                             k2={s: KV.fit_vocabulary(pca, psc, pool_std, s) for s in KV.SEEDS},
                             k0={s: MiniBatchKMeans(n_clusters=625, random_state=s, batch_size=4096, n_init=5, max_iter=300).fit(pool) for s in KV.SEEDS})
        F = fits[key]; ho, Vho = X[reg], V[reg]
        P_tr, P_ho = RP.project(F["pca"], X["train"]), RP.project(F["pca"], ho)
        sc = RP.scaler_on([P_tr, P_ho], [V["train"], Vho])
        ids = ((RP.fsq_round(sc.transform(P_ho.reshape(-1, 4))).astype(np.int64) + 2) * W5).sum(-1).reshape(len(ho), ho.shape[1])
        out.append(dict(row=row, code="rounded PCA", seed="na", bits_all=ent(ids), bits_valid=ent(ids[Vho])))
        for seed in KV.SEEDS:
            l2 = F["k2"][seed].codes(ho)
            out.append(dict(row=row, code="k-means K2 (letter)", seed=seed, bits_all=ent(l2), bits_valid=ent(l2[Vho])))
            l0 = F["k0"][seed].predict(ho.reshape(-1, 10)).reshape(len(ho), ho.shape[1])
            out.append(dict(row=row, code="k-means K0 (ten-band)", seed=seed, bits_all=ent(l0), bits_valid=ent(l0[Vho])))
        print(row, "done", flush=True)
    return pd.DataFrame(out)


def summary(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby(["row", "code"])
    s = g.agg(bits_all_mean=("bits_all", "mean"), bits_all_min=("bits_all", "min"), bits_all_max=("bits_all", "max"),
              bits_valid_mean=("bits_valid", "mean"), n_fits=("bits_all", "size")).reset_index()
    return s


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--compare", action="store_true")
    a = ap.parse_args(argv)
    df = per_fit(); s = summary(df)
    paths.OUTPUT.mkdir(parents=True, exist_ok=True)
    df.round(4).to_csv(paths.OUTPUT / "bits_per_month_per_fit.csv", index=False)
    s.round(4).to_csv(paths.OUTPUT / "bits_per_month_summary.csv", index=False)
    print(s.round(3).to_string(index=False))
    if a.compare:
        ok = True
        ref = pd.read_csv(paths.SUPPLEMENT_S3 / "bits_per_month_per_fit.csv", dtype={"seed": str})
        mine = df.copy(); mine["seed"] = mine["seed"].astype(str)
        m = ref.merge(mine, on=["row", "code", "seed"], suffixes=("_ref", ""))
        ok &= len(m) == len(ref)
        for code, g in m.groupby("code"):
            d = max(float((g[c + "_ref"] - g[c].round(4)).abs().max()) for c in ("bits_all", "bits_valid"))
            if code.startswith("k-means K2"):
                means = g.groupby("row")[["bits_all", "bits_all_ref", "bits_valid", "bits_valid_ref"]].mean()
                dm = float(max((means.bits_all - means.bits_all_ref).abs().max(), (means.bits_valid - means.bits_valid_ref).abs().max()))
                ok &= dm <= KMEANS_BITS
                print(f"{code}: single fits max |d| {d:.1e}; five-fit means max |d| {dm:.1e} (tolerance {KMEANS_BITS:g})")
            else:
                ok &= d <= 1e-4 + 1e-12  # the stored values have four decimals; a rounding tie moves the last digit
                print(f"{code}: {'identical at four decimals' if d == 0 else f'max |d| {d:.1e} (last stored digit)'}")
        print(f"{'PASS' if ok else 'FAIL'}: bits per month against results/supplement_S3_controls/")
        return 0 if ok else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
