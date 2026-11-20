"""Supplement S4.1: Table I without label harmonization.

    python -m fsq_matched_controls.analysis.no_harmonization [--compare]     -> outputs/no_label_harmonization.csv

Every probe is fitted on all training classes (classes missing on the evaluation side are not merged into a residual class) and
scored on the raw evaluation labels: the letter's ten runs per objective (token arrays cached by
python -m fsq_matched_controls.analysis.probe_runs), rounded PCA, the k-means vocabulary (five fits, scores averaged) and raw band
means. Table I's tests on the new scores; verdicts beside the letter's. --compare: every verdict equal to the stored table's.
Needs the token cache of all 80 runs and the monthly band means.
"""
from __future__ import annotations

import os

for _v in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import sys

import numpy as np
import pandas as pd

from fsq_matched_controls import paths
from fsq_matched_controls.analysis import letter_runs as L
from fsq_matched_controls.analysis.control_variants import STORED_STANDARDIZE, Row, analyse_columns, learned
from fsq_matched_controls.analysis.probe_runs import token_cache_path
from fsq_matched_controls.controls import features, kmeans_vocabulary as KV
from fsq_matched_controls.controls.__main__ import probe
from fsq_matched_controls.stats.tests import holm_adjust, welch_greater

REGION_KEY = {"FRA": "FRA", "AUS": "AUS", "estonia": "Estonia"}


def run_dirs(row, obj):
    """The run directories of the letter's ten runs of an objective in a row, in letter_runs' order."""
    setting, _ = L.SETTING[row]
    src = pd.concat([L._gate, L._new], ignore_index=True)
    return [src[(src.objective == obj) & (src.setting == setting) & (src.tag == t)].run_dir.iloc[0] for t in L.runs_df(row, obj).tag]


def run():
    data = {"cropglobe": features.cropglobe_monthly(), "eurocropsml": features.eurocropsml_monthly()}
    scores, ctl = {}, {}
    for r in L.ROWS:
        o = Row(r, data)
        ytr, yte = data[o.ds]["y"]["train"], data[o.ds]["y"][o.reg]
        scores[r] = {}
        for obj in ("mp", "recon"):
            out = []
            for run_dir in run_dirs(r, obj):
                z = np.load(token_cache_path(run_dir))
                reg = REGION_KEY[o.reg]
                assert np.array_equal(z["y_train"], ytr) and np.array_equal(z[f"y_{reg}"], yte), run_dir
                out.append(probe(z["tok_train"].astype(np.float32), ytr, z[f"tok_{reg}"].astype(np.float32), yte))
            scores[r][obj] = np.array(out)
        Ftr, Fho = o.pca_features(STORED_STANDARDIZE[r], "round")
        pca_s = probe(Ftr, ytr, Fho, yte)
        raw_s = probe(o.X["train"].reshape(len(o.X["train"]), -1), ytr, o.X[o.reg].reshape(len(o.X[o.reg]), -1), yte)
        pool = np.concatenate([o.X[k][o.V[k]] for k in o.pool])
        pca, psc, pool_std = KV.fit_pool_projection(pool)
        ks = []
        for seed in KV.SEEDS:
            voc = KV.fit_vocabulary(pca, psc, pool_std, seed)
            ks.append(probe(voc.represent(o.X["train"], o.V["train"]), ytr, voc.represent(o.X[o.reg], o.V[o.reg]), yte))
        ctl[r] = dict(PCA=pca_s, kmeans=tuple(np.mean(ks, 0)), raw=raw_s)
        print(r, {k: tuple(round(x, 4) for x in v) for k, v in ctl[r].items()}, flush=True)
    a, va = analyse_columns(scores, {r: ctl[r]["PCA"] for r in L.ROWS}, {"MP-PCA": "mp", "R-PCA": "recon"}, "t")
    b, vb = analyse_columns(scores, {r: ctl[r]["kmeans"] for r in L.ROWS}, {"MP-kmeans": "mp", "R-kmeans": "recon"}, "t")
    w = []
    for r in L.ROWS:
        for i, met in enumerate(("acc", "f1")):
            p = welch_greater(scores[r]["mp"][:, i], scores[r]["recon"][:, i])
            w.append(dict(panel=r, metric=met, p=min(1.0, p * L.LOOKS[r])))
    vw = {}
    for met in ("acc", "f1"):
        fam = [x for x in w if x["metric"] == met]
        for x, ad in zip(fam, holm_adjust([x["p"] for x in fam])):
            vw.setdefault((x["panel"], "MP-recon"), []).append(bool(ad <= L.ALPHA))
    vw = {k: ("E" if all(v) else ("S" if any(v) else "N")) for k, v in vw.items()}
    vt = {**va, **vb, **vw}
    _, lv_all, _ = L.analyse(learned("tok"), L.LOOKS)
    df = pd.DataFrame([dict(row=r, column=c, verdict_no_harm=vt[(r, c)], verdict_letter=lv_all[(r, c)]) for r in L.ROWS for c in L.COLS])
    paths.OUTPUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(paths.OUTPUT / "no_label_harmonization.csv", index=False)
    return df


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--compare", action="store_true")
    a = ap.parse_args(argv)
    df = run()
    print(df.to_string(index=False))
    if a.compare:
        ref = pd.read_csv(paths.SUPPLEMENT_S4 / "no_label_harmonization.csv")
        same = df.astype(str).equals(ref.astype(str))
        print(f"verdicts identical to results/supplement_S4_sensitivity/no_label_harmonization.csv: {same}")
        return 0 if same else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
