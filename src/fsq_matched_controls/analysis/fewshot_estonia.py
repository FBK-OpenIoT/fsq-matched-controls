"""Supplement S4.2: a probe with Estonian labels, on the official EuroCropsML few-shot split.

    python -m fsq_matched_controls.analysis.fewshot_estonia [--compare]     -> outputs/estonia_official_fewshot.csv

The official split (split/latvia_portugal_vs_estonia/finetune/region_split_{k}.json): nested training sets of 123 / 486 / 840 /
1,402 / 4,393 / 7,264 / 14,179 parcels for k = 1, 5, 10, 20, 100, 200, 500, and one test set of 35,182 parcels (the letter's
Estonia evaluation set). The letter's ten masked-prediction and ten reconstruction runs (tokens), rounded PCA, the k-means vocabulary
(five fits, scores averaged) and the raw band means are probed with the letter's probe fitted on the k-shot training parcels only
and scored on the test set; labels = every Estonian class of the training and test parcels (no harmonization to the pretraining
pool). Tests: per-run margins against the control, one-sided t and the exact sign-flip test, p x 2, Holm over the seven k per
column and metric, verdict E / S / N. The controls keep the Estonia row's fits (pool: Latvia + Portugal, valid months).
Needs FSQ_DATA_ROOT with eurocropml/preprocess/ (at least the k-shot training parcels), eurocropml/split/ and the data module's
cache of the test set (eurocropml/cache/latvia_portugal_vs_estonia_wmax10_wmin2/finetune_test.npz).
--compare: every verdict equal to the stored table; margins within analysis.tolerances (UNCONVERGED for the rounded-PCA columns,
FEWSHOT_KMEANS for the k-means columns).
"""
from __future__ import annotations

import os

for _v in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import sys

import numpy as np
import pandas as pd
import torch

from fsq_matched_controls import paths
from fsq_matched_controls.analysis import letter_runs as L
from fsq_matched_controls.analysis.check_end_to_end import load_tokenizer
from fsq_matched_controls.analysis.tolerances import FEWSHOT_KMEANS, UNCONVERGED
from fsq_matched_controls.analysis.control_variants import STORED_STANDARDIZE, Row
from fsq_matched_controls.analysis.no_harmonization import run_dirs
from fsq_matched_controls.analysis.signflip_tests import signflip_p
from fsq_matched_controls.controls import features, kmeans_vocabulary as KV
from fsq_matched_controls.controls.__main__ import probe
from fsq_matched_controls.data.eurocropml import _npz_to_padded_months, _parse_ec_hcat_c
from fsq_matched_controls.data.normalization import NormStats
from fsq_matched_controls.stats.tests import holm_adjust, one_sided_t

KS = [1, 5, 10, 20, 100, 200, 500]
SPLIT = "latvia_portugal_vs_estonia"


def _inputs():
    root = paths.DATA_ROOT / "eurocropml"
    spl = root / "split" / SPLIT / "finetune"
    splits = {k: json.loads((spl / f"region_split_{k}.json").read_text()) for k in KS}
    union, seen = [], set()
    for k in KS:
        for f in splits[k]["train"]:
            if f not in seen:
                seen.add(f); union.append(f)
    r = [_npz_to_padded_months(root / "preprocess" / f, w_max=10, w_min=2) for f in union]
    win, obsm, valid = np.stack([a[0] for a in r]), np.stack([a[1] for a in r]), np.stack([a[2] for a in r])
    codes_tr = [_parse_ec_hcat_c(f) for f in union]
    tc = np.load(root / "cache" / f"{SPLIT}_wmax10_wmin2" / "finetune_test.npz", allow_pickle=False)
    return splits, {f: i for i, f in enumerate(union)}, (win, obsm, valid, codes_tr), (tc["windows"], tc["obs_mask"], tc["valid_mask"], tc["codes"].tolist())


@torch.no_grad()
def _tokens(run_dir, w, om):
    enc, fsq, ck = load_tokenizer(run_dir)
    ns = NormStats(mean=np.asarray(ck["norm_stats"]["mean"], np.float32), std=np.asarray(ck["norm_stats"]["std"], np.float32),
                   dn_scale=float(ck["norm_stats"]["dn_scale"]))
    out = []
    for i in range(0, len(w), 256):
        x = torch.as_tensor(ns.apply(w[i:i + 256]).astype(np.float32)); o = torch.as_tensor(om[i:i + 256]); B, Wn, obs, bands = x.shape
        zq, _ = fsq(enc(x.view(B * Wn, obs, bands), obs_mask=o.reshape(B * Wn, obs)))
        out.append(zq.numpy().reshape(B, -1))
    return np.concatenate(out)


def run():
    splits, row_of, (win, obsm, valid, codes_tr), (win_te, obs_te, val_te, codes_te) = _inputs()
    vocab = {c: i for i, c in enumerate(sorted(set(codes_tr) | set(codes_te)))}
    y_tr = np.array([vocab[c] for c in codes_tr]); y_te = np.array([vocab[c] for c in codes_te])
    data = {"cropglobe": None, "eurocropsml": features.eurocropsml_monthly()}
    o = Row("Estonia", data)
    ck = torch.load(paths.CHECKPOINTS / f"{features.NORM_SOURCE['eurocropsml']}.pt", map_location="cpu", weights_only=True)["norm_stats"]
    ns = NormStats(mean=np.asarray(ck["mean"], np.float32), std=np.asarray(ck["std"], np.float32), dn_scale=float(ck["dn_scale"]))

    def monthly(w, om):
        x = ns.apply(w).astype(np.float32) * om[..., None]; cnt = om.sum(2, keepdims=True)
        return (x.sum(2) / np.maximum(cnt, 1)).astype(np.float32)
    M_tr, M_te = monthly(win, obsm), monthly(win_te, obs_te)
    # rounded PCA with the Estonia row's fits (PCA on the pool, standardization of the letter's construction)
    from fsq_matched_controls.controls import rounded_pca as RP
    sc4 = RP.scaler_on([o.P[k] for k in STORED_STANDARDIZE["Estonia"]], [o.V[k] for k in STORED_STANDARDIZE["Estonia"]])
    pca_feat = lambda M, V: RP.rounded(sc4, RP.project(o.pca, M), V)
    F_pca_tr, F_pca_te = pca_feat(M_tr, valid), pca_feat(M_te, val_te)
    pool = o.X["train"][o.V["train"]]
    pca, psc, pool_std = KV.fit_pool_projection(pool)
    vocs = [KV.fit_vocabulary(pca, psc, pool_std, s) for s in KV.SEEDS]
    F_k2 = [(v.represent(M_tr, valid), v.represent(M_te, val_te)) for v in vocs]
    F_raw_tr, F_raw_te = M_tr.reshape(len(M_tr), -1), M_te.reshape(len(M_te), -1)
    tok = {}
    for obj in ("mp", "recon"):
        for rd in run_dirs("Estonia", obj):
            tok[(obj, rd)] = (_tokens(rd, win, obsm), _tokens(rd, win_te, obs_te))
            print("tokens", obj, rd, flush=True)
    res = {}
    for k in KS:
        ix = np.array([row_of[f] for f in splits[k]["train"]]); yk = y_tr[ix]
        res[(k, "PCA")] = probe(F_pca_tr[ix], yk, F_pca_te, y_te)
        res[(k, "raw")] = probe(F_raw_tr[ix], yk, F_raw_te, y_te)
        res[(k, "kmeans")] = tuple(np.mean([probe(a[ix], yk, b, y_te) for a, b in F_k2], 0))
        for obj in ("mp", "recon"):
            res[(k, obj)] = np.array([probe(tok[(obj, rd)][0][ix], yk, tok[(obj, rd)][1], y_te) for rd in run_dirs("Estonia", obj)])
        print("k", k, "done", flush=True)
    tests, verd = {}, {}
    for col, obj, ctl in (("MP-PCA", "mp", "PCA"), ("MP-kmeans", "mp", "kmeans"), ("R-PCA", "recon", "PCA"), ("R-kmeans", "recon", "kmeans")):
        for k in KS:
            for i, met in enumerate(("acc", "f1")):
                m = res[(k, obj)][:, i] - res[(k, ctl)][i]
                tests[(col, k, met)] = dict(margin=float(m.mean()), p_t=one_sided_t(m), p_sf=signflip_p(m))
    for test in ("p_t", "p_sf"):
        for col in ("MP-PCA", "MP-kmeans", "R-PCA", "R-kmeans"):
            for met in ("acc", "f1"):
                for k, a in zip(KS, holm_adjust([min(1.0, 2 * tests[(col, k, met)][test]) for k in KS])):
                    tests[(col, k, met)][f"holm_{test}"] = float(a)
            for k in KS:
                ok = [tests[(col, k, met)][f"holm_{test}"] <= L.ALPHA for met in ("acc", "f1")]
                verd[(col, k, test)] = "E" if all(ok) else ("S" if any(ok) else "N")
    rows = []
    for col in ("MP-PCA", "MP-kmeans", "R-PCA", "R-kmeans"):
        for k in KS:
            a, f = tests[(col, k, "acc")], tests[(col, k, "f1")]
            rows.append(dict(column=col, k=k, margin_acc=a["margin"], margin_f1=f["margin"], p_t_acc=a["p_t"], p_t_f1=f["p_t"],
                             verdict_t=verd[(col, k, "p_t")], verdict_sf=verd[(col, k, "p_sf")]))
    df = pd.DataFrame(rows)
    paths.OUTPUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(paths.OUTPUT / "estonia_official_fewshot.csv", index=False)
    return df


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--compare", action="store_true")
    a = ap.parse_args(argv)
    df = run()
    print(df.to_string(index=False))
    if a.compare:
        ref = pd.read_csv(paths.SUPPLEMENT_S4 / "estonia_official_fewshot.csv")
        same_v = df[["verdict_t", "verdict_sf"]].astype(str).equals(ref[["verdict_t", "verdict_sf"]].astype(str))
        ok = same_v
        for cols, tol in ((("MP-PCA", "R-PCA"), UNCONVERGED), (("MP-kmeans", "R-kmeans"), FEWSHOT_KMEANS)):
            m = df.column.isin(cols)
            d = max(float((df[m][c] - ref[m][c]).abs().max()) for c in ("margin_acc", "margin_f1"))
            ok &= d <= tol
            print(f"{' / '.join(cols)}: margins {'identical' if d == 0 else f'max |d| {d:.2e}'} (tolerance {tol:g})")
        print(f"verdicts identical {same_v} -> {'PASS' if ok else 'FAIL'}")
        return 0 if ok else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
