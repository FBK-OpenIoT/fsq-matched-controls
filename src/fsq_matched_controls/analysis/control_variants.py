"""Supplement S4.1 and S5.2: variants of the rounded-PCA control, and a spectral-index control, tested as in Table I.

    python -m fsq_matched_controls.analysis.control_variants [--compare]

    continuous_pca_vs_latent.csv            the learned continuous latent (per-run scores before rounding) against continuous PCA-4
                                            (standardized as the rounded control, not clipped, not rounded; beside it, clipped at +-2,
                                            not rounded)
    rounded_pca_fitpool_standardization.csv Table I's PCA columns with rounded PCA standardized on the encoder's fit pool instead of
                                            the probe's training and evaluation samples
    spectral_index_control.csv              a width-matched spectral-index control: NDVI (B8, B4), NDRE (B8A, B5), NDMI (B8, B11) and
                                            (B11 - B12) / (B11 + B12) from the monthly mean reflectance, each standardized on the fit pool,
                                            clipped at +-2 and rounded to the five-level grid (four numbers per month; a denominator below
                                            1e-6 gives 0)

Tests as Table I: per-run margins of the letter's ten runs per objective against the control, one-sided t and the exact sign-flip
test, p multiplied by the row's number of interim analyses, Holm within column and metric over the five rows, verdict E / S / N.
The control scores come from the probe (lbfgs, 300 iterations, unconverged on these features), so they vary across environments
by up to about 5e-4; --compare checks the stored tables: every verdict identical, every control score within the tolerance of
its table (analysis.tolerances: GENERAL; UNCONVERGED for the spectral-index control).
Needs the monthly band means (python -m fsq_matched_controls.controls builds and caches them).
"""
from __future__ import annotations

import os

for _v in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

from fsq_matched_controls import paths
from fsq_matched_controls.analysis import letter_runs as L
from fsq_matched_controls.analysis.tolerances import GENERAL, UNCONVERGED
from fsq_matched_controls.analysis.signflip_tests import signflip_p
from fsq_matched_controls.controls import features, rounded_pca as RP
from fsq_matched_controls.controls.__main__ import probe
from fsq_matched_controls.controls.rows import ROWS as SPEC, STANDARDIZE
from fsq_matched_controls.probe.linear_probe import harmonize_label_sets
from fsq_matched_controls.stats.tests import holm_adjust, one_sided_t

# The supplement's sensitivity tables were computed with Estonia's rounded PCA standardized on the probe pool plus Estonia
# (valid months); the letter's own Estonia control is standardized on the probe pool only (controls.rows.STANDARDIZE). The
# tables are reproduced with the construction they were computed with; for the CropGlobe rows the two are the same.
STORED_STANDARDIZE = dict(STANDARDIZE, Estonia=["train", "estonia"])

BAND = {"cropglobe": dict(B2=0, B3=1, B4=2, B5=3, B6=4, B7=5, B8=6, B8A=7, B11=8, B12=9),
        "eurocropsml": dict(B2=0, B3=1, B4=2, B8=3, B5=4, B6=5, B7=6, B8A=7, B11=8, B12=9)}
PAIRS = [("NDVI", "B8", "B4"), ("NDRE", "B8A", "B5"), ("NDMI", "B8", "B11"), ("SWIR", "B11", "B12")]
EPS = 1e-6


class Row:
    def __init__(self, row, data):
        s = SPEC[row]; d = data[s["dataset"]]
        self.row, self.ds, self.reg = row, s["dataset"], s["region"]
        self.X, self.V = d["X"], d["V"]
        kw = {"other_class": -1} if self.ds == "eurocropsml" else {}
        self.ytr, self.yho, _ = harmonize_label_sets(d["y"]["train"], d["y"][self.reg], **kw)
        self.pool = s["pool"]
        self.Vtr, self.Vho = self.V["train"], self.V[self.reg]
        self.Vtr_m = None if self.ds == "cropglobe" else self.Vtr
        self.pca = RP.fit_pca(np.concatenate([self.X[k][self.V[k]] for k in self.pool]))
        self.P = {k: RP.project(self.pca, self.X[k]) for k in set(self.pool) | {"train", self.reg} | set(STORED_STANDARDIZE[row])}

    def pca_features(self, standardize_on, mode):
        sc = RP.scaler_on([self.P[k] for k in standardize_on], [self.V[k] for k in standardize_on])
        out = []
        for k in ("train", self.reg):
            z = sc.transform(self.P[k].reshape(-1, 4)).reshape(self.P[k].shape)
            z = np.clip(z, -2, 2).round() if mode == "round" else (np.clip(z, -2, 2) if mode == "clip" else z)
            z[~self.V[k]] = 0.0
            out.append(z.reshape(len(z), -1))
        return out

    def score(self, Ftr, Fho):
        return probe(Ftr, self.ytr, Fho, self.yho)


def analyse_columns(data, ctl, cols, test):
    out = []
    for r in L.ROWS:
        for col, obj in cols.items():
            a = data[r][obj]
            for i, met in enumerate(("acc", "f1")):
                m = a[:, i] - ctl[r][i]
                p = one_sided_t(m) if test == "t" else signflip_p(m)
                out.append(dict(panel=r, col=col, metric=met, n=len(a), margin=float(m.mean()), p_raw=p, p=min(1.0, p * L.LOOKS[r]),
                                above=int((m > 0).sum()), ctl=float(ctl[r][i])))
    for col in cols:
        for met in ("acc", "f1"):
            fam = [x for x in out if x["col"] == col and x["metric"] == met]
            for x, ad in zip(fam, holm_adjust([x["p"] for x in fam])):
                x["p_holm"] = float(ad); x["h_col"] = bool(ad <= L.ALPHA)
    v = {}
    for x in out:
        v.setdefault((x["panel"], x["col"]), []).append(x["h_col"])
    return out, {k: ("E" if all(b) else ("S" if any(b) else "N")) for k, b in v.items()}


def learned(kind):
    return {r: dict(mp=L.arr(r, "mp", kind), recon=L.arr(r, "recon", kind)) for r in L.ROWS}


def _two_tests(dat, ctl, cols, extra):
    rt, vt = analyse_columns(dat, ctl, cols, "t"); rs, vs = analyse_columns(dat, ctl, cols, "sf")
    rows = []
    for col in cols:
        for r in L.ROWS:
            for met in ("acc", "f1"):
                xt, xs = L.find(rt, r, col, met), L.find(rs, r, col, met)
                rows.append(dict(**extra, column=col, row=r, metric=met, margin=xt["margin"], control_score=xt["ctl"], p_raw_t=xt["p_raw"],
                                 p_raw_signflip=xs["p_raw"], p_holm_t=xt["p_holm"], p_holm_signflip=xs["p_holm"], runs_above=xt["above"],
                                 verdict_t=vt[(r, col)], verdict_signflip=vs[(r, col)]))
    return rows


def _norm(dataset):
    ck = torch.load(paths.CHECKPOINTS / f"{features.NORM_SOURCE[dataset]}.pt", map_location="cpu", weights_only=True)["norm_stats"]
    return np.asarray(ck["mean"], np.float64), np.asarray(ck["std"], np.float64)


def indices(X, dataset):
    mean, std = _norm(dataset)
    R = X.astype(np.float64) * (std + 1e-8) + mean          # normalised monthly means back to reflectance
    b = BAND[dataset]; out = np.zeros(X.shape[:-1] + (4,))
    for k, (_, a_, c_) in enumerate(PAIRS):
        a, c = R[..., b[a_]], R[..., b[c_]]; den = a + c; bad = ~(np.abs(den) > EPS)
        out[..., k] = np.where(bad, 0.0, (a - c) / np.where(bad, 1.0, den))
    return out


def run():
    data = {"cropglobe": features.cropglobe_monthly(), "eurocropsml": features.eurocropsml_monthly()}
    R = {r: Row(r, data) for r in L.ROWS}
    sc = {}
    for r, o in R.items():
        fit_pool = o.pool if o.ds == "cropglobe" else ["train"]
        sc[(r, "round")] = o.score(*o.pca_features(STORED_STANDARDIZE[r], "round"))
        sc[(r, "cont")] = o.score(*o.pca_features(STORED_STANDARDIZE[r], "cont"))
        sc[(r, "clip")] = o.score(*o.pca_features(STORED_STANDARDIZE[r], "clip"))
        sc[(r, "fitpool")] = o.score(*o.pca_features(fit_pool, "round"))
        I = {k: indices(o.X[k], o.ds) for k in set(o.pool) | {"train", o.reg}}
        isc = StandardScaler().fit(np.concatenate([I[k][o.V[k]] for k in o.pool]))
        F = []
        for k in ("train", o.reg):
            z = np.clip(isc.transform(I[k].reshape(-1, 4)).reshape(I[k].shape), -2, 2).round(); z[~o.V[k]] = 0.0; F.append(z.reshape(len(z), -1))
        sc[(r, "index")] = o.score(*F)
        print(f"{r:24s} rounded {sc[(r, 'round')][0]:.6f}  continuous {sc[(r, 'cont')][0]:.6f}  clipped {sc[(r, 'clip')][0]:.6f}  "
              f"fit-pool {sc[(r, 'fitpool')][0]:.6f}  spectral index {sc[(r, 'index')][0]:.6f}", flush=True)
    dc, tok = learned("cont"), learned("tok")
    r1 = []
    for tag, mode in (("continuous", "cont"), ("clipped", "clip")):
        r1 += _two_tests(dc, {r: sc[(r, mode)] for r in L.ROWS}, {"MP-PCAc": "mp", "R-PCAc": "recon"}, dict(control=tag))
    colsT = {"MP-PCA": "mp", "R-PCA": "recon"}
    old_t, vold_t = analyse_columns(tok, {r: sc[(r, "round")] for r in L.ROWS}, colsT, "t")
    new_t, vnew_t = analyse_columns(tok, {r: sc[(r, "fitpool")] for r in L.ROWS}, colsT, "t")
    old_s, vold_s = analyse_columns(tok, {r: sc[(r, "round")] for r in L.ROWS}, colsT, "sf")
    new_s, vnew_s = analyse_columns(tok, {r: sc[(r, "fitpool")] for r in L.ROWS}, colsT, "sf")
    r3 = []
    for col in colsT:
        for r in L.ROWS:
            for met in ("acc", "f1"):
                xo, xn, xs = L.find(old_t, r, col, met), L.find(new_t, r, col, met), L.find(new_s, r, col, met)
                r3.append(dict(column=col, row=r, metric=met, control_old=xo["ctl"], control_new=xn["ctl"], margin_old=xo["margin"], margin_new=xn["margin"],
                               p_raw_t_old=xo["p_raw"], p_raw_t_new=xn["p_raw"], p_raw_sf_new=xs["p_raw"], verdict_t_old=vold_t[(r, col)],
                               verdict_t_new=vnew_t[(r, col)], verdict_sf_old=vold_s[(r, col)], verdict_sf_new=vnew_s[(r, col)]))
    r5 = _two_tests(tok, {r: sc[(r, "index")] for r in L.ROWS}, {"MP-IDX": "mp", "R-IDX": "recon"}, {})
    out = {"continuous_pca_vs_latent": pd.DataFrame(r1), "rounded_pca_fitpool_standardization": pd.DataFrame(r3), "spectral_index_control": pd.DataFrame(r5)}
    paths.OUTPUT.mkdir(parents=True, exist_ok=True)
    for k, df in out.items():
        df.to_csv(paths.OUTPUT / f"{k}.csv", index=False)
    return out


TOLERANCE = {"continuous_pca_vs_latent": GENERAL, "rounded_pca_fitpool_standardization": GENERAL, "spectral_index_control": UNCONVERGED}
STORED = {"continuous_pca_vs_latent": paths.TABLE2, "rounded_pca_fitpool_standardization": paths.SUPPLEMENT_S4, "spectral_index_control": paths.SUPPLEMENT_S4}


def compare(out) -> bool:
    ok = True
    for name, df in out.items():
        ref = pd.read_csv(STORED[name] / f"{name}.csv")
        vcols = [c for c in df.columns if c.startswith("verdict")]
        ccols = [c for c in df.columns if c.startswith("control_") and c != "control"] or ["control_score"]
        same_v = df[vcols].astype(str).equals(ref[vcols].astype(str))
        d = max(float((df[c] - ref[c]).abs().max()) for c in ccols)
        tol = TOLERANCE[name]
        good = same_v and d <= tol
        ok &= good
        print(f"{name}: verdicts identical {same_v}; control scores {'identical' if d == 0 else f'max |d| {d:.2e}'} (tolerance {tol:g}) -> {'PASS' if good else 'FAIL'}")
    return ok


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--compare", action="store_true", help="compare with the stored tables")
    a = ap.parse_args(argv)
    out = run()
    if a.compare:
        return 0 if compare(out) else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
