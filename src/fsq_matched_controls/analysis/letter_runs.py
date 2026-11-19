"""The letter's ten runs per objective and setting, its controls, and the Table I analysis.

Per-run scores: the first runs of each panel come from figure1_data_pre_ten_runs (tokens) and results/runs/n_probe_gate_*.csv
(continuous latent, training-pool accuracy, training facts); the seed-extension runs from results/runs/n_probe_new_*.csv.
Controls: rounded PCA from figure1_data_pre_ten_runs; the k-means vocabulary is the mean of five fits (results/runs/diagnostic72_k_readout.csv,
variant K2, seeds 1904, 42, 101, 2024, 4242).
Tests (Table I): one-sided one-sample t on per-run margins against a control, one-sided Welch between objectives; p multiplied by the
number of looks of the row (2 for the transductive rows and Estonia, 3 for the strict rows) and capped at 1; Holm within a column over the
five rows per metric; E (both metrics) / S (one) / N (neither). Holm-adjusted p = the smallest level at which the comparison survives.

Moved from the original analysis script without changes to the computation; the per-run tables are read when the module is imported.
"""
from __future__ import annotations

import glob

import numpy as np
import pandas as pd

from fsq_matched_controls import paths
from fsq_matched_controls.analysis import figure1_data_pre_ten_runs as F1
from fsq_matched_controls.stats.tests import holm_adjust, one_sided_t, welch_greater

PS = paths.RUN_RESULTS
ROWS = ["France, transductive", "Australia, transductive", "France, strict", "Australia, strict", "Estonia"]
SETTING = {"France, transductive": ("cropglobe_transductive", "FRA"), "Australia, transductive": ("cropglobe_transductive", "AUS"), "France, strict": ("cropglobe_strict_FRA", "FRA"),
           "Australia, strict": ("cropglobe_strict_AUS", "AUS"), "Estonia": ("eurocropml_estonia", "Estonia")}
LOOKS = {"France, transductive": 2, "Australia, transductive": 2, "France, strict": 3, "Australia, strict": 3, "Estonia": 2}
OLD_LOOKS = {"France, transductive": 1, "Australia, transductive": 1, "France, strict": 2, "Australia, strict": 2, "Estonia": 1}
KEY = {"mp": "predictive", "recon": "reconstruction"}
ALPHA = 0.05
SEEDS5 = [1904, 42, 101, 2024, 4242]

_gate = pd.concat([pd.read_csv(p) for p in sorted(glob.glob(str(PS / "n_probe_gate_*of*.csv")))], ignore_index=True); _gate["tag"] = _gate["tag"].astype(str)
_new = pd.concat([pd.read_csv(p) for p in sorted(glob.glob(str(PS / "n_probe_new_*of*.csv")))], ignore_index=True); _new["tag"] = _new["tag"].astype(str)
_d72 = pd.read_csv(PS / "diagnostic72_k_readout.csv"); _d72 = _d72[_d72["analysis"].isna()].copy(); _d72["seed"] = _d72["seed"].astype(str)


def k2_mean(row):
    d = _d72[(_d72.row == row) & (_d72.variant == "K2") & _d72.seed.isin([str(s) for s in SEEDS5])]
    assert len(d) == 5
    return float(d.acc.mean()), float(d.f1.mean())


CTL = {r: dict(PCA=tuple(F1.PANELS[r]["pca"]), kmeans=k2_mean(r), raw=tuple(F1.PANELS[r]["raw"]), majority=tuple(F1.PANELS[r]["majority"])) for r in ROWS}


def runs_df(row, obj, which="all"):
    """One line per run: tag, new, tok_acc, tok_f1, cont_acc, cont_f1, train_acc, codes_used, entropy_bits, epoch, global_step, best_val_loss, stop_reason, wall_clock_hours."""
    setting, reg = SETTING[row]
    out = []
    ex_tags = list(F1.PANELS[row][KEY[obj]].keys())
    for t in ex_tags:
        g = _gate[(_gate.objective == obj) & (_gate.setting == setting) & (_gate.region == reg) & (_gate.tag == t)]
        gt, gc = g[g.repr_kind == "tokens"].iloc[0], g[g.repr_kind == "continuous"].iloc[0]
        ta, tf = F1.PANELS[row][KEY[obj]][t]
        out.append(dict(tag=t, new=False, tok_acc=ta, tok_f1=tf, cont_acc=float(gc.accuracy), cont_f1=float(gc.macro_f1), train_acc=float(gt.train_acc), codes_used=int(gt.codes_used),
                        entropy_bits=float(gt.entropy_bits), epoch=int(gt.epoch), global_step=int(gt.global_step), best_val_loss=float(gt.best_val_loss) if pd.notna(gt.best_val_loss) else np.nan,
                        stop_reason="n/a", wall_clock_hours=np.nan))
    n = _new[(_new.objective == obj) & (_new.setting == setting) & (_new.region == reg)]
    for t in sorted(set(n.tag), key=int):
        g = n[n.tag == t]; gt, gc = g[g.repr_kind == "tokens"].iloc[0], g[g.repr_kind == "continuous"].iloc[0]
        out.append(dict(tag=t, new=True, tok_acc=float(gt.accuracy), tok_f1=float(gt.macro_f1), cont_acc=float(gc.accuracy), cont_f1=float(gc.macro_f1), train_acc=float(gt.train_acc),
                        codes_used=int(gt.codes_used), entropy_bits=float(gt.entropy_bits), epoch=int(gt.epoch), global_step=int(gt.global_step), best_val_loss=float(gt.best_val_loss),
                        stop_reason=gt.stop_reason, wall_clock_hours=float(gt.wall_clock_hours)))
    d = pd.DataFrame(out)
    return d if which == "all" else (d[~d.new] if which == "existing" else d[d.new])


def arr(row, obj, kind="tok", which="all"):
    d = runs_df(row, obj, which)
    return d[[f"{kind}_acc", f"{kind}_f1"]].values


COLS = ["MP-PCA", "MP-kmeans", "MP-recon", "R-PCA", "R-kmeans"]


def analyse(data, looks, ctl=None, cols=COLS):
    """data[row] = dict(mp=(n,2), recon=(n,2)) of (accuracy, macro-F1); ctl[row] = dict(PCA=(a,f), kmeans=(a,f)); returns (list of dict rows, verdict dict per (row, col))."""
    ctl = ctl or CTL
    out = []
    for r in ROWS:
        P, R = data[r]["mp"], data[r]["recon"]
        for col, a, c in (("MP-PCA", P, "PCA"), ("MP-kmeans", P, "kmeans"), ("R-PCA", R, "PCA"), ("R-kmeans", R, "kmeans")):
            if col not in cols:
                continue
            for i, met in enumerate(("acc", "f1")):
                m = a[:, i] - ctl[r][c][i]
                p = one_sided_t(m)
                out.append(dict(panel=r, col=col, metric=met, n=len(a), margin=float(m.mean()), p_raw=p, p=min(1.0, p * looks[r]), above=int((m > 0).sum()), ctl=float(ctl[r][c][i])))
        if "MP-recon" in cols:
            for i, met in enumerate(("acc", "f1")):
                p = welch_greater(P[:, i], R[:, i])
                out.append(dict(panel=r, col="MP-recon", metric=met, n=len(P), margin=float(P[:, i].mean() - R[:, i].mean()), p_raw=p, p=min(1.0, p * looks[r]), above=None, ctl=float("nan")))
    for col in cols:
        for met in ("acc", "f1"):
            fam = [x for x in out if x["col"] == col and x["metric"] == met]
            adj = holm_adjust([x["p"] for x in fam])
            for x, a in zip(fam, adj):
                x["p_holm"] = float(a); x["h_col"] = bool(a <= ALPHA)
    for met in ("acc", "f1"):
        fam = [x for x in out if x["metric"] == met]
        adj = holm_adjust([x["p"] for x in fam])
        for x, a in zip(fam, adj):
            x["p_holm_pool"] = float(a); x["h_pool"] = bool(a <= ALPHA)
    def verd(key):
        v = {}
        for x in out:
            v.setdefault((x["panel"], x["col"]), []).append(x[key])
        return {k: ("E" if all(b) else ("S" if any(b) else "N")) for k, b in v.items()}
    return out, verd("h_col"), verd("h_pool")


def collect(which="all"):
    return {r: dict(mp=arr(r, "mp", "tok", which), recon=arr(r, "recon", "tok", which)) for r in ROWS}


def cell(x):
    return next(z for z in x)


def find(rows, panel, col, met):
    return next(x for x in rows if x["panel"] == panel and x["col"] == col and x["metric"] == met)


def pfmt(p):
    return "<0.001" if p < 0.001 else (f"{p:.2f}" if p >= 0.1 else f"{p:.3f}")


def mean_sd(a):
    a = np.asarray(a, float)
    return float(a.mean()), float(a.std(ddof=1)) if len(a) > 1 else float("nan")
