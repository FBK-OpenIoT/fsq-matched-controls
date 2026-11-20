"""Supplement S4.1 / S7.1: the exact one-sided sign-flip test beside the letter's t-tests, on the same per-run margins.

    python -m fsq_matched_controls.analysis.signflip_tests [--compare]     -> outputs/signflip_tests.csv

Table I's tests (one-sided one-sample t on the per-run margins against the fixed control; one-sided Welch for masked prediction
against reconstruction; p multiplied by the row's number of interim analyses, capped at 1; Holm within each column and metric over
the five rows; a metric survives below 0.05; verdict E both metrics, S one, N neither) next to the exact one-sided sign-flip
permutation test (all 2^10 sign patterns, statistic = mean margin) in the one-sample columns. Masked prediction against
reconstruction stays the Welch test in both versions.
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from fsq_matched_controls import paths
from fsq_matched_controls.analysis import letter_runs as L
from fsq_matched_controls.stats.tests import holm_adjust, one_sided_t, welch_greater

LAYOUT = ["France, transductive", "France, strict", "Australia, transductive", "Australia, strict", "Estonia"]
_PATTERNS = {}


def signflip_p(m):
    m = np.asarray(m, float); n = len(m)
    if n not in _PATTERNS:
        _PATTERNS[n] = np.array(list(itertools.product((-1.0, 1.0), repeat=n)))
    return float(((_PATTERNS[n] * np.abs(m)).mean(axis=1) >= m.mean() - 1e-12).mean())


def analyse(data, test):
    out = []
    for r in L.ROWS:
        P, R = data[r]["mp"], data[r]["recon"]
        for col, a, c in (("MP-PCA", P, "PCA"), ("MP-kmeans", P, "kmeans"), ("R-PCA", R, "PCA"), ("R-kmeans", R, "kmeans")):
            for i, met in enumerate(("acc", "f1")):
                m = a[:, i] - L.CTL[r][c][i]
                p = one_sided_t(m) if test == "t" else signflip_p(m)
                out.append(dict(panel=r, col=col, metric=met, margin=float(m.mean()), p_raw=p, p=min(1.0, p * L.LOOKS[r]), above=int((m > 0).sum())))
        for i, met in enumerate(("acc", "f1")):
            p = welch_greater(P[:, i], R[:, i])
            out.append(dict(panel=r, col="MP-recon", metric=met, margin=float(P[:, i].mean() - R[:, i].mean()), p_raw=p, p=min(1.0, p * L.LOOKS[r]), above=None))
    for col in L.COLS:
        for met in ("acc", "f1"):
            fam = [x for x in out if x["col"] == col and x["metric"] == met]
            for x, ad in zip(fam, holm_adjust([x["p"] for x in fam])):
                x["p_holm"] = float(ad); x["h_col"] = bool(ad <= L.ALPHA)
    v = {}
    for x in out:
        v.setdefault((x["panel"], x["col"]), []).append(x["h_col"])
    return out, {k: ("E" if all(b) else ("S" if any(b) else "N")) for k, b in v.items()}


def table():
    data = L.collect("all")
    for r in L.ROWS:
        assert len(data[r]["mp"]) == 10 and len(data[r]["recon"]) == 10
    rt, vt = analyse(data, "t")
    rs, vs = analyse(data, "sf")
    rows = []
    for col in L.COLS:
        for row in LAYOUT:
            for met in ("acc", "f1"):
                xt, xs = L.find(rt, row, col, met), L.find(rs, row, col, met)
                rows.append(dict(row=row, column=col, metric=met, margin=xt["margin"], p_raw_t=xt["p_raw"], p_raw_signflip=xs["p_raw"],
                                 p_used_t=xt["p"], p_used_signflip=xs["p"], p_holm_t=xt["p_holm"], p_holm_signflip=xs["p_holm"],
                                 verdict_t=vt[(row, col)], verdict_signflip=vs[(row, col)]))
    return pd.DataFrame(rows)


def main():
    df = table()
    paths.OUTPUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(paths.OUTPUT / "signflip_tests.csv", index=False)
    chg = df[df.verdict_t != df.verdict_signflip][["row", "column"]].drop_duplicates()
    print(f"verdict changes, t to sign-flip: {'none' if chg.empty else chg.to_string(index=False)}")
    print(f"-> {paths.OUTPUT / 'signflip_tests.csv'}")


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
        sys.exit(0 if compare([("signflip_tests", paths.SUPPLEMENT_S4)]) else 1)
