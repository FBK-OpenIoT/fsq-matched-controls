"""Table I: tests against matched controls (ten runs per objective and setting).

For every row (region, setting) and metric: the control scores (rounded PCA, k-means vocabulary), and the margins of masked prediction
minus each control, masked prediction minus reconstruction, and reconstruction minus each control, with the one-sided raw p and the
verdict after the interim-analysis multiplier and Holm's correction within the column and metric (letter_runs.analyse).

    python -m fsq_matched_controls.analysis.table1     writes outputs/table1.csv (one line per row, column and metric)
                                                       and outputs/table1_s71.csv (the supplement S7.1 record format of the original analysis)
"""
from __future__ import annotations

import pandas as pd

from fsq_matched_controls import paths
from fsq_matched_controls.analysis import letter_runs as L

LAYOUT = [("France", "transductive", "France, transductive"), ("France", "strict", "France, strict"),
          ("Queensland", "transductive", "Australia, transductive"), ("Queensland", "strict", "Australia, strict"), ("Estonia", "strict", "Estonia")]
LETTER_COLS = ["MP-PCA", "MP-kmeans", "MP-recon", "R-PCA", "R-kmeans"]


def table1():
    rows_p, vp, vp_pool = L.analyse(L.collect("all"), L.LOOKS)
    out = []
    for region, setting, row in LAYOUT:
        for mi, met in enumerate(("acc", "f1")):
            c = L.CTL[row]
            for col in LETTER_COLS:
                x = L.find(rows_p, row, col, met)
                out.append(dict(region=region, setting=setting, row=row, metric=met, control_pca=c["PCA"][mi], control_kmeans=c["kmeans"][mi], column=col,
                                margin=x["margin"], p_raw=x["p_raw"], p_after_looks=x["p"], p_holm=x["p_holm"], bold=x["h_col"], runs_above=x["above"],
                                verdict=vp[(row, col)]))
    return pd.DataFrame(out), rows_p, vp, vp_pool


def s71_records(rows_p, vp, vp_pool):
    """The 'S71' records exactly as the original analysis wrote them (column order MP-kmeans, MP-PCA, MP-recon, R-PCA, R-kmeans)."""
    rec = []
    for col in ("MP-kmeans", "MP-PCA", "MP-recon", "R-PCA", "R-kmeans"):
        for r in L.ROWS:
            a, f = L.find(rows_p, r, col, "acc"), L.find(rows_p, r, col, "f1")
            rec.append(dict(item="S71", col=col, row=r, margin_acc=a["margin"], margin_f1=f["margin"], p_raw_acc=a["p_raw"], p_raw_f1=f["p_raw"], p_used_acc=a["p"], p_used_f1=f["p"], p_holm_acc=a["p_holm"], p_holm_f1=f["p_holm"],
                            p_holm_pool25_acc=a["p_holm_pool"], p_holm_pool25_f1=f["p_holm_pool"], above_acc=a["above"], above_f1=f["above"], verdict=vp[(r, col)], verdict_pooled25=vp_pool[(r, col)]))
    return pd.DataFrame(rec)


def main():
    paths.OUTPUT.mkdir(parents=True, exist_ok=True)
    T, rows_p, vp, vp_pool = table1()
    T.to_csv(paths.OUTPUT / "table1.csv", index=False)
    s71_records(rows_p, vp, vp_pool).to_csv(paths.OUTPUT / "table1_s71.csv", index=False)
    for (region, setting, met), g in T.groupby(["region", "setting", "metric"], sort=False):
        cells = []
        for _, x in g.iterrows():
            s = f"{x.margin:+.3f}"
            s = f"*{s}*" if x.bold else s
            cells.append(s + (f" ({L.pfmt(x.p_raw)})" if x.column.startswith("MP") else ""))
        print(f"{region:<11}{setting:<13}{'Acc.' if met == 'acc' else 'F1':<5}{g.control_pca.iloc[0]:.3f}  {g.control_kmeans.iloc[0]:.3f}  " + "  ".join(cells))
    print("(*bold*: significant after the interim-analysis multiplier and Holm's correction within the column and metric)")


if __name__ == "__main__":
    main()
