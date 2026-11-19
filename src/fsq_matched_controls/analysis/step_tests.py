"""Tokens against models trained without the quantizer (Table II, supplement S5.3): the step tests with the twins' ORIGINAL runs.

  (a) continuous latent of the model trained without the quantizer (twin_noquant for reconstruction; twin_teacher, slow-teacher target, and twin_bands, band-value target, for masked prediction)
  (b) continuous latent (before rounding) of the FSQ-trained encoder of the same objective, ten runs (letter_runs.runs_df: cont_acc / cont_f1)
  (c) FSQ tokens of the same runs (tok_acc / tok_f1)
Test: two-sided Welch, ten runs against ten, Holm over the five rows per twin, metric and comparison (no multiplier on the number of looks).
Reads results/table2_decomposition/twin_probe_rows_transductive.csv (transductive CropGlobe) and twin_probe_rows_strict_estonia.csv (strict CropGlobe, EuroCropsML).

    python -m fsq_matched_controls.analysis.step_tests        writes outputs/twin_step_tests.csv

Moved from the original analysis script; the computation is unchanged. Its cross-checks against superseded manuscript files are not part of
the package (see the README).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from fsq_matched_controls import paths
from fsq_matched_controls.analysis import letter_runs as L
from fsq_matched_controls.stats.tests import holm_adjust_capped as holm_adj, welch_two_sided

OBJ = [("Reconstruction", "recon", "twin_noquant", "twin_noquant"), ("Masked prediction", "mp", "twin_teacher", "twin_teacher"), ("Masked prediction", "mp", "twin_bands", "twin_bands")]
MET = [("accuracy", "acc"), ("macro-F1", "f1")]


def twin_rows():
    tr = pd.read_csv(paths.TABLE2 / "twin_probe_rows_transductive.csv"); ex = pd.read_csv(paths.TABLE2 / "twin_probe_rows_strict_estonia.csv")
    return {"France, transductive": (tr, None, "FRA"), "Australia, transductive": (tr, None, "AUS"), "France, strict": (ex, "FRA", "FRA"),
            "Australia, strict": (ex, "AUS", "AUS"), "Estonia": (ex, "EC", "Estonia")}


def nofsq(src, row, kind):
    d, setting, reg = src[row]
    d = d[(d.kind == kind) & (d.region == reg) & (d["repr"] == "continuous")]
    if setting is not None:
        d = d[d.setting == setting]
    return d.sort_values("seed")


def step_tests() -> pd.DataFrame:
    src = twin_rows()
    rec = []
    for objname, ref, kind, twin in OBJ:
        for row in L.ROWS:
            a = nofsq(src, row, kind); f = L.runs_df(row, ref)
            assert len(a) == 10 and len(f) == 10 and a.seed.is_unique, (row, kind, len(a), len(f))
            for mname, m in MET:
                A = a["accuracy" if m == "acc" else "macro_f1"].values; B = f[f"cont_{m}"].values; C = f[f"tok_{m}"].values
                raw = L.CTL[row]["raw"][0 if m == "acc" else 1]
                r = dict(objective=objname, twin=twin, row=row, metric=mname, n_a=len(A), n_fsq=len(B), raw=raw, mean_a=A.mean(), mean_b=B.mean(), mean_c=C.mean(), sd_a=A.std(ddof=1), sd_b=B.std(ddof=1), sd_c=C.std(ddof=1),
                         width=A.mean() - raw, step_b_minus_a=B.mean() - A.mean(), rounding_c_minus_b=C.mean() - B.mean(), disc_c_minus_a=C.mean() - A.mean(), chain_gap=(A.mean() - raw) + (B.mean() - A.mean()) + (C.mean() - B.mean()) - (C.mean() - raw))
                for tag, X in (("ba", B), ("ca", C)):
                    t = welch_two_sided(X, A); r[f"welch_p_{tag}"] = float(t.pvalue); r[f"welch_t_{tag}"] = float(t.statistic)
                rec.append(r)
    R = pd.DataFrame(rec)
    for tag in ("ba", "ca"):
        R[f"holm_p_{tag}"] = np.nan
        for (twin, mname), idx in R.groupby(["twin", "metric"]).groups.items():
            R.loc[idx, f"holm_p_{tag}"] = holm_adj(R.loc[idx, f"welch_p_{tag}"].values)
    return R


def main():
    paths.OUTPUT.mkdir(parents=True, exist_ok=True)
    R = step_tests(); R.to_csv(paths.OUTPUT / "twin_step_tests.csv", index=False)
    print(f"wrote {paths.OUTPUT / 'twin_step_tests.csv'} ({len(R)} lines)")


if __name__ == "__main__":
    main()
