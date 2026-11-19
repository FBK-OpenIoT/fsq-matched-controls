"""Table II: tokens against models trained without the quantizer (CropGlobe transductive, EuroCropsML strict).

Columns (accuracy / macro-F1, mean of ten runs): no-quantizer model minus raw bands; tokens minus the no-quantizer model; of which rounding.
Rows per region:
  Reconstruction      (a) = twin `twin_noquant`, original runs (step_tests); bold = Holm p < 0.05 for (c) - (a) over the five rows, per twin and metric
  MP, slow teacher    (a) = twin `twin_teacher` at its latest finished stage, best checkpoint (twins_continued.teacher_twin_step_tests, variant "best")
  MP, band values     (a) = twin `twin_bands`, continued runs' best checkpoint (twins_continued.bands_twin_step_tests, stage "NEW"); no rounding column
  Rounded PCA         (a) = the continuous (unrounded) principal-component control (results/table2_decomposition/continuous_pca_vs_latent.csv);
                      tokens = rounded PCA (the Table I control), so tokens - (a) is the rounding cost itself
Raw bands as in letter_runs.CTL.

    python -m fsq_matched_controls.analysis.table2      writes outputs/table2.csv
"""
from __future__ import annotations

import pandas as pd

from fsq_matched_controls import paths
from fsq_matched_controls.analysis import letter_runs as L
from fsq_matched_controls.analysis.step_tests import step_tests
from fsq_matched_controls.analysis.twins_continued import bands_twin_step_tests, teacher_twin_step_tests

LAYOUT = [("France", "France, transductive"), ("Queensland", "Australia, transductive"), ("Estonia", "Estonia")]
MET = {"accuracy": 0, "macro-F1": 1}


def table2():
    noq = step_tests(); teach = teacher_twin_step_tests(); teach = teach[teach.variant == "best"]; bands = bands_twin_step_tests(); bands = bands[bands.stage == "NEW"]
    pca = pd.read_csv(paths.TABLE2 / "continuous_pca_vs_latent.csv"); pca = pca[pca.column == "MP-PCAc"]
    out = []
    for region, row in LAYOUT:
        for mname, mi in MET.items():
            raw = L.CTL[row]["raw"][mi]
            s = noq[(noq.twin == "twin_noquant") & (noq.row == row) & (noq.metric == mname)].iloc[0]
            out.append(dict(region=region, row=row, model="Reconstruction", metric=mname, no_quantizer_minus_raw=s.width, tokens_minus_no_quantizer=s.disc_c_minus_a,
                            rounding=s.rounding_c_minus_b, bold=bool(s.holm_p_ca < 0.05)))
            s = teach[(teach.row == row) & (teach.metric == mname)].iloc[0]
            out.append(dict(region=region, row=row, model="MP, slow teacher", metric=mname, no_quantizer_minus_raw=s.width, tokens_minus_no_quantizer=s.disc_c_minus_a,
                            rounding=s.rounding_c_minus_b, bold=bool(s.holm_p_ca < 0.05)))
            s = bands[(bands.row == row) & (bands.metric == ("accuracy" if mi == 0 else "macro_f1"))].iloc[0]
            out.append(dict(region=region, row=row, model="MP, band values", metric=mname, no_quantizer_minus_raw=s.mean_a - raw, tokens_minus_no_quantizer=s.disc_c_minus_a,
                            rounding=float("nan"), bold=bool(s.sig_ca)))
            cont = float(pca[(pca.row == row) & (pca.metric == ("acc" if mi == 0 else "f1"))].control_score.iloc[0]); rnd = L.CTL[row]["PCA"][mi]
            out.append(dict(region=region, row=row, model="Rounded PCA", metric=mname, no_quantizer_minus_raw=cont - raw, tokens_minus_no_quantizer=rnd - cont,
                            rounding=rnd - cont, bold=False))
    return pd.DataFrame(out)


def main():
    paths.OUTPUT.mkdir(parents=True, exist_ok=True)
    T = table2(); T.to_csv(paths.OUTPUT / "table2.csv", index=False)
    f = lambda v, b=False: "" if pd.isna(v) else (f"*{v:+.3f}*" if b else f"{v:+.3f}")
    for (region, model), g in T.groupby(["region", "model"], sort=False):
        a, m = g.iloc[0], g.iloc[1]
        print(f"{region:<11}{model:<18}{f(a.no_quantizer_minus_raw)} / {f(m.no_quantizer_minus_raw)}   {f(a.tokens_minus_no_quantizer, a.bold)} / {f(m.tokens_minus_no_quantizer, m.bold)}   "
              f"{f(a.rounding)} / {f(m.rounding)}")


if __name__ == "__main__":
    main()
