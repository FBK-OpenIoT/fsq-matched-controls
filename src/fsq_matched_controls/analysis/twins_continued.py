"""The masked-prediction twins continued to early stopping (Table II, supplement S5.3).

Slow-teacher twin (twin_teacher): (a) is taken from each twin's LATEST FINISHED STAGE (second continuation where it finished and was probed,
else the first continuation, else the original run), at the best-validation checkpoint ("best") and at the final checkpoint ("final");
results/table2_decomposition/teacher_twin_continued_probe.csv. Band-value twin (twin_bands): OLD = the original (capped) runs, NEW = the capped runs replaced by their continued
runs' best checkpoint, FINAL = their final-epoch checkpoint; results/table2_decomposition/bands_twin_continued_probe_best.csv and bands_twin_continued_probe_final.csv.
(b), (c) = the FSQ-trained masked-prediction runs (letter_runs.runs_df). Test: two-sided Welch, ten against ten, Holm over the five rows per
metric (and stage).

    python -m fsq_matched_controls.analysis.twins_continued     writes outputs/teacher_twin_continued_step_tests.csv and outputs/bands_twin_continued_step_tests.csv

Moved from the original analysis scripts; the computation is unchanged. Their cross-checks against superseded manuscript files, and the
report text, are not part of the package (see the README).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from fsq_matched_controls import paths
from fsq_matched_controls.analysis import letter_runs as L
from fsq_matched_controls.stats.tests import holm_adjust_capped as holm_adj, welch_two_sided

ROWS = L.ROWS
PENDING = [11, 4242, 101]   # strict-France seeds whose second continuation finished last; held at the first continuation only with pin_pending=True
TWIN = "twin_teacher"
MET = [("accuracy", "acc"), ("macro-F1", "f1")]


# ------------------------------------------------------------------ slow-teacher twin (twin_teacher)
def _inputs():
    tr = pd.read_csv(paths.TABLE2 / "twin_probe_rows_transductive.csv"); ex = pd.read_csv(paths.TABLE2 / "twin_probe_rows_strict_estonia.csv")
    pr = pd.read_csv(paths.TABLE2 / "teacher_twin_continued_probe.csv")
    src = {"France, transductive": (tr, None, "FRA", "transductive CropGlobe"), "Australia, transductive": (tr, None, "AUS", "transductive CropGlobe"), "France, strict": (ex, "FRA", "FRA", "strict FRA"),
           "Australia, strict": (ex, "AUS", "AUS", "strict AUS"), "Estonia": (ex, "EC", "Estonia", "EuroCropsML (Estonia)")}
    return src, pr


def orig_rows(src, row, kind="twin_teacher"):
    d, setting, reg, _ = src[row]
    d = d[(d.kind == kind) & (d.region == reg)]
    if setting is not None:
        d = d[d.setting == setting]
    c = d[d["repr"] == "continuous"].sort_values("seed").reset_index(drop=True)
    r = d[d["repr"] == "rounded"].sort_values("seed").reset_index(drop=True)
    assert (c.seed.values == r.seed.values).all()
    return pd.DataFrame(dict(seed=c.seed, acc=c.accuracy, f1=c.macro_f1, racc=r.accuracy, rf1=r.macro_f1, rank=c.eff_rank, stage="orig"))


def twin(src, pr, row, variant, pin_pending=False):
    """(a) per seed for one row: variant orig | best | final; latest finished stage (second continuation > first continuation > original)."""
    d = orig_rows(src, row)
    if variant == "orig":
        return d
    pset = src[row][3]; reg = src[row][2]
    for i, s in enumerate(d.seed):
        for which, stage in ((f"{variant}_cont2", "cont2"), (variant, "cont1")):
            if stage == "cont2" and pin_pending and row == "France, strict" and int(s) in PENDING:
                continue
            q = pr[(pr.which == which) & (pr.setting == pset) & (pr.seed == s) & (pr.region == reg)]
            if len(q):
                c = q[q["repr"] == "continuous"].iloc[0]; r_ = q[q["repr"] == "rounded"].iloc[0]
                d.loc[i, ["acc", "f1", "racc", "rf1", "rank", "stage"]] = [c.accuracy, c.macro_f1, r_.accuracy, r_.macro_f1, c.eff_rank, stage]
                break
    return d


def step_table(src, pr, variant, pin_pending=False):
    rec = []
    for row in ROWS:
        a = twin(src, pr, row, variant, pin_pending); f = L.runs_df(row, "mp")
        assert len(a) == 10 and len(f) == 10 and a.seed.is_unique
        for mname, m in MET:
            A = a[m].values.astype(float); B = f[f"cont_{m}"].values; C = f[f"tok_{m}"].values
            raw = L.CTL[row]["raw"][0 if m == "acc" else 1]
            r = dict(objective="Masked prediction", twin=TWIN, row=row, metric=mname, n_a=len(A), n_fsq=len(B), raw=raw, mean_a=A.mean(), mean_b=B.mean(), mean_c=C.mean(), sd_a=A.std(ddof=1), sd_b=B.std(ddof=1),
                     sd_c=C.std(ddof=1), width=A.mean() - raw, step_b_minus_a=B.mean() - A.mean(), rounding_c_minus_b=C.mean() - B.mean(), disc_c_minus_a=C.mean() - A.mean(),
                     chain_gap=(A.mean() - raw) + (B.mean() - A.mean()) + (C.mean() - B.mean()) - (C.mean() - raw))
            for tag, X in (("ba", B), ("ca", C)):
                t = welch_two_sided(X, A); r[f"welch_p_{tag}"] = float(t.pvalue); r[f"welch_t_{tag}"] = float(t.statistic)
            rec.append(r)
    R = pd.DataFrame(rec)
    for tag in ("ba", "ca"):
        R[f"holm_p_{tag}"] = np.nan
        for mname, idx in R.groupby("metric").groups.items():
            R.loc[idx, f"holm_p_{tag}"] = holm_adj(R.loc[idx, f"welch_p_{tag}"].values)
    return R


def teacher_twin_step_tests(pin=False) -> pd.DataFrame:
    """The table written to teacher_twin_continued_step_tests.csv: variants best and final, with the original twin's values and the stage counts beside."""
    src, pr = _inputs()
    Ro = step_table(src, pr, "orig")
    R = {v: step_table(src, pr, v, pin) for v in ("best", "final")}

    def flags(Rv):
        out = []
        for _, r in Rv.iterrows():
            o = Ro[(Ro.row == r.row) & (Ro.metric == r.metric)].iloc[0]
            f = []
            for nm, col, hp in (("(c)-(a)", "disc_c_minus_a", "holm_p_ca"), ("(b)-(a)", "step_b_minus_a", "holm_p_ba")):
                if np.sign(r[col]) != np.sign(o[col]):
                    f.append(f"{nm} sign {o[col]:+.3f} -> {r[col]:+.3f}")
                if (r[hp] < 0.05) != (o[hp] < 0.05):
                    f.append(f"{nm} Holm {o[hp]:.3f} -> {r[hp]:.3f} ({'sig' if o[hp] < 0.05 else 'n.s.'} -> {'sig' if r[hp] < 0.05 else 'n.s.'})")
            if np.sign(r.width) != np.sign(o.width):
                f.append(f"width sign {o.width:+.3f} -> {r.width:+.3f}")
            out.append("; ".join(f))
        return out

    parts = []
    for v in ("best", "final"):
        d = R[v].copy(); d.insert(0, "variant", v); d["flags_vs_original"] = flags(R[v])
        for c in ("mean_a", "width", "step_b_minus_a", "disc_c_minus_a", "holm_p_ba", "holm_p_ca", "welch_p_ca"):
            d["orig_" + c] = [Ro[(Ro.row == r.row) & (Ro.metric == r.metric)].iloc[0][c] for _, r in d.iterrows()]
        d["n_replaced"] = [int((twin(src, pr, r.row, v, pin).stage != "orig").sum()) for _, r in d.iterrows()]
        d["n_cont1"] = [int((twin(src, pr, r.row, v, pin).stage == "cont1").sum()) for _, r in d.iterrows()]
        d["n_cont2"] = [int((twin(src, pr, r.row, v, pin).stage == "cont2").sum()) for _, r in d.iterrows()]
        parts.append(d)
    return pd.concat(parts, ignore_index=True)


# ------------------------------------------------------------------ band-value twin (twin_bands)
def bands_twin_step_tests() -> pd.DataFrame:
    tr = pd.read_csv(paths.TABLE2 / "twin_probe_rows_transductive.csv"); ex = pd.read_csv(paths.TABLE2 / "twin_probe_rows_strict_estonia.csv")
    best = pd.read_csv(paths.TABLE2 / "bands_twin_continued_probe_best.csv"); fin = pd.read_csv(paths.TABLE2 / "bands_twin_continued_probe_final.csv")
    MET_B = [("accuracy", "acc"), ("macro_f1", "f1")]
    SRC = {"France, transductive": (tr, None, "FRA"), "Australia, transductive": (tr, None, "AUS"), "France, strict": (ex, "FRA", "FRA"), "Australia, strict": (ex, "AUS", "AUS"), "Estonia": (ex, "EC", "Estonia")}
    SETNAME = {"France, transductive": ("transductive CropGlobe", "FRA"), "Australia, transductive": ("transductive CropGlobe", "AUS"), "France, strict": ("strict FRA", "FRA"), "Australia, strict": ("strict AUS", "AUS"), "Estonia": ("EuroCropsML (Estonia)", "Estonia")}

    def old_rows(row, rep="continuous"):
        d, setting, reg = SRC[row]; d = d[(d.kind == "twin_bands") & (d.region == reg) & (d["repr"] == rep)]
        if setting is not None:
            d = d[d.setting == setting]
        return d.sort_values("seed").set_index("seed")

    def new_rows(row, df, rep="continuous"):
        st, reg = SETNAME[row]; d = df[(df.setting == st) & (df.region == reg) & (df["repr"] == rep)]
        return d.set_index("seed")

    rec = []
    for row in ROWS:
        f = L.runs_df(row, "mp"); B = {m: f[f"cont_{k}"].values for m, k in (("accuracy", "acc"), ("macro_f1", "f1"))}; C = {m: f[f"tok_{k}"].values for m, k in (("accuracy", "acc"), ("macro_f1", "f1"))}
        o = old_rows(row); orr = old_rows(row, "rounded"); assert len(o) == 10, (row, len(o))
        for label, src in (("OLD", None), ("NEW", best), ("FINAL", fin)):
            cont = o.copy(); rnd = orr.copy(); nrep = 0
            if src is not None:
                n = new_rows(row, src); nr = new_rows(row, src, "rounded")
                for s in n.index:
                    assert s in cont.index, (row, s)
                    for m, _ in MET_B:
                        cont.loc[s, m] = n.loc[s, m]; rnd.loc[s, m] = nr.loc[s, m]
                    nrep += 1
            for m, _ in MET_B:
                A = cont[m].values
                rec.append(dict(row=row, stage=label, metric=m, n_replaced=nrep, mean_a=A.mean(), sd_a=A.std(ddof=1), min_a=A.min(), max_a=A.max(), mean_b=B[m].mean(), mean_c=C[m].mean(),
                                step_b_minus_a=B[m].mean() - A.mean(), disc_c_minus_a=C[m].mean() - A.mean(), rounding_a=(rnd[m].values - A).mean(),
                                welch_p_ba=float(welch_two_sided(B[m], A).pvalue), welch_p_ca=float(welch_two_sided(C[m], A).pvalue)))
    R = pd.DataFrame(rec)
    for tag in ("ba", "ca"):
        R[f"holm_p_{tag}"] = np.nan
        for (stage, m), idx in R.groupby(["stage", "metric"]).groups.items():
            R.loc[idx, f"holm_p_{tag}"] = holm_adj(R.loc[idx, f"welch_p_{tag}"].values)
    R["sig_ca"] = R.holm_p_ca < 0.05; R["sig_ba"] = R.holm_p_ba < 0.05
    return R


def main():
    paths.OUTPUT.mkdir(parents=True, exist_ok=True)
    teacher_twin_step_tests().to_csv(paths.OUTPUT / "teacher_twin_continued_step_tests.csv", index=False)
    bands_twin_step_tests().to_csv(paths.OUTPUT / "bands_twin_continued_step_tests.csv", index=False)
    print(f"wrote {paths.OUTPUT / 'teacher_twin_continued_step_tests.csv'} and {paths.OUTPUT / 'bands_twin_continued_step_tests.csv'}")


if __name__ == "__main__":
    main()
