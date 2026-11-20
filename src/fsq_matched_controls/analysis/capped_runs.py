"""Supplement S8: the masked-prediction runs stopped at the epoch cap, and the estimate of the work to continue them.

    python -m fsq_matched_controls.analysis.capped_runs [--compare]     -> outputs/capped_runs_{timing_groups,estimate,work,check_summary}.csv

Inputs (results/supplement_S8_training/): capped_runs_list.csv (the 89 capped runs: 20 FSQ-trained, 32 band-value twins, 37
slow-teacher twins), capped_runs_check.csv (what each run's last checkpoint holds), capped_runs_timing.csv (seconds per epoch of
the node's training runs and the number of other runs at each run's midpoint; runs other than the capped ones are labelled
"other run on the node") and training_times_per_run.csv.
Estimate: work to add per run between one patience window without improvement (lower bound) and running from the cap to twice the
cap (upper bound); GPU-seconds per run-epoch g = seconds per epoch / (runs sharing the node at its midpoint + 1), taken from the
runs measured while the node was crowded (>= 10 other runs for the twins, >= 7 for the FSQ-trained runs; all runs of a set if none);
days = GPU hours / 24 / f, with f = 1 (GPU free; columns days_*_free) or 0.5 (the GPU shared with the other jobs of the time;
columns days_*_shared).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from fsq_matched_controls import paths


def build():
    D = paths.SUPPLEMENT_S8
    ck = pd.read_csv(D / "capped_runs_check.csv"); tm = pd.read_csv(D / "capped_runs_timing.csv")
    dt = pd.read_csv(D / "training_times_per_run.csv"); L = pd.read_csv(D / "capped_runs_list.csv")
    bench = lambda s: "EuroCropsML" if s.startswith("Euro") else "CropGlobe"
    L["benchmark"] = L.setting.map(bench); L["patience"] = np.where(L.benchmark == "EuroCropsML", 25, 15)
    L = L.merge(ck[["experiment", "epoch"]].rename(columns={"epoch": "ckpt_epoch"}), on="experiment", how="left")
    L["resume_redo_epochs"] = (L.cap - 1 - L.ckpt_epoch).clip(lower=0)
    L["add_upper"] = 2 * L.cap - (L.ckpt_epoch + 1); L["add_lower"] = L.patience
    ck["group"] = ck.side + " " + ck.objective
    chk = ck.groupby("group").apply(lambda g: pd.Series(dict(
        runs=len(g), final_ckpt_present=int(g.final_ckpt.notna().sum()), last_ckpt_at_or_after_cap_epoch=int((g.epoch >= g.cap - 1).sum()),
        last_ckpt_before_cap=int((g.epoch < g.cap - 1).sum()), redo_epochs_median=float((g.cap - 1 - g.epoch).clip(lower=0).median()),
        redo_epochs_max=int((g.cap - 1 - g.epoch).clip(lower=0).max()), wait_count_zero=int((g.es_wait_count == 0).sum()),
        optimizer_state=int((g.n_optimizer_states.fillna(0) > 0).sum()), scheduler_state=int((g.n_schedulers.fillna(0) > 0).sum()),
        scheduler_is_plateau=int(g.scheduler_is_plateau.fillna(False).sum()), early_stopping_state=int(g.es_patience.notna().sum()),
        config_says_plateau=int((g.config_lr_scheduler.astype(str).str.lower() == "plateau").sum()) if "config_lr_scheduler" in g else 0,
        dirpath_is_original_folder=int(g.dirpath_is_this_folder.fillna(False).sum()),
        teacher_keys_in_state_dict=int((g.n_teacher_keys.fillna(0) > 0).sum()) if "n_teacher_keys" in g else 0))).reset_index()
    tw = tm[tm.experiment.isin(L[L.side == "twin"].experiment)].copy()
    g_rows = []
    for exp, x in tw.groupby("experiment"):
        s = (x.end - x.start).sum() / x.n_epochs.sum(); c = np.average(x.others_at_midpoint, weights=x.n_epochs)
        g_rows.append(dict(side="twin", experiment=exp, sec_per_epoch=s, others=c, g_sec=s / (c + 1)))
    for r in L[L.side == "fsq"].itertuples():
        m = dt[dt.run_dir == r.experiment]
        if len(m):
            m = m.iloc[0]; s = float(m.hours_per_epoch) * 3600; c = float(m.concurrent_runs_at_midpoint)
            g_rows.append(dict(side="fsq", experiment=r.experiment, sec_per_epoch=s, others=c, g_sec=s / (c + 1)))
    G = pd.DataFrame(g_rows).merge(L[["experiment", "set", "setting", "objective", "benchmark"]].drop_duplicates("experiment"), on="experiment")

    def gset(grp):
        thr = 10 if grp.side.iloc[0] == "twin" else 7; sel = grp[grp.others >= thr]
        sel = sel if len(sel) else grp
        return pd.Series(dict(runs_timed=len(grp), runs_used=len(sel), sec_per_epoch=sel.sec_per_epoch.mean(), others=sel.others.mean(), g_sec=sel.g_sec.mean()))
    GS = G.groupby(["side", "benchmark", "objective"]).apply(gset).reset_index()
    L = L.merge(GS[["side", "benchmark", "objective", "g_sec"]], on=["side", "benchmark", "objective"], how="left")
    for k in ("upper", "lower"):
        L[f"work_h_{k}"] = L[f"add_{k}"] * L.g_sec / 3600
    opts = {"(i) twin_teacher + FSQ-trained capped runs": lambda x: x.objective.isin(["twin_teacher", "FSQ-mp"]),
            "(ii) (i) + twin_bands": lambda x: x.objective.isin(["twin_teacher", "twin_bands", "FSQ-mp"]),
            "(iii-a) CropGlobe only, twin_teacher + FSQ-trained": lambda x: (x.benchmark == "CropGlobe") & x.objective.isin(["twin_teacher", "FSQ-mp"]),
            "(iii-b) CropGlobe only, twin_teacher + twin_bands + FSQ-trained": lambda x: (x.benchmark == "CropGlobe")}
    est = []
    for name, sel in opts.items():
        x = L[sel(L)]
        e = dict(option=name, runs=len(x), runs_EuroCropsML=int((x.benchmark == "EuroCropsML").sum()), gpu_hours_upper=x.work_h_upper.sum(), gpu_hours_lower=x.work_h_lower.sum())
        for lab, f in (("free", 1.0), ("shared", 0.5)):         # shared: the GPU shared with the other jobs of the time
            e[f"days_upper_{lab}"] = e["gpu_hours_upper"] / 24 / f; e[f"days_lower_{lab}"] = e["gpu_hours_lower"] / 24 / f
        est.append(e)
    return dict(timing_groups=GS, estimate=pd.DataFrame(est), work=L, check_summary=chk)


def main():
    out = build()
    paths.OUTPUT.mkdir(parents=True, exist_ok=True)
    for k, df in out.items():
        df.to_csv(paths.OUTPUT / f"capped_runs_{k}.csv", index=False)
    print(out["estimate"].round(2).to_string(index=False))


def compare():
    from fsq_matched_controls.analysis.compare_tables import report
    ok = report([(paths.OUTPUT / f"capped_runs_{k}.csv", paths.SUPPLEMENT_S8 / f"capped_runs_{k}.csv")
                 for k in ("timing_groups", "estimate", "work", "check_summary")])
    print("PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    import sys
    main()
    if "--compare" in sys.argv:
        sys.exit(0 if compare() else 1)
