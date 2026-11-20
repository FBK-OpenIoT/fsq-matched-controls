"""Supplement S8: training wall-clock of the letter's 80 runs, from the MLflow records of the GPU node.

    python -m fsq_matched_controls.analysis.training_times [--compare]     -> outputs/training_times_per_run.csv, outputs/training_times_summary.csv

Input: results/supplement_S8_training/training_runs_mlflow_dump.csv (one line per MLflow run of the node's stores: start, end,
status, last logged epoch and step) and, for the checkpoint columns (Lightning version, epoch and step of the best checkpoint,
read from the full training checkpoints, which are not part of the repository), training_times_per_run.csv itself.
For each run directory the MLflow run with the most steps is taken (a run that failed in a post-training step still logs its
epochs); wall-clock = time of the last logged epoch metric minus the run's start (it excludes the final test and save steps);
epochs trained = last logged epoch + 1; concurrency = MLflow runs of the node running at the run's midpoint (the GPU was shared).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from fsq_matched_controls import paths

CKPT_COLS = ["objective", "setting", "tag", "run_dir", "lightning", "ckpt_epoch", "ckpt_step"]


def per_run() -> pd.DataFrame:
    d = pd.read_csv(paths.SUPPLEMENT_S8 / "training_runs_mlflow_dump.csv", float_precision="round_trip")
    d["t_end"] = d["last_epoch_ms"]; d["t_start"] = d["start_ms"]
    ck = pd.read_csv(paths.SUPPLEMENT_S8 / "training_times_per_run.csv", dtype={"tag": str}, float_precision="round_trip")[CKPT_COLS]
    rows = []
    for c in ck.itertuples(index=False):
        m = d[d["name"] == c.run_dir]
        rec = dict(c._asdict(), n_mlflow_runs=len(m))
        if len(m):
            r = m.sort_values(["last_step", "start_ms"]).iloc[-1]
            h = (r["t_end"] - r["t_start"]) / 3.6e6; mid = (r["t_start"] + r["t_end"]) / 2
            conc = int(((d["t_start"] <= mid) & (d["t_end"].fillna(d["end_ms"]) >= mid) & (d["run_uuid"] != r["run_uuid"])).sum())
            rec.update(status=r["status"], epochs_trained=int(r["last_epoch"]) + 1, last_step=int(r["last_step"]), hours=h, hours_per_epoch=h / (r["last_epoch"] + 1),
                       concurrent_runs_at_midpoint=conc, start_utc=pd.to_datetime(r["t_start"], unit="ms").strftime("%Y-%m-%d %H:%M"), mlflow_db=r["db"],
                       step_consistent=bool(r["last_step"] >= c.ckpt_step))
        else:
            rec.update(status="no MLflow record on the node", hours=np.nan)
        rows.append(rec)
    return pd.DataFrame(rows)


def summary(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["benchmark"] = np.where(df.setting.str.startswith("cropglobe"), "CropGlobe", "EuroCropsML")
    agg = lambda g: pd.Series(dict(n_runs=len(g), n_timed=int(g.hours.notna().sum()), mean_h=g.hours.mean(), min_h=g.hours.min(), max_h=g.hours.max(),
                                   median_h=g.hours.median(), mean_epochs=g.epochs_trained.mean(), min_epochs=g.epochs_trained.min(),
                                   max_epochs=g.epochs_trained.max(), mean_concurrent=g.concurrent_runs_at_midpoint.mean()))
    s1 = df.groupby(["objective", "setting"]).apply(agg).reset_index(); s2 = df.groupby(["objective", "benchmark"]).apply(agg).reset_index()
    s3 = df.groupby(["objective"]).apply(agg).reset_index()
    return pd.concat([s1.assign(level="setting"), s2.assign(level="benchmark", setting=s2.benchmark), s3.assign(level="objective", setting="all")])


def main():
    paths.OUTPUT.mkdir(parents=True, exist_ok=True)
    df = per_run(); df.to_csv(paths.OUTPUT / "training_times_per_run.csv", index=False)
    s = summary(df); s.to_csv(paths.OUTPUT / "training_times_summary.csv", index=False)
    print(s[s.level != "setting"].round(2).to_string(index=False))
    print(f"total hours {df.hours.sum():.1f}, runs timed {int(df.hours.notna().sum())} of {len(df)}")


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
        sys.exit(0 if compare([("training_times_per_run", paths.SUPPLEMENT_S8), ("training_times_summary", paths.SUPPLEMENT_S8)]) else 1)
