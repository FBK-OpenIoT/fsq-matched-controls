"""The letter's probe on every training run: tokens and continuous latent from the shipped checkpoints, and the raw band means.

    python -m fsq_matched_controls.analysis.probe_runs --sample          # one run per objective and setting (8 runs), compared
    python -m fsq_matched_controls.analysis.probe_runs --all             # all 80 runs, compared
    python -m fsq_matched_controls.analysis.probe_runs --run <run_dir>   # one or more runs (repeatable)
    python -m fsq_matched_controls.analysis.probe_runs --raw             # raw band means of every row, compared with Figure 1's values

Per run: the encoder and FSQ from checkpoints/<run_dir>.pt and the run's normalisation; the month windows of the probe pool and the
evaluation region(s) from the package's data modules (a transductive CropGlobe run is evaluated on France and Australia, a strict
run on its own region, a EuroCropsML run on Estonia); the probe reads the four FSQ values per month ("tokens") or the encoder's
four continuous values before the quantizer ("continuous"), flattened over months. Probe: StandardScaler + logistic regression
(lbfgs, C 1, at most 300 iterations, random state 1904), labels harmonized between pool and region.

Every score is compared with the run's row in results/runs/n_probe_*.csv: "identical", or within analysis.tolerances (GENERAL;
ESTONIA_MACRO_F1 for macro-F1 in Estonia). The raw-band probe stops at the iteration limit, so its scores vary across environments by up to 0.0015 in
accuracy and 0.0022 in macro-F1 (supplement, probe convergence); they are compared with Figure 1's four-decimal values within
RAW_TOLERANCE. Writes outputs/probe_runs.csv.
"""
from __future__ import annotations

import os

for _v in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import sys
import time

import numpy as np
import pandas as pd
import torch

from fsq_matched_controls import paths
from fsq_matched_controls.analysis.check_end_to_end import PROBE_HOLDOUT, REGIONS, load_tokenizer, probe, stored_rows
from fsq_matched_controls.analysis.tolerances import ESTONIA_MACRO_F1, GENERAL as PLATFORM_TOLERANCE
from fsq_matched_controls.data.cropglobe import CropGlobeDataModule
from fsq_matched_controls.data.eurocropml import EuroCropsMLDataModule

RAW_TOLERANCE = 0.0025
OUT = paths.OUTPUT / "probe_runs.csv"


def token_cache_path(run_dir: str):
    """FSQ_FEATURES/runs/<run_dir>.npz: tok_train, y_train, tok_<region>, y_<region> (written by every probe_run)."""
    return paths.FEATURES / "runs" / f"{run_dir}.npz"


def all_runs() -> pd.DataFrame:
    import glob
    df = pd.concat([pd.read_csv(p) for p in sorted(glob.glob(str(paths.RUN_RESULTS / "n_probe_*.csv")))], ignore_index=True)
    return df.drop_duplicates("run_dir")[["objective", "setting", "tag", "run_dir"]].reset_index(drop=True)


def sample_runs() -> list[str]:
    """One run per objective and setting: the first stored one."""
    return all_runs().groupby(["objective", "setting"], sort=True).first().run_dir.tolist()


@torch.no_grad()
def _extract(enc, fsq, dl, ec: bool):
    z_l, zq_l, lab, ctry = [], [], [], []
    for batch in dl:
        windows, lbl = batch[0], batch[1]
        B, W, obs, bands = windows.shape
        x = windows.view(B * W, obs, bands)
        z = enc(x, obs_mask=batch[4].reshape(B * W, obs)) if ec else enc(x)
        z_q, _ = fsq(z)
        z_l.append(z.numpy().reshape(B, -1)); zq_l.append(z_q.numpy().reshape(B, -1)); lab.append(lbl.numpy())
        if not ec:
            ctry.extend(batch[2])
    return np.concatenate(zq_l), np.concatenate(z_l), np.concatenate(lab), np.array(ctry)


def probe_run(run_dir: str) -> list[dict]:
    stored = stored_rows(run_dir)
    setting = stored.setting.iloc[0]
    enc, fsq, ck = load_tokenizer(run_dir)
    work = paths.OUTPUT / "probe_runs"; work.mkdir(parents=True, exist_ok=True)
    norm = work / f"{run_dir}_norm_stats.json"; norm.write_text(json.dumps(ck["norm_stats"]), encoding="utf-8")
    out = []
    if setting == "eurocropml_estonia":
        dm = EuroCropsMLDataModule(data_root=str(paths.DATA_ROOT), split_name="latvia_portugal_vs_estonia", finetune_variant="all",
                                   batch_size=64, norm_stats_path=str(norm), w_max=10, w_min=2)
        dm.setup()
        q_tr, c_tr, ytr, _ = _extract(enc, fsq, dm.parcel_dataloader(split="train", batch_size=64), ec=True)
        q_ho, c_ho, yho, _ = _extract(enc, fsq, dm.parcel_dataloader(split="held_out", batch_size=64), ec=True)
        tests = {"Estonia": (yho, q_ho, c_ho)}
        other = -1
    else:
        dm = CropGlobeDataModule(raw_dir=paths.DATA_ROOT / "cropglobe" / "raw", hold_out_countries=list(PROBE_HOLDOUT),
                                 norm_stats_path=str(norm), source="Median", train_countries=None, batch_size=64, days_per_token=30,
                                 overlap_days=0, dn_scale=10_000.0)
        dm.setup()
        q_tr, c_tr, ytr, _ = _extract(enc, fsq, dm.parcel_dataloader(split="all_train", batch_size=64), ec=False)
        q_ho, c_ho, yho, cho = _extract(enc, fsq, dm.parcel_dataloader(split="held_out", batch_size=64), ec=False)
        tests = {r: (yho[cho == r], q_ho[cho == r], c_ho[cho == r]) for r in REGIONS[setting]}
        other = None
    # token cache for the supplement analyses (FSQ values are multiples of 0.5, exact in float16)
    cache = token_cache_path(run_dir); cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, tok_train=q_tr.astype(np.float16), y_train=ytr,
             **{f"tok_{r}": q.astype(np.float16) for r, (y, q, c) in tests.items()}, **{f"y_{r}": y for r, (y, q, c) in tests.items()})
    for reg, (y, q, c) in tests.items():
        for kind, Xtr, Xte in (("tokens", q_tr, q), ("continuous", c_tr, c)):
            acc, f1 = probe(Xtr, ytr, Xte, y, other_class=other)
            out.append(dict(run_dir=run_dir, setting=setting, region=reg, repr_kind=kind, accuracy=acc, macro_f1=f1))
    return out


def compare(df: pd.DataFrame, tol: float = PLATFORM_TOLERANCE) -> bool:
    ok = True
    for r in df.itertuples():
        s = stored_rows_all(r.run_dir, r.repr_kind)
        s = s[s.region == r.region].iloc[0]
        da, df_ = abs(r.accuracy - float(s.accuracy)), abs(r.macro_f1 - float(s.macro_f1))
        tol_f1 = ESTONIA_MACRO_F1 if r.region == "Estonia" else tol
        if da == 0 and df_ == 0:
            v = "identical"
        elif da <= tol and df_ <= tol_f1:
            v = f"within tolerance {tol:g}{f' (macro-F1 {tol_f1:g})' if tol_f1 != tol else ''} (|d| {da:.2e} / {df_:.2e})"
        else:
            v = f"DIFFERENT (|d| {da:.2e} / {df_:.2e})"; ok = False
        print(f"  {r.run_dir[:70]:70s} {r.region:8s} {r.repr_kind:10s} {r.accuracy:.6f} / {r.macro_f1:.6f}  {v}")
    return ok


def stored_rows_all(run_dir, kind):
    import glob
    df = pd.concat([pd.read_csv(p) for p in sorted(glob.glob(str(paths.RUN_RESULTS / "n_probe_*.csv")))], ignore_index=True)
    return df[(df.run_dir == run_dir) & (df.repr_kind == kind)]


def raw_bands() -> bool:
    """Raw band means of the five rows against Figure 1's (four-decimal) values."""
    from fsq_matched_controls.analysis.figure1_data_pre_ten_runs import PANELS
    from fsq_matched_controls.controls import features
    from fsq_matched_controls.controls.__main__ import probe as probe_dense
    from fsq_matched_controls.probe.linear_probe import harmonize_label_sets
    ok = True
    rows = {"France, transductive": ("cropglobe", "FRA"), "Australia, transductive": ("cropglobe", "AUS"), "Estonia": ("eurocropsml", "estonia")}
    data = {}
    for row, (ds, reg) in rows.items():
        d = data.get(ds) or data.setdefault(ds, features.cropglobe_monthly() if ds == "cropglobe" else features.eurocropsml_monthly())
        X, y = d["X"], d["y"]
        yt, yh, _ = (harmonize_label_sets(y["train"], y[reg], other_class=-1) if ds == "eurocropsml" else harmonize_label_sets(y["train"], y[reg]))
        a, f = probe_dense(X["train"].reshape(len(X["train"]), -1), yt, X[reg].reshape(len(X[reg]), -1), yh)
        ref = PANELS[row]["raw"]
        da, df_ = abs(a - ref[0]), abs(f - ref[1])
        good = max(da, df_) <= RAW_TOLERANCE
        ok &= good
        print(f"  raw bands {row:24s} {a:.6f} / {f:.6f}  (Figure 1 {ref[0]:.4f} / {ref[1]:.4f})  |d| {da:.1e} / {df_:.1e}  "
              f"{'within' if good else 'BEYOND'} raw-band tolerance {RAW_TOLERANCE:g}")
    return ok


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--sample", action="store_true")
    g.add_argument("--all", action="store_true")
    g.add_argument("--run", action="append")
    g.add_argument("--raw", action="store_true")
    a = ap.parse_args(argv)
    if a.raw:
        ok = raw_bands()
        print(f"\n{'PASS' if ok else 'FAIL'}: raw band means")
        return 0 if ok else 1
    runs = a.run or (sample_runs() if a.sample else all_runs().run_dir.tolist())
    rows, t0 = [], time.time()
    for i, r in enumerate(runs, 1):
        rows += probe_run(r)
        print(f"[{i}/{len(runs)}] {r} ({time.time() - t0:.0f} s)", flush=True)
        OUT.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(OUT, index=False)
    df = pd.DataFrame(rows)
    ok = compare(df)
    print(f"\n{'PASS' if ok else 'FAIL'}: {len(df)} probe scores of {len(runs)} runs (identical, or within {PLATFORM_TOLERANCE:g}) -> {OUT}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
