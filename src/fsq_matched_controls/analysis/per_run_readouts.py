"""Per-run readouts of the supplement: rounding curve and grid distance (S6), gradient boosting (S4.3), one-hot codes (S4.1).

    python -m fsq_matched_controls.analysis.per_run_readouts --mode curve|gb|onehot (--sample | --all | --run <run_dir> ...) [--compare]

From each run's checkpoint (encoder outputs extracted as in probe_runs):
    curve   S6.1: the continuous latent re-quantized at 3, 5, 9 and 17 levels (per-column minimum and maximum of the probe pool,
            uniform bins, snap to the nearest), read by the letter's probe; S6.2: mean and median distance of the continuous codes to
            the nearest clipped, rounded grid point (5 levels, half-range 2), per dimension, on the probe pool.
            Runs: transductive CropGlobe and EuroCropsML, both objectives.
    gb      S4.3: HistGradientBoostingClassifier (max_iter 200, random_state 1904, early_stopping False) on the token values;
            transductive CropGlobe masked prediction.
    onehot  S4.1: the tokens read as one-hot codes over 625 per month (sparse, no standardization, logistic regression lbfgs,
            C 1, max_iter 300, random state 1904); masked prediction.
--compare: every score against the stored per-run rows (results/runs/n2/n2_<mode>_*.csv), identical or within the platform
tolerance; the grid distances against the same files.
"""
from __future__ import annotations

import os

for _v in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import glob
import json
import sys
import time
import warnings

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler

from fsq_matched_controls import paths
from fsq_matched_controls.analysis.check_end_to_end import PROBE_HOLDOUT, REGIONS, load_tokenizer, stored_rows
from fsq_matched_controls.analysis.tolerances import GENERAL, UNCONVERGED

TOLERANCE = {"curve": GENERAL, "onehot": UNCONVERGED, "gb": None}   # gb: no numeric bound (see analysis.tolerances)
from fsq_matched_controls.analysis.probe_runs import _extract, all_runs
from fsq_matched_controls.data.cropglobe import CropGlobeDataModule
from fsq_matched_controls.data.eurocropml import EuroCropsMLDataModule
from fsq_matched_controls.probe.linear_probe import harmonize_label_sets

SEED = 1904
WGT = np.array([125, 25, 5, 1])
SETTINGS = {"curve": ("cropglobe_transductive", "eurocropml_estonia"), "gb": ("cropglobe_transductive",),
            "onehot": ("cropglobe_transductive", "cropglobe_strict_FRA", "cropglobe_strict_AUS", "eurocropml_estonia")}
OBJECTIVES = {"curve": ("mp", "recon"), "gb": ("mp",), "onehot": ("mp",)}


def extract(run_dir):
    setting = stored_rows(run_dir).setting.iloc[0]
    enc, fsq, ck = load_tokenizer(run_dir)
    work = paths.OUTPUT / "per_run_readouts"; work.mkdir(parents=True, exist_ok=True)
    norm = work / f"{run_dir}_norm_stats.json"; norm.write_text(json.dumps(ck["norm_stats"]), encoding="utf-8")
    if setting == "eurocropml_estonia":
        dm = EuroCropsMLDataModule(data_root=str(paths.DATA_ROOT), split_name="latvia_portugal_vs_estonia", finetune_variant="all",
                                   batch_size=64, norm_stats_path=str(norm), w_max=10, w_min=2)
        dm.setup()
        q_tr, c_tr, ytr, _ = _extract(enc, fsq, dm.parcel_dataloader(split="train", batch_size=64), ec=True)
        q_ho, c_ho, yho, _ = _extract(enc, fsq, dm.parcel_dataloader(split="held_out", batch_size=64), ec=True)
        regs = {"Estonia": np.ones(len(yho), bool)}
    else:
        dm = CropGlobeDataModule(raw_dir=paths.DATA_ROOT / "cropglobe" / "raw", hold_out_countries=list(PROBE_HOLDOUT), norm_stats_path=str(norm),
                                 source="Median", train_countries=None, batch_size=64, days_per_token=30, overlap_days=0, dn_scale=10_000.0)
        dm.setup()
        q_tr, c_tr, ytr, _ = _extract(enc, fsq, dm.parcel_dataloader(split="all_train", batch_size=64), ec=False)
        q_ho, c_ho, yho, cho = _extract(enc, fsq, dm.parcel_dataloader(split="held_out", batch_size=64), ec=False)
        regs = {r: cho == r for r in REGIONS[setting]}
    return setting, dict(q_tr=q_tr, c_tr=c_tr, ytr=ytr), {r: dict(q=q_ho[m], c=c_ho[m], y=yho[m]) for r, m in regs.items()}


def _labels(setting, ytr, yte):
    return (harmonize_label_sets(ytr, yte, other_class=-1) if setting == "eurocropml_estonia" else harmonize_label_sets(ytr, yte))[:2]


def round_to_levels(Xtr, Xhos, L):
    lo, hi = Xtr.min(axis=0, keepdims=True), Xtr.max(axis=0, keepdims=True)
    step = np.clip(hi - lo, 1e-8, None) / (L - 1)
    snap = lambda X: lo + np.clip(np.round((X - lo) / step), 0, L - 1) * step
    return snap(Xtr), [snap(h) for h in Xhos]


def onehot(ids, n_codes=625):
    n, m = ids.shape
    return sparse.csr_matrix((np.ones(n * m, np.float32), (np.repeat(np.arange(n), m), (np.tile(np.arange(m), n) * n_codes + ids.reshape(-1)))),
                             shape=(n, m * n_codes))


def decode(zq4):
    return ((np.rint(zq4).astype(np.int64) + 2) * WGT).sum(-1)


def readouts(run_dir, mode, reference=None):
    r = all_runs().set_index("run_dir").loc[run_dir]
    setting, tr, tests = extract(run_dir)
    if reference is not None:   # the letter's token arrays of this run (keys tr and one per region): count the flipped cells
        ref = np.load(reference)
        n = int((ref["tr"].astype(np.float32) != tr["q_tr"]).sum()) + sum(int((ref[k.lower() if k.lower() in ref else k].astype(np.float32) != t["q"]).sum()) for k, t in tests.items())
        tot = ref["tr"].size + sum(t["q"].size for t in tests.values())
        print(f"  token cells differing from the reference: {n} of {tot}", flush=True)
    W = tr["q_tr"].shape[1] // 4
    rows = []
    if mode == "curve":
        Z = tr["c_tr"].reshape(len(tr["c_tr"]), W, 4); dist = np.abs(Z - np.round(np.clip(Z, -2, 2)))
        for k in range(4):
            rows.append(dict(objective=r.objective, setting=setting, tag=str(r.tag), region="-", n_levels=0, kind="grid_distance", dim=k,
                             mean_dist=float(dist[..., k].mean()), median_dist=float(np.median(dist[..., k]))))
    for reg, t in tests.items():
        ytr, yte = _labels(setting, tr["ytr"], t["y"])
        if mode == "curve":
            for L in (3, 5, 9, 17):
                Xr, (Hr,) = round_to_levels(tr["c_tr"], [t["c"]], L)
                sc = StandardScaler(); clf = LogisticRegression(max_iter=300, C=1.0, solver="lbfgs", random_state=SEED).fit(sc.fit_transform(Xr), ytr)
                p = clf.predict(sc.transform(Hr))
                rows.append(dict(objective=r.objective, setting=setting, tag=str(r.tag), region=reg, n_levels=L, kind="curve",
                                 acc=float(accuracy_score(yte, p)), f1=float(f1_score(yte, p, average="macro")), n_iter=int(np.max(clf.n_iter_))))
        elif mode == "gb":
            clf = HistGradientBoostingClassifier(max_iter=200, random_state=SEED, early_stopping=False).fit(tr["q_tr"], ytr)
            p = clf.predict(t["q"])
            rows.append(dict(objective=r.objective, setting=setting, tag=str(r.tag), region=reg, n_levels=0, kind="gb",
                             acc=float(accuracy_score(yte, p)), f1=float(f1_score(yte, p, average="macro"))))
        else:
            Xtr = onehot(decode(tr["q_tr"].reshape(len(tr["q_tr"]), W, 4))); Xte = onehot(decode(t["q"].reshape(len(t["q"]), W, 4)))
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                clf = LogisticRegression(max_iter=300, C=1.0, solver="lbfgs", random_state=SEED).fit(Xtr, ytr)
            p = clf.predict(Xte)
            rows.append(dict(objective=r.objective, setting=setting, tag=str(r.tag), region=reg, n_levels=0, kind="onehot",
                             acc=float(accuracy_score(yte, p)), f1=float(f1_score(yte, p, average="macro")), n_iter=int(np.max(clf.n_iter_)),
                             warned=bool(any("converge" in str(x.message).lower() for x in w))))
    return rows


def stored(mode) -> pd.DataFrame:
    fs = sorted(glob.glob(str(paths.RUN_RESULTS / "n2" / f"n2_{mode}_*.csv")))
    d = pd.concat([pd.read_csv(f, dtype={"tag": str}) for f in fs], ignore_index=True)
    return d


def candidates(mode) -> list[str]:
    """Runs of the mode's settings and objectives that have stored rows to compare with."""
    st = stored(mode)
    a = all_runs(); a["tag"] = a.tag.astype(str)
    a = a[a.setting.isin(SETTINGS[mode]) & a.objective.isin(OBJECTIVES[mode])]
    keys = set(zip(st.objective, st.setting, st.tag))
    return [r.run_dir for r in a.itertuples() if (r.objective, r.setting, r.tag) in keys]


def compare(df, mode) -> bool:
    st = stored(mode); ok = True; tol = TOLERANCE[mode]
    k = ["objective", "setting", "tag", "region", "n_levels", "kind"] + (["dim"] if "dim" in df else [])
    df = df.copy(); st = st.copy()
    for d in (df, st):
        d["tag"] = d["tag"].astype(str); d["n_levels"] = d["n_levels"].astype(int)
        if "dim" in d:
            d["dim"] = d["dim"].fillna(-1).astype(int)
    m = df.merge(st, on=k, suffixes=("", "_ref"))
    for c in [c for c in ("acc", "f1", "mean_dist", "median_dist") if c in m]:
        x = m[m[c].notna()]
        if len(x):
            d = float((x[c] - x[c + "_ref"]).abs().max())
            if tol is None:
                diff = x[(x[c] - x[c + "_ref"]).abs() > 0]
                print(f"  {c}: {len(x)} values, {'identical' if d == 0 else f'{len(diff)} differ, max |d| {d:.2e} (no numeric bound; differing cells below)'}")
                for r in diff.itertuples():
                    print(f"    {r.objective} {r.setting} {r.tag} {r.region}: {getattr(r, c):.6f} vs stored {getattr(r, c + '_ref'):.6f}")
                continue
            good = d <= tol
            ok &= good
            print(f"  {c}: {len(x)} values, {'identical' if d == 0 else f'max |d| {d:.2e}'} (tolerance {tol:g}) -> {'PASS' if good else 'FAIL'}")
    print(f"  {len(m)} of {len(df)} rows had a stored counterpart")
    return ok and len(m) == len(df)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--mode", required=True, choices=["curve", "gb", "onehot"])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--sample", action="store_true", help="two runs with stored rows")
    g.add_argument("--all", action="store_true")
    g.add_argument("--run", action="append")
    ap.add_argument("--compare", action="store_true")
    ap.add_argument("--reference", help="with a single --run: npz of the letter's token arrays of that run (tr, and one key per region): count flipped token cells")
    a = ap.parse_args(argv)
    runs = a.run or (candidates(a.mode)[:2] if a.sample else candidates(a.mode))
    rows, t0 = [], time.time()
    out = paths.OUTPUT / f"per_run_{a.mode}.csv"; out.parent.mkdir(parents=True, exist_ok=True)
    for i, rd in enumerate(runs, 1):
        rows += readouts(rd, a.mode, reference=a.reference if len(runs) == 1 else None)
        pd.DataFrame(rows).to_csv(out, index=False)
        print(f"[{i}/{len(runs)}] {rd} ({time.time() - t0:.0f} s)", flush=True)
    df = pd.DataFrame(rows)
    if a.compare:
        ok = compare(df, a.mode)
        print(f"{'PASS' if ok else 'FAIL'}: {a.mode}, {len(runs)} runs -> {out}")
        return 0 if ok else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
