"""End-to-end check: from a shipped tokenizer checkpoint to the letter's per-run probe score.

    python -m fsq_matched_controls.analysis.check_end_to_end [--run <run_dir>] [--eurocropsml]

For one letter run (default: masked prediction, strict Queensland, seed 11; with --eurocropsml also masked prediction on EuroCropsML,
seed 11) it
1. builds the encoder and FSQ from checkpoints/<run_dir>.pt and the run's own normalisation statistics;
2. tokenizes the probe pool and the evaluation region with the package's data modules (CropGlobe: the fixed eight-country pool, France
   and Australia held out; EuroCropsML: Latvia + Portugal, Estonia held out);
3. runs the letter's probe: the month codes flattened over months, StandardScaler + LogisticRegression (lbfgs, C 1, max_iter 300,
   random_state 1904), labels harmonized between pool and region;
4. compares accuracy and macro-F1 with that run's row in results/runs/n_probe_*.csv (repr_kind "tokens"), and the token arrays with
   SHA-256 fingerprints of the letter's own arrays (TOKEN_SHA256).
"identical": every score equals the stored one. Otherwise the check passes within PLATFORM_TOLERANCE (see there) and prints |d| per
metric. With --reference <npz> (the letter's encoder outputs for this run: Xtr_tokens, Xq_<region>) it also counts the token cells
that differ. The datasets are read from FSQ_DATA_ROOT (see the README). Exit status 0 if every run is identical or within tolerance.
"""
from __future__ import annotations

import os

for _v in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")   # the letter's probes ran single-threaded

import argparse
import glob
import hashlib
import json
import sys
import time

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler

from fsq_matched_controls import paths
from fsq_matched_controls.analysis.tolerances import GENERAL
from fsq_matched_controls.data.cropglobe import CropGlobeDataModule
from fsq_matched_controls.data.eurocropml import EuroCropsMLDataModule
from fsq_matched_controls.models.encoder import TemporalCNNEncoder
from fsq_matched_controls.models.fsq import FSQ
from fsq_matched_controls.probe.linear_probe import harmonize_label_sets

SEED = 1904
PROBE_HOLDOUT = ["FRA", "AUS"]
PROBE_POOL = ["ARG", "BEL", "CHN", "GBR", "NLD", "USA", "USA_17", "USA_20"]
DEFAULT_RUN = "cropglobe_predictive_ssl_ov0_625_holdout_AUS_armN_seed11_vm_holdout_AUS"
EC_RUN = "eurocropml_predictive_ssl_625_heldout_estonia_gradclip_armN_seed11_vm_holdout_Estonia"
# Exact equality holds on the platform the letter's probes ran on (Windows, CPU, PyTorch 2.7.0). On another platform a few FSQ codes can
# flip at the quantizer's rounding boundaries, and the lbfgs fit can differ slightly with the BLAS library. Observed on Linux (CPU,
# same pins), EuroCropsML masked prediction, seed 11: 3 of about 9.9 million token cells differ, accuracy |d| 8.5e-5, macro-F1 |d| 5.8e-4;
# CropGlobe (strict Queensland, seed 11) gave identical scores there, although its token fingerprint differed too.
# A run passes if every |d| is at most this tolerance (analysis.tolerances.GENERAL).
PLATFORM_TOLERANCE = GENERAL
# SHA-256 of the letter's token arrays (float32, months flattened) for the default runs: (probe pool, {evaluation region: array}).
TOKEN_SHA256 = {
    DEFAULT_RUN: ("5afe4fdca1990b2a707daf7b901c409da529cc5cb2fc2bf68a13d6d7ffab5714",
                  {"AUS": "c795377cf47b8ae1de1c1ba1cce5294acc3aeeda6d0779a24f7f20280ba97004"}),
    EC_RUN: ("8f3c893bb0897d42c5eda481235bbed7f936ba59aa03b2738c5691763fed94ac",
             {"Estonia": "b18eeeacc982a2fd25c9869d0cfbbc31fb87da22750c20280b2abdce6c65a8f1"}),
}
REGIONS = {"cropglobe_transductive": ["FRA", "AUS"], "cropglobe_strict_FRA": ["FRA"], "cropglobe_strict_AUS": ["AUS"], "eurocropml_estonia": ["Estonia"]}


def load_tokenizer(run_dir):
    ck = torch.load(paths.CHECKPOINTS / f"{run_dir}.pt", map_location="cpu", weights_only=True)
    c = ck["config"]
    enc = TemporalCNNEncoder(n_bands=c["n_bands"], channels=c["encoder_channels"], kernel_size=c["kernel_size"], n_dims=c["n_dims"],
                             n_levels=c["n_levels"], dropout=c["dropout"], norm_type=c["norm_type"])
    enc.load_state_dict(ck["encoder_state_dict"]); enc.eval()
    fsq = FSQ(n_dims=c["n_dims"], n_levels=c["n_levels"]); fsq.load_state_dict(ck["fsq_state_dict"]); fsq.eval()
    return enc, fsq, ck


@torch.no_grad()
def tokens_cropglobe(enc, fsq, dl):
    zq_l, lab, ctry = [], [], []
    for windows, lbl, c in dl:
        B, W, obs, bands = windows.shape
        z_q, _ = fsq(enc(windows.view(B * W, obs, bands)))
        zq_l.append(z_q.numpy().reshape(B, W, -1)); lab.append(lbl.numpy()); ctry.extend(c)
    zq = np.concatenate(zq_l)
    return zq.reshape(len(zq), -1), np.concatenate(lab), np.array(ctry)


@torch.no_grad()
def tokens_eurocropsml(enc, fsq, dl):
    zq_l, lab = [], []
    for windows, lbl, _regions, _valid, obs_mask in dl:
        B, W, obs, bands = windows.shape
        z_q, _ = fsq(enc(windows.view(B * W, obs, bands), obs_mask=obs_mask.reshape(B * W, obs)))
        zq_l.append(z_q.numpy().reshape(B, W, -1)); lab.append(lbl.numpy())
    zq = np.concatenate(zq_l)
    return zq.reshape(len(zq), -1), np.concatenate(lab)


def probe(Xtr, ytr, Xte, yte, other_class=None):
    yt, yh, _ = harmonize_label_sets(ytr, yte) if other_class is None else harmonize_label_sets(ytr, yte, other_class=other_class)
    sc = StandardScaler()
    clf = LogisticRegression(max_iter=300, C=1.0, solver="lbfgs", random_state=SEED)
    clf.fit(sc.fit_transform(Xtr), yt)
    pred = clf.predict(sc.transform(Xte))
    return float(accuracy_score(yh, pred)), float(f1_score(yh, pred, average="macro"))


def _sha(a):
    return hashlib.sha256(np.ascontiguousarray(a, dtype=np.float32).tobytes()).hexdigest()


def stored_rows(run_dir):
    df = pd.concat([pd.read_csv(p) for p in sorted(glob.glob(str(paths.RUN_RESULTS / "n_probe_*.csv")))], ignore_index=True)
    return df[(df.run_dir == run_dir) & (df.repr_kind == "tokens")]


def check(run_dir, reference=None):
    t0 = time.time()
    stored = stored_rows(run_dir)
    if stored.empty:
        raise SystemExit(f"no stored probe row for {run_dir} in {paths.RUN_RESULTS}")
    setting = stored.setting.iloc[0]
    enc, fsq, ck = load_tokenizer(run_dir)
    work = paths.OUTPUT / "check_end_to_end"; work.mkdir(parents=True, exist_ok=True)
    norm = work / f"{run_dir}_norm_stats.json"
    norm.write_text(json.dumps(ck["norm_stats"]), encoding="utf-8")
    results = {}
    if setting == "eurocropml_estonia":
        dm = EuroCropsMLDataModule(data_root=str(paths.DATA_ROOT), split_name="latvia_portugal_vs_estonia", finetune_variant="all", batch_size=64,
                                   norm_stats_path=str(norm), w_max=10, w_min=2)
        dm.setup()
        Xtr, ytr = tokens_eurocropsml(enc, fsq, dm.parcel_dataloader(split="train", batch_size=64))
        Xho, yho = tokens_eurocropsml(enc, fsq, dm.parcel_dataloader(split="held_out", batch_size=64))
        print(f"  tokenized: probe pool {len(ytr)} parcels, Estonia {len(yho)} parcels ({time.time() - t0:.0f} s)")
        tokens = {"Estonia": Xho}
        results["Estonia"] = probe(Xtr, ytr, Xho, yho, other_class=-1)
    else:
        dm = CropGlobeDataModule(raw_dir=paths.DATA_ROOT / "cropglobe" / "raw", hold_out_countries=list(PROBE_HOLDOUT), norm_stats_path=str(norm),
                                 source="Median", train_countries=None, batch_size=64, days_per_token=30, overlap_days=0, dn_scale=10_000.0)
        dm.setup()
        Xtr, ytr, _ = tokens_cropglobe(enc, fsq, dm.parcel_dataloader(split="all_train", batch_size=64))
        Xho, yho, cho = tokens_cropglobe(enc, fsq, dm.parcel_dataloader(split="held_out", batch_size=64))
        print(f"  tokenized: probe pool {len(ytr)} pixels ({', '.join(PROBE_POOL)}), held out {len(yho)} ({time.time() - t0:.0f} s)")
        tokens = {}
        for reg in REGIONS[setting]:
            m = cho == reg
            tokens[reg] = Xho[m]
            results[reg] = probe(Xtr, ytr, Xho[m], yho[m])
    if run_dir in TOKEN_SHA256:
        pool_sha, reg_sha = TOKEN_SHA256[run_dir]
        same_tok = _sha(Xtr) == pool_sha and all(_sha(tokens[r]) == h for r, h in reg_sha.items())
        print(f"  token arrays identical to the letter's (SHA-256): {same_tok}")
    if reference is not None:
        ref = np.load(reference)
        n_diff = int((ref["Xtr_tokens"] != Xtr).sum()) + sum(int((ref[f"Xq_{r}"] != tokens[r]).sum()) for r in tokens)
        n_all = ref["Xtr_tokens"].size + sum(tokens[r].size for r in tokens)
        print(f"  token cells differing from {reference}: {n_diff} of {n_all}")
    ok = True
    for reg, (acc, f1) in results.items():
        s = stored[stored.region == reg].iloc[0]
        da, df = abs(acc - float(s.accuracy)), abs(f1 - float(s.macro_f1))
        if da == 0 and df == 0:
            verdict = "identical"
        elif max(da, df) <= PLATFORM_TOLERANCE:
            verdict = f"within platform tolerance {PLATFORM_TOLERANCE:g} (|d| accuracy {da:.2e}, macro-F1 {df:.2e})"
        else:
            verdict = f"DIFFERENT, beyond platform tolerance (|d| accuracy {da:.2e}, macro-F1 {df:.2e})"
            ok = False
        print(f"  {reg}: accuracy {acc!r} (stored {float(s.accuracy)!r})  macro-F1 {f1!r} (stored {float(s.macro_f1)!r})  {verdict}")
    print(f"  {time.time() - t0:.0f} s")
    return ok


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run", action="append", help=f"run_dir of a letter run (repeatable; default {DEFAULT_RUN})")
    ap.add_argument("--eurocropsml", action="store_true", help=f"also check {EC_RUN}")
    ap.add_argument("--reference", help="with a single run: npz of the letter's encoder outputs for it (Xtr_tokens, Xq_<region>); "
                                        "counts the token cells that differ")
    a = ap.parse_args(argv)
    runs = (a.run or [DEFAULT_RUN]) + ([EC_RUN] if a.eurocropsml else [])
    print(f"data root: {paths.DATA_ROOT}")
    fails = []
    for r in runs:
        print(f"\n{r}")
        if not check(r, reference=a.reference if len(runs) == 1 else None):
            fails.append(r)
    print(f"\n{'PASS' if not fails else 'FAIL'}: {len(runs) - len(fails)} of {len(runs)} runs reproduce the stored probe scores "
          f"(identical, or within the platform tolerance {PLATFORM_TOLERANCE:g})")
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main())
