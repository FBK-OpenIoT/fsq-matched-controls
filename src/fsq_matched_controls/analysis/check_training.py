"""Training check: a short real training run of each objective through the package's entry points, data modules and paths.

    python -m fsq_matched_controls.analysis.check_training [--eurocropsml] [--train-batches 5] [--val-batches 2]

For masked prediction and reconstruction (strict Queensland, seed 11; with --eurocropsml also the EuroCropsML runs of seed 11) it runs
the letter's configuration for one epoch on a few batches (--limit-train-batches / --limit-val-batches; the configs are not edited),
then checks that optimizer steps were taken and that the best checkpoint and norm_stats.json were written. Runs go to
outputs/check_training/runs/ (git-ignored) unless FSQ_RUNS_DIR is set. The datasets are read from FSQ_DATA_ROOT (see the README).
Exit status 0 only if every run trains and saves its checkpoint.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import torch

from fsq_matched_controls import paths

RUNS = {
    "masked prediction, CropGlobe": ("fsq_matched_controls.training.train_predictive",
                                     "configs/letter/cropglobe_predictive_ssl_ov0_625_holdout_AUS_armN_seed11_vm.yaml", "predictive_best.ckpt"),
    "reconstruction, CropGlobe": ("fsq_matched_controls.training.train",
                                  "configs/letter/cropglobe_fsq_ov0_625_reconstruction_matched_holdout_AUS_armN_seed11_vm.yaml", "best.ckpt"),
}
RUNS_EC = {
    "masked prediction, EuroCropsML": ("fsq_matched_controls.training.train_predictive",
                                       "configs/letter/eurocropml_predictive_ssl_625_heldout_estonia_gradclip_armN_seed11_vm.yaml", "predictive_best.ckpt"),
    "reconstruction, EuroCropsML": ("fsq_matched_controls.training.train",
                                    "configs/letter/eurocropml_fsq_baseline_625_reconstruction_varobs_armN_seed11_vm.yaml", "best.ckpt"),
}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--eurocropsml", action="store_true", help="also run the two EuroCropsML configurations")
    ap.add_argument("--train-batches", type=int, default=5)
    ap.add_argument("--val-batches", type=int, default=2)
    a = ap.parse_args(argv)
    env = dict(os.environ)
    runs_dir = Path(env.get("FSQ_RUNS_DIR") or (paths.OUTPUT / "check_training" / "runs"))
    env["FSQ_RUNS_DIR"] = str(runs_dir)
    todo = dict(RUNS, **(RUNS_EC if a.eurocropsml else {}))
    print(f"data root: {paths.DATA_ROOT}\nruns dir:  {runs_dir}\n")
    fails = []
    for name, (module, cfg, ckpt_name) in todo.items():
        cmd = [sys.executable, "-m", module, "--config", str(paths.REPO / cfg), "--max-epochs", "1",
               "--limit-train-batches", str(a.train_batches), "--limit-val-batches", str(a.val_batches), "--no-eval"]
        t0 = time.time()
        log = paths.OUTPUT / "check_training" / (name.replace(", ", "_").replace(" ", "_") + ".log")
        log.parent.mkdir(parents=True, exist_ok=True)
        with open(log, "w", encoding="utf-8") as fh:
            rc = subprocess.run(cmd, cwd=paths.REPO, env=env, stdout=fh, stderr=subprocess.STDOUT).returncode
        dt = time.time() - t0
        found = sorted(runs_dir.glob(f"*/checkpoints/{ckpt_name}"), key=lambda p: p.stat().st_mtime)
        ck = found[-1] if found and found[-1].stat().st_mtime >= t0 else None
        steps = None
        if ck is not None:
            c = torch.load(ck, map_location="cpu", weights_only=False)
            steps = int(c.get("global_step", 0))
        ok = rc == 0 and ck is not None and steps and (ck.parent.parent / "norm_stats.json").exists()
        print(f"{name:32s} exit {rc}  {dt / 60:5.1f} min  optimizer steps {steps}  checkpoint {ck if ck else '-'}  log {log}")
        if not ok:
            fails.append(name)
    print(f"\n{'PASS' if not fails else 'FAIL'}: {len(todo) - len(fails)} of {len(todo)} runs trained and saved a checkpoint")
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main())
