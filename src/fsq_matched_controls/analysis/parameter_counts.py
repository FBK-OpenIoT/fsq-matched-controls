"""Method / supplement S8: learnable parameter counts of the tokenizers and of the probe.

    python -m fsq_matched_controls.analysis.parameter_counts [--probe]    -> outputs/parameter_counts.csv [, outputs/probe_parameter_counts.csv]

Models: one run per objective and benchmark, built from its configuration in configs/letter/ with the package's model classes
(month windows: CropGlobe 6 observations x 7 months, EuroCropsML 10 x 9); parameters with requires_grad, by module; buffers
(normalisation statistics, FSQ levels) apart. Compared with results/supplement_S8_training/parameter_counts.csv.
Probe (--probe, needs the datasets): features = months x 4, classes = classes of the harmonized probe-training pool;
StandardScaler has statistics, not learnable parameters.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from fsq_matched_controls import paths
from fsq_matched_controls.models.autoencoder import FSQAutoencoder, FSQAutoencoderConfig
from fsq_matched_controls.models.predictive_ssl import FSQPredictiveSSL, FSQPredictiveSSLConfig
from fsq_matched_controls.training._train_utils import load_config

RUNS = {("mp", "CropGlobe"): "cropglobe_predictive_ssl_ov0_625_armN_seed11_vm_holdout_none",
        ("recon", "CropGlobe"): "cropglobe_fsq_ov0_625_reconstruction_matched_transductive_seed42_vm_holdout_none",
        ("mp", "EuroCropsML"): "eurocropml_predictive_ssl_625_heldout_estonia_gradclip_armN_seed11_vm_holdout_Estonia",
        ("recon", "EuroCropsML"): "eurocropml_fsq_baseline_625_reconstruction_varobs_seed42_vm_holdout_Estonia"}
WINDOWS = {"CropGlobe": (6, 7), "EuroCropsML": (10, 9)}       # (observations per month window, months)


def build(obj, bench):
    idx = pd.read_csv(paths.CHECKPOINTS / "index.csv").set_index("file")
    cfg = load_config(paths.REPO / idx.loc[f"{RUNS[(obj, bench)]}.pt", "config"])
    obs, months = WINDOWS[bench]
    common = dict(n_bands=cfg.get("n_bands", 10), obs_per_window=obs, encoder_channels=cfg.get("encoder_channels", [32, 64, 128]),
                  kernel_size=cfg.get("kernel_size", 3), dropout=cfg.get("dropout", 0.1), norm_type=cfg.get("norm_type", "batch"),
                  n_dims=cfg.get("n_dims", 4), n_levels=cfg.get("n_levels", 5))
    if obj == "recon":
        return FSQAutoencoder(config=FSQAutoencoderConfig(**common))
    return FSQPredictiveSSL(config=FSQPredictiveSSLConfig(
        **common, n_tokens=months, factorized_head=cfg.get("factorized_head", False), head_mode=cfg.get("head_mode", "joint"),
        d_model=cfg.get("d_model", 128), n_heads=cfg.get("n_heads", 4), n_layers=cfg.get("n_layers", 3),
        dim_feedforward=cfg.get("dim_feedforward", 256), transformer_dropout=cfg.get("transformer_dropout", 0.1), max_seq_len=cfg.get("max_seq_len", 100)))


def model_counts() -> pd.DataFrame:
    def cnt(mod):
        return sum(p.numel() for p in mod.parameters() if p.requires_grad)
    rows = []
    for (obj, bench), d in RUNS.items():
        m = build(obj, bench)
        rec = dict(objective=obj, benchmark=bench, run_dir=d, learnable_total=cnt(m), non_learnable_params=sum(p.numel() for p in m.parameters() if not p.requires_grad),
                   buffers_total=sum(b.numel() for b in m.buffers()), encoder=cnt(m.encoder), fsq=cnt(m.fsq), fsq_buffers=sum(b.numel() for b in m.fsq.buffers()),
                   encoder_buffers=sum(b.numel() for b in m.encoder.buffers()))
        if obj == "mp":
            ct = m.context_transformer
            rec.update(context_transformer=cnt(ct), ct_input_proj=cnt(ct.input_proj), ct_mask_token=ct.mask_token.numel(), ct_pos_emb=cnt(ct.pos_emb),
                       ct_transformer=cnt(ct.transformer), ct_head=cnt(ct.head), decoder=0)
        else:
            rec.update(decoder=cnt(m.decoder), context_transformer=0)
        rec["modules_top"] = ";".join(f"{n}={cnt(c)}" for n, c in m.named_children())
        rows.append(rec)
    return pd.DataFrame(rows)


def probe_counts() -> pd.DataFrame:
    from fsq_matched_controls.controls import features
    from fsq_matched_controls.probe.linear_probe import harmonize_label_sets
    rows = []
    for ds, regs, months, kw in (("cropglobe", ("FRA", "AUS"), 7, {}), ("eurocropsml", ("estonia",), 9, {"other_class": -1})):
        d = features.cropglobe_monthly() if ds == "cropglobe" else features.eurocropsml_monthly()
        for reg in regs:
            yt, _, _ = harmonize_label_sets(d["y"]["train"], d["y"][reg], **kw)
            nf, nc = months * 4, len(np.unique(yt))
            rows.append(dict(dataset=ds, region=reg, features=nf, classes=nc, coef=nf * nc, intercepts=nc, params=nf * nc + nc))
    return pd.DataFrame(rows)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--probe", action="store_true", help="also the probe's parameters (needs the datasets)")
    a = ap.parse_args(argv)
    paths.OUTPUT.mkdir(parents=True, exist_ok=True)
    M = model_counts()
    M.to_csv(paths.OUTPUT / "parameter_counts.csv", index=False)
    ref = pd.read_csv(paths.SUPPLEMENT_S8 / "parameter_counts.csv")
    cols = ["learnable_total", "encoder", "context_transformer", "decoder", "ct_transformer", "ct_head"]
    same = M[cols].fillna(-1).astype(float).equals(ref[cols].fillna(-1).astype(float))
    print(M[["objective", "benchmark", "learnable_total", "encoder", "context_transformer", "decoder"]].to_string(index=False))
    print(f"learnable counts identical to results/supplement_S8_training/parameter_counts.csv: {same}")
    if a.probe:
        P = probe_counts(); P.to_csv(paths.OUTPUT / "probe_parameter_counts.csv", index=False); print(P.to_string(index=False))
    return 0 if same else 1


if __name__ == "__main__":
    raise SystemExit(main())
