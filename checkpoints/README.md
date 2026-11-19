# Tokenizer checkpoints of the letter's 80 runs

One file per run, `<run_dir>.pt`: 80 files, 11.5 MB in total. Each file holds only the **tokenizer**, the temporal CNN encoder and FSQ, which turns a month of Sentinel-2 observations into one of 625 codes. The parts used only during training are not included: the masked-prediction context transformer and the reconstruction decoder.

| Key | Content |
|---|---|
| `encoder_state_dict` | Weights of `fsq_matched_controls.models.encoder.TemporalCNNEncoder` |
| `fsq_state_dict` | Buffers of `fsq_matched_controls.models.fsq.FSQ` (it has no learned parameters) |
| `config` | The architecture keys needed to build both: `n_bands`, `encoder_channels`, `kernel_size`, `n_dims`, `n_levels`, `dropout`, `norm_type`, `obs_per_window` |
| `norm_stats` | The run's own per-band normalisation (`mean`, `std`, `dn_scale`), computed on its training data; apply it before the encoder |
| `run` | `run_dir` (the key of the per-run tables in `results/runs/`), `config_file` (the training configuration in `configs/letter/`), `objective` (`mp` masked prediction, `recon` reconstruction), `setting`, `seed_tag`, `best_epoch` |

`index.csv` lists every file with its objective, setting, seed and configuration. Every tensor was checked against the training checkpoint it was taken from: all are identical.

## Loading

```python
import torch
from fsq_matched_controls.models.encoder import TemporalCNNEncoder
from fsq_matched_controls.models.fsq import FSQ

ck = torch.load("checkpoints/<run_dir>.pt", map_location="cpu", weights_only=True)
c = ck["config"]
encoder = TemporalCNNEncoder(n_bands=c["n_bands"], channels=c["encoder_channels"], kernel_size=c["kernel_size"],
                             n_dims=c["n_dims"], n_levels=c["n_levels"], dropout=c["dropout"], norm_type=c["norm_type"])
encoder.load_state_dict(ck["encoder_state_dict"]); encoder.eval()
fsq = FSQ(n_dims=c["n_dims"], n_levels=c["n_levels"]); fsq.load_state_dict(ck["fsq_state_dict"])
```

Inputs are the month windows that the data modules build (`fsq_matched_controls.data`), normalised with `ck["norm_stats"]`. Five-level FSQ on four dimensions gives 5^4 = 625 codes.

## Not included

The twins, i.e. the models trained without the quantizer (`twin_noquant`, `twin_bands`, `twin_teacher`; Table II, supplement S5.3), are not included. Their encoder weights would add about 120 files of about 0.15 MB each (encoder only), roughly 18 MB in all. They can be attached as a release asset; their training configurations are in `configs/twins/`.
