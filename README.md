# Learned Discrete Tokenizers Against Matched Non-Learned Controls for Satellite Image Time Series

Code, run configurations, tokenizer checkpoints and per-run results of the IEEE Geoscience and Remote Sensing Letters letter of the same title.

**What the letter does.** It evaluates a finite scalar quantization (FSQ) tokenizer that encodes each month of a Sentinel-2 series into one of 625 codes. The tokenizer is trained without labels, by reconstruction or by masked prediction. It is compared with two non-learned controls fitted on the same data and matched in dimensionality and vocabulary:
- rounded principal components of the monthly band means;
- a k-means vocabulary.

All codes are read by a linear probe for crop-type classification on three held-out regions (France and Queensland from CropGlobe, Estonia from EuroCropsML), with ten training runs per objective and setting.

## Install

Python 3.10 or later (the analyses ran on 3.12).

```
git clone https://github.com/FBK-OpenIoT/fsq-matched-controls
cd fsq-matched-controls
```

Then, from the repository root, create an environment and install the package.

Windows (PowerShell):
```
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e .
```

Linux / macOS:
```
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

This installs the package `fsq_matched_controls` and the pinned dependencies (`pyproject.toml`, also in `requirements.txt`), the versions the analyses ran with. Training ran on a GPU node with PyTorch 2.14.0 (CUDA 13.0) and Lightning 2.6.6; see `requirements.txt`. If PowerShell refuses to run `Activate.ps1`, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` first, or call `.venv\Scripts\python.exe` directly.

Without installing, the commands below also run from the repository root with the source folder on the path, given the dependencies (`pip install -r requirements.txt`):
```
$env:PYTHONPATH = "src"        # Windows PowerShell
export PYTHONPATH=src          # Linux / macOS
```

## Repository structure

```
src/fsq_matched_controls/     the Python package
  models/                     encoder, FSQ, decoder; the reconstruction and masked-prediction objectives; the continuous twins
  data/                       CropGlobe and EuroCropsML data modules, normalisation, label definitions
  training/                   training entry points and shared training helpers
  probe/                      the logistic probe
  stats/                      tests: one-sided t, Welch, Holm, interim-analysis multiplier
  analysis/                   Table I, Table II, Figure 1, the decomposition and twin tables, the letter's printed values, the smoke test
  paths.py                    every file location (see "Paths")
configs/
  letter/                     the 80 runs of the letter (40 masked prediction, 40 reconstruction)
  twins/                      120 runs of the models trained without the quantizer (Table II, supplement S5.3)
  own_recipe/                 3 reconstruction runs on reconstruction's own recipe (supplement S2)
checkpoints/                  tokenizer weights (encoder + FSQ) of the 80 letter runs; see checkpoints/README.md
results/                      per-run result tables and aggregates (CSV / JSON), the inputs of every regenerated table
  runs/                       the letter runs' probe rows (n_probe_*), control tables (diagnostic*), probe diagnostics
  letter_numbers/             the analysis output tables behind the letter's numbers (Table I tests: B.csv, item S71)
  table2_decomposition/       Table II: the twins' probe rows, step tests, continued twins, continuous PCA
  supplement_S3_controls/     k-means fits, bits per month
  supplement_S4_sensitivity/  sensitivity analyses, Estonian few-shot probes, continued runs
  supplement_S5_runs/         per-class F1 in France, training stop epochs
  supplement_S8_training/     capped runs, parameter counts, training times
manuscript/                   the supplement (.md/.tex/.pdf), the extended supplement and Figure 1 (.pdf/.png)
```

## Reproduce Table I, Table II and Figure 1 from `results/`

No dataset, GPU or checkpoint is needed for this.

```
python -m fsq_matched_controls.analysis.table1           # Table I    -> outputs/table1.csv (+ printed table)
python -m fsq_matched_controls.analysis.table2           # Table II   -> outputs/table2.csv (+ printed table)
python -m fsq_matched_controls.analysis.figure1 --palette okabe-ito --layout column --au-label Queensland --group-titles
                                                         # Figure 1   -> outputs/figure1_column.pdf / .png
python -m fsq_matched_controls.analysis.step_tests       # supplement S5.3, original twins   -> outputs/twin_step_tests.csv
python -m fsq_matched_controls.analysis.twins_continued  # supplement S5.3, continued twins  -> outputs/teacher_twin_continued_step_tests.csv,
                                                         #                                      outputs/bands_twin_continued_step_tests.csv
python -m fsq_matched_controls.analysis.smoke_test       # all of the above, compared with the letter
```

`smoke_test` checks four things and exits non-zero on any mismatch:
- every printed value, raw p and bold flag of Table I and Table II against the letter's printed values, stored in `src/fsq_matched_controls/analysis/expected_letter_values.csv` (generated once from the letter's LaTeX source with `python -m fsq_matched_controls.analysis.expected_letter_values --tex <letter .tex>`);
- the regenerated step-test, twin and Table I S7.1 tables byte for byte against the original analysis outputs shipped in `results/`;
- the Figure 1 PDF byte for byte against `manuscript/figure1_column.pdf`;
- the figure's per-run constants against the per-run tables.

**Display rounding.** One value is a rounding case, not a computation difference. The Estonia rounded-PCA macro-F1 control is stored as 0.0475 (four decimals). It prints as 0.048; the letter prints 0.047, from the unrounded value.

## Paths

Every location is defined in `src/fsq_matched_controls/paths.py`. Each can be overridden with an environment variable:

| Variable | Default | Content |
|---|---|---|
| `FSQ_RESULTS` | `results/` | the shipped per-run tables |
| `FSQ_OUTPUT` | `outputs/` | regenerated tables and figures |
| `FSQ_DATA_ROOT` | `data/` | the public datasets (below) |
| `FSQ_RUNS_DIR` | `runs/` | training outputs |
| `FSQ_FEATURES` | `features/` | monthly band-mean arrays rebuilt from the datasets |

## Data

The repository contains no satellite data and no per-pixel or per-parcel arrays. Download the public benchmarks:

- **CropGlobe** (Tong and Wang, 2025). Sentinel-2 median composites per country split, `median_S2_{CC}_month5-11_day5.npz`, in `data/cropglobe/raw/Median/`.
- **EuroCropsML** (Reuss et al., 2025). The preprocessed parcel files in `data/eurocropml/preprocess/`, and the benchmark's split files in `data/eurocropml/split/latvia_portugal_vs_estonia/`:
  - `pretrain/region_split.json`;
  - `finetune/region_split_{k}.json`, the official few-shot splits used in supplement S4.2.

The arrays the controls and probes read are rebuilt from these files: the package's data modules build the month windows, and the probe and control code (see "Contents of this version") rebuilds the arrays. The monthly band means are the masked mean of each month's observations.

## Retrain

```
python -m fsq_matched_controls.training.train_predictive --config configs/letter/<run>.yaml   # masked prediction
python -m fsq_matched_controls.training.train            --config configs/letter/<run>.yaml   # reconstruction
python -m fsq_matched_controls.training.train_twin_noquant --config configs/twins/<...nofsq...>.yaml                 # twin_noquant
python -m fsq_matched_controls.training.train_twin_bands   --config configs/twins/<...continuous_ssl_d4...>.yaml     # twin_bands, CropGlobe
python -m fsq_matched_controls.training.train_twin_teacher --config configs/twins/<...continuous_teacher_d4...>.yaml # twin_teacher, CropGlobe
python -m fsq_matched_controls.training.train_twins_eurocropsml --variant ssl|teacher --config configs/twins/eurocropml_...yaml
                                                                    # twin_bands (ssl) / twin_teacher (teacher), EuroCropsML
```

Run the commands from the repository root. The configs set `data_root: "data"` and a local MLflow store (`sqlite:///_mlruns/...`). Each run writes its best checkpoint and `norm_stats.json` to its run directory.

## Run-name key

The configuration file names and `experiment_name`s are kept as trained, because they key the per-run tables in `results/`.

| Part | Meaning |
|---|---|
| `cropglobe_predictive_ssl_ov0_625` | Masked prediction on CropGlobe: month windows without overlap (`ov0`), 625 codes (d = 4, L = 5) |
| `cropglobe_fsq_ov0_625_reconstruction_matched` | Reconstruction on CropGlobe, on the masked-prediction recipe (matched recipe) |
| `eurocropml_predictive_ssl_625_heldout_estonia_gradclip` | Masked prediction on EuroCropsML (Estonia held out), with gradient clipping |
| `eurocropml_fsq_baseline_625_reconstruction_varobs` | Reconstruction on EuroCropsML, with a variable number of real observations per month |
| `holdout_FRA`, `holdout_AUS` | Strict setting: the evaluation region (France; the Australian split, i.e. Queensland) is excluded from pretraining |
| `holdout_none` | Transductive setting (CropGlobe) |
| `armN` | The second group of runs (seeds 11, 23, 37, 53, 71, 89, 97), which brought every objective and setting to ten runs |
| `seed…` | The training seed. Runs without a seed in the name use seed 1904 |
| `_vm` | Trained on the GPU node |
| `_cineca` | The one run trained on an HPC A100 (EuroCropsML masked prediction, seed 1904) |
| `1904-new`, `1904-gc` (result tags) | Seed 1904 retrained with the final recipe (CropGlobe), and with gradient clipping (EuroCropsML) |
| `nofsq`, `continuous_ssl_d4`, `continuous_teacher_d4` | The twins trained without the quantizer: `twin_noquant` (reconstruction without the quantizer), `twin_bands` (masked prediction of the band values) and `twin_teacher` (masked prediction of a slow teacher's latent). The same names label the twins in `results/table2_decomposition/` |
| `twin-noquant`, `twin-bands`, `twin-teacher` (MLflow experiment names) | The same three twins |

## Relation to the code the letter was produced with

The package contains the same computations as the code the letter's numbers were produced with, organised as modules. The smoke test checks this: the regenerated tables are byte-identical to the stored outputs in `results/`, and the figure to the letter's. Differences that do not affect any number:
- **Paths:** every file location is in `paths.py`, and every analysis runs with `python -m`.
- **Cross-checks:** checks against earlier drafts of the letter are not included. The smoke test checks against the submitted letter's printed values instead (`analysis/expected_letter_values.csv`).
- **Configs:** comments are removed, and `data_root` is `data` in every file. Every parsed configuration is otherwise identical to the one used for training.
- **MLflow default:** when a config does not set an MLflow store, the default is a local relative path.

## Contents of this version

This version contains:
- the training code and every run configuration;
- the tokenizer checkpoints;
- the per-run results;
- the analyses behind Table I, Table II and Figure 1.

The following analyses are available as per-run results in `results/`; their code modules will be added:
- the probe and control code that rebuilds the monthly arrays from the datasets and the checkpoints;
- the supplement's sensitivity analyses (S4), the gradient-boosting probe (S4.3), the bits per month (S3), the rounding grid (S6), and the continued-run tables (S8).

## How to cite

Please cite the letter this code accompanies:

> J. Calisto, M. Vecchio, M. Pincheira, and F. Antonelli, "Learned Discrete Tokenizers Against Matched Non-Learned Controls for Satellite Image Time Series," submitted to *IEEE Geoscience and Remote Sensing Letters*, 2026.

`CITATION.cff` holds the same reference in machine-readable form; GitHub shows it as "Cite this repository".

## License

MIT; see `LICENSE`.
