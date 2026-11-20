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
  controls/                   the two matched controls: rounded PCA and the k-means vocabulary, built from the datasets
  analysis/                   Table I, Table II, Figure 1, the supplement's analyses, the checks, the letter's printed values, the smoke test
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

The repository contains no satellite data and no per-pixel or per-parcel arrays. The two public benchmarks are downloaded into `FSQ_DATA_ROOT` (default `data/`), in the layout the data modules read:

```
pip install -e ".[download]"                                   # adds Prefect
python -m fsq_matched_controls.data.download.flows all         # Prefect flow: both datasets (or: cropglobe, eurocropsml)
```

The same downloads without Prefect:

```
python -m fsq_matched_controls.data.download all               # or: cropglobe [--countries FRA AUS ...], eurocropsml
python -m fsq_matched_controls.data.download all --dry-run     # list files, sizes, checksums and destinations; write nothing
```

- **CropGlobe** (Tong and Wang, 2025), from the Hugging Face dataset `x-ytong/CropGlobe`: the Sentinel-2 median composites of the ten country splits, `Median/median_S2_{CC}_month5-11_day5.npz` (391 MB). They go to `data/cropglobe/raw/Median/`.
- **EuroCropsML** (Reuss et al., 2025), from Zenodo record 13789558: `preprocess.zip` (1.45 GB, one file per parcel) and `split.zip` (10 MB, the benchmark's split files). They are extracted into `data/eurocropml/`, which gives:
  - `preprocess/`;
  - `split/latvia_portugal_vs_estonia/pretrain/region_split.json`;
  - `split/latvia_portugal_vs_estonia/finetune/region_split_{k}.json`, the official few-shot splits used in supplement S4.2.

  The record's `raw_data.zip` is not needed.

Every file is checked against the checksum its source publishes (SHA-256 for CropGlobe, MD5 for EuroCropsML), and a rerun skips files already present and verified. No conversion step is needed. On first use, the EuroCropsML data module gathers the parcel files it needs into `data/eurocropml/cache/`, a one-time read.

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

Run the commands from the repository root. The configs set `data_root: "data"` and a local MLflow store (`sqlite:///_mlruns/...`). Each run writes its best checkpoint and `norm_stats.json` to its run directory. `--max-epochs`, `--limit-train-batches` and `--limit-val-batches` shorten a run without editing its configuration.

## Check training and the checkpoints with the datasets

Both checks need the datasets (see "Data"); point `FSQ_DATA_ROOT` at them. They run on CPU.

```
python -m fsq_matched_controls.analysis.check_training [--eurocropsml]
python -m fsq_matched_controls.analysis.check_end_to_end [--eurocropsml] [--run <run_dir>]
```

`check_training` trains the letter's configuration of each objective (strict Queensland, seed 11; with `--eurocropsml` also the EuroCropsML runs of seed 11) for one epoch of five batches. It checks that optimizer steps were taken and that the checkpoint and `norm_stats.json` were written, to `outputs/check_training/`. Each run takes a few minutes.

`check_end_to_end` starts from a shipped checkpoint (default: masked prediction, strict Queensland, seed 11; with `--eurocropsml` also masked prediction on EuroCropsML, seed 11). It tokenizes the probe pool and the evaluation region with the package's data modules and runs the letter's probe. It then compares accuracy and macro-F1 with the run's row in `results/runs/n_probe_*.csv`, and the token arrays with fingerprints of the letter's own arrays. Each check takes one to two minutes and needs about 5 GB of memory.

**Platform.** Exact equality holds on the platform the letter's probes ran on (Windows, CPU, PyTorch 2.7.0); the check then prints "identical". On other platforms a few FSQ codes can flip at the quantizer's rounding boundaries, and the lbfgs fit can differ slightly with the BLAS library. The check then passes within a tolerance of 0.001 per metric (`PLATFORM_TOLERANCE` in the module) and prints the differences. Observed on Linux (CPU, same pins): for EuroCropsML masked prediction, seed 11, 3 of about 9.9 million token cells differed, accuracy by 8.5e-5 and macro-F1 by 5.8e-4. CropGlobe (strict Queensland, seed 11) gave identical scores, although a few of its token cells differed too.

## Rebuild the controls and the supplement's analyses

These need the datasets (see "Data"; `FSQ_DATA_ROOT`). The monthly band means are built once with the package's data modules and cached in `FSQ_FEATURES`. All run on CPU.

```
python -m fsq_matched_controls.controls                              # rounded PCA and the k-means vocabulary, every row, the letter's probe
python -m fsq_matched_controls.analysis.check_controls               # the same, compared with the stored control scores and Table I's verdicts
python -m fsq_matched_controls.analysis.probe_runs --sample | --all  # the letter's probe on the runs' tokens and continuous latent, compared with results/runs/
python -m fsq_matched_controls.analysis.control_variants --compare   # S4.1, S5.2: continuous PCA, fit-pool standardization, spectral-index control
python -m fsq_matched_controls.analysis.no_harmonization --compare   # S4.1: without label harmonization (after probe_runs --all)
python -m fsq_matched_controls.analysis.per_run_readouts --mode curve|onehot|gb --sample --compare   # S6 rounding curve, S4.1 one-hot, S4.3 boosting
python -m fsq_matched_controls.analysis.fewshot_estonia --compare    # S4.2: Estonian labels on the official few-shot split
python -m fsq_matched_controls.analysis.gradient_boosting --compare  # S4.3: boosting on the raw band means (--kmeans: and the ten-band k-means)
python -m fsq_matched_controls.analysis.bits_per_month --compare     # S3: entropy of the controls' month codes
```

From `results/` alone (no datasets):

```
python -m fsq_matched_controls.analysis.kmeans_fits --compare        # S3: the five k-means fits per row
python -m fsq_matched_controls.analysis.signflip_tests --compare     # S4.1, S7: exact sign-flip test beside Table I's t-tests
python -m fsq_matched_controls.analysis.continued_runs --compare     # S4.1: Table I and the decomposition with the continued runs
python -m fsq_matched_controls.analysis.parameter_counts             # S2: parameters of the tokenizers (from the configurations and the model code)
python -m fsq_matched_controls.analysis.training_times --compare     # S2: training wall-clock from the MLflow records
python -m fsq_matched_controls.analysis.capped_runs --compare        # S8: the capped runs and the estimate of their continuation
```

With `--compare`, these commands check their tables against the stored ones: byte-identical on the letter's platform (Windows); elsewhere identical apart from line endings, or equal to floating-point precision (the last one or two digits of some p-values; `analysis/compare_tables.py`). `parameter_counts` prints whether every count equals the stored one.

**Run time and memory** (one CPU thread, measured on Linux):

| Command | Time | Peak memory |
|---|---|---|
| `check_training --eurocropsml` | 2 min 47 s | 4.7 GB |
| `check_end_to_end --eurocropsml` | 2 min 11 s | 4.3 GB |
| `check_controls` | 13 min 51 s | 1.8 GB |
| `probe_runs --sample` (8 runs) | 8 min | 4.6 GB |
| `probe_runs --all` (80 runs) | 1 h 17 min | 4.4 GB |
| `control_variants --compare` | 6 min 21 s | 1.5 GB |
| `per_run_readouts --sample`, curve / onehot / gb | 1 min 21 s / 1 min 10 s / 2 min 38 s | 4.2 GB |
| `fewshot_estonia --compare` | 5 min 41 s | 1.3 GB |
| `gradient_boosting --compare` (`--kmeans`) | 6 min 51 s (41 min) | 0.9 GB (1.0 GB) |
| `bits_per_month --compare` | 8 min 55 s | 1.7 GB |
| `no_harmonization --compare` (after `probe_runs --all`) | 54 min | 1.4 GB |

**Reproduction on another platform.** The letter's probes and controls ran on Windows (CPU, PyTorch 2.7.0, the pinned versions). Every check prints "identical" or the largest difference per metric, and compares every verdict. It passes only if every verdict is identical and every difference is within the tolerance of its analysis (`analysis/tolerances.py`). Observed on Linux with the same pinned versions:

| Analysis | Tolerance | Observed on Linux | Cause |
|---|---|---|---|
| Probe scores of the runs (tokens, continuous) | 0.001 | up to 8.0e-4; 60 of the 200 scores identical | lbfgs stops at 300 iterations; a few FSQ codes flip at the rounding boundaries |
| Macro-F1 in Estonia (run probes) | 0.002 | 1.12e-3 (one run: 6 of 8.6 million token cells of the probe pool flipped) | macro-F1 over many small classes, about 0.08 |
| Rounded PCA | 0.001 | up to 2.4e-4; strict France accuracy identical | lbfgs |
| k-means vocabulary, mean of five fits | 0.003 | up to 1.2e-3 (single fits up to 4.2e-3) | the vocabulary is fitted on principal-component scores, whose tiny float differences (BLAS) change the mini-batch path; k-means fitted on the band means directly is identical on Linux |
| k-means vocabulary, entropy (bits per month), mean of five fits | 0.01 bits | 0.0077 bits (single fits up to 0.046) | as above |
| Few-shot probe (S4.2), k-means columns / rounded-PCA columns | 0.005 / 0.002 | 4.6e-3 (k = 1, 123 training parcels) / 1.3e-3 | as above; lbfgs on small training sets |
| Continuous PCA, fit-pool standardization | 0.001 | up to 3.8e-4 | lbfgs |
| Spectral-index control, one-hot readout | 0.002 | 1.3e-3, 1.1e-3 | lbfgs far from convergence |
| Rounding curve, grid distance | 0.001 | 1.2e-4, 5e-7 | lbfgs; float summation |
| Gradient boosting on the tokens | none (cells listed) | seed 11 identical; seed 97 France macro-F1 1.2e-2 (2 of 10.2 million token cells flipped) | splits on exact token values |
| Gradient boosting on the raw band means / ten-band k-means | 0.001 | identical to the Linux rebuild (`gradient_boosting_raw_linux.csv`); the Windows rebuild differs in the k-means rows by up to 1.3e-2 | the platform |
| Rounded PCA and ten-band k-means, entropy (bits per month) | last stored digit | identical (one rounding tie at the fourth decimal) | – |
| Smoke test: regenerated statistics tables (Table I's S7.1 records, Table II's step tests) | relative 1e-12 | 5.6e-16 (last one or two digits of some p-values); byte-identical on Windows | floating-point arithmetic of the platform |
| Monthly band means, k-means fit statistics, parameter counts, training times, capped runs | exact | identical (k-means fit statistics and training times: apart from line endings) | – |
| Sign-flip tests, continued runs (tables from `results/`) | relative 1e-12 | 4.3e-16 | floating-point arithmetic of the platform |

Table I's verdicts recomputed with the rebuilt controls equal the letter's.

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
- the analyses behind Table I, Table II and Figure 1;
- the two controls and the probe, rebuilt from the datasets and the checkpoints;
- the supplement's analyses: S2 (parameters, training times), S3 (k-means fits, bits per month), S4 (sensitivity, Estonian few-shot, gradient boosting, continued runs), S5.2 (continuous PCA), S6 (rounding curve, grid distance), S7 (sign-flip tests).

Kept as stored results, without a regenerating module: the rows of earlier diagnostics that feed these tables (`results/runs/diagnostic*`), the stop epochs and validation losses read from the training records (`results/supplement_S5_runs/`), and the per-class F1 in France.

## How to cite

Please cite the letter this code accompanies:

> J. Calisto, M. Vecchio, M. Pincheira, and F. Antonelli, "Learned Discrete Tokenizers Against Matched Non-Learned Controls for Satellite Image Time Series," submitted to *IEEE Geoscience and Remote Sensing Letters*, 2026.

`CITATION.cff` holds the same reference in machine-readable form; GitHub shows it as "Cite this repository".

## License

MIT; see `LICENSE`.
