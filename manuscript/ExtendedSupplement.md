# Extended Supplementary Material

## Learned Discrete Tokenizers Against Matched Non-Learned Controls for Satellite Image Time Series

This is the full report behind the letter's 2–3-page supplementary material, released with the code
and the per-run result tables (repository, https://github.com/FBK-OpenIoT/fsq-matched-controls). Its section numbers are its own; the
short supplement renumbers the parts it keeps.

This document accompanies the letter and is intended to be read on its own by a reader or
reader who does not have access to the authors' working files. Citations (e.g. [7], [17]) use the
same numbering as the letter's own reference list; see the letter for full entries. Every number
below comes from a per-run result table, a training configuration or a checkpoint's own saved
state; Section S11 describes their availability.

---

## S1. Data and preprocessing per benchmark

### S1.1 Input pipelines side by side

CropGlobe [7] provides Sentinel-2 pixel samples labelled by country split; the Australian split covers
Queensland only [7]. EuroCropsML [11] provides Sentinel-2 parcels labelled by NUTS3 region.
The interpolation shares were measured against the per-acquisition series that the authors link from their repository; that series is used only for this check.

| | CropGlobe | EuroCropsML |
|---|---|---|
| Source files | Fixed-cadence per-country composites | Per-parcel time series, one file per (region, parcel, crop code) |
| Acquisition | Fixed 5-day grid as distributed (Hugging Face release [7]): per-band median of the acquisitions in each 5-day bin; bins with no acquisition are linearly interpolated in the distributed product (34 % to 67 % of bins per country); no value is missing | Variable, irregular real acquisition dates per parcel; some calendar months structurally empty |
| Calendar months covered | May–November (7 months) — see §S1.3 for what this window means per hemisphere | March–November (9 months); December–February excluded (§S1.2) |
| Tokens per sample (CropGlobe) / parcel (EuroCropsML) | 7 (one per calendar month) | 9 (one per calendar month) |
| Token window | 30 calendar days, 0-day overlap → 6 grid values per token | Variable real observations per month, padded/truncated to `w_max = 10` |
| Observations per window | Always 6 grid values, about half of them interpolated on average (34 % to 67 % of bins per country) | 0–15 real, padded to 10; an observation mask marks real vs. padding within a window |
| Minimum real observations for a valid month | N/A — no position is missing (empty bins are interpolated in the distributed product) | 2 (`w_min`); below that the whole month is marked invalid, not imputed |
| Band selection / ordering | Native 10-band order `[B2,B3,B4,B5,B6,B7,B8,B8A,B11,B12]` (red at index 2, NIR at index 6) | 10 of the 13 raw Sentinel-2 bands, ordered `[B2,B3,B4,B8,B5,B6,B7,B8A,B11,B12]` (red at index 2, NIR at index 3) |
| Missing whole-month handling | Does not occur (full calendar grid by construction) | A validity mask marks missing months; §S2.2 describes how the predictive objective treats them |

Normalization statistics: see §S1.4.

### S1.2 Empirical basis for the EuroCropsML choices

- **`w_max = 10`, `w_min = 2`.** A 500-parcel survey of per-(parcel, month) observation counts
  (March–November, Latvia+Portugal) found the 90th percentile at 9 real observations, the 95th at
  10, and a maximum of 15, with about 49% mean padding fraction already at `w_max = 10` — growing
  `w_max` further buys negligible extra coverage for a lot more padding. `w_max = 10` keeps
  truncation to about 1.7% of month-observations. `w_min = 2` (at least 2 real looks for a month to
  count as observed) matches the project's "do not impute" principle.
- **Parcels with zero 20 m bands.** In some scans of the distributed files the six 20 m bands
  (B5, B6, B7, B8A, B11, B12) and the three 60 m bands are exactly zero while B2, B3, B4 and B8 carry
  values: missing data coded as zero, concentrated in a few parcels, for most of their series. Over all
  706,683 files, 1.12% of parcels have such a scan between March and November: 0.98% in Estonia, 0.30%
  in Latvia and 4.90% in Portugal. In the letter's pools that is 4,363 of the 239,538 Latvia+Portugal
  parcels (1.82%) and 366 of the 35,182 Estonian parcels (1.04%); in the Latvia+Portugal pool, 36,649 of
  the 2,039,231 valid months (1.8%) have the 20 m bands at zero, and a further 1,446 (0.07%) the four
  10 m bands. The cause cannot be confirmed from the files. These months have at least two scans, so the
  validity rule above keeps them, and every representation reads them alike. Removing the flagged parcels
  from the probe's training pool and from the Estonia evaluation set (encoders and control fits
  unchanged) raises every Estonia score by 0.002 to 0.004 in accuracy and changes no outcome of
  Table I, under the t-test or the sign-flip test (margins of masked prediction +0.042 / +0.032 over
  the rounded PCA, +0.033 / +0.027 over k-means, +0.057 / +0.045 over reconstruction).
- **March–November token grid, December–February excluded.** A 300-parcel sample of the
  Latvia+Portugal pretraining split shows March–November at 95.7–100% parcel coverage;
  December/January/February sit at 28–49% — excluded from the fixed token grid entirely rather
  than carried as near-always-missing positions.
- **Band selection.** From EuroCropsML's standard 13-band Sentinel-2 order, the same ten bands as
  CropGlobe are selected, but in a different order (NIR fourth rather than seventh; §S1.1 table).
  Every encoder, control and probe is fitted and evaluated within one benchmark, so the order
  affects no result; it only means that the two benchmarks' tokens are not comparable position by
  position.

### S1.3 CropGlobe's May–November window

CropGlobe's raw per-country files are all named on the same `month5-11` convention (May–November,
5-day cadence), for every country including the Australian split (Queensland). Per Tong and Wang [7], CropGlobe's Southern
Hemisphere series are collected over a window shifted by six months relative to the Northern
Hemisphere (July 2022 – June 2023 for the Southern Hemisphere collection), so the fixed
"May–November" filename label denotes the same relative position in the local growing season for
every country, not the same calendar months. CropGlobe's month-tokens are therefore season-aligned
across hemispheres.

We checked this directly against the raw files rather than relying on the citation alone.
Loading the Queensland file and a Northern-Hemisphere file (France) shows both hold exactly two
arrays: pixel-sample features (samples × 10 bands × 43 five-day composites) and an integer class
label per sample — no per-observation date or timestamp field of any kind. The 5-day cadence and
the seven-month span are encoded only in the fixed array shape and the filename, identically for
every country; nothing in the array contents distinguishes a Northern- from a Southern-Hemisphere
acquisition calendar. The season-alignment claim therefore rests on [7]'s description of the
underlying collection windows, not on an inspectable date field in these particular files — which
is exactly what this check was designed to determine, and the files hold no dates.

### S1.4 Normalization statistics

**Training-time normalization.** Every CropGlobe and EuroCropsML training configuration leaves
its normalization-statistics path unset, so each training run computes its own statistics fresh
from its own training split — the training countries after hold-out exclusion for CropGlobe, the
pretraining split for EuroCropsML — and saves them to that run's own statistics file. This is
uniform across every run, transductive and strict, on both benchmarks: no run ever reads a shared
or externally supplied statistics file at training time.

**Feature-extraction-time normalization.** Extracting continuous or token features from a trained
checkpoint for the probe is a separate step, and it is here — not at training time — that a shared
statistics file is deliberately used for one setting. Transductive checkpoints are extracted using the
shared all-countries statistics file, computed over all ten countries — correct and non-leaking,
because the transductive encoder was itself trained on all ten countries, so the extraction
statistics match what the encoder saw during training. Strict checkpoints are extracted using
their own auto-saved, run-specific statistics file (computed from that run's actual training
countries only, i.e. excluding the evaluation region) — never the shared ten-country file, which
would otherwise leak the held-out region's statistics into a strict-setting evaluation. Every
feature-based number in this document follows this rule.

---

## S2. Encoder and objectives

### S2.1 Encoder adaptations for irregular sampling

The temporal CNN encoder's default path (no observation mask, batch normalization) is CropGlobe's,
untouched, and existing CropGlobe checkpoints load unchanged. Supplying an observation mask
(EuroCropsML) activates two changes, both needed together:

- **Masked mean pooling** instead of a plain average, so padded zeros do not dilute the pooled
  result.
- **Per-timestep layer normalization** in place of batch normalization. Batch normalization pools
  statistics over the batch and time axes jointly — with a padded fraction measured at about 49%
  mean at the `w_max = 10` actually used (§S1.2's 500-parcel survey), that would materially bias the learned running statistics toward zero.
  Layer normalization normalizes across channels only, independently at each timestep, so padding
  in one position cannot contaminate another's statistics.

A third adaptation, not gated by the normalization choice: padded positions are re-zeroed after
every convolutional block (not just at the initial input), bounding how much a padded zero can
bleed into neighbouring real timesteps as the convolutional receptive field compounds across
layers.

### S2.2 Missing months in the predictive objective

When a validity mask is supplied (EuroCropsML only; CropGlobe passes none, since no position is
missing):

- The transformer's input mask becomes "masked-by-training-policy OR missing" — missing months are
  always hidden from the transformer as masked context, exactly like a training-time mask, whether
  or not the random mask happened to select them.
- The supervision mask becomes "masked-by-training-policy AND present" — missing months are never
  used as cross-entropy supervision targets and never counted in accuracy, since there is no
  ground-truth token there.

A structurally missing month is therefore never fed to the transformer as if it were real content,
and never scored as if a prediction were expected of it.

**Missing months in the probe.** The sequence always has nine positions, March to November, so a
missing month keeps its calendar slot and the spacing between months is preserved. In the probe's
input that slot carries a constant placeholder, the same for every parcel: zero for raw band means
(in normalized units), and for the encoders and controls whatever an empty month maps to (the
encoder's output for a window with no observations; for PCA and k-means, the code of the zero
vector). The probe therefore sees "month missing" identically across representations. In Estonia
11.9% of parcel-months are missing.

### S2.3 End-to-end training of the masked-prediction tokenizer, and collapse

The masked-prediction tokenizer is trained in one stage: encoder, quantizer and transformer are
optimized jointly on the masked-prediction loss, with no reconstruction pretraining and no frozen
tokenizer. The prediction head is a single joint softmax over all codes (625 at `d = 4`, 15,625 at
`d = 6`). The "continuous" representation of the main text is the same trained encoder's
four-dimensional output read before rounding, not a separately trained model. After training, the
transformer and its prediction head are discarded: every probe reads the encoder's output, quantized
or, for the continuous representation, before rounding, and never the transformer's output.

Because the encoder also produces the prediction targets, mapping every month to one code would make
prediction trivial; this collapse is the known failure mode of such objectives. No explicit
anti-collapse term (entropy regularizer, momentum target encoder) is used. The targets are integer
code indices obtained by rounding, so no gradient flows through them: the encoder is trained only
through the visible tokens it passes to the transformer, which rewards informative codes, and never
directly toward targets that are easy to predict. This makes collapse unlikely but does not exclude
it, so code usage and per-dimension code entropy were logged every epoch. Every CropGlobe
masked-prediction checkpoint uses all 625 codes (S5.1). One of the ten EuroCropsML masked-prediction
checkpoints (seed 23) collapsed early: its best validation loss came at epoch 0 and it maps the probe
pool onto 308 of the 625 codes (token entropy 6.2 bits, against 9.0 to 9.1 for the other nine). It is
kept in every table and test. The twenty France-evaluated runs have mean per-dimension entropy between
2.13 and 2.29 bits, against a maximum of log2 5 ≈ 2.32: all four dimensions are used, close to uniformly.

## S3. Training recipes per benchmark and objective, and checkpoint provenance

Within each benchmark the two objectives share encoder, quantizer, width and data pipeline (§S2)
and differ only in the objective head. Recipes are matched on both benchmarks: reconstruction is
trained under masked prediction's schedule, batch definition and number of optimizer steps per
epoch. On CropGlobe, reconstruction was also run on its own recipe (cosine schedule, batches of 512 windows); §S9.2 compares the two.

### S3.1 Training recipe per benchmark and objective

Optimizer is AdamW throughout. "Plateau" halves the learning rate (factor 0.5) after the stated
number of epochs without validation-loss improvement. Steps per epoch is the training-set size
divided by the batch size. Batch size for reconstruction counts 30-day windows; for masked
prediction it counts whole series (CropGlobe pixel samples, or EuroCropsML parcels), each
contributing 7 (CropGlobe) or up to 9 (EuroCropsML) windows.

| Benchmark | Objective / setting | LR | Schedule | Schedule params | Batch size | Counts | Steps/epoch | Early-stop patience | Max epochs | Grad. clip | Weight decay |
|---|---|---|---|---|---|---|---|---|---|---|---|
| CropGlobe | Masked prediction, transductive | 1e-3 | plateau | factor 0.5, patience 10 | 64 | series (= 448 windows) | ≈5,123 | 15 | 100 | none | 1e-4 |
| CropGlobe | Masked prediction, strict (France held out) | 1e-3 | plateau | factor 0.5, patience 10 | 64 | series | ≈3,596 | 15 | 100 | none | 1e-4 |
| CropGlobe | Masked prediction, strict (Queensland held out) | 1e-3 | plateau | factor 0.5, patience 10 | 64 | series | ≈4,858 | 15 | 100 | none | 1e-4 |
| CropGlobe | Reconstruction (transductive, strict France, strict Queensland) | 1e-3 | plateau | factor 0.5, patience 10 | 448 | windows | ≈5,123 | 15 | 100 | none | 1e-4 |
| EuroCropsML | Reconstruction | 5e-4 | plateau | factor 0.5, patience 10 | 545 | windows (matches the masked-prediction steps per epoch) | ≈3,743 | 25 | 200 | 1.0 | 1e-4 |
| EuroCropsML | Masked prediction | 5e-4 | plateau | factor 0.5, patience 10 | 64 | parcels | ≈3,743 | 25 | 200 | 1.0 | 1e-4 |

The six-dimension runs (§S9.4) use the same recipe as their four-dimension counterparts; only the
latent width and the random seed differ.

### S3.1b Parameters, compute, software and where each setting came from

**Learnable parameters.** These were counted from the checkpoints. All 80 runs share one architecture per
objective and benchmark.

| Module | Masked prediction | Reconstruction |
|---|---|---|
| Encoder (temporal CNN, channels 32/64/128, kernel 3, pooling, linear to d = 4, tanh): the tokenizer at inference | 32,868 | 32,868 |
| FSQ (5 levels, d = 4) | 0 (no codebook) | 0 |
| Context transformer (3 layers, d_model 128, 4 heads, feed-forward 256; joint softmax head over 625 codes), training only | 508,401 | – |
| Decoder, training only | – | 31,978 |
| **Total during training** | **541,269** | **64,846** |

The token probe has 145 (France), 87 (Queensland) and 3,071 (Estonia) parameters: 28 or 36 inputs times the
harmonized classes, plus the biases.

**Training time and hardware.**
- **Hardware.** 79 of the 80 runs were trained on one NVIDIA L40S (46 GB; 16 vCPUs of a Xeon Platinum
  8592+; Ubuntu 24.04; PyTorch 2.14.0 with CUDA 13.0; Lightning 2.6.6).
- **Shared GPU.** The GPU was shared: on average 7 to 11 other runs were open at a run's midpoint. The
  wall-clock times below are therefore upper bounds, not isolated times.
- **Wall-clock per run** (mean, range):

| Objective | CropGlobe | EuroCropsML |
|---|---|---|
| Masked prediction | 5.1 h (2.1–9.3) | 8.1 h (1.3–11.7), nine runs |
| Reconstruction | 2.1 h (1.1–3.4) | 4.9 h (3.6–6.1) |

- **The one other run.** EuroCropsML masked prediction, seed 1904, ran on one NVIDIA A100 (CINECA
  Leonardo): 6.4 h elapsed (6:22:19) under a 12 h limit, ended by a manual job cancellation (§S8,
  deviation 6).

**Inference.** Tokenizing the 364,286 series of the CropGlobe probe pool, encoder plus FSQ, without data
loading:
- **CPU:** 0.28 s per 1,000 series on a laptop (i7-1370P);
- **GPU:** 0.014 s per 1,000 series on the shared L40S.

**Analysis software.** Probes, controls and statistics ran on Windows 11 with Python 3.12.10, PyTorch
2.7.0 (CPU), scikit-learn 1.9.0 and NumPy 2.1.1. The gradient-boosting fits agree to four decimals with
scikit-learn 1.9.1 on the training node (§S9.3).

**Where each setting came from.**

| Setting | Origin | Tuned? |
|---|---|---|
| d = 4, L = 5 (625 codes) | One cell of the design grid (d and L from 3 to 6) written before any run; kept as the grid's low-cost reference | No. A later vocabulary sweep on EuroCropsML (frozen d = 16 backbone) did not select it |
| 15% of month-tokens masked | Fixed default of the objective (BERT's ratio; BEiT masks about 40% of image patches, blockwise) | No |
| Three transformer layers | Code default; the design asked for a small context model of 2–3 layers | No depth sweep |
| Probe C = 1 | scikit-learn default | No |
| Probe 300 iterations | Fixed for the letter; no recorded reason. The 2,000-iteration check (§S7, "Probe convergence") shows the raw-band scores move by at most 0.003 in accuracy and 0.006 in macro-F1 | No |
| Learning rate, schedule, batch, patience, clipping | S3.1. The EuroCropsML gradient clipping and lower learning rate followed an instability diagnosis | Not swept |

### S3.2 Checkpoint provenance

Verified from each checkpoint's own saved state, not from names or configurations. A checkpoint
stores its epoch at save time and its cumulative optimizer step count, so the implied batch size is
`N_train × (epoch + 1) / global_step`, where `N_train` is that checkpoint's exact training-set size
(windows for reconstruction, series or parcels for masked prediction). Applied to every checkpoint
feeding a number in the letter or this supplement, tolerance 10%.

| Checkpoint | Epoch | Global step | Implied batch | Expected batch | LR | Check |
|---|---|---|---|---|---|---|
| CropGlobe reconstruction, `d=4`, seed 42 (original recipe, §S9.2) | 36 | 165,871 | 511.9 | 512 | 0.001 | pass |
| CropGlobe reconstruction, `d=4`, seed 101 (original recipe, §S9.2) | 30 | 138,973 | 511.9 | 512 | 0.001 | pass |
| CropGlobe reconstruction, `d=4`, seed 1904 (original recipe, §S9.2) | 21 | 98,626 | 511.9 | 512 | 0.001 | pass |
| CropGlobe reconstruction, `d=6`, seed 1904 (original recipe, §S9.4) | 35 | 161,388 | 511.9 | 512 | 0.001 | pass |
| CropGlobe reconstruction, `d=6`, seed 42 (original recipe, §S9.4) | 19 | 89,660 | 511.9 | 512 | 0.001 | pass |
| CropGlobe reconstruction, `d=6`, seed 101 (original recipe, §S9.4) | 26 | 121,041 | 511.9 | 512 | 0.001 | pass |
| CropGlobe reconstruction, strict, France (original recipe, §S9.2) | 27 | 88,116 | 511.9 | 512 | 0.001 | pass |
| CropGlobe reconstruction, strict, Queensland (original recipe, §S9.2) | 19 | 85,020 | 511.9 | 512 | 0.001 | pass |
| CropGlobe reconstruction, transductive, seed 42 | 62 | 322,749 | 448.0 | 448 | 0.001 | pass |
| CropGlobe reconstruction, transductive, seed 101 | 23 | 122,952 | 448.0 | 448 | 0.001 | pass |
| CropGlobe reconstruction, transductive, seed 1904 | 41 | 215,166 | 448.0 | 448 | 0.001 | pass |
| CropGlobe reconstruction, strict, France, seed 42 | 92 | 334,428 | 448.0 | 448 | 0.001 | pass; ran all 100 epochs |
| CropGlobe reconstruction, strict, France, seed 101 | 51 | 186,992 | 448.0 | 448 | 0.001 | pass |
| CropGlobe reconstruction, strict, France, seed 1904 | 53 | 194,184 | 448.0 | 448 | 0.001 | pass |
| CropGlobe reconstruction, strict, Queensland, seed 42 | 33 | 165,172 | 448.0 | 448 | 0.001 | pass |
| CropGlobe reconstruction, strict, Queensland, seed 101 | 32 | 160,314 | 448.0 | 448 | 0.001 | pass |
| CropGlobe reconstruction, `d=6`, transductive, seed 42 | 45 | 235,658 | 448.0 | 448 | 0.001 | pass |
| CropGlobe reconstruction, `d=6`, transductive, seed 101 | 69 | 358,610 | 448.0 | 448 | 0.001 | pass |
| CropGlobe reconstruction, `d=6`, transductive, seed 1904 | 35 | 184,428 | 448.0 | 448 | 0.001 | pass |
| CropGlobe reconstruction, strict, Queensland, seed 1904 | 36 | 179,746 | 448.0 | 448 | 0.001 | pass |
| CropGlobe masked prediction, `d=4`, seed 2024 | 85 | 440,578 | 64.0 | 64 | 0.001 | pass |
| CropGlobe masked prediction, `d=4`, seed 4242 | 59 | 307,380 | 64.0 | 64 | 0.001 | pass |
| CropGlobe masked prediction, `d=4`, seed 1904 | 97 | 502,054 | 64.0 | 64 | 0.001 | pass |
| CropGlobe masked prediction, `d=6`, seed 1904 | 98 | 507,177 | 64.0 | 64 | 0.001 | pass |
| CropGlobe masked prediction, `d=6`, seed 42 | 92 | 476,439 | 64.0 | 64 | 0.001 | pass |
| CropGlobe masked prediction, `d=6`, seed 101 | 99 | 512,300 | 64.0 | 64 | 0.001 | pass |
| CropGlobe masked prediction, `d=4`, `L=3` (single run, §S9.1) | 8 | 46,107 | 64.0 | 64 | 0.001 | pass |
| CropGlobe masked prediction, `d=4`, `L=7` (single run, §S9.1) | 95 | 491,808 | 64.0 | 64 | 0.001 | pass |
| CropGlobe masked prediction, `d=4`, `L=9` (single run, §S9.1) | 98 | 507,177 | 64.0 | 64 | 0.001 | pass |
| CropGlobe masked prediction, strict, France, seeds 1904/2024/4242/42/101 | 65 / 84 / 96 / 98 / 95 | 237,336 / 305,660 / 348,812 / 356,004 / 345,216 | 64.0 (all five) | 64 | 0.001 | pass (all five) |
| CropGlobe masked prediction, strict, Queensland, seeds 1904/2024/4242/42/101 | 72 / 42 / 98 / 97 / 92 | 354,634 / 208,894 / 480,942 / 476,084 / 451,794 | 64.0 (all five) | 64 | 0.001 | pass (all five) |
| EuroCropsML reconstruction, seed 42 | 152 | 572,526 | 544.96 | 545 | 0.0005 | pass |
| EuroCropsML reconstruction, seed 101 | 155 | 583,752 | 544.96 | 545 | 0.0005 | pass |
| EuroCropsML reconstruction, seed 1904 | 137 | 516,396 | 544.96 | 545 | 0.0005 | pass |
| EuroCropsML masked prediction, seed 1904 | 137 | 516,534 | 64.0 | 64 | 0.0005 | pass |
| EuroCropsML masked prediction, seed 2024 | 121 | 456,646 | 64.0 | 64 | 0.0005 | pass |
| EuroCropsML masked prediction, seed 4242 | 178 | 669,997 | 64.0 | 64 | 0.0005 | pass |
| CropGlobe masked prediction, transductive, seeds 11 / 23 / 37 / 53 / 71 / 89 / 97 | 97 / 91 / 98 / 50 / 67 / 67 / 99 | 502,054 / 471,316 / 507,177 / 261,273 / 348,364 / 348,364 / 512,300 | 64.00 (all 7) | 64 | 0.001 | pass (all 7); 4 ran to the epoch limit |
| CropGlobe masked prediction, strict, France, seeds 11 / 23 / 37 / 53 / 71 | 74 / 63 / 99 / 36 / 89 | 269,700 / 230,144 / 359,600 / 133,052 / 323,640 | 63.99 (all 5) | 64 | 0.001 | pass (all 5); 2 ran to the epoch limit |
| CropGlobe masked prediction, strict, Queensland, seeds 11 / 23 / 37 / 53 / 71 | 88 / 91 / 98 / 54 / 68 | 432,362 / 446,936 / 480,942 / 267,190 / 335,202 | 63.99 (all 5) | 64 | 0.001 | pass (all 5); 3 ran to the epoch limit |
| EuroCropsML masked prediction, seeds 11 / 23 / 37 / 53 / 71 / 89 / 97 | 125 / 0 / 152 / 113 / 164 / 158 / 191 | 471,618 / 3,743 / 572,679 / 426,702 / 617,595 / 595,137 / 718,656 | 64.00 (all 7) | 64 | 0.0005 | pass (all 7); 1 ran to the epoch limit |
| CropGlobe reconstruction, transductive, seeds 11 / 23 / 37 / 53 / 71 / 89 / 97 | 53 / 33 / 71 / 38 / 16 / 52 / 33 | 276,642 / 174,182 / 368,856 / 199,797 / 87,091 / 271,519 / 174,182 | 447.99 (all 7) | 448 | 0.001 | pass (all 7) |
| CropGlobe reconstruction, strict, France, seeds 11 / 23 / 37 / 53 / 71 / 89 / 97 | 79 / 55 / 32 / 75 / 36 / 27 / 39 | 287,680 / 201,376 / 118,668 / 273,296 / 133,052 / 100,688 / 143,840 | 447.96 (all 7) | 448 | 0.001 | pass (all 7) |
| CropGlobe reconstruction, strict, Queensland, seeds 11 / 23 / 37 / 53 / 71 / 89 / 97 | 37 / 34 / 48 / 24 / 71 / 35 / 65 | 184,604 / 170,030 / 238,042 / 121,450 / 349,776 / 174,888 / 320,628 | 447.97 (all 7) | 448 | 0.001 | pass (all 7) |
| EuroCropsML reconstruction, seeds 11 / 23 / 37 / 53 / 71 / 89 / 97 | 95 / 104 / 132 / 158 / 103 / 81 / 124 | 359,232 / 392,910 / 497,686 / 594,978 / 389,168 / 306,844 / 467,750 | 544.96 (all 7) | 545 | 0.0005 | pass (all 7) |

All 97 checkpoints pass. Reconstruction runs listed without a recipe note use the matched recipe. One matched-recipe
reconstruction run (strict France, seed 42) ran the full 100 epochs without triggering early stopping; all other
reconstruction runs stopped early. Of the masked-prediction runs in the last eight rows, ten ran to the epoch limit (100 epochs on
CropGlobe, 200 on EuroCropsML); the Check column marks them.

## S4. Non-learned controls

### S4.1 Construction and fit pools

**Construction.** The dimensionality-matched control (rounded PCA) projects each month's 10-band vector onto its
first four principal components (fitted without labels) and rounds them onto the same five-level
grid FSQ uses; it differs from the FSQ tokenizer only in that its projection is not learned, and it
is deterministic given its fit data. The vocabulary-matched control (k-means-625) is mini-batch
k-means with 625 centroids (`batch_size=4096`, `n_init=5`, `max_iter=300`) fitted on the months'
four principal-component scores, standardized on the fit pool; in the probe each month is
represented by its centroid, four numbers, like the tokenizer and the rounded PCA. Because the
result depends on centroid initialization, it is fitted five times per fit pool and reported as the
mean; the fit-to-fit spread is at most 0.0016 in accuracy (§S4.4). §S4.4 also reports a k-means
vocabulary fitted on the ten band means, which is not dimensionality-matched.

**Fit pools.** Both controls are fit on exactly the pretraining data the compared encoder saw,
never on labels.

- *CropGlobe, transductive setting:* fit on all 10 country splits, matching the transductive encoder,
  which also sees all 10 during self-supervised pretraining.
- *CropGlobe, strict setting:* fit on **9 splits — every split except the evaluation region** —
  matching the strict encoder for that region. The France-strict control is fit on the 8-country
  probe pool plus Queensland (France excluded); the Queensland-strict control is fit on the
  8-country probe pool plus France (Queensland excluded). These are two separate 9-split fits, one
  per held-out region, not one pool shared between them.
- *EuroCropsML:* both controls are fit on the Latvia+Portugal pretraining pool, matching the
  (strict-by-construction) encoder; there is no transductive/strict distinction to draw here.

Each strict control excludes only its own evaluation region, giving 9 fit splits. France's strict
rounded PCA control scores 0.673 and Queensland's 0.555 (0.677 and 0.559 for the 10-split transductive
fits).

### S4.2 Alternative quantizations of the PCA control

The letter's rounded PCA control standardizes each month's four principal-component scores (zero
mean, unit variance, fitted jointly over the probe's training and evaluation pools), clips them at
±2 standard deviations and rounds to the nearest integer: five levels per component, the same grid
FSQ uses. Two alternatives were compared with it, each fitted on the setting's own pool (§S4.1): the
same standardized scores without rounding (continuous), and an equal-mass grid that cuts each
component at the 20/40/60/80 percentiles of the fit pool.

**Accuracy / macro-F1, logistic probe, harmonized labels:**

| Setting | Continuous PCA | Equal-mass PCA | Rounded PCA (letter) |
|---|---|---|---|
| France, transductive | 0.7032 / 0.6395 | 0.7073 / 0.6431 | 0.6771 / 0.6043 |
| Queensland, transductive | 0.5713 / 0.4394 | 0.5550 / 0.4213 | 0.5587 / 0.4102 |
| France, strict | 0.7036 / 0.6414 | 0.7036 / 0.6389 | 0.6733 / 0.5994 |
| Queensland, strict | 0.5713 / 0.4396 | 0.5569 / 0.4222 | 0.5549 / 0.4049 |
| Estonia | 0.1782 / 0.0567 | 0.1668 / 0.0507 | 0.1611 / 0.0475 |

The continuous scores are a little higher everywhere. The equal-mass grid is within about 0.006 of
the rounded grid in Queensland and Estonia and about 0.030 higher in both France settings, so the
rounded grid is the weakest of the three in France.

**Code usage: joint-code entropy on the evaluation region, in bits** (ceiling log₂ 625 = 9.288).
Masked-prediction tokens are the mean of the ten runs. The ten-band k-means column is one fit (seed 1904) of the k-means fitted on the ten band means (the form of §S4.4); the letter's k-means column is the mean of the five fits of the letter's control (four standardized principal-component scores), recomputed with the same estimator on the same samples (scripts/bits_k2.py; aggregates in results/bits_k2_summary.csv). Single masked-prediction runs reach 8.78 / 8.53 / 8.41 / 8.41 / 8.18 bits per row.

| Setting | Rounded PCA | Equal-mass PCA | k-means-625, ten-band (one fit) | k-means-625, letter's (mean of five fits) | Masked-prediction tokens |
| --- | --- | --- | --- | --- | --- |
| France, transductive | 6.383 | 8.268 | 8.819 | 8.752 | 8.483 |
| Queensland, transductive | 5.770 | 7.911 | 8.423 | 8.337 | 8.154 |
| France, strict | 6.381 | 8.024 | 8.723 | 8.551 | 8.292 |
| Queensland, strict | 5.780 | 7.844 | 8.404 | 8.291 | 8.164 |
| Estonia | 6.066 | 7.590 | 8.383 | 8.342 | 7.829 |

The Estonia rounded-PCA value here (6.066) comes from this table's original arrays; rebuilt with the letter's control pipeline (task_k_readouts.py construction) it is 5.958, the value in the submitted supplement. All Estonia values count all nine months, including the 11.9 % invalid parcel-months (valid months only: letter's k-means 8.887, rounded PCA 6.304).

Uniform rounding uses little of rounded PCA's code space; the equal-mass grid uses 1.5 to 2.1
bits more, closer to k-means and the tokens.

**Masked-prediction tokens against the equal-mass grid**, tests as in §S7.1 (one-sided one-sample
t-test on per-run accuracy margins; p multiplied by the number of analyses of the row, §S7):

| Region, setting | n | Mean margin (acc.) | p (raw) | p (used) | At 0.05 |
| --- | --- | --- | --- | --- | --- |
| France, transductive | 10 | −0.009 | 0.73 | 1.00 | not established |
| Queensland, transductive | 10 | +0.122 | <0.001 | <0.001 | established |
| France, strict | 10 | +0.021 | 0.051 | 0.15 | not established |
| Queensland, strict | 10 | +0.093 | <0.001 | <0.001 | established |
| Estonia | 10 | +0.037 | <0.001 | <0.001 | established |

Against the equal-mass grid, masked prediction stays ahead in accuracy in Queensland and Estonia
(p < 0.001) and is not established in France in either setting (transductive margin −0.009, strict +0.021 with
p = 0.15 after the multiplier). The strict France result therefore depends on how rounded PCA is quantized.

**Standardization fitted on the encoder's data.** The letter's rounded PCA standardizes its four scores on the probe's training and evaluation samples. Refitting that standardization on
the encoder's fit pool instead (the principal components themselves are unchanged) changes one outcome of Table I: strict Queensland, reconstruction against rounded PCA, E → N under the t-test (it
is already N under the sign-flip test, §S7.1b). Every masked-prediction outcome is unchanged.

### S4.3 Controls on the month's full window of observations

The encoder sees each month's individual values (six grid values on CropGlobe, about half of them interpolated; two to ten real observations on EuroCropsML),
while both controls in the main text act on the month's ten band means. To check that this input
difference does not produce the learned tokenizer's advantage, both controls were refitted on the
month's full window: the six grid values concatenated into a
60-dimensional month vector (CropGlobe), or each band linearly interpolated from the parcel's
acquisitions onto six evenly spaced days of the month (EuroCropsML; months with fewer than two
observations keep the placeholder of S2.2). Fit pools, rounding, seeds, probe and tests are those of
the main text; the masked-prediction scores are unchanged.

| Setting | Rounded PCA, means / window | k-means, means / window |
|---|---|---|
| France, transductive | 0.677 / 0.602 | 0.696 / 0.646 |
| Queensland, transductive | 0.559 / 0.520 | 0.566 / 0.550 |
| France, strict | 0.673 / 0.597 | 0.690 / 0.642 |
| Queensland, strict | 0.555 / 0.517 | 0.566 / 0.550 |
| Estonia | 0.161 / 0.159 | 0.169 / 0.163 |

Accuracy; k-means is the mean of five fits, fitted in both versions on four standardized
principal-component scores (of the ten band means, or of the 60-dimensional month vectors).

No control gains from the full window: k-means loses 0.016 to 0.050 and the rounded PCA 0.04 to 0.08 on CropGlobe
(k-means 0.006 and the rounded PCA 0.002 on EuroCropsML). Against the window controls, the outcomes of Table I are
unchanged or stronger: eight of the ten control outcomes are unchanged and two are stronger (France, transductive: against
the rounded PCA S → E and against k-means N → E). Both changes reflect a weaker control, not a stronger tokenizer.
Estonia's outcome against k-means stays E, with margins of +0.040 in accuracy and +0.035 in macro-F1. The tokenizer's advantage
is therefore not explained by its access to the individual observations.

**Missing months in Estonia.** Appending the nine month-validity bits to the features of the rounded PCA, the k-means control and raw bands changes their Estonia scores by at most 0.0011 in
accuracy (rounded PCA 0.162, k-means 0.170, raw bands 0.245) and changes no outcome: masked prediction stays E against both controls (margins +0.041 / +0.031 and +0.034 / +0.027) and
reconstruction stays N. (The recomputed raw-band score without the bits differs from the stored one by 0.0008, beyond the 0.0005 tolerance of this check; no outcome involves raw bands.)

### S4.4 Forms of the k-means control

A k-means vocabulary can be fitted and read in more than one way, and the choice matters for a
linear probe. Seven forms were compared, with the letter's pools, five fits, probe and tests; the
masked-prediction scores are those of the ten runs, except under the one-hot readout, where the tokens are
read as one-hot codes too.

- **Ten-band:** fitted on the ten band means, each month read as its centroid's ten values. Not dimensionality-matched.
- **Ten-band, four numbers:** the same partition, each centroid read as its four
  principal-component scores.
- **Four-dimensional (the letter's control):** fitted on the four standardized principal-component
  scores, each month read as its centroid.
- **One-hot, ten-band partition** and **one-hot, four-dimensional partition:** code identity only,
  a one-hot vector over 625 codes per month for the tokens, the rounded PCA and k-means alike.
- **Gaussian mixture:** a mixture of 625 Gaussian components with full covariances, fitted on the same four standardized
  principal-component scores (a random subsample of 300,000 months per fit); each month is assigned to its most probable component
  and read as the component's mean, four numbers.
- **BIRCH:** a tree clustering of the full fit pool into 625 clusters; each month is read as its cluster's mean, four numbers.
  The Gaussian mixture and BIRCH are fitted five times, like k-means.

Control accuracy / macro-F1 (mean of five fits), masked prediction's margin over it (mean of the ten runs; accuracy / macro-F1), runs above the control
(accuracy / macro-F1, out of 10), and the outcome (Holm within the column over five rows, p multiplied by the number of analyses of the row, §S7):

| Row | Form | Control score | Prediction − control | Runs above | Outcome |
|---|---|---|---|---|---|
| France, transductive | Ten-band | 0.730 / 0.675 | −0.032 / −0.009 | 4 / 4 | N |
| France, transductive | Ten-band, four numbers | 0.684 / 0.616 | +0.015 / +0.051 | 4 / 7 | N |
| France, transductive | Four-dimensional (the letter's control) | 0.696 / 0.627 | +0.002 / +0.039 | 4 / 6 | N |
| France, transductive | One-hot, ten-band partition | 0.738 / 0.686 | −0.003 / +0.029 | 4 / 6 | N |
| France, transductive | One-hot, four-dimensional partition | 0.744 / 0.686 | −0.008 / +0.029 | 4 / 6 | N |
| France, transductive | Gaussian mixture, 625 components | 0.697 / 0.628 | +0.001 / +0.039 | 4 / 6 | N |
| France, transductive | BIRCH, 625 clusters | 0.671 / 0.595 | +0.027 / +0.071 | 6 / 8 | S |
| Queensland, transductive | Ten-band | 0.615 / 0.486 | +0.062 / +0.145 | 10 / 10 | E |
| Queensland, transductive | Ten-band, four numbers | 0.564 / 0.421 | +0.113 / +0.210 | 10 / 10 | E |
| Queensland, transductive | Four-dimensional (the letter's control) | 0.566 / 0.430 | +0.111 / +0.202 | 10 / 10 | E |
| Queensland, transductive | One-hot, ten-band partition | 0.656 / 0.588 | +0.028 / +0.050 | 10 / 10 | E |
| Queensland, transductive | One-hot, four-dimensional partition | 0.646 / 0.560 | +0.038 / +0.078 | 10 / 10 | E |
| Queensland, transductive | Gaussian mixture, 625 components | 0.568 / 0.433 | +0.109 / +0.198 | 10 / 10 | E |
| Queensland, transductive | BIRCH, 625 clusters | 0.557 / 0.411 | +0.120 / +0.220 | 10 / 10 | E |
| France, strict | Ten-band | 0.716 / 0.668 | +0.009 / +0.052 | 7 / 7 | S |
| France, strict | Ten-band, four numbers | 0.674 / 0.605 | +0.050 / +0.115 | 8 / 10 | E |
| France, strict | Four-dimensional (the letter's control) | 0.690 / 0.620 | +0.035 / +0.100 | 7 / 10 | E |
| France, strict | One-hot, ten-band partition | 0.730 / 0.678 | +0.018 / +0.069 | 7 / 10 | S |
| France, strict | One-hot, four-dimensional partition | 0.738 / 0.678 | +0.010 / +0.068 | 7 / 10 | S |
| France, strict | Gaussian mixture, 625 components | 0.690 / 0.619 | +0.035 / +0.101 | 7 / 10 | E |
| France, strict | BIRCH, 625 clusters | 0.665 / 0.588 | +0.059 / +0.132 | 9 / 10 | E |
| Queensland, strict | Ten-band | 0.615 / 0.488 | +0.035 / +0.117 | 8 / 10 | S |
| Queensland, strict | Ten-band, four numbers | 0.566 / 0.422 | +0.084 / +0.183 | 10 / 10 | E |
| Queensland, strict | Four-dimensional (the letter's control) | 0.566 / 0.430 | +0.083 / +0.176 | 10 / 10 | E |
| Queensland, strict | One-hot, ten-band partition | 0.653 / 0.585 | +0.029 / +0.055 | 10 / 10 | E |
| Queensland, strict | One-hot, four-dimensional partition | 0.640 / 0.546 | +0.043 / +0.094 | 10 / 10 | E |
| Queensland, strict | Gaussian mixture, 625 components | 0.569 / 0.433 | +0.081 / +0.172 | 10 / 10 | E |
| Queensland, strict | BIRCH, 625 clusters | 0.559 / 0.414 | +0.091 / +0.192 | 10 / 10 | E |
| Estonia | Ten-band | 0.180 / 0.058 | +0.024 / +0.021 | 10 / 10 | E |
| Estonia | Ten-band, four numbers | 0.168 / 0.052 | +0.036 / +0.027 | 10 / 10 | E |
| Estonia | Four-dimensional (the letter's control) | 0.169 / 0.051 | +0.035 / +0.028 | 10 / 10 | E |
| Estonia | One-hot, ten-band partition | 0.179 / 0.055 | +0.037 / +0.028 | 10 / 10 | E |
| Estonia | One-hot, four-dimensional partition | 0.176 / 0.055 | +0.041 / +0.028 | 10 / 10 | E |
| Estonia | Gaussian mixture, 625 components | 0.167 / 0.051 | +0.036 / +0.028 | 10 / 10 | E |
| Estonia | BIRCH, 625 clusters | 0.163 / 0.048 | +0.041 / +0.031 | 10 / 10 | E |

Estonia and transductive Queensland are established under every form. Strict Queensland is established under every form except the ten-band
vocabulary read as ten numbers (macro-F1 only). Transductive France is not established under any form except BIRCH (macro-F1 only). Strict France
is established in macro-F1 only under the ten-band vocabulary and both one-hot forms, and in both metrics under the other four. A one-hot readout
raises every representation, the tokens included (France, transductive: tokens 0.735, rounded PCA 0.735), and the tokens' outcomes against the rounded
PCA are then those of Table I in every row except strict France, where one metric survives. The Gaussian mixture reproduces the four-dimensional
k-means outcomes in every row; BIRCH, a weaker partition in France, is the only form under which transductive France reaches a outcome above N.
The multiplier of §S7 changes five outcomes of this table, each by one level (without it: France transductive, ten-band with four numbers N → S, both
one-hot forms N → S, BIRCH S → E; strict Queensland, ten-band S → E). Fit-to-fit standard deviation of the letter's control: 0.0006–0.0016 in
accuracy and 0.0010–0.0026 in macro-F1 (ten-band: up to 0.0038 and 0.0057).

### S4.5 A spectral-index control

A dimensionality-matched control built from domain knowledge rather than from the data: four monthly indices computed from the monthly mean reflectance, NDVI (B8, B4), a red-edge index NDRE (B8A, B5),
NDMI (B8, B11) and a short-wave infrared ratio (B11 − B12) / (B11 + B12), each standardized on the encoder's fit pool, clipped at ±2 sd and rounded onto the five-level grid: four numbers per month,
like the tokenizer. Same probe and tests as Table I, as its own Holm family.

| Region, setting | Index control | Masked prediction − index control | Outcome | Reconstruction − index control | Outcome |
|---|---|---|---|---|---|
| France, transductive | 0.631 / 0.561 | +0.067 / +0.106 | E | −0.011 / −0.005 | N |
| Queensland, transductive | 0.568 / 0.497 | +0.109 / +0.135 | E | +0.003 / −0.072 | N |
| France, strict | 0.627 / 0.554 | +0.098 / +0.166 | E | −0.014 / −0.008 | N |
| Queensland, strict | 0.568 / 0.496 | +0.082 / +0.110 | E | +0.003 / −0.070 | N |
| Estonia | 0.151 / 0.041 | +0.053 / +0.038 | E | −0.005 / −0.006 | N |

The index control is weaker than both controls of Table I in France and Estonia, and slightly above both in Queensland (accuracy 0.568 against 0.566 for k-means and 0.559 for the rounded PCA; macro-F1
0.497 against 0.430 and 0.410). In about 1.9% of the valid EuroCropsML months the 20 m or the 10 m bands are zero (§S1.2), so NDRE and the
short-wave infrared ratio, or NDVI, have a vanishing denominator (and NDMI equals +1 where B11 is zero); the Estonia row sets the undefined indices to 0. Treating such a month instead like an invalid month (excluded from the
standardization fit, its four features set to 0) gives an index control of 0.157 / 0.042 and margins of +0.047 / +0.037 (masked prediction, E) and −0.011 / −0.006
(reconstruction, N): the same outcomes, also under the exact sign-flip test.

### S4.6 A probe with Estonian labels (k-shot)

The letter's probe uses no Estonian labels. Here it is fitted on Estonian labels from EuroCropsML's official few-shot split
(k = 1, 5, 10, 20, 100, 200, 500 parcels per class; fixed nested training sets of 123 to 14,179 parcels, one draw per k) and tested on the
official test set of 35,182 parcels, the letter's Estonia evaluation set. Labels are the Estonian vocabulary of 124 classes, without
harmonization. Encoders and controls are unchanged. Tests per k on the ten per-run margins against the fixed control, p × 2, Holm over
the seven k. Accuracy / macro-F1:

| k | Masked prediction | Reconstruction | Rounded PCA | k-means | Raw bands |
|---|---|---|---|---|---|
| 1 | 0.092 / 0.031 | 0.036 / 0.011 | 0.036 / 0.014 | 0.035 / 0.016 | 0.034 / 0.015 |
| 5 | 0.136 / 0.048 | 0.068 / 0.027 | 0.080 / 0.030 | 0.097 / 0.031 | 0.101 / 0.035 |
| 10 | 0.164 / 0.060 | 0.086 / 0.030 | 0.097 / 0.034 | 0.116 / 0.040 | 0.119 / 0.045 |
| 20 | 0.223 / 0.074 | 0.128 / 0.040 | 0.143 / 0.044 | 0.175 / 0.051 | 0.200 / 0.065 |
| 100 | 0.331 / 0.103 | 0.220 / 0.062 | 0.234 / 0.068 | 0.273 / 0.075 | 0.366 / 0.102 |
| 200 | 0.373 / 0.109 | 0.250 / 0.066 | 0.277 / 0.077 | 0.308 / 0.078 | 0.429 / 0.115 |
| 500 | 0.481 / 0.117 | 0.337 / 0.065 | 0.370 / 0.078 | 0.394 / 0.079 | 0.545 / 0.136 |

Masked prediction beats both controls at every k (E, ten of ten runs above, t and sign-flip). Its margins are +0.056 to +0.110 in accuracy
over the rounded PCA and +0.039 to +0.087 over k-means, and grow with k. Reconstruction is N against both controls at every k. Against raw
bands, masked prediction is ahead up to k = 20 (by 0.058, 0.035, 0.045 and 0.023 in accuracy). From k = 100 raw bands are ahead in
accuracy, by 0.035, 0.055 and 0.065; in macro-F1 they are level at k = 100 and ahead from k = 200 (+0.006, +0.020).

**For scale, not a comparison of methods.** On the same official test set, Reuss et al. [17] pretrain a
Transformer with crop labels on Latvia and Portugal, fine-tune it with Estonian labels over 127 classes,
and reach overall accuracies of 0.08 to 0.16 with one labelled parcel per class and 0.58 to 0.64 with 500.
Here, frozen masked-prediction tokens read by a logistic probe score 0.092 at k = 1 and 0.481 at k = 500,
over 124 classes, from an encoder pretrained without labels. The protocols differ in pretraining labels,
fine-tuning and class list (S10).

**Beside: our earlier random draws** (k Estonian parcels per class drawn by us from the 83 harmonized classes, k = 1, 5, 10, 20, five
draws, tested on the other Estonian parcels; tests on the ten per-run margins averaged over the draws, p × 2, Holm over the four k):

| k | Masked prediction | Reconstruction | Rounded PCA | k-means | Raw bands |
|---|---|---|---|---|---|
| 1 | 0.094 / 0.034 | 0.056 / 0.016 | 0.051 / 0.020 | 0.058 / 0.022 | 0.055 / 0.022 |
| 5 | 0.168 / 0.063 | 0.082 / 0.028 | 0.100 / 0.039 | 0.129 / 0.043 | 0.125 / 0.044 |
| 10 | 0.193 / 0.074 | 0.101 / 0.035 | 0.128 / 0.049 | 0.141 / 0.053 | 0.151 / 0.058 |
| 20 | 0.222 / 0.081 | 0.134 / 0.043 | 0.158 / 0.054 | 0.179 / 0.060 | 0.205 / 0.071 |

With these draws, masked prediction also beats both controls at every k (E, ten of ten runs above, t and sign-flip; margins +0.044 to
+0.068 in accuracy over the rounded PCA, +0.036 to +0.052 over k-means). Reconstruction is N against k-means at every k and S against the
rounded PCA at k = 1 only. Raw bands are below masked prediction at every k, by 0.039, 0.043, 0.042 and 0.017 in accuracy.

### S4.7 Without label harmonization

The probe is fitted on all training classes and scored on the evaluation region's own labels, without merging absent classes into the residual class.

| Region, setting | Masked prediction | Reconstruction | Rounded PCA | k-means | Raw bands |
|---|---|---|---|---|---|
| France, transductive | 0.699 / 0.488 | 0.641 / 0.424 | 0.683 / 0.444 | 0.700 / 0.459 | 0.800 / 0.575 |
| Queensland, transductive | 0.467 / 0.244 | 0.300 / 0.156 | 0.236 / 0.127 | 0.229 / 0.124 | 0.410 / 0.234 |
| France, strict | 0.729 / 0.526 | 0.635 / 0.419 | 0.680 / 0.441 | 0.695 / 0.454 | 0.800 / 0.575 |
| Queensland, strict | 0.497 / 0.251 | 0.292 / 0.153 | 0.235 / 0.125 | 0.229 / 0.124 | 0.410 / 0.234 |
| Estonia | 0.204 / 0.069 | 0.146 / 0.033 | 0.162 / 0.043 | 0.169 / 0.046 | 0.247 / 0.084 |

Accuracy rises slightly in France, falls strongly in Queensland (to 0.23 to 0.50, from 0.56 to 0.71) and barely moves in Estonia; macro-F1 falls everywhere. Every
masked-prediction outcome of Table I is unchanged. Three reconstruction outcomes change from N to E, all in Queensland: against the rounded PCA and k-means in the
transductive setting and against k-means in the strict setting. We read this as an effect of scoring a probe that predicts classes absent from the region, which
costs the controls more than the encoders, not as evidence that reconstruction works there; we did not test that reading.

## S5. Full per-seed results and standard deviations

### S5.1 Seed-stability table

All checkpoints below use the fixed 8-country probe pool, logistic probe, harmonized labels. Every CropGlobe masked-prediction run uses all 625 codes. One EuroCropsML
masked-prediction run (seed 23) maps the probe pool onto 308 codes (§S2.3); it is kept in every row, and its removal would change the Estonia mean accuracy by 0.001.

| Representation | Region | n | Seeds | Accuracy mean ± sd | Macro-F1 mean ± sd |
|---|---|---|---|---|---|
| Masked prediction, transductive | France | 10 | 1904, 2024, 4242, 11, 23, 37, 53, 71, 89, 97 | 0.6983 ± 0.0444 | 0.6667 ± 0.0808 |
| Masked prediction, transductive | Queensland | 10 | 1904, 2024, 4242, 11, 23, 37, 53, 71, 89, 97 | 0.6770 ± 0.0327 | 0.6314 ± 0.0580 |
| Masked prediction, strict | France | 10 | 1904, 2024, 4242, 42, 101, 11, 23, 37, 53, 71 | 0.7246 ± 0.0364 | 0.7200 ± 0.0492 |
| Masked prediction, strict | Queensland | 10 | 1904, 2024, 4242, 42, 101, 11, 23, 37, 53, 71 | 0.6497 ± 0.0410 | 0.6054 ± 0.0592 |
| Masked prediction, strict (EuroCropsML) | Estonia | 10 | 1904, 2024, 4242, 11, 23, 37, 53, 71, 89, 97 | 0.2035 ± 0.0055 | 0.0791 ± 0.0060 |

Per-run accuracy of the nine low France runs, lowest first: transductive seed 71 0.6563, transductive seed 11 0.6581, transductive seed 97 0.6590, strict seed 101 0.6638, transductive seed 1904 0.6665, transductive seed 23 0.6713, strict seed 4242 0.6730, transductive seed 37 0.6766, strict seed 71 0.6842. A run is called low if its token accuracy is not above the k-means control of its setting (0.696
transductive, 0.690 strict); the letter's "nine of twenty France runs" refers to exactly these.

**What separates the low France runs.** Epoch of best validation loss, the loss itself, per-dimension code entropy and the probe's own training accuracy on the 8-split pool,
for every France run:

| Run | Epoch | Best val. loss | Mean entropy | Train acc. (8-split pool) | Accuracy |
|---|---|---|---|---|---|
| Transductive, seed 53 | 50 | 2.6644 | 2.2001 | 0.8020 | 0.7580 (strong) |
| Strict, seed 37 | 99 | 2.5874 | 2.1741 | 0.8017 | 0.7571 (strong) |
| Transductive, seed 2024 | 85 | 2.6489 | 2.1300 | 0.8069 | 0.7557 (strong) |
| Strict, seed 2024 | 84 | 2.6118 | 2.1558 | 0.7984 | 0.7536 (strong) |
| Strict, seed 1904 | 65 | 2.6409 | 2.1745 | 0.8069 | 0.7504 (strong) |
| Transductive, seed 89 | 67 | 2.6438 | 2.1749 | 0.7943 | 0.7495 (strong) |
| Strict, seed 53 | 36 | 2.6210 | 2.1746 | 0.7992 | 0.7474 (strong) |
| Strict, seed 42 | 98 | 2.6411 | 2.1802 | 0.8000 | 0.7465 (strong) |
| Strict, seed 23 | 63 | 2.6672 | 2.1874 | 0.7984 | 0.7432 (strong) |
| Transductive, seed 4242 | 59 | 2.6659 | 2.1347 | 0.8008 | 0.7322 (strong) |
| Strict, seed 11 | 74 | 2.7329 | 2.2372 | 0.8024 | 0.7266 (strong) |
| Strict, seed 71 | 89 | 2.8625 | 2.2908 | 0.7523 | 0.6842 (low) |
| Transductive, seed 37 | 98 | 2.6889 | 2.2260 | 0.7731 | 0.6766 (low) |
| Strict, seed 4242 | 96 | 2.7397 | 2.2173 | 0.7535 | 0.6730 (low) |
| Transductive, seed 23 | 91 | 2.8482 | 2.2490 | 0.7704 | 0.6713 (low) |
| Transductive, seed 1904 | 97 | 2.8095 | 2.2247 | 0.7765 | 0.6665 (low) |
| Strict, seed 101 | 95 | 2.6524 | 2.1970 | 0.7470 | 0.6638 (low) |
| Transductive, seed 97 | 99 | 2.5990 | 2.1943 | 0.7594 | 0.6590 (low) |
| Transductive, seed 11 | 97 | 2.8411 | 2.2586 | 0.7779 | 0.6581 (low) |
| Transductive, seed 71 | 67 | 2.7436 | 2.2297 | 0.7506 | 0.6563 (low) |

The probe's training accuracy on the pool separates the nine low runs (0.747–0.778) from the eleven others (0.794–0.807); the ranges do not overlap, and the Spearman correlation
between training-pool accuracy and held-out accuracy is +0.78 over the twenty runs. Per-dimension entropy does not separate them (2.194 to 2.291 against 2.130 to 2.237). The best validation
epoch does: eight of the nine low runs reached it at epoch 89 or later of the 100-epoch cap, against two of the eleven others (Fisher exact p = 0.0055), and their best validation loss is higher by
rank (Mann–Whitney p = 0.0098; low runs 2.60 to 2.86, other runs 2.59 to 2.73).

**Continuing the capped runs.** To test whether the cap explains the low runs, the 18 CropGlobe masked-prediction runs that stopped at the 100-epoch cap, eight of the nine low runs among them, and
transductive seed 71 were continued with the same recipe up to 200 epochs, each resumed from its best checkpoint. All 19 stopped by early stopping before 200 epochs, and 16 reached a lower validation
loss (mean −0.020). None of the nine low runs then exceeds its k-means control (0.696 transductive, 0.690 strict); their mean France accuracy and macro-F1 move from 0.668 / 0.623 to 0.667 / 0.623.
With the continued runs every Table I margin moves by at most 0.003. Under the t-test one outcome changes: strict France, masked prediction against k-means, E → S (accuracy Holm-adjusted p 0.045 →
0.066; macro-F1 still survives). Under the sign-flip test (§S7.1) nothing changes with the best checkpoints; with the final-epoch checkpoints, transductive France against the rounded PCA goes from S to
N (macro-F1 Holm-adjusted p 0.047 → 0.051). The two capped Estonian masked-prediction runs (seeds 4242 and 97) were continued the same way, up to 600 epochs: both stopped by early stopping, and all six Estonian margins of Table I are identical to three decimals. The letter's tables keep the original runs. The cap is therefore not the cause of the low runs, and the cause remains untested.

**Per-class F1 in France.** Classes (harmonized labels; share of France pixels): corn 0.23, soybeans 0.19, rice 0.02, wheat 0.26, other 0.30. Mean F1 of the strong runs, the low runs and the
k-means control (five fits):

| Setting | Class | Strong runs | Low runs | k-means | Share of the strong − low macro-F1 gap |
|---|---|---|---|---|---|
| Transductive (4 strong, 6 low) | Corn | 0.824 | 0.760 | 0.752 | 0.09 |
| | Soybeans | 0.567 | 0.317 | 0.466 | 0.35 |
| | Rice | 0.807 | 0.525 | 0.376 | 0.40 |
| | Wheat | 0.867 | 0.828 | 0.873 | 0.05 |
| | Other | 0.694 | 0.618 | 0.669 | 0.11 |
| Strict (7 strong, 3 low) | Corn | 0.824 | 0.711 | 0.750 | 0.23 |
| | Soybeans | 0.544 | 0.529 | 0.446 | 0.03 |
| | Rice | 0.819 | 0.588 | 0.367 | 0.46 |
| | Wheat | 0.871 | 0.810 | 0.872 | 0.12 |
| | Other | 0.692 | 0.613 | 0.663 | 0.16 |

The low runs lose most on rice in both settings, then on soybeans (transductive) or corn (strict). Even the low runs stay well above k-means on rice; the strong runs are level with k-means on
wheat. Queensland's classes are sugarcane (5,098 pixels), cotton (4,825) and other (8,938); Estonia has 83 classes, 82 shared with the Latvia and Portugal pool.

### S5.2 Scores behind Table II — standard deviations

Same protocol, runs and source data as Table II in the letter; mean over ten runs ± sd (n − 1).

| Region | Objective | (b) continuous acc. sd | (b) continuous F1 sd | (c) tokens acc. sd | (c) tokens F1 sd |
|---|---|---|---|---|---|
| France | Reconstruction | 0.0118 | 0.0165 | 0.0103 | 0.0148 |
| France | Masked prediction | 0.0458 | 0.0822 | 0.0444 | 0.0808 |
| Queensland | Reconstruction | 0.0167 | 0.0214 | 0.0199 | 0.0257 |
| Queensland | Masked prediction | 0.0324 | 0.0588 | 0.0327 | 0.0580 |
| Estonia | Reconstruction | 0.0020 | 0.0014 | 0.0016 | 0.0011 |
| Estonia | Masked prediction | 0.0064 | 0.0066 | 0.0055 | 0.0060 |

Masked prediction's France spread is the largest cell in the table by a wide margin (macro-F1 sd 0.082 for the continuous latent, 0.081 for tokens) — this is the same run-to-run split documented in §S5.1, not a
separate finding.

**Table II in the strict CropGlobe settings.** Same protocol, runs and raw bands as Table II; accuracy / macro-F1, mean over ten runs. (b) − raw is the width plus the training step (b) − (a) (the twins (a) of the strict settings are in §S5.4); rounding is (c) − (b).

| Region, setting | Objective | Raw bands | (b) continuous, d = 4 | (c) tokens | (b) − raw | Rounding |
|---|---|---|---|---|---|---|
| France, strict | Reconstruction | 0.782 / 0.791 | 0.628 / 0.564 | 0.613 / 0.545 | −0.154 / −0.226 | −0.014 / −0.019 |
| France, strict | Masked prediction | 0.782 / 0.791 | 0.734 / 0.734 | 0.725 / 0.720 | −0.047 / −0.057 | −0.010 / −0.014 |
| Queensland, strict | Reconstruction | 0.707 / 0.636 | 0.575 / 0.432 | 0.571 / 0.426 | −0.132 / −0.204 | −0.005 / −0.006 |
| Queensland, strict | Masked prediction | 0.707 / 0.636 | 0.652 / 0.608 | 0.650 / 0.605 | −0.055 / −0.029 | −0.003 / −0.002 |

Standard deviations (n − 1) and the rounding cost per run (mean ± sd):

| Region, setting | Objective | (b) continuous acc. sd | (b) continuous F1 sd | (c) tokens acc. sd | (c) tokens F1 sd | Rounding acc. | Rounding F1 |
|---|---|---|---|---|---|---|---|
| France, strict | Reconstruction | 0.0089 | 0.0118 | 0.0098 | 0.0102 | −0.0144 ± 0.0046 | −0.0191 ± 0.0060 |
| France, strict | Masked prediction | 0.0351 | 0.0450 | 0.0364 | 0.0492 | −0.0097 ± 0.0029 | −0.0140 ± 0.0046 |
| Queensland, strict | Reconstruction | 0.0140 | 0.0169 | 0.0140 | 0.0174 | −0.0047 ± 0.0019 | −0.0062 ± 0.0030 |
| Queensland, strict | Masked prediction | 0.0405 | 0.0598 | 0.0410 | 0.0592 | −0.0027 ± 0.0026 | −0.0021 ± 0.0042 |

The largest rounding cost over all rows of Table II and this table is 0.019 (strict France, reconstruction, macro-F1): the "at most 0.019" of the letter.

**The learned continuous latent against continuous PCA.** The principal-component control of Table I, standardized as in the letter but neither clipped nor rounded, against the learned
continuous latent (d = 4) of each run; tests as Table I (t-test and sign-flip give the same outcomes here). Rounding costs the principal components 0.013 to 0.030 in accuracy (continuous minus rounded:
France 0.026 / 0.030, Queensland 0.013 / 0.016, Estonia 0.017), more than rounding costs the trained latents in every row.

| Region, setting | Continuous PCA | Masked prediction − rounded PCA | Outcome | Reconstruction − rounded PCA | Outcome |
|---|---|---|---|---|---|
| France, transductive | 0.703 / 0.639 | +0.002 / +0.037 (4 / 6 runs above) | N | −0.068 / −0.067 | N |
| Queensland, transductive | 0.571 / 0.439 | +0.105 / +0.189 (10 / 10) | E | +0.003 / −0.009 | N |
| France, strict | 0.704 / 0.641 | +0.031 / +0.093 (7 / 10) | S | −0.076 / −0.077 | N |
| Queensland, strict | 0.571 / 0.440 | +0.081 / +0.168 (10 / 10) | E | +0.004 / −0.007 | N |
| Estonia | 0.178 / 0.057 | +0.032 / +0.027 (10 / 10) | E | −0.027 / −0.019 | N |

### S5.3 Per-seed values underlying Figure 1

Token-level accuracy / macro-F1, logistic probe, harmonized labels — every dot plotted in the letter's Figure 1, plus each panel's control lines. Runs the letter calls out as "low" France runs (§S5.1) are marked with †. A masked-prediction and a reconstruction run with the same seed number share a row.

**France, transductive**

| Seed | Masked prediction (acc / F1) | Reconstruction (acc / F1) |
|---|---|---|
| 1904 † | 0.6665 / 0.5622 | 0.6258 / 0.5516 |
| 2024 | 0.7557 / 0.7608 | — |
| 4242 | 0.7322 / 0.7290 | — |
| 42 | — | 0.6284 / 0.5732 |
| 101 | — | 0.6088 / 0.5433 |
| 11 † | 0.6581 / 0.5594 | 0.6189 / 0.5491 |
| 23 † | 0.6713 / 0.6030 | 0.6222 / 0.5644 |
| 37 † | 0.6766 / 0.6720 | 0.6144 / 0.5461 |
| 53 | 0.7580 / 0.7618 | 0.6111 / 0.5478 |
| 71 † | 0.6563 / 0.6352 | 0.6442 / 0.5865 |
| 89 | 0.7495 / 0.7560 | 0.6192 / 0.5401 |
| 97 † | 0.6590 / 0.6272 | 0.6153 / 0.5495 |

Controls: rounded PCA 0.6771 / 0.6043; k-means-625 0.6960 / 0.6274 (ten-band form: 0.7300 / 0.6752); raw bands 0.7817 / 0.7907; majority class 0.3012 / 0.0926.

**Queensland, transductive**

| Seed | Masked prediction (acc / F1) | Reconstruction (acc / F1) |
|---|---|---|
| 1904 | 0.6624 / 0.6331 | 0.5298 / 0.3770 |
| 2024 | 0.6960 / 0.6669 | — |
| 4242 | 0.7327 / 0.7148 | — |
| 42 | — | 0.5638 / 0.4177 |
| 101 | — | 0.5794 / 0.4351 |
| 11 | 0.6557 / 0.5633 | 0.5777 / 0.4407 |
| 23 | 0.6325 / 0.5288 | 0.5635 / 0.4182 |
| 37 | 0.6691 / 0.6435 | 0.5713 / 0.4299 |
| 53 | 0.7095 / 0.6850 | 0.5890 / 0.4485 |
| 71 | 0.6324 / 0.5761 | 0.5942 / 0.4513 |
| 89 | 0.6844 / 0.6530 | 0.5498 / 0.3860 |
| 97 | 0.6949 / 0.6494 | 0.5897 / 0.4454 |

Controls: rounded PCA 0.5587 / 0.4102; k-means-625 0.5662 / 0.4297 (ten-band form: 0.6154 / 0.4863); raw bands 0.7074 / 0.6363; majority class 0.4739 / 0.2143.

**France, strict**

| Seed | Masked prediction (acc / F1) | Reconstruction (acc / F1) |
|---|---|---|
| 1904 | 0.7504 / 0.7515 | 0.6195 / 0.5377 |
| 2024 | 0.7536 / 0.7618 | — |
| 4242 † | 0.6730 / 0.6552 | — |
| 42 | 0.7465 / 0.7536 | 0.6128 / 0.5423 |
| 101 † | 0.6638 / 0.6481 | 0.6216 / 0.5504 |
| 11 | 0.7266 / 0.7230 | 0.5987 / 0.5239 |
| 23 | 0.7432 / 0.7486 | 0.6142 / 0.5447 |
| 37 | 0.7571 / 0.7598 | 0.6073 / 0.5483 |
| 53 | 0.7474 / 0.7504 | 0.5994 / 0.5449 |
| 71 † | 0.6842 / 0.6477 | 0.6240 / 0.5593 |
| 89 | — | 0.6078 / 0.5430 |
| 97 | — | 0.6265 / 0.5586 |

Controls: rounded PCA 0.6733 / 0.5994; k-means-625 0.6900 / 0.6196 (ten-band form: 0.7159 / 0.6678); raw bands 0.7817 / 0.7907; majority class 0.3012 / 0.0926.

**Queensland, strict**

| Seed | Masked prediction (acc / F1) | Reconstruction (acc / F1) |
|---|---|---|
| 1904 | 0.6078 / 0.5360 | 0.5739 / 0.4302 |
| 2024 | 0.6924 / 0.6599 | — |
| 4242 | 0.6058 / 0.5333 | — |
| 42 | 0.7053 / 0.6807 | 0.5590 / 0.4239 |
| 101 | 0.6426 / 0.6221 | 0.5381 / 0.3846 |
| 11 | 0.6636 / 0.6382 | 0.5778 / 0.4377 |
| 23 | 0.6205 / 0.5601 | 0.5872 / 0.4466 |
| 37 | 0.6288 / 0.5948 | 0.5831 / 0.4379 |
| 53 | 0.7128 / 0.6822 | 0.5747 / 0.4298 |
| 71 | 0.6177 / 0.5468 | 0.5659 / 0.4197 |
| 89 | — | 0.5698 / 0.4157 |
| 97 | — | 0.5775 / 0.4366 |

Controls: rounded PCA 0.5549 / 0.4049; k-means-625 0.5662 / 0.4299 (ten-band form: 0.6146 / 0.4881); raw bands 0.7074 / 0.6363; majority class 0.4739 / 0.2143.

**Estonia**

| Seed | Masked prediction (acc / F1) | Reconstruction (acc / F1) |
|---|---|---|
| 1904 | 0.2020 / 0.0812 | 0.1473 / 0.0362 |
| 2024 | 0.2053 / 0.0772 | — |
| 4242 | 0.2086 / 0.0785 | — |
| 42 | — | 0.1453 / 0.0343 |
| 101 | — | 0.1450 / 0.0339 |
| 11 | 0.2063 / 0.0875 | 0.1481 / 0.0362 |
| 23 | 0.1908 / 0.0644 | 0.1432 / 0.0333 |
| 37 | 0.2059 / 0.0832 | 0.1470 / 0.0348 |
| 53 | 0.2014 / 0.0779 | 0.1469 / 0.0354 |
| 71 | 0.2008 / 0.0807 | 0.1480 / 0.0369 |
| 89 | 0.2110 / 0.0819 | 0.1469 / 0.0359 |
| 97 | 0.2031 / 0.0783 | 0.1446 / 0.0349 |

Controls: rounded PCA 0.1611 / 0.0475; k-means-625 0.1687 / 0.0513 (ten-band form: 0.1796 / 0.0577); raw bands 0.2461 / 0.0975; majority class 0.0764 / 0.0017.

### S5.4 Tokens against models trained without the quantizer

Referenced from Table II and Sec. IV-A. Rounding a trained latent measures what rounding costs that latent, not what discretization costs the model. Each comparison, per objective at d = 4, sets the FSQ-trained model beside a model trained without the quantizer, with the same early-stopping patience and ten seeds (caps: see "Training budget" below), probed as in the letter (logistic probe, harmonized labels, its continuous latent of four numbers per month). For reconstruction that model is the same encoder, so (c) − (a) is the cost of discretization. For masked prediction the quantizer defines the target, so the comparison is exploratory.

- (a) trained without FSQ, continuous latent. Reconstruction: the same encoder and objective without the quantizer (the only exact twin). Masked prediction predicts the code of the masked month, so its quantizer defines the target and no twin with the same objective exists. Its main twin, the slow-teacher twin (`twin_teacher`), predicts the masked month's continuous latent from a slowly updated (exponential-moving-average) copy of the encoder, the closest continuous counterpart; the band-value twin (`twin_bands`), which predicts the masked month's band values, is reported beside.
- (b) trained through FSQ, latent before rounding.
- (c) trained through FSQ, tokens.

(c) − (a) is Table II's middle column. Diagnostics: (b) − (a), what training through the quantizer does to the continuous latent (the training step: (c) − (a) minus the rounding), and (c) − (b), Table II's rounding. Test: two-sided Welch t-test, ten runs against ten, of (c) against (a) and, as a separate family, of (b) against (a) (`results/table2_decomposition/twin_step_tests.csv`; continued twins: `results/table2_decomposition/teacher_twin_continued_step_tests.csv`, `results/table2_decomposition/bands_twin_continued_step_tests.csv`); Holm's correction over the five rows, per objective, twin, comparison and metric. The tests are post hoc (§S8, deviation 7). Bold: Holm p < 0.05 for that comparison and metric; a significant step need not be large (reconstruction's training step in Estonia is +0.003 / +0.002). Post-hoc rounding of (a) (clip at ±2, round to the five-level grid) is a diagnostic only. Accuracy / macro-F1; per-seed range in accuracy.

**Reconstruction, (a) = encoder trained without the quantizer.**

| Row | (a) continuous (sd; per seed) | (b) | (c) | (c) − (a) | Welch p | Holm p | (b) − (a) | (b) − (a) Welch p | (b) − (a) Holm p | (c) − (b) | Rounding of (a): mean; per seed |
|---|---|---|---|---|---|---|---|---|---|---|---|
| France, transductive | 0.664 / 0.614 (0.004 / 0.005; 0.659–0.671) | 0.635 / 0.572 | 0.621 / 0.555 | **−0.043 / −0.059** | <0.001 / <0.001 | <0.001 / <0.001 | **−0.029 / −0.042** | <0.001 / <0.001 | <0.001 / <0.001 | −0.015 / −0.017 | −0.071 / −0.093; −0.135 to −0.011 |
| Queensland, transductive | 0.563 / 0.406 (0.006 / 0.009; 0.557–0.579) | 0.574 / 0.431 | 0.571 / 0.425 | +0.008 / +0.019 | 0.27 / 0.045 | 0.27 / 0.091 | +0.011 / **+0.025** | 0.082 / 0.005 | 0.082 / 0.005 | −0.003 / −0.006 | −0.055 / −0.087; −0.113 to +0.024 |
| France, strict | 0.663 / 0.612 (0.009 / 0.012; 0.645–0.675) | 0.628 / 0.564 | 0.613 / 0.545 | **−0.050 / −0.067** | <0.001 / <0.001 | <0.001 / <0.001 | **−0.036 / −0.048** | <0.001 / <0.001 | <0.001 / <0.001 | −0.014 / −0.019 | −0.066 / −0.086; −0.110 to −0.021 |
| Queensland, strict | 0.559 / 0.400 (0.005 / 0.007; 0.551–0.565) | 0.575 / 0.432 | 0.571 / 0.426 | +0.011 / **+0.027** | 0.035 / <0.001 | 0.11 / 0.002 | **+0.016 / +0.033** | 0.006 / <0.001 | 0.011 / <0.001 | −0.005 / −0.006 | −0.024 / −0.036; −0.098 to +0.046 |
| Estonia | 0.148 / 0.036 (0.001 / 0.001; 0.146–0.150) | 0.151 / 0.037 | 0.146 / 0.035 | −0.001 / 0.000 | 0.052 / 0.40 | 0.11 / 0.40 | **+0.003 / +0.002** | <0.001 / 0.002 | 0.002 / 0.004 | −0.005 / −0.002 | −0.024 / −0.012; −0.040 to −0.016 |

**Masked prediction, (a) = `twin_teacher` (continuous target from a slowly updated copy of the encoder), each run at its latest finished stage.**

| Row | (a) continuous (sd; per seed) | (b) | (c) | (c) − (a) | Welch p | Holm p | (b) − (a) | (b) − (a) Welch p | (b) − (a) Holm p | (c) − (b) | Rounding of (a): mean; per seed |
|---|---|---|---|---|---|---|---|---|---|---|---|
| France, transductive | 0.707 / 0.682 (0.038 / 0.062; 0.669–0.764) | 0.705 / 0.677 | 0.698 / 0.667 | −0.008 / −0.016 | 0.66 / 0.64 | 1 / 1 | −0.002 / −0.006 | 0.92 / 0.87 | 1 / 1 | −0.007 / −0.010 | −0.068 / −0.102; −0.138 to −0.035 |
| Queensland, transductive | 0.582 / 0.446 (0.033 / 0.053; 0.541–0.632) | 0.676 / 0.628 | 0.677 / 0.631 | **+0.095 / +0.186** | <0.001 / <0.001 | <0.001 / <0.001 | **+0.093 / +0.183** | <0.001 / <0.001 | <0.001 / <0.001 | +0.001 / +0.003 | −0.007 / −0.008; −0.030 to +0.014 |
| France, strict | 0.726 / 0.718 (0.036 / 0.061; 0.660–0.761) | 0.734 / 0.734 | 0.725 / 0.720 | −0.001 / +0.002 | 0.95 / 0.93 | 1 / 1 | +0.009 / +0.016 | 0.60 / 0.51 | 1 / 1 | −0.010 / −0.014 | −0.073 / −0.097; −0.109 to −0.030 |
| Queensland, strict | 0.583 / 0.448 (0.031 / 0.045; 0.546–0.629) | 0.652 / 0.608 | 0.650 / 0.605 | **+0.067 / +0.157** | <0.001 / <0.001 | 0.002 / <0.001 | **+0.070 / +0.159** | <0.001 / <0.001 | 0.002 / <0.001 | −0.003 / −0.002 | −0.016 / −0.026; −0.042 to +0.031 |
| Estonia | 0.222 / 0.089 (0.009 / 0.007; 0.198–0.230) | 0.210 / 0.084 | 0.204 / 0.079 | **−0.019 / −0.010** | <0.001 / 0.003 | <0.001 / 0.008 | **−0.012** / −0.005 | 0.003 / 0.12 | 0.009 / 0.36 | −0.006 / −0.005 | −0.040 / −0.028; −0.045 to −0.036 |

**Masked prediction beside, (a') = `twin_bands` (band-value target), continued to early stopping.**

| Row | (a') continuous (sd; per seed) | (c) − (a') | Welch p | Holm p | (b) − (a') | (b) − (a') Welch p | (b) − (a') Holm p | Rounding of (a'): mean; per seed |
|---|---|---|---|---|---|---|---|---|
| France, transductive | 0.693 / 0.642 (0.012 / 0.011; 0.663–0.702) | +0.005 / +0.025 | 0.74 / 0.36 | 0.74 / 0.72 | +0.012 / +0.035 | 0.46 / 0.22 | 0.46 / 0.22 | −0.055 / −0.070; −0.069 to −0.040 |
| Queensland, transductive | 0.611 / 0.477 (0.011 / 0.013; 0.586–0.622) | **+0.066 / +0.155** | <0.001 / <0.001 | <0.001 / <0.001 | **+0.065 / +0.152** | <0.001 / <0.001 | <0.001 / <0.001 | −0.028 / −0.033; −0.043 to −0.006 |
| France, strict | 0.672 / 0.638 (0.011 / 0.014; 0.657–0.687) | **+0.053 / +0.082** | 0.001 / <0.001 | 0.005 / 0.001 | **+0.062 / +0.096** | <0.001 / <0.001 | <0.001 / <0.001 | −0.047 / −0.060; −0.060 to −0.029 |
| Queensland, strict | 0.609 / 0.475 (0.009 / 0.011; 0.598–0.622) | **+0.041 / +0.130** | 0.012 / <0.001 | 0.037 / <0.001 | **+0.043 / +0.132** | 0.008 / <0.001 | 0.024 / <0.001 | −0.028 / −0.037; −0.050 to −0.008 |
| Estonia | 0.207 / 0.078 (0.001 / 0.002; 0.205–0.209) | −0.003 / +0.001 | 0.088 / 0.57 | 0.18 / 0.72 | +0.003 / **+0.006** | 0.20 / 0.018 | 0.39 / 0.037 | −0.027 / −0.019; −0.029 to −0.024 |

**Reading.** Rounding alone, (c) − (b), stays within 0.015 in accuracy and 0.019 in macro-F1 in every row; the large effects, in both directions, arise in training through the quantizer, (b) − (a). Discretization costs reconstruction 0.043 to 0.050 in accuracy in France, about three times its rounding, and nothing elsewhere in accuracy (strict Queensland gains 0.027 in macro-F1). For masked prediction the comparison is exploratory: against `twin_teacher` the tokens are level in France, lose 0.019 / 0.010 in Estonia and gain 0.067 to 0.095 / 0.157 to 0.186 in Queensland. The twin's target matters in two rows: against `twin_bands`, strict France becomes a gain and Estonia becomes level.

**Where the training step and the full cost differ.** In six cells (b) − (a) and (c) − (a) get different outcomes, because rounding offsets or adds to a small training step: reconstruction in transductive Queensland, macro-F1 (step +0.025, Holm 0.005; cost +0.019, Holm 0.091); reconstruction in strict Queensland, accuracy (+0.016, 0.011; +0.011, 0.11); reconstruction in Estonia, both metrics (+0.003 / +0.002, 0.002 / 0.004; −0.001 / 0.000, 0.11 / 0.40); masked prediction against `twin_teacher` in Estonia, macro-F1 (−0.005, 0.36; −0.010, 0.008); and against `twin_bands` in Estonia, macro-F1 (+0.006, 0.037; +0.001, 0.72).

**Estonia, masked prediction.** Nine of the ten `twin_teacher` runs (0.219–0.230 in accuracy) score above every token run (0.191–0.211), in both metrics; the tenth (seed 89, 0.198 / 0.072) is `twin_teacher`'s lowest, and without it the gap widens to −0.021 / −0.012. Mann–Whitney, two-sided: p = 0.002 in both metrics. Every Estonian score here lies between 0.146 and 0.222 in accuracy, against raw bands 0.246 and the majority class 0.076.

**Stability.** No twin collapses: the effective rank of the latent (1 to 4) on the probe pool is 3.37–3.91 for `twin_teacher`, 3.56–3.86 for `twin_bands` and 2.88–3.58 for the reconstruction twin over all fifty runs of each. `twin_teacher`'s accuracy sd (0.038, 0.033, 0.036, 0.031, 0.009 in the order of the tables) is 2.9 to 8.1 times `twin_bands`'s and close to that of the FSQ-trained masked-prediction tokens (0.044, 0.033, 0.036, 0.041, 0.006). No run is dropped.

**Training budget.** Both sides started with the same cap and patience (CropGlobe 100 epochs and 15, EuroCropsML 200 and 25). Reconstruction runs stop early (79 of 80). In masked prediction the cap ends 20 of 40 FSQ-trained, 32 `twin_bands` and 37 `twin_teacher` runs (16, 26 and 37 with their best epoch in the last 10). On EuroCropsML, three of the ten `twin_bands` runs stopped early, at epochs 179, 197 and 186 (seeds 37, 89 and 97; 0-based); the other seven reached the 200-epoch cap. The 32 capped `twin_bands` twins were then continued to early stopping (all stopped before a tripled cap), and no outcome changed; the `twin_bands` table above shows the continued runs. The 20 capped FSQ-trained runs were likewise resumed from their best checkpoint and trained to early stopping (§S5.1); the only outcome change is strict France, masked prediction − k-means, E → S. The 37 capped `twin_teacher` twins were continued up to four times the cap (27 stopped early; 10 ended at a cap, 2 of them with their best epoch in the last 10): validation loss fell, probe scores barely moved and no outcome changed. The `twin_teacher` table shows them at their latest finished stage; the three strict-Queensland runs that had stopped early (seeds 11, 71 and 101) keep their original values. The twin tables print best-checkpoint values, as for the original twins; final-checkpoint values lead to the same outcomes. Setting the continued FSQ-trained runs, instead of the original ones printed here, against the continued `twin_teacher` twins moves (c) − (a) by at most 0.003 and changes no sign or Holm outcome, with best or final checkpoints (`results/table2_decomposition/continued_fsq_vs_continued_teacher_twin.csv`).

## S6. Rounding curve and grid-distance control

### S6.1 Rounding curve

Protocol for every row below: harmonized labels, logistic probe, the same checkpoints as everywhere else in this document. Data-driven per-dimension min/max from each checkpoint's own
training split, uniform bins at the stated number of levels, snap to nearest — a post-hoc
resolution sweep on a fixed trained continuous encoder, independent of the training-time
grid-resolution question (§S9.1, dropped as a claim). Mean ± sd across 10 runs. Every checkpoint used
passed its own batch-size provenance check (§S3.2) before extraction; the arrays extracted for this
table reproduce the token-level accuracies of the same runs at 5 levels (e.g. CropGlobe masked prediction, seed 2024, France: 5-level accuracy 0.7557 / F1 0.7608, matching the token value exactly) — an independent check that the extraction pipeline is correct.

**CropGlobe:**

| Objective | Region | Levels | Accuracy mean ± sd | Macro-F1 mean ± sd |
|---|---|---|---|---|
| Reconstruction | France | 3 | 0.589 ± 0.016 | 0.519 ± 0.015 |
| Reconstruction | France | 5 | 0.621 ± 0.010 | 0.555 ± 0.015 |
| Reconstruction | France | 9 | 0.633 ± 0.012 | 0.570 ± 0.016 |
| Reconstruction | France | 17 | 0.635 ± 0.012 | 0.572 ± 0.017 |
| Reconstruction | Queensland | 3 | 0.559 ± 0.012 | 0.408 ± 0.017 |
| Reconstruction | Queensland | 5 | 0.571 ± 0.020 | 0.425 ± 0.026 |
| Reconstruction | Queensland | 9 | 0.572 ± 0.015 | 0.428 ± 0.020 |
| Reconstruction | Queensland | 17 | 0.573 ± 0.016 | 0.430 ± 0.021 |
| Masked prediction | France | 3 | 0.681 ± 0.040 | 0.644 ± 0.075 |
| Masked prediction | France | 5 | 0.698 ± 0.044 | 0.667 ± 0.081 |
| Masked prediction | France | 9 | 0.703 ± 0.045 | 0.674 ± 0.082 |
| Masked prediction | France | 17 | 0.705 ± 0.046 | 0.676 ± 0.082 |
| Masked prediction | Queensland | 3 | 0.672 ± 0.032 | 0.626 ± 0.057 |
| Masked prediction | Queensland | 5 | 0.677 ± 0.033 | 0.631 ± 0.058 |
| Masked prediction | Queensland | 9 | 0.676 ± 0.033 | 0.629 ± 0.060 |
| Masked prediction | Queensland | 17 | 0.675 ± 0.033 | 0.628 ± 0.060 |

**EuroCropsML (Estonia):**

| Objective | Levels | Accuracy mean ± sd | Macro-F1 mean ± sd |
|---|---|---|---|
| Reconstruction | 3 | 0.139 ± 0.002 | 0.031 ± 0.001 |
| Reconstruction | 5 | 0.146 ± 0.002 | 0.035 ± 0.001 |
| Reconstruction | 9 | 0.150 ± 0.002 | 0.037 ± 0.001 |
| Reconstruction | 17 | 0.150 ± 0.002 | 0.037 ± 0.001 |
| Masked prediction | 3 | 0.193 ± 0.005 | 0.070 ± 0.004 |
| Masked prediction | 5 | 0.204 ± 0.005 | 0.079 ± 0.006 |
| Masked prediction | 9 | 0.208 ± 0.006 | 0.083 ± 0.006 |
| Masked prediction | 17 | 0.210 ± 0.006 | 0.084 ± 0.006 |

Five levels already captures nearly all the accuracy achievable at 17 levels. The largest 5→17 gaps are CropGlobe reconstruction in France (+0.014 accuracy, +0.017 macro-F1; 5→9: +0.013 and +0.015); every other cell moves by
at most 0.009. Going from 3 to 5 levels recovers more than going from 5 to 17 in every row and metric, consistent with the real FSQ grid (5 levels) already sitting past the knee of this curve.

### S6.2 Grid-distance control

Protocol: the real trained FSQ grid (5 levels, half-range 2), continuous codes from each checkpoint's own training split. Units are level-spacing, since FSQ's own grid has spacing 1:
distance is the per-dimension, per-token absolute deviation from the nearest clipped, rounded grid point. A uniform, uninformative spread within one level's cell has an expected mean absolute
distance of 0.25 — the reference point, not zero. Mean over 4 dimensions and 10 runs per objective:

| Objective | CropGlobe mean distance | EuroCropsML mean distance |
|---|---|---|
| Reconstruction | 0.264 (≈ uniform reference 0.25) | 0.257 (≈ uniform reference 0.25) |
| Masked prediction | 0.200 (below the 0.25 uniform reference) | 0.217 (below the 0.25 uniform reference) |

Reconstruction's continuous codes sit close to the uniform-spread reference on both benchmarks (a little above it on CropGlobe under the matched recipe) — the encoder is not organizing its
continuous output around the eventual grid points. Masked prediction's codes sit meaningfully below 0.25 on both benchmarks (further below on CropGlobe than EuroCropsML), meaning the predictive
objective already pulls its continuous representation toward the quantization grid before any rounding happens; every masked-prediction run is closer to the grid than every reconstruction run of its
benchmark (single runs 0.195 to 0.221 against 0.251 to 0.274). **This does not translate into a consistently smaller rounding cost, however.** Rounding cost here is the continuous-minus-tokens accuracy / macro-F1 gap, per run,
averaged over the ten runs: on CropGlobe France, masked prediction's cost (accuracy −0.0066, macro-F1 −0.0099) is smaller than matched-recipe reconstruction's (accuracy −0.0146, macro-F1 −0.0170); **on Estonia it is the other way around** — masked prediction's rounding cost (accuracy −0.0062, macro-F1 −0.0049) is larger in magnitude than reconstruction's (accuracy −0.0046, macro-F1 −0.0023). Grid distance and rounding cost track together on CropGlobe but not on EuroCropsML, so grid distance is not a general predictor of rounding cost across benchmarks.

## S7. Statistics

Every comparison of a learned representation with a control is a one-sided one-sample t-test on the per-run margins, the control being fixed given its data; masked prediction against reconstruction is a one-sided Welch test. Each objective has ten independent runs in every setting. The tests were first computed with three to five runs per setting. The sample was then extended in stages to ten. Every p-value is therefore multiplied by the number of analyses of its row: 2 for the transductive rows and for Estonia, 3 for the strict rows. Holm's correction is applied within each column of Table I over its five rows, separately per metric. No outcome of Table I depends on the multiplier: all 25 are the same without it. The runs added in the extension (seven per objective and setting, five for strict masked prediction), analysed on their own and without the multiplier, give the same outcome in 23 of the 25 comparisons. The two that differ are masked prediction against rounded PCA in transductive France (macro-F1 only with all ten runs, neither metric in the added runs alone) and reconstruction against k-means in strict Queensland (neither metric with all ten runs, accuracy only in the added runs alone). Analysis of the added runs alone, margins in accuracy / macro-F1, uncorrected one-sided p, outcome after Holm within the column:

| Row | n | Masked prediction − PCA | Masked prediction − k-means | Masked prediction − reconstruction | Reconstruction − PCA | Reconstruction − k-means |
|---|---|---|---|---|---|---|
| France, transductive | 7 | +0.013 / +0.055; p 0.238 / 0.052; N | −0.006 / +0.032; p 0.638 / 0.156; N | +0.069 / +0.104; p 0.003 / 0.005; E | −0.056 / −0.050; p 1.000 / 1.000; N | −0.075 / −0.073; p 1.000 / 1.000; N |
| Queensland, transductive | 7 | +0.110 / +0.204; p <0.001 / <0.001; E | +0.102 / +0.185; p <0.001 / <0.001; E | +0.092 / +0.183; p <0.001 / <0.001; E | +0.018 / +0.021; p 0.013 / 0.026; N | +0.010 / +0.002; p 0.072 / 0.423; N |
| France, strict | 5 | +0.058 / +0.127; p 0.005 / 0.002; E | +0.042 / +0.106; p 0.016 / 0.003; E | +0.121 / +0.180; p <0.001 / <0.001; E | −0.062 / −0.053; p 1.000 / 1.000; N | −0.079 / −0.074; p 1.000 / 1.000; N |
| Queensland, strict | 5 | +0.094 / +0.200; p 0.003 / <0.001; E | +0.082 / +0.175; p 0.005 / 0.001; E | +0.072 / +0.172; p 0.008 / 0.001; E | +0.022 / +0.027; p <0.001 / <0.001; E | +0.010 / +0.002; p 0.005 / 0.313; S |
| Estonia | 7 | +0.042 / +0.032; p <0.001 / <0.001; E | +0.034 / +0.028; p <0.001 / <0.001; E | +0.056 / +0.044; p <0.001 / <0.001; E | −0.015 / −0.012; p 1.000 / 1.000; N | −0.022 / −0.016; p 1.000 / 1.000; N |

**Probe convergence.** The probe (standardization, then logistic regression with lbfgs and at most 300 iterations) stops at the iteration limit for raw bands (in Queensland
on some machines only), so their scores depend slightly on the numerical environment (library versions, threads). Across the machines and runs checked, Estonia raw bands score 0.2460 to
0.2462 in accuracy and 0.0953 to 0.0975 in macro-F1 (Table II prints 0.246 / 0.097); the other raw-band scores move by at most 0.0015 in accuracy and 0.0022 in macro-F1.
Converged fits (300 to 900 iterations) agree across environments to 0.0013: France 0.783 / 0.791, Queensland 0.708 / 0.638 to 0.639, Estonia 0.249 / 0.102. The letter keeps
the 300-iteration protocol as declared. Raw bands enter no outcome of Table I, only the width column of Table II. The controls' and the encoders' probe scores
reproduced within 0.0005 in every check of this document. At that size two outcomes sit at the threshold, the same two that change under the sign-flip test (§S7.1b):
raising the rounded PCA control by 0.0005 turns strict-Queensland reconstruction against it from E to S, and raising the k-means control by 0.001 turns strict
France masked prediction against it from E to S. No other outcome changes for a shift of either sign up to 0.002.

### S7.1 Tests against matched controls, with run counts

Tests of Table I, with run counts. One-sided one-sample t-test on per-run margins (learned minus control), the k-means control (§S4.1) treated as fixed (mean of five fits; fit-to-fit sd at most 0.0016 in accuracy, §S4.4); masked prediction against reconstruction is a one-sided Welch test. "p raw" is the uncorrected p, "p used" is p multiplied by the number of analyses of the row (§S7). "Runs above" is accuracy / macro-F1, out of 10. The Holm-adjusted p is the level at which a comparison just survives the correction within its column, computed from unrounded p-values; a metric survives below 0.05. The outcomes for the other forms of the k-means control are in §S4.4.

**Masked prediction minus the rounded PCA control**

| Region, setting | n | Margin acc. / F1 | p raw acc. / F1 | p used acc. / F1 | Holm-adjusted p acc. / F1 | Runs above | Outcome (per-column Holm) |
|---|---|---|---|---|---|---|---|
| France, transductive | 10 | +0.021 / +0.062 | 0.083 / 0.019 | 0.165 / 0.037 | 0.165 / 0.037 | 4 / 7 | S |
| Queensland, transductive | 10 | +0.118 / +0.221 | <0.001 / <0.001 | <0.001 / <0.001 | <0.001 / <0.001 | 10 / 10 | E |
| France, strict | 10 | +0.051 / +0.121 | <0.001 / <0.001 | 0.002 / <0.001 | 0.005 / <0.001 | 8 / 10 | E |
| Queensland, strict | 10 | +0.095 / +0.201 | <0.001 / <0.001 | <0.001 / <0.001 | <0.001 / <0.001 | 10 / 10 | E |
| Estonia | 10 | +0.042 / +0.032 | <0.001 / <0.001 | <0.001 / <0.001 | <0.001 / <0.001 | 10 / 10 | E |

**Masked prediction minus k-means**

| Region, setting | n | Margin acc. / F1 | p raw acc. / F1 | p used acc. / F1 | Holm-adjusted p acc. / F1 | Runs above | Outcome (per-column Holm) |
|---|---|---|---|---|---|---|---|
| France, transductive | 10 | +0.002 / +0.039 | 0.437 / 0.080 | 0.874 / 0.159 | 0.874 / 0.159 | 4 / 6 | N |
| Queensland, transductive | 10 | +0.111 / +0.202 | <0.001 / <0.001 | <0.001 / <0.001 | <0.001 / <0.001 | 10 / 10 | E |
| France, strict | 10 | +0.035 / +0.100 | 0.007 / <0.001 | 0.022 / <0.001 | 0.045 / <0.001 | 7 / 10 | E |
| Queensland, strict | 10 | +0.083 / +0.176 | <0.001 / <0.001 | <0.001 / <0.001 | <0.001 / <0.001 | 10 / 10 | E |
| Estonia | 10 | +0.035 / +0.028 | <0.001 / <0.001 | <0.001 / <0.001 | <0.001 / <0.001 | 10 / 10 | E |

**Masked prediction minus reconstruction**

| Region, setting | n | Margin acc. / F1 | p raw acc. / F1 | p used acc. / F1 | Holm-adjusted p acc. / F1 | Runs above | Outcome (per-column Holm) |
|---|---|---|---|---|---|---|---|
| France, transductive | 10 | +0.078 / +0.111 | <0.001 / <0.001 | <0.001 / 0.002 | <0.001 / 0.002 | — | E |
| Queensland, transductive | 10 | +0.106 / +0.206 | <0.001 / <0.001 | <0.001 / <0.001 | <0.001 / <0.001 | — | E |
| France, strict | 10 | +0.111 / +0.175 | <0.001 / <0.001 | <0.001 / <0.001 | <0.001 / <0.001 | — | E |
| Queensland, strict | 10 | +0.079 / +0.179 | <0.001 / <0.001 | <0.001 / <0.001 | <0.001 / <0.001 | — | E |
| Estonia | 10 | +0.057 / +0.044 | <0.001 / <0.001 | <0.001 / <0.001 | <0.001 / <0.001 | — | E |

**Reconstruction minus the rounded PCA control**

| Region, setting | n | Margin acc. / F1 | p raw acc. / F1 | p used acc. / F1 | Holm-adjusted p acc. / F1 | Runs above | Outcome (per-column Holm) |
|---|---|---|---|---|---|---|---|
| France, transductive | 10 | −0.056 / −0.049 | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 | 0 / 0 | N |
| Queensland, transductive | 10 | +0.012 / +0.015 | 0.043 / 0.051 | 0.087 / 0.103 | 0.348 / 0.411 | 8 / 8 | N |
| France, strict | 10 | −0.060 / −0.054 | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 | 0 / 0 | N |
| Queensland, strict | 10 | +0.016 / +0.021 | 0.003 / 0.002 | 0.009 / 0.005 | 0.046 / 0.027 | 9 / 9 | E |
| Estonia | 10 | −0.015 / −0.012 | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 | 0 / 0 | N |

**Reconstruction minus k-means**

| Region, setting | n | Margin acc. / F1 | p raw acc. / F1 | p used acc. / F1 | Holm-adjusted p acc. / F1 | Runs above | Outcome (per-column Holm) |
|---|---|---|---|---|---|---|---|
| France, transductive | 10 | −0.075 / −0.072 | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 | 0 / 0 | N |
| Queensland, transductive | 10 | +0.005 / −0.005 | 0.243 / 0.710 | 0.487 / 1.000 | 1.000 / 1.000 | 6 / 6 | N |
| France, strict | 10 | −0.077 / −0.074 | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 | 0 / 0 | N |
| Queensland, strict | 10 | +0.004 / −0.004 | 0.171 / 0.736 | 0.513 / 1.000 | 1.000 / 1.000 | 7 / 5 | N |
| Estonia | 10 | −0.022 / −0.016 | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 | 0 / 0 | N |

Reconstruction against k-means: below on average in France and Estonia in both metrics (margins, accuracy / macro-F1: France −0.075 / −0.072 transductive and −0.077 / −0.074 strict; Estonia −0.022 / −0.016); all ten runs fall below it in both metrics in France and Estonia. In Queensland the mean margin is +0.005 / −0.005 (transductive) and +0.004 / −0.004 (strict), with 6 / 6 and 7 / 5 of ten runs above it.

Masked prediction against the rounded PCA control, runs above it in accuracy / macro-F1: France 4 / 7 of 10 (transductive) and 8 / 10 of 10 (strict); Queensland 10 / 10 and 10 / 10; Estonia 10 / 10.

Reconstruction against the rounded PCA control: all ten runs below it in both metrics in France (both settings) and Estonia; in Queensland 8 and 9 of ten runs lie above it (transductive, strict), with mean margins +0.012 / +0.015 (not established) and +0.016 / +0.021 (established per column, not when the correction is pooled, §S7.2).

**Transductive pairing.** One transductive encoder produces both the France and the Queensland score. Accuracy / macro-F1 per encoder (masked prediction):

| Run (seed) | France | Queensland |
|---|---|---|
| 1904 | 0.666 / 0.562 | 0.662 / 0.633 |
| 2024 | 0.756 / 0.761 | 0.696 / 0.667 |
| 4242 | 0.732 / 0.729 | 0.733 / 0.715 |
| 11 | 0.658 / 0.559 | 0.656 / 0.563 |
| 23 | 0.671 / 0.603 | 0.632 / 0.529 |
| 37 | 0.677 / 0.672 | 0.669 / 0.644 |
| 53 | 0.758 / 0.762 | 0.710 / 0.685 |
| 71 | 0.656 / 0.635 | 0.632 / 0.576 |
| 89 | 0.749 / 0.756 | 0.684 / 0.653 |
| 97 | 0.659 / 0.627 | 0.695 / 0.649 |

Across the ten encoders the two accuracies correlate (Pearson r = +0.70, Spearman ρ = +0.75; macro-F1 r = +0.72, ρ = +0.82; descriptive, n = 10). The lowest France run (seed 71, 0.656) is also the lowest in Queensland (0.632), and every run clears Queensland's rounded-PCA (0.559) and k-means (0.566) controls. Strict encoders differ per region, so strict runs cannot be paired this way.

### S7.1b Exact sign-flip permutation test

The t-tests above assume roughly normal per-run margins, and the France runs are bimodal (§S5.1). The same comparisons were therefore repeated with the exact one-sided sign-flip permutation test
(all 2^10 sign patterns of the ten per-run margins; statistic the mean margin), with the same multiplier and the same per-column Holm correction. Masked prediction against reconstruction keeps its
Welch test. With ten runs the smallest attainable p is 1/1024, so after the multiplier and Holm the strongest results read 0.010 rather than < 0.001.

| Comparison | Region, setting | p raw acc. / F1, sign-flip | Holm-adjusted p acc. / F1, sign-flip | Outcome t | Outcome sign-flip |
|---|---|---|---|---|---|
| Masked prediction − rounded PCA | France, transductive | 0.076 / 0.023 | 0.15 / 0.047 | S | S |
| | France, strict | 0.004 / < 0.001 | 0.023 / 0.010 | E | E |
| | Queensland, both settings | < 0.001 / < 0.001 | 0.010 / 0.010 | E | E |
| | Estonia | < 0.001 / < 0.001 | 0.010 / 0.010 | E | E |
| Masked prediction − k-means | France, transductive | 0.44 / 0.079 | 0.87 / 0.16 | N | N |
| | France, strict | 0.010 / < 0.001 | 0.059 / 0.010 | E | **S** |
| | Queensland, both settings | < 0.001 / < 0.001 | 0.010 / 0.010 | E | E |
| | Estonia | < 0.001 / < 0.001 | 0.010 / 0.010 | E | E |
| Reconstruction − rounded PCA | Queensland, transductive | 0.044 / 0.050 | 0.35 / 0.40 | N | N |
| | Queensland, strict | 0.006 / 0.005 | 0.088 / 0.073 | E | **N** |
| | France (both), Estonia | 1.00 / 1.00 | 1.00 / 1.00 | N | N |
| Reconstruction − k-means | every row | 0.17 to 1.00 | 1.00 | N | N |

Two outcomes change: strict France, masked prediction against k-means, E → S, and strict Queensland, reconstruction against the rounded PCA, E → N. Every other outcome of Table I is unchanged.

### S7.2 Multiplicity correction: per-column versus pooled Holm

Holm's method at 0.05, separately per metric, with two family definitions: per column (one family per question — five comparisons for each of the five columns of Table I, the letter's primary reading), and pooled (the 15 masked-prediction comparisons — 5 against the rounded PCA control, 5 against k-means, 5 against reconstruction — in one family, reported alongside so no reader has to compute it). E = both metrics survive, S = one, N = neither.

| Comparison | Per-column Holm | Pooled Holm (15) |
|---|---|---|
| France transductive — PCA / k-means / reconstruction | S / N / E | N / N / E |
| Queensland transductive — PCA / k-means / reconstruction | E / E / E | E / E / E |
| France strict — PCA / k-means / reconstruction | E / E / E | E / S / E |
| Queensland strict — PCA / k-means / reconstruction | E / E / E | E / E / E |
| Estonia — PCA / k-means / reconstruction | E / E / E | E / E / E |

In the pooled family of 15, the accuracy step-down first fails at France strict against k-means (p = 0.0223 after the multiplier, against a Holm threshold of 0.0167), with 12 of 15 comparisons surviving; the macro-F1 step-down retains 13 of 15, first failing at France transductive against PCA (p = 0.0374 against 0.0250), which is why some outcomes are S. Pooled over the 15 comparisons, every comparison in Queensland and Estonia, every comparison with reconstruction and strict France against rounded PCA stay established; strict France against k-means survives in macro-F1 only; transductive France against either control does not survive.

The two columns of reconstruction against a control (per-column / pooled over all 25 comparisons of Table I):

| Row | Reconstruction − PCA | Reconstruction − k-means |
|---|---|---|
| France, transductive | N / N | N / N |
| Queensland, transductive | N / N | N / N |
| France, strict | N / N | N / N |
| Queensland, strict | E / N | N / N |
| Estonia | N / N | N / N |

Pooled over all 25 comparisons, 12 survive in accuracy and 13 in macro-F1. Reconstruction against rounded PCA in strict Queensland is established per column and does not survive the pooled correction. With the ten-band k-means vocabulary (§S4.4) the k-means entries are, per column / pooled, N / N in transductive France, E / E in transductive Queensland, S / S in strict France, S / S in strict Queensland and E / E in Estonia, and the pooled macro-F1 step-down retains 13 of 15 (accuracy 11 of 15).

## S8. Analysis plan and its deviations

**Where the rules come from.** The decision rules are the pre-declared readings for the four numbered claims (C1–C4), plus the transductive predictive-versus-PCA rule (one-sided one-sample t-test on per-run margins, established if p < 0.05 in both accuracy and macro-F1). The plan was kept in working files and first committed to version control after the first results were known. Its earlier history is therefore the authors' own record, not externally verifiable, and the plan was never publicly registered. Everything decided after that commit carries a version-control timestamp, which becomes externally verifiable when the repository is published.

**Rules.**
- **Masked prediction vs. PCA-4, transductive:** one-sided one-sample t-test on per-run margins; established if p < 0.05 in accuracy and macro-F1.
- **C1 (reconstruction vs. PCA-4):** "falls below PCA-4" if all ten runs are below it in both metrics; if any run reaches it, "no better than PCA-4"; if the mean exceeds it, withdrawn.
- **C1b (masked prediction minus reconstruction):** one-sided Welch t-test, established if p < 0.05 in both metrics.
- **C2 (strict CropGlobe masked prediction vs. strict-fit PCA-4):** established if p < 0.05 in both metrics.
- **C3 (training-time grid resolution):** results at 7 and 9 levels within 0.030 of the 5-level mean means "below 5 costly, above flat"; more than 0.030 above means "under-resolved; re-examine the default level".
- **C4 (widening from 4 to 6 dimensions):** widening helps reconstruction if its 6-dimension mean exceeds its 4-dimension mean by more than twice the pooled run-to-run sd; the claim requires that masked prediction's 6-dimension mean does *not* exceed its 4-dimension mean.

**Deviations.**
1. **C1 clarified and C1b added after partial data.** After one reconstruction run had been seen, C1 was clarified to apply to each region separately (the strictest reading) and C1b was added. No C1b comparison could be computed at that point, but one run was already known.
2. **The sample was extended to ten runs per objective and setting** after the results with three or five runs were known. It was extended in stages, from three to five runs per setting and then to ten, and every p-value is multiplied by the number of analyses of its row (§S7). No outcome of Table I depends on the multiplier.
3. **C3 fired and its prescribed action was not taken.** Its 0.030 threshold assumed a run-to-run sd of 0.015; the measured sd was about 0.045. Rather than recalibrate after seeing the data, the claim was dropped and the single runs at 3, 7 and 9 levels are reported as inconclusive (§S9.1).
4. **The multiplicity correction, the tests against k-means and the tests of reconstruction against the controls were defined after the first results were known.** Both family definitions (per column and pooled) are reported (§S7.2).
5. **The dimensionality-matched form of the k-means control was fixed after the results with the ten-band vocabulary were known:** a k-means vocabulary fitted on the ten band means and read as ten values per month is not matched in width to the tokenizer and the rounded PCA. The letter's control is fitted and read in the four principal-component dimensions; every form is reported in §S4.4, including the ten-band one.
6. **One run was stopped by hand.** One EuroCropsML masked-prediction run (seed 1904) was trained on a different machine (one NVIDIA A100) and was stopped by a manual job cancellation at epoch 145, eight epochs after its best validation epoch (137) and before early stopping (patience 25) could trigger. Its best-validation checkpoint is used, as for every other run. Continuing the run from its checkpoint (on one L40S) until early stopping at epoch 165 moves the best epoch to 140 (validation loss 3.2517 against 3.2525, small beside the epoch-to-epoch range of 3.2517–3.2665) and changes its Estonia probe by at most 0.0005 in accuracy and 0.0003 in macro-F1 (tokens 0.2020 / 0.0810 → 0.2025 / 0.0807), with all 625 codes in use; the letter keeps the original checkpoint.
7. **The cost of discretization against encoders trained without the quantizer (§S5.4) was added after the letter's results were known.** The encoders without the quantizer, the choice of `twin_teacher` as masked prediction's main twin (`twin_bands` beside), the two-sided Welch tests of (c) − (a) and of the training step (b) − (a), and Holm's correction over the five rows are post hoc; every row and both twins are reported.
8. **Continuing the capped runs (§S5.1, §S5.4)** was decided after the results. Table I keeps the original runs; the twin tables print the continued twins, because the cap had cut them short. Under the letter's tests, no outcome changed beyond the strict-France cell of §S5.1.

C4's outcome is reported in §S9.4; the letter states only that the comparison does not isolate width, which is a change of interpretation, not of a rule.

**Matched-recipe runs (pre-declared).** The matched-recipe runs and their reading rules were committed to version control before any run started. Reading rule 3 required every masked-prediction run to keep more of the raw-band score through the reduction step than every reconstruction run, in both regions and both metrics. With ten runs per objective it holds in Queensland in both metrics, and also in strict France, strict Queensland and Estonia, but fails in transductive France in both metrics: the weakest masked-prediction run's continuous accuracy there is 0.6611 (seed 71) against 0.6623 for the best reconstruction run (seed 71), and its continuous macro-F1 is 0.5624 (seed 11) against 0.6099 (seed 71); two of the ten masked-prediction runs are at or below the best reconstruction run in accuracy. The letter therefore reports the CropGlobe reduction numbers without attributing them to the objective in transductive France. The other rules applied as written: the matched-recipe runs replace the original-recipe CropGlobe reconstruction runs in every table, figure and claim of the letter, and fill the strict reconstruction cells. The width comparison (C4) is on the same footing: reconstruction is also trained at six dimensions on the matched recipe, so the letter compares both widths on the matched recipe (§S9.4).

## S9. Inconclusive and context results

### S9.1 Single-run grid-resolution results — inconclusive

Referenced from the letter as data that does not support a training-time grid-resolution
conclusion; each row other than the 5-level reference is a single run (n = 1), CropGlobe
transductive, seed 1904, harmonized logistic probe. The 5-level reference is the ten-run pool mean used throughout this document.

| Levels | France acc. / F1 | Queensland acc. / F1 | Δ vs. 5-level pool mean (France / Queensland, acc.) |
|---|---|---|---|
| 3 | 0.622 / 0.551 | 0.660 / 0.626 | −0.076 / −0.017 |
| 5 (pool mean, n = 10) | 0.698 / 0.667 | 0.677 / 0.631 | — |
| 7 | 0.623 / 0.560 | 0.572 / 0.491 | −0.075 / −0.105 |
| 9 | 0.766 / 0.774 | 0.681 / 0.647 | +0.068 / +0.004 |

No monotonic or otherwise interpretable pattern: 3 and 7 levels both sit well below the 5-level reference on France, by similar margins (−0.076 and −0.075), despite 7 having more resolution than 3; 9 levels sits above it (+0.068, 1.5 run-to-run sd); and Queensland does not track France at any level. This is reported as inconclusive single-run data, not used to draw or revise any grid-resolution claim (§S8, deviation 3).

### S9.2 Original versus matched recipe, CropGlobe reconstruction

CropGlobe reconstruction was trained on two recipes: its own (cosine schedule, batches of 512 windows) and the masked-prediction recipe (plateau schedule, batches of 448 windows; §S3.1). The letter reports the matched-recipe runs, ten per setting. The original recipe was run three times in the transductive setting (seeds 42, 101, 1904; its strict runs were not probed). Those runs are compared here with the matched-recipe runs of the same seeds, and with all ten. Logistic probe, harmonized labels, fixed 8-country pool.

**France**

| Seed | Representation | Matched recipe acc / F1 | Original recipe acc / F1 |
|---|---|---|---|
| 42 | Tokens | 0.6284 / 0.5732 | 0.6279 / 0.5540 |
| 101 | Tokens | 0.6088 / 0.5433 | 0.6094 / 0.5422 |
| 1904 | Tokens | 0.6258 / 0.5516 | 0.6129 / 0.5467 |
| 42 | Continuous | 0.6388 / 0.5826 | 0.6396 / 0.5673 |
| 101 | Continuous | 0.6234 / 0.5600 | 0.6298 / 0.5649 |
| 1904 | Continuous | 0.6330 / 0.5581 | 0.6205 / 0.5532 |

**Queensland**

| Seed | Representation | Matched recipe acc / F1 | Original recipe acc / F1 |
|---|---|---|---|
| 42 | Tokens | 0.5638 / 0.4177 | 0.5813 / 0.4475 |
| 101 | Tokens | 0.5794 / 0.4351 | 0.5835 / 0.4375 |
| 1904 | Tokens | 0.5298 / 0.3770 | 0.5687 / 0.4229 |
| 42 | Continuous | 0.5667 / 0.4235 | 0.5703 / 0.4317 |
| 101 | Continuous | 0.5764 / 0.4304 | 0.5820 / 0.4356 |
| 1904 | Continuous | 0.5408 / 0.3965 | 0.5796 / 0.4363 |

Averaged over the three paired seeds, the matched recipe changes France by +0.002 to +0.008 (accuracy and macro-F1, both representations) and Queensland by −0.016 to −0.026 (tokens accuracy −0.020, tokens F1 −0.026, continuous accuracy −0.016, continuous F1 −0.018). Over all ten matched-recipe runs the means are: France tokens 0.6208 / 0.5552, continuous 0.6355 / 0.5722; Queensland tokens 0.5708 / 0.4250, continuous 0.5738 / 0.4309, against the original recipe's three-run means of 0.6167 / 0.5476 and 0.6300 / 0.5618 (France), 0.5778 / 0.4360 and 0.5773 / 0.4345 (Queensland). The reduction cost (continuous minus raw bands, accuracy) is 0.146 in France and 0.134 in Queensland on the matched recipe, against 0.152 and 0.130 on the original recipe. On the original recipe, reconstruction's tokens were 0.019 above the rounded PCA in transductive Queensland; on the matched recipe the ten-run mean is 0.012 above it, which is not established (§S7.1).

### S9.3 Gradient-boosting probe

The letter and this document use the logistic probe throughout. Gradient-boosting results exist for part of the grid and are compiled here for context only; they are not tested.

**Masked-prediction tokens** (accuracy / macro-F1):

| Setting | Seed | France | Queensland |
|---|---|---|---|
| Transductive | 1904 | 0.7463 / 0.7020 | 0.7040 / 0.6807 |
| Transductive | 2024 | 0.7916 / 0.7775 | 0.7468 / 0.7267 |
| Transductive | 4242 | 0.7949 / 0.7899 | 0.7365 / 0.7151 |
| Transductive | 11 | 0.7392 / 0.6765 | 0.7108 / 0.6700 |
| Transductive | 23 | 0.7439 / 0.7012 | 0.7034 / 0.6512 |
| Transductive | 37 | 0.7184 / 0.7325 | 0.7178 / 0.7060 |
| Transductive | 53 | 0.7963 / 0.7971 | 0.7383 / 0.7138 |
| Transductive | 71 | 0.6825 / 0.6902 | 0.6972 / 0.6577 |
| Transductive | 89 | 0.7740 / 0.7702 | 0.7277 / 0.7027 |
| Transductive | 97 | 0.6963 / 0.6992 | 0.7619 / 0.7396 |
| Transductive | mean (ten) | 0.7483 / 0.7336 | 0.7245 / 0.6964 |
| Strict | 42 | 0.7925 / 0.7919 | 0.7311 / 0.7068 |
| Strict | 101 | 0.6907 / 0.7086 | 0.7203 / 0.7070 |

Of the ten strict runs per region, only two were probed with gradient boosting.

**Controls and raw bands** (accuracy / macro-F1; k-means read as its ten band values, mean of five
fits):

| Representation | Setting | France | Queensland |
|---|---|---|---|
| Rounded PCA | Transductive | 0.7570 / 0.6943 | 0.6788 / 0.6199 |
| Rounded PCA | Strict | 0.7517 / 0.6877 | 0.6786 / 0.6260 |
| k-means-625 | Transductive | 0.7628 / 0.7234 | 0.7214 / 0.6881 |
| k-means-625 | Strict | 0.7566 / 0.7142 | 0.7167 / 0.6814 |
| Raw bands (mean of five random states, see the note) | both | 0.8039 / 0.7894 | 0.7536 / 0.7226 |

In transductive France the tokens' mean over the ten runs (0.748) is below both controls (0.757, 0.763) and below raw bands (0.804); 4 of the 10 runs are above k-means. In transductive Queensland it is above the rounded PCA (0.679) and level with k-means (0.724 against 0.721), and below raw bands (0.754). Both statements also hold against the lowest of the five raw-band states (France 0.796, Queensland 0.750).

*Note on the raw-band row (changed 2026-10-12).* The raw-band values printed here before (0.8019 / 0.7841 in France, 0.7581 / 0.7279 in Queensland) came from a single fit whose script was not kept, and none of the five random states tried reproduces them. The gradient-boosting probe bins continuous features from a random subsample when the training pool has more than 200,000 rows (here 236,826), so scores on continuous inputs such as raw bands vary with the random state, by about 0.01. The row now gives the mean of five random states (1904, 42, 101, 2024, 4242); the range over the five states is 0.796 to 0.808 (France accuracy), 0.776 to 0.794 (France macro-F1), 0.750 to 0.759 (Queensland accuracy) and 0.718 to 0.729 (Queensland macro-F1). The earlier values lie inside these ranges. The five-state fits are identical on the letter's machine (Windows, scikit-learn 1.9.0) and on the GPU node (Linux, scikit-learn 1.9.1) to four decimals. Integer-valued codes (the tokens, rounded PCAs) are not affected by the random state (checked on the seed-1904 token run and on the rounded PCA with ten components: identical to six decimals under every random state, row order and data type tried). The k-means row (continuous: its ten band values) keeps its stored values; they reproduce exactly on the letter's machine at random state 1904, and over the five random states the five-fit mean ranges from 0.763 to 0.765 (France accuracy), 0.721 to 0.726 (France macro-F1), 0.721 to 0.724 (Queensland accuracy) and 0.688 to 0.693 (Queensland macro-F1). Script and every fit: `scripts/task_n3_raw_gb_rebuild.py`, `results/n3_raw_gb_rebuild.csv` and `results/n3_raw_gb_rebuild_gpu_node.csv`.

### S9.4 Width comparison: four versus six dimensions

Both objectives were also trained at six dimensions and five levels, three runs each, in the transductive setting (tokens, logistic probe); the four-dimension reference is the ten runs per objective. The plan's rule (C4, §S8): widening helps an objective if its six-dimension mean exceeds its four-dimension mean by more than twice the pooled run-to-run standard deviation.

| Objective | France, 6 − 4 dimensions | Queensland, 6 − 4 dimensions |
|---|---|---|
| Masked prediction | +0.029 accuracy, within twice the pooled spread (0.041) | −0.035 accuracy, −0.070 macro-F1, within twice the pooled spread (0.030, 0.052) |
| Reconstruction, matched recipe | +0.037 accuracy, beyond twice the pooled spread (0.010) | +0.012 accuracy, within it (0.018) |
| Reconstruction, original recipe (§S9.2) | +0.037, beyond twice the pooled spread in both metrics | +0.002, within it |

Widening raises reconstruction in France and does not raise masked prediction. For masked prediction the comparison does not isolate width: at five levels the wider latent has 15,625 codes, so the prediction target grows 25-fold, while the reconstruction target is unchanged. The Queensland drop may therefore reflect a harder prediction task rather than width itself; it is within twice the pooled spread. Six dimensions at three levels (729 codes) would separate the two; it was not run.

## S10. Published results for scale

Reuss et al. [17], "Benchmarking for Practice: Few-Shot Time-Series Crop-Type Classification on the
EuroCropsML Dataset," pretrain on Latvia (LV, 103 classes) or Latvia+Portugal (LV+PT, 142 classes)
and fine-tune and evaluate on Estonia, 127 classes, with `k` labelled samples per class,
`k ∈ {1, 5, 10, 20, 100, 200, 500}`, reporting overall accuracy (aOA) micro-averaged over instances.
The paper reports no zero-shot result. Further protocol detail, all from the paper itself:

- **Backbone.** One Transformer block, four heads, embedding 128, feed-forward 256, sinusoidal
  positional encoding over up to 366 days, a single linear layer on top for supervised methods. The
  "PRESTO" row reuses the published PRESTO weights, a larger encoder–decoder.
- **Pretraining.** Transfer learning and the meta-learning family (ANIL, MAML, FOMAML, TIML) — i.e.
  every row of the LV and LV+PT blocks — are pretrained *with crop labels* on the EuroCropsML
  pretraining data. The PRESTO variants are self-supervised on a different dataset, not on Latvia
  or Portugal.
- **Fine-tuning.** Three learning-rate strategies are searched (backbone and head at one rate, at
  separate rates, or head only with a frozen backbone); each reported cell is the best strategy on
  a 1,000-parcel validation set, and the paper does not say which strategy won per cell.
- **Splits.** A 60/20/20 split of Estonia; the k-shot samples come from the training part, and
  every reported accuracy is on the disjoint 35,182-parcel test set. Our Estonia evaluation reads
  the dataset's own predefined finetune-test split and holds the same 35,182 parcels, so
  both papers score the same parcels.
- **Classes.** All 127 Estonia classes are scored, including 34 (LV+PT) not seen in pretraining.

Their Table 4a ("Fine-tuning classification results for EuroCropsML S2", aOA), `k = 1` and
`k = 500`:

| Method | k = 1 | k = 500 |
|---|---|---|
| No pre-training | 0.17 ± 0.14 | 0.6317 ± 0.0062 |
| Transfer learning (LV) | 0.143 ± 0.070 | 0.575 ± 0.048 |
| ANIL (LV) | 0.189 ± 0.047 | 0.652 ± 0.011 |
| MAML (LV) | 0.176 ± 0.077 | 0.648 ± 0.017 |
| FOMAML (LV) | 0.21 ± 0.11 | 0.6191 ± 0.0069 |
| TIML (encoder) (LV) | 0.247 ± 0.092 | 0.591 ± 0.021 |
| TIML (no encoder) (LV) | 0.18 ± 0.12 | 0.637 ± 0.010 |
| Transfer learning (LV+PT) | 0.0787 ± 0.0091 | 0.583 ± 0.028 |
| ANIL (LV+PT) | 0.154 ± 0.050 | 0.623 ± 0.019 |
| MAML (LV+PT) | 0.117 ± 0.013 | 0.637 ± 0.014 |
| FOMAML (LV+PT) | 0.103 ± 0.011 | 0.623 ± 0.011 |
| TIML (encoder) (LV+PT) | 0.128 ± 0.031 | 0.587 ± 0.024 |
| TIML (no encoder) (LV+PT) | 0.162 ± 0.074 | 0.622 ± 0.012 |
| PRESTO | 0.157 ± 0.078 | 0.5528 ± 0.0037 |
| PRESTO-XTS | 0.050 ± 0.025 | 0.622 ± 0.021 |
| CrossPRESTO-XTS | 0.09 ± 0.10 | 0.599 ± 0.018 |

The letter quotes only the range over the LV+PT block, the setting whose pretraining regions match
ours: 0.08–0.16 at `k = 1` and 0.58–0.64 at `k = 500`. **Extraction caveat:** the rows were read
with a text-extraction tool, without a rendered page. Row order was cross-checked against two
statements in the paper's prose, and a range over a block does not depend on how rows line up
inside it, only on which rows belong to the block. Verified by the authors by eye against the PDF (LV+PT block, `k = 1` and `k = 500`).

Not comparable with our numbers: their evaluation uses Estonian labels (`k` per class) over 127
classes; ours uses none, over 82 shared classes plus a residual class, with a frozen logistic
probe. The figures show the range of the task, not a ranking of methods.

---

## S11. Data and code availability

**Data.** Both benchmarks are public: CropGlobe [7] and EuroCropsML [11].

**Code and results.** The code and the per-run result tables are public in this repository, https://github.com/FBK-OpenIoT/fsq-matched-controls, under the MIT License. The release holds the
training configuration of every run in §S3, the probe and analysis scripts, the per-run result
tables behind every table and figure of the letter and this supplement, and the scripts that build
Figure 1 and the tests of Table I from those tables.

**Bibliography.** Citation numbers in this document match the letter's own reference list; see the
letter for full entries.
