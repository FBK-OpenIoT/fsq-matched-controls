# Supplementary Material

## Learned Discrete Tokenizers Against Matched Non-Learned Controls for Satellite Image Time Series

This material supports the letter's claims; it is not needed to follow them. Citations ([7], [11]) use the letter's reference list. Accuracy / macro-F1 throughout; logistic probe, harmonized labels, ten runs per objective and setting.

## S1. Data and preprocessing

**CropGlobe [7]:** Sentinel-2 pixel samples per country split (the Australian split covers Queensland only), fixed 5-day composites with 34 % to 67 % of bins interpolated in the product, and seven season-aligned month-tokens (May–November) of 6 grid values. **EuroCropsML [11]:** Sentinel-2 parcels per NUTS3 region with real, irregular acquisition dates, and nine month-tokens (March–November) of up to 10 real observations, valid with at least 2 and never imputed. Both use the same ten bands. In Estonia 11.9 % of parcel-months are missing; each keeps its calendar slot and reaches the probe as a constant placeholder, the same for every parcel and representation.

## S2. Encoder and training

**Encoder.** Temporal CNN (channels 32/64/128, kernel 3, pooling, linear to d = 4, tanh) followed by FSQ at five levels. On EuroCropsML it is adapted to the observation mask. The masked-prediction tokenizer is trained end to end with a joint softmax over the 625 codes and no anti-collapse term; the transformer is discarded after training, and every probe reads the encoder's output.

**Recipe.** AdamW (weight decay 1e-4), learning rate halved after 10 epochs without validation improvement, early stopping on validation loss. Reconstruction is trained on masked prediction's schedule and number of steps per epoch.
**Reconstruction on its own recipe.** On CropGlobe, reconstruction was also trained three times (transductive) on its own recipe (cosine schedule, batches of 512 windows): it stays below both controls in France (0.609 to 0.628 in accuracy) and lies above k-means in Queensland (0.569 to 0.584 against 0.566; three runs, not tested). The matched recipe changes France by +0.002 to +0.008 and Queensland by −0.016 to −0.026.

**Code use and collapse.** Every CropGlobe masked-prediction run uses all 625 codes. One EuroCropsML run (seed 23) collapsed early onto 308 codes; it is kept in every table and test.

## S3. Matched controls

**Fit pools.** Both controls are fitted, without labels, on exactly the data the compared encoder saw: in CropGlobe transductive, all ten country splits; in CropGlobe strict, the nine splits other than the evaluation region, one fit per held-out region; in EuroCropsML, the Latvia and Portugal pretraining pool.

**Rounded PCA** uses scikit-learn PCA with four components (random state 1904), fitted on the fit pool; each component is standardized on the probe's training and evaluation samples together (on EuroCropsML, on the training samples only), clipped at ±2 standard deviations and rounded onto the five-level grid. On EuroCropsML only valid months enter the fits, and a missing month is zero.

**The k-means vocabulary** is scikit-learn mini-batch k-means with 625 centroids on the four principal-component scores, standardized on the fit pool; each month is represented by the coordinates of its centroid. It is fitted five times per row with seeds 1904, 42, 101, 2024 and 4242, and the letter uses the mean of the five fits. The fit-to-fit standard deviation is 0.0006 to 0.0016 in accuracy and 0.0010 to 0.0026 in macro-F1.

**Code usage.** Bits per month are the entropy of the empirical distribution of the joint month code on the evaluation region (ceiling log2 625 = 9.29). Rounded PCA uses 5.8 to 6.4 bits, the masked-prediction tokens (mean of the ten runs) 7.8 to 8.5 and k-means (mean of the five fits) 8.3 to 8.8; single masked-prediction runs reach up to 8.78, so k-means uses more bits than the tokens' mean, not than every run.

## S4. Sensitivity of the outcomes

### S4.1 Outcomes under each alternative

Table I's outcomes (established = both metrics bold in Table I; here E = both metrics, S = one, N = neither, per-column Holm) under each alternative. Masked prediction's outcomes in transductive Queensland and Estonia, and against reconstruction, never change.

| Alternative | Outcomes that change (Table I → alternative) |
|---|---|
| Equal-mass grid (20/40/60/80 percentiles) for rounded PCA | Masked prediction against it is not established in France in either setting (accuracy test; margins −0.009 and +0.021); Queensland and Estonia stay established |
| Rounded PCA standardized on the encoder's fit pool | Strict Queensland, reconstruction − rounded PCA: E → N |
| Both controls on the month's full window of values | France transductive, masked prediction − rounded PCA S → E, − k-means N → E (both controls weaker); reconstruction lies above the window form of rounded PCA in every CropGlobe row |
| k-means on the ten band means, read as ten values (not dimensionality-matched) | Masked prediction − k-means: strict France and strict Queensland E → S |
| One-hot code readout, for tokens and controls alike (raises every score) | Masked prediction − k-means: strict France E → S; − rounded PCA: strict France E → S. The Queensland margins over k-means fall to 0.03–0.04 in accuracy |
| Gaussian mixture or BIRCH (625 components) instead of k-means | Gaussian mixture: none. BIRCH: France transductive N → S |
| Spectral-index control (NDVI, NDRE, NDMI, SWIR ratio; rounded like rounded PCA) | New control: masked prediction E in every row, reconstruction N in every row |
| No label harmonization | Masked prediction: none. Reconstruction in Queensland: three N → E (an effect of scoring absent classes; untested). Queensland accuracies fall to 0.23–0.50 |
| Exact sign-flip permutation test | Strict France, masked prediction − k-means E → S; strict Queensland, reconstruction − rounded PCA E → N |
| Holm pooled over the 15 masked-prediction comparisons (25 with reconstruction) | France transductive − rounded PCA S → N; strict France − k-means E → S; strict Queensland, reconstruction − rounded PCA E → N (pooled over 25) |
| All 20 capped masked-prediction runs resumed from their best checkpoint and trained to early stopping | Strict France, masked prediction − k-means E → S (t-test); with final-epoch checkpoints under the sign-flip test, France transductive − rounded PCA S → N; margins move by at most 0.003; Estonia unchanged |
| Control scores shifted by up to ±0.002 (probe-convergence tolerance) | At +0.0005: strict Queensland, reconstruction − rounded PCA E → S. At +0.001: strict France, masked prediction − k-means E → S. Nothing else |

### S4.2 A probe with Estonian labels

EuroCropsML's official few-shot split (k parcels per class, one draw per k, 124 Estonian classes, no harmonization), tested on the official test set of 35,182 parcels. Encoders and controls are unchanged. Accuracy / macro-F1 at five of the seven k:

| k | Masked prediction | Reconstruction | Rounded PCA | k-means | Raw bands |
|---|---|---|---|---|---|
| 1 | 0.092 / 0.031 | 0.036 / 0.011 | 0.036 / 0.014 | 0.035 / 0.016 | 0.034 / 0.015 |
| 10 | 0.164 / 0.060 | 0.086 / 0.030 | 0.097 / 0.034 | 0.116 / 0.040 | 0.119 / 0.045 |
| 20 | 0.223 / 0.074 | 0.128 / 0.040 | 0.143 / 0.044 | 0.175 / 0.051 | 0.200 / 0.065 |
| 100 | 0.331 / 0.103 | 0.220 / 0.062 | 0.234 / 0.068 | 0.273 / 0.075 | 0.366 / 0.102 |
| 500 | 0.481 / 0.117 | 0.337 / 0.065 | 0.370 / 0.078 | 0.394 / 0.079 | 0.545 / 0.136 |

At every k of the official split (1, 5, 10, 20, 100, 200, 500; the table shows five of the seven k), **masked prediction** beats both controls (E, ten of ten runs, t-test and sign-flip; p × 2, Holm over the seven k), **reconstruction** is N against both, and **raw bands** are behind masked prediction up to k = 20 and ahead in accuracy from k = 100.

### S4.3 A gradient-boosting probe

A partial check, not tested: gradient boosting replaces the logistic probe for masked prediction only, on the ten transductive runs, against rounded PCA and the ten-band k-means vocabulary. In accuracy the tokens' mean is below both controls and raw bands in France (0.748 against 0.757, 0.763 and 0.804) and level with k-means in Queensland (0.725 against 0.721; rounded PCA 0.679, raw bands 0.754).
## S5. Run-to-run spread and the decomposition

### S5.1 The low French runs

**The low runs.** Nine of the twenty French runs are not above the k-means control of their setting (0.696 transductive, 0.690 strict): accuracy 0.656 to 0.684. What separates them:
- **Probe training accuracy.** On the probe's own training pool it separates them without overlap (0.747–0.778 against 0.794–0.807).
- **Best epoch.** Eight of the nine reached their best validation epoch at 89 or later of the 100-epoch cap, against two of the eleven others.

**The cap is not the cause.** Continuing the capped runs to early stopping lowered the validation loss of most of them but lifted none of the low runs above k-means: their French accuracy moved by at most 0.008 (0.656–0.684 before, 0.655–0.684 after).

**Where they lose.** In per-class F1, most on rice, then on soybeans (transductive) or corn (strict). Rice, 2 % of French pixels, gives about three quarters of the strict macro-F1 margin over k-means (rice F1 0.750 against 0.367).

### S5.2 Scores behind Table II, with standard deviations, and the strict settings

Raw bands: France 0.782 / 0.791, Queensland 0.707 / 0.636, Estonia 0.246 / 0.097. These are the scores behind Table II's differences, with sd (n − 1) in parentheses; (a), (b) and (c) are as in S5.3, and for rounded PCA (b) is the continuous and (c) the rounded score (table below). Rounding, (c) − (b), costs at most 0.015 / 0.019 (no single run loses more than 0.031).

| Row | Objective | (b) continuous (sd) | (c) tokens (sd) | (b) − raw | Rounding |
|---|---|---|---|---|---|
| France, transductive | Reconstruction | 0.635 / 0.572 (0.012 / 0.016) | 0.621 / 0.555 (0.010 / 0.015) | −0.146 / −0.219 | −0.015 / −0.017 |
| France, transductive | Masked prediction | 0.705 / 0.677 (0.046 / 0.082) | 0.698 / 0.667 (0.044 / 0.081) | −0.077 / −0.114 | −0.007 / −0.010 |
| Queensland, transductive | Reconstruction | 0.574 / 0.431 (0.017 / 0.021) | 0.571 / 0.425 (0.020 / 0.026) | −0.134 / −0.205 | −0.003 / −0.006 |
| Queensland, transductive | Masked prediction | 0.676 / 0.628 (0.032 / 0.059) | 0.677 / 0.631 (0.033 / 0.058) | −0.032 / −0.008 | +0.001 / +0.003 |
| France, strict | Reconstruction | 0.628 / 0.564 (0.009 / 0.012) | 0.613 / 0.545 (0.010 / 0.010) | −0.154 / −0.226 | −0.014 / −0.019 |
| France, strict | Masked prediction | 0.734 / 0.734 (0.035 / 0.045) | 0.725 / 0.720 (0.036 / 0.049) | −0.047 / −0.057 | −0.010 / −0.014 |
| Queensland, strict | Reconstruction | 0.575 / 0.432 (0.014 / 0.017) | 0.571 / 0.426 (0.014 / 0.017) | −0.132 / −0.204 | −0.005 / −0.006 |
| Queensland, strict | Masked prediction | 0.652 / 0.608 (0.040 / 0.060) | 0.650 / 0.605 (0.041 / 0.059) | −0.055 / −0.029 | −0.003 / −0.002 |
| Estonia | Reconstruction | 0.151 / 0.037 (0.002 / 0.001) | 0.146 / 0.035 (0.002 / 0.001) | −0.095 / −0.060 | −0.005 / −0.002 |
| Estonia | Masked prediction | 0.210 / 0.084 (0.006 / 0.007) | 0.204 / 0.079 (0.006 / 0.006) | −0.036 / −0.014 | −0.006 / −0.005 |

**Retention above the majority class** (abstract, Conclusion): (tokens − majority) / (raw − majority) in accuracy, with majority-class accuracy 0.301 (France), 0.474 (Queensland), 0.076 (Estonia). Masked-prediction tokens: 0.83, 0.87, 0.88, 0.75, 0.75 (France transductive, Queensland transductive, France strict, Queensland strict, Estonia); k-means: 0.82, 0.40, 0.81, 0.40, 0.54.

**The learned continuous latent against continuous principal components.** The principal-component control is compared standardized but neither clipped nor rounded (tests as Table I); rounding costs the principal components 0.013 to 0.030 in accuracy, more than it costs the trained latents in every row:

| Row | Continuous PCA | Rounded PCA | Masked prediction − continuous PCA | Reconstruction − continuous PCA |
|---|---|---|---|---|
| France, transductive | 0.703 / 0.639 | 0.677 / 0.604 | +0.002 / +0.037, N | −0.068 / −0.067, N |
| Queensland, transductive | 0.571 / 0.439 | 0.559 / 0.410 | +0.105 / +0.189, E | +0.003 / −0.009, N |
| France, strict | 0.704 / 0.641 | 0.673 / 0.599 | +0.031 / +0.093, S | −0.076 / −0.077, N |
| Queensland, strict | 0.571 / 0.440 | 0.555 / 0.405 | +0.081 / +0.168, E | +0.004 / −0.007, N |
| Estonia | 0.178 / 0.057 | 0.161 / 0.047 | +0.032 / +0.027, E | −0.027 / −0.019, N |

### S5.3 Tokens against models trained without the quantizer

Per objective at d = 4, **(c)** the FSQ tokens are set against **(a)** the continuous latent of a model trained without the quantizer, with the same patience and seeds (caps: "Training budget"). For reconstruction, (a) is the same encoder and objective without the quantizer, so (c) − (a) is the cost of discretization. For masked prediction the quantizer defines the target, so two continuous-target twins stand in (slow teacher; band values) and the comparison is exploratory.

**Test.** (c) − (a) is Table II's middle column; (b) − (a) is the training step; (c) − (b) is Table II's rounding. Two-sided Welch test, ten runs against ten, with Holm's correction over the five rows per objective and metric (post hoc, S8). The same test is applied to (b) − (a), the training step, as its own family; its Holm p sits beside it. Bold: Holm p < 0.05 for that comparison.

**Reconstruction.**

| Row | (a) no quantizer | (c) − (a) | Holm p | (b) − (a), training | Holm p | (c) − (b), rounding |
|---|---|---|---|---|---|---|
| France, transductive | 0.664 / 0.614 | **−0.043 / −0.059** | <0.001 / <0.001 | **−0.029 / −0.042** | <0.001 / <0.001 | −0.015 / −0.017 |
| Queensland, transductive | 0.563 / 0.406 | +0.008 / +0.019 | 0.27 / 0.091 | +0.011 / **+0.025** | 0.082 / 0.005 | −0.003 / −0.006 |
| France, strict | 0.663 / 0.612 | **−0.050 / −0.067** | <0.001 / <0.001 | **−0.036 / −0.048** | <0.001 / <0.001 | −0.014 / −0.019 |
| Queensland, strict | 0.559 / 0.400 | +0.011 / **+0.027** | 0.11 / 0.002 | **+0.016 / +0.033** | 0.011 / <0.001 | −0.005 / −0.006 |
| Estonia | 0.148 / 0.036 | −0.001 / 0.000 | 0.11 / 0.40 | **+0.003 / +0.002** | 0.002 / 0.004 | −0.005 / −0.002 |

**Masked prediction (slow-teacher twin, each run at its latest finished stage).**

| Row | (a) no quantizer | (c) − (a) | Holm p | (b) − (a), training | Holm p | (c) − (b), rounding |
|---|---|---|---|---|---|---|
| France, transductive | 0.707 / 0.682 | −0.008 / −0.016 | 1 / 1 | −0.002 / −0.006 | 1 / 1 | −0.007 / −0.010 |
| Queensland, transductive | 0.582 / 0.446 | **+0.095 / +0.186** | <0.001 / <0.001 | **+0.093 / +0.183** | <0.001 / <0.001 | +0.001 / +0.003 |
| France, strict | 0.726 / 0.718 | −0.001 / +0.002 | 1 / 1 | +0.009 / +0.016 | 1 / 1 | −0.010 / −0.014 |
| Queensland, strict | 0.583 / 0.448 | **+0.067 / +0.157** | 0.002 / <0.001 | **+0.070 / +0.159** | 0.002 / <0.001 | −0.003 / −0.002 |
| Estonia | 0.222 / 0.089 | **−0.019 / −0.010** | <0.001 / 0.008 | **−0.012** / −0.005 | 0.009 / 0.36 | −0.006 / −0.005 |

**Masked prediction (band-value twin, continued to early stopping).**

| Row | (a) no quantizer | (c) − (a) | Holm p | (b) − (a), training | Holm p | (c) − (b), rounding |
|---|---|---|---|---|---|---|
| France, transductive | 0.693 / 0.642 | +0.005 / +0.025 | 0.74 / 0.72 | +0.012 / +0.035 | 0.46 / 0.22 | −0.007 / −0.010 |
| Queensland, transductive | 0.611 / 0.477 | **+0.066 / +0.155** | <0.001 / <0.001 | **+0.065 / +0.152** | <0.001 / <0.001 | +0.001 / +0.003 |
| France, strict | 0.672 / 0.638 | **+0.053 / +0.082** | 0.005 / 0.001 | **+0.062 / +0.096** | <0.001 / <0.001 | −0.010 / −0.014 |
| Queensland, strict | 0.609 / 0.475 | **+0.041 / +0.130** | 0.037 / <0.001 | **+0.043 / +0.132** | 0.024 / <0.001 | −0.003 / −0.002 |
| Estonia | 0.207 / 0.078 | −0.003 / +0.001 | 0.18 / 0.72 | +0.003 / **+0.006** | 0.39 / 0.037 | −0.006 / −0.005 |

**The twin matters.** Against both twins the tokens are ahead in Queensland. In strict France they are level against the slow-teacher twin and ahead of the band-value twin; in Estonia they are behind the slow-teacher twin and level with the band-value twin.

**Training budget.** Both sides started with the same cap and patience (CropGlobe 100 epochs and 15, EuroCropsML 200 and 25); the capped runs were then continued, the twin tables show the continued runs, and no outcome changed.

## S6. Rounding curve and grid distance

**Five levels are enough for post-hoc rounding.** With the trained continuous latents rounded post hoc at 3 to 17 levels, going from 5 to 17 levels gains at most 0.014 in accuracy and 0.017 in macro-F1.

**Grid distance.** The mean distance of the latent values to the nearest grid point, in level spacings, has a uniform-spread reference of 0.25. It is 0.264 (CropGlobe) and 0.257 (EuroCropsML) for **reconstruction** and 0.200 and 0.217 for **masked prediction**. Rounding therefore does real work. Being closer to the grid does not consistently lower the rounding cost.

## S7. Statistics

**The run extension.** The sample was extended in stages, from three to five runs per setting and then to ten. The number of these interim looks was not fixed in advance. Every p is therefore multiplied by the number of analyses of its row (2 for the transductive rows and Estonia, 3 for the strict rows). No outcome depends on this multiplier, and the added runs alone give the same outcome in 23 of the 25 comparisons.

**Distribution-free check.** The French margins are bimodal (S5.1), so every comparison was repeated with the exact sign-flip permutation test (all 2^10 sign patterns), which assumes that the margins are symmetric about zero under the null hypothesis. Two outcomes change (S4).

**Probe convergence.** The probe (standardization, then lbfgs logistic regression, at most 300 iterations) stops at the iteration limit for raw bands, whose scores therefore vary by up to 0.0015 in accuracy and 0.0022 in macro-F1 across environments; raw bands enter no outcome.

## S8. Choices made after the first results

Decided or defined after the first results: the per-region reading of the reconstruction rule and the masked-prediction-against-reconstruction rule; the extension to ten runs (S7); dropping the grid-resolution claim, whose rule fired on a spread three times the one assumed; the multiplicity correction and the tests against k-means and of reconstruction against the controls; the dimensionality-matched k-means form (S4); one run cancelled by hand (EuroCropsML; at most 0.0005 on its probe); the discretization comparison (S5.3); and continuing the capped runs (S4). The plan was kept in working files and first committed to version control after the first results. Its earlier history is the authors' own record.
