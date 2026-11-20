# Results

Per-run tables and aggregates behind the letter (Table I, Table II, Figure 1) and the supplementary material (S1–S8). Section numbers refer to `manuscript/SupplementaryMaterial.md`. The module that reads or regenerates each file is given after the arrow.

## runs/: the letter's runs and controls

- `n_probe_gate_*.csv`, `n_probe_new_*.csv`, `n_probe_new_n_mp_eurocropml_estonia_s97of1.csv`: the letter's 80 runs. The probe's accuracy and macro-F1 on tokens and on the continuous latent, per evaluation region, with training provenance: epoch, step, best validation loss, codes used and token entropy on the probe pool. Table I, Table II, Figure 1, S5. → `analysis.probe_runs`, `analysis.check_end_to_end`, `analysis.letter_runs`
- `n_extract_new_*.csv`: provenance of the same runs from the extraction pass (no scores).
- `n_runs.json`: the configurations of the second group of runs (seeds 11, 23, 37, 53, 71, 89, 97).
- `n2_mp_train_diag.csv`: training diagnostics of the masked-prediction runs: best epoch, best validation loss, code entropy, distinct codes (S2, S3).
- `n2/n2_curve_*.csv`: rounding curve (3, 5, 9, 17 levels) and grid distance of the second group of runs (S6). → `analysis.per_run_readouts --mode curve`
- `n2/n2_gb_0of1.csv`: gradient-boosting probe on the tokens, transductive masked prediction (S4.3). → `analysis.per_run_readouts --mode gb`
- `n2/n2_onehot_0of1.csv`: tokens read as one-hot codes (S4.1). → `analysis.per_run_readouts --mode onehot`
- `diagnostic40_cropglobe_transductive_pca_kmeans.csv`: transductive controls; the rounded-PCA row is Table I's control. → `controls`, `analysis.check_controls`
- `diagnostic44_cropglobe_strict_pca_kmeans_refit.csv`: strict controls, fitted on the pool each strict encoder saw; the rounded-PCA rows are Table I's controls. → `controls`, `analysis.check_controls`
- `diagnostic62_pca_grid_variants.csv`: other quantizations of the PCA control (S4.1); its Estonia rounded-PCA row is Table I's control. → `analysis.check_controls`
- `diagnostic72_k_readout*.csv`: k-means readouts. Variant K2 is the letter's k-means vocabulary (five fits per row; Table I); K0 is the ten-band form, K1 and K3 are the other readouts (S4.1). → `controls`, `analysis.check_controls`, `analysis.kmeans_fits`
- `diagnostic74_k2_switch.csv`: a refit of the k-means vocabulary that reproduces every stored fit (S3).
- `diagnostic55_cropglobe_transductive_kmeans_5seed.csv`: five fits of the ten-band k-means, transductive (S4.1).
- `diagnostic66_w_window_controls.csv`: controls on the month's full window of observations (S4.1).
- `diagnostic46_*`, `diagnostic57_*`, `diagnostic63_*`: reconstruction on its own recipe, on the matched recipe, and at six dimensions (S2, S4.1).
- `diagnostic50_final_predictive_batch_probe.csv`: masked-prediction runs read with the gradient-boosting probe (S4.3).
- `diagnostic51_decomposition_continuous_probe.csv`: continuous-latent scores behind Table II's decomposition (S5.2).
- `diagnostic53_*`, `diagnostic54_*`, `diagnostic59_*`: rounding curve and grid distance of the first runs (S6).
- `diagnostic78_g_clustering.csv`: other clustering methods for the k-means control (S4.1).
- `diagnostic79_n_seeds.csv`: Table I's tests with per-column against pooled Holm correction and other run counts (S7).

## letter_numbers/: the analysis outputs behind the letter's numbers

- `B.csv`: Table I's tests (item S71). → `analysis.table1`
- `A.csv`, `CD.csv`, `E.csv`, `FH.csv`: the other numbers of the letter and the supplement, keyed by item (`item` column, numbered as the extended supplement's sections): Table I and Table II means, rounding and the share of the raw-band gain kept (A); weakest and strongest runs, spread (CD); bits per month, quantizations, forms of the k-means control (E); rounding curve, grid distance, gradient boosting, width (FH).

## table2_decomposition/: Table II and S5.3

- `twin_probe_rows_transductive.csv`, `twin_probe_rows_strict_estonia.csv`: probe scores of the models trained without the quantizer (`twin_noquant`, `twin_bands`, `twin_teacher`).
- `twin_step_tests.csv`: Table II's tests. → `analysis.step_tests`
- `twin_stop_epochs.csv`: stop epochs of the twins (S5.3, training budget).
- `teacher_twin_continued_*.csv`, `teacher_twin_second_continuation_decisions.csv`: the slow-teacher twins continued to early stopping. → `analysis.twins_continued`
- `bands_twin_continued_*.csv`: the band-value twins continued to early stopping. → `analysis.twins_continued`
- `continued_fsq_vs_continued_teacher_twin.csv`: both sides continued (S5.3).
- `strict_decomposition.csv`: the decomposition in the strict settings (S5.2).
- `continuous_pca_vs_latent.csv`: the learned continuous latent against continuous PCA (S5.2). → `analysis.control_variants`

## supplement_S3_controls/

- `kmeans_fits.csv`, `kmeans_fits_per_fit.csv`: the five fits of the k-means vocabulary per row: mean, sd and range. → `analysis.kmeans_fits`
- `bits_per_month_per_fit.csv`, `bits_per_month_summary.csv`: entropy of the controls' month codes on the evaluation region. → `analysis.bits_per_month`

## supplement_S4_sensitivity/

- `rounded_pca_fitpool_standardization.csv`: rounded PCA standardized on the encoder's fit pool (S4.1). → `analysis.control_variants`
- `spectral_index_control.csv`: a width-matched spectral-index control (S4.1). → `analysis.control_variants`
- `no_label_harmonization.csv`: Table I without label harmonization (S4.1). → `analysis.no_harmonization`
- `signflip_tests.csv`: the exact sign-flip test beside Table I's t-tests (S4.1, S7). → `analysis.signflip_tests`
- `estonia_official_fewshot.csv`: probe with Estonian labels on the official few-shot split (S4.2). → `analysis.fewshot_estonia`
- `estonia_kshot.csv`: the same on random draws of k parcels per class (S4.2).
- `gradient_boosting_raw.csv`, `gradient_boosting_raw_linux.csv`: gradient-boosting probe on the raw band means and the ten-band k-means, five random states, on Windows and on Linux (S4.3). → `analysis.gradient_boosting`
- `continued_runs_probe.csv`, `continued_runs_registry.csv`: the capped masked-prediction runs continued to early stopping, CropGlobe (S4.1).
- `continued_runs_estonia_*.csv`: the same for EuroCropsML (S4.1).
- `continued_runs_matched_tests.csv`, `continued_runs_matched_tests_estonia.csv`, `continued_runs_decomposition.csv`, `continued_runs_catchup.csv`: Table I and the decomposition with the continued runs (S4.1). → `analysis.continued_runs`

## supplement_S5_runs/

- `training_runs_list.csv`, `training_stop_epochs.csv`, `training_val_loss.csv`: stop epoch, stop reason, best epoch and validation loss of the 80 runs, from the training records (S5.1).
- `france_per_class_f1_runs.csv`, `france_per_class_f1_kmeans.csv`: per-class F1 in France (S5.1).

## supplement_S8_training/

- `parameter_counts.csv`, `probe_parameter_counts.csv`: learnable parameters of the tokenizers and of the probe (Method, S2). → `analysis.parameter_counts`
- `training_runs_mlflow_dump.csv`: the training records of the GPU node (start, end, status, last epoch and step per MLflow run). → `analysis.training_times`
- `training_times_per_run.csv`, `training_times_summary.csv`: training wall-clock of the 80 runs (S2). → `analysis.training_times`
- `capped_runs_*.csv`: the runs stopped at the epoch cap, and the estimate of the work to continue them (S8). In `capped_runs_estimate.csv`, `days_*_shared` assume the GPU shared with the other jobs of the time (half of it), `days_*_free` a free GPU.

## Names in these files

- `diagnosticNN_…`: NN is the analysis's number in the order the analyses were run; it carries no meaning beyond the file name.
- `w6` (`diagnostic63_w6_…`): six latent dimensions instead of four.
- The MLflow experiment names in the configurations and in `training_runs_mlflow_dump.csv` keep the original project prefix `croplang-fsq`, because they key the MLflow records.
- `runs/n_runs.json` and `supplement_S4_sensitivity/continued_runs_registry.csv` keep the original configuration paths (`src/training/configs/<name>.yaml`); the same files are now in `configs/letter/`.
