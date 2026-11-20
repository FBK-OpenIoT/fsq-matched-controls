"""Tolerances of the reproduction checks on a platform other than the letter's (Windows, CPU, PyTorch 2.7.0).

Every check prints "identical" or the largest difference per metric, and compares every verdict; it passes only if every verdict
is identical and every difference is within the tolerance of its analysis. Values observed on Linux (CPU, the same pinned versions).
"""

# General: probe scores of the runs and of rounded PCA. lbfgs stops at 300 iterations before converging, and FSQ codes can flip at
# the rounding boundaries; observed up to 4.3e-4 (EuroCropsML masked prediction, seed 11: 3 of about 9.9 million token cells
# differ, macro-F1 5.8e-4).
GENERAL = 1e-3

# Probes that are far from convergence at 300 iterations: the spectral-index control (observed 1.32e-3) and the one-hot readout of
# the tokens (sparse features, observed 1.10e-3 in macro-F1).
UNCONVERGED = 2e-3

# The k-means vocabulary, mean of its five fits. The mini-batch k-means fits themselves differ across platforms (single fits up to
# 4.2e-3; a refit on the letter's platform reproduces every stored fit exactly); the means differ by up to 1.2e-3, against a
# fit-to-fit sd of up to 0.0016 (supplement S3).
KMEANS_MEAN = 3e-3

# Gradient boosting on the token values has no numeric bound: its splits fall on exact token values, so a flipped code can move a
# split, and macro-F1 over rare classes amplifies it (observed: masked prediction seed 97, France, macro-F1 1.15e-2; seed 11 and
# Australia identical). The check lists the differing cells and the number of flipped token cells instead.

# Why the k-means vocabulary differs: k-means fitted on the ten band means directly (results/runs, the ten-band form) is identical
# on Linux; the letter's vocabulary is fitted on principal-component scores, whose tiny float differences (BLAS) change the
# mini-batch path. Its entropy (bits per month, S3): single fits up to 0.046 bits, five-fit means up to 0.0077 bits.
KMEANS_BITS = 0.01

# Few-shot probe (S4.2), columns against the k-means vocabulary: the fits differ as above, and at k = 1 the probe trains on 123
# parcels; observed up to 4.6e-3 (accuracy margin, k = 1). The rounded-PCA columns use UNCONVERGED (observed 1.3e-3).
FEWSHOT_KMEANS = 5e-3

# Macro-F1 in Estonia (the run probes): it averages over many small classes and is about 0.08, so a small change in the probe's fit
# moves it by about 1e-3. Observed: masked prediction seed 37, tokens, 1.12e-3 (accuracy 2.8e-5); 6 of 8.6 million token cells of the
# probe pool differ from the letter's arrays (6 parcels), none of the Estonian evaluation parcels.
ESTONIA_MACRO_F1 = 2e-3

# Smoke test, regenerated statistics tables (Table I's S7.1 records, Table II's step tests): byte-identical on the letter's platform;
# on Linux some p-values differ in the last one or two floating-point digits (observed: relative 5.6e-16, 3 to 6 lines per table).
FLOAT_RELATIVE = 1e-12
