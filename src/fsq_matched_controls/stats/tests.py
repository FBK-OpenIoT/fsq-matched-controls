"""The letter's tests, moved unchanged from the analysis scripts.

- one_sided_t: learned-versus-control, one-sided one-sample t on the per-run margins (Table I).
- welch_greater: learned-versus-learned, one-sided Welch (Table I, masked prediction minus reconstruction).
- holm_adjust: Holm step-down adjusted p-values, as used for Table I (each term capped at 1 inside the running maximum).
- holm_adjust_capped: the same adjustment written as in the decomposition scripts (Table II, supplement S5.3):
  the running maximum is capped at 1 afterwards. Both give the same numbers; each script keeps its own form.
- welch_two_sided: two-sided Welch (Table II).
The interim-analysis multiplier (p times the number of looks of the row, capped at 1) is applied where the p-values are formed
(analysis.letter_runs.analyse).
"""
from __future__ import annotations

import numpy as np
from scipy import stats


def one_sided_t(x):
    return float(stats.ttest_1samp(x, 0.0, alternative="greater").pvalue)


def welch_greater(a, b):
    return float(stats.ttest_ind(a, b, equal_var=False, alternative="greater").pvalue)


def holm_adjust(p):
    """Holm step-down adjusted p-values (monotone), same order as the input."""
    p = np.asarray(p, float); m = len(p); order = np.argsort(p); adj = np.empty(m); run = 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (m - rank) * p[i])); adj[i] = run
    return adj


def holm_adjust_capped(p):
    p = np.asarray(p, float); o = np.argsort(p); n = len(p); adj = np.empty(n); run = 0.0
    for rank, i in enumerate(o):
        run = max(run, (n - rank) * p[i]); adj[i] = min(1.0, run)
    return adj


def welch_two_sided(x, a):
    """scipy ttest_ind(x, a, equal_var=False): returns the result object (statistic, pvalue)."""
    return stats.ttest_ind(x, a, equal_var=False)
