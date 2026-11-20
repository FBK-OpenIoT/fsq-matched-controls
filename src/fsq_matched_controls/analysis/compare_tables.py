"""Compare a regenerated table with the stored one.

The stored tables were written on the letter's platform (Windows), mostly with CRLF line endings. On the same platform a regenerated
table is byte-identical. Elsewhere it can differ in line endings, and computed statistics (p-values) can differ in their last one or
two floating-point digits; such a table counts as equal when every non-numeric cell is identical and every number is within
FLOAT_RELATIVE (analysis.tolerances) of the stored one.
"""
from __future__ import annotations

import filecmp
from pathlib import Path

import numpy as np
import pandas as pd

from fsq_matched_controls.analysis.tolerances import FLOAT_RELATIVE


def equal_to_float_precision(a: pd.DataFrame, b: pd.DataFrame) -> tuple[bool, float]:
    """(equal, largest relative difference): same shape and columns, non-numeric cells identical, numbers within FLOAT_RELATIVE."""
    if a.shape != b.shape or list(a.columns) != list(b.columns):
        return False, float("inf")
    num = [c for c in a.columns if a[c].dtype.kind in "fi" and b[c].dtype.kind in "fi"]
    other = [c for c in a.columns if c not in num]
    if not a[other].astype(str).equals(b[other].astype(str)):
        return False, float("inf")
    x, y = a[num].to_numpy(float), b[num].to_numpy(float)
    if not np.array_equal(np.isnan(x), np.isnan(y)):
        return False, float("inf")
    m = ~np.isnan(y)
    d = float(np.max(np.abs(x[m] - y[m]) / np.maximum(np.abs(y[m]), np.finfo(float).tiny))) if m.any() else 0.0
    return d <= FLOAT_RELATIVE, d


def compare_csv(out: Path, ref: Path) -> tuple[bool, str]:
    """(equal, how): 'byte-identical', 'identical apart from line endings', or 'equal to floating-point precision (...)'."""
    if filecmp.cmp(out, ref, shallow=False):
        return True, "byte-identical"
    if Path(out).read_bytes().replace(b"\r\n", b"\n") == Path(ref).read_bytes().replace(b"\r\n", b"\n"):
        return True, "identical apart from line endings"
    a = pd.read_csv(out, float_precision="round_trip"); b = pd.read_csv(ref, float_precision="round_trip")
    same, d = equal_to_float_precision(a, b)
    return same, (f"equal to floating-point precision (max relative difference {d:.1e})" if same else
                  f"DIFFERENT (max relative difference {d:.1e})")


def report(pairs) -> bool:
    """Print one line per (regenerated, stored) pair; True if all are equal."""
    ok = True
    for out, ref in pairs:
        same, how = compare_csv(out, ref)
        ok &= same
        print(f"{Path(out).name}: {how}")
    return ok
