"""Per-band z-score normalisation utilities.

Stats are computed from training countries only and saved alongside
model checkpoints so evaluation can reproduce the exact same transform.

CropGlobe raw reflectance values are Sentinel-2 DNs (0–~16000).
The recommended pipeline is:
    1. Divide by ``dn_scale`` (default 10 000) → float in [0, ~1.6]
    2. Z-score per band (subtract mean, divide by std)

Usage
-----
    stats = compute_norm_stats(train_features)   # (N, T, B) float32
    stats.save("<run dir>/norm_stats.json")

    x_norm = stats.apply(x)                      # any (..., B) array
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class NormStats:
    """Z-score statistics for a single dataset split."""

    mean: np.ndarray   # (n_bands,)
    std: np.ndarray    # (n_bands,)
    dn_scale: float = 10_000.0

    # ── apply ────────────────────────────────────────────────────────────────

    def apply(self, x: np.ndarray) -> np.ndarray:
        """Normalise ``x`` in-place-compatible way.

        Parameters
        ----------
        x : ndarray, last dimension = n_bands (any leading shape is fine).
        """
        x = x / self.dn_scale
        x = (x - self.mean) / (self.std + 1e-8)
        return x

    def inverse(self, x: np.ndarray) -> np.ndarray:
        """Inverse-transform from normalised space back to DN scale."""
        x = x * (self.std + 1e-8) + self.mean
        x = x * self.dn_scale
        return x

    # ── persistence ──────────────────────────────────────────────────────────

    def save(self, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        obj = {
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
            "dn_scale": self.dn_scale,
        }
        path.write_text(json.dumps(obj, indent=2))

    @classmethod
    def load(cls, path: Path | str) -> "NormStats":
        obj = json.loads(Path(path).read_text())
        return cls(
            mean=np.array(obj["mean"], dtype=np.float32),
            std=np.array(obj["std"], dtype=np.float32),
            dn_scale=float(obj["dn_scale"]),
        )


def compute_norm_stats(
    features: np.ndarray,
    dn_scale: float = 10_000.0,
) -> NormStats:
    """Compute per-band mean and std from a feature array.

    Parameters
    ----------
    features : ndarray of shape (N, T, B) or (N, n_months, obs, B).
               All leading dimensions are flattened; stats computed over dim B.
    dn_scale : Divisor applied before computing stats (converts DN → [0, ~1.6]).
    """
    x = features.astype(np.float32) / dn_scale
    # Flatten everything except band dimension
    x_flat = x.reshape(-1, x.shape[-1])           # (N*T, B)
    mean = x_flat.mean(axis=0)                     # (B,)
    std = x_flat.std(axis=0) + 1e-8                # (B,) — never zero
    return NormStats(mean=mean, std=std, dn_scale=dn_scale)

