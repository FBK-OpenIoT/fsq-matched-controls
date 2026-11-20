"""Rounded PCA: the first control, matched to the tokenizer in dimensionality (four numbers per month) and grid (five levels).

The monthly band means are projected onto four principal components (scikit-learn PCA, random state 1904) fitted on the fit pool;
each component is standardized, clipped at +-2 standard deviations and rounded onto the integer grid {-2, ..., 2}, the grid FSQ
uses. Which samples the standardization is fitted on is set per row in controls.rows (STANDARDIZE). On EuroCropsML only valid
months enter the fits, and a missing month is zero.
"""
from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

SEED = 1904
N_COMPONENTS = 4
HALF = 2          # five levels: -2 .. 2


def fsq_round(z: np.ndarray) -> np.ndarray:
    return np.clip(z, -HALF, HALF).round()


def fit_pca(pool_rows: np.ndarray) -> PCA:
    """PCA on the fit pool's monthly vectors (rows of ten band means)."""
    return PCA(n_components=N_COMPONENTS, random_state=SEED).fit(pool_rows)


def project(pca: PCA, X3: np.ndarray) -> np.ndarray:
    """(N, months, 10) -> (N, months, 4) principal-component scores."""
    return pca.transform(X3.reshape(-1, X3.shape[-1])).reshape(X3.shape[0], X3.shape[1], N_COMPONENTS)


def rounded(scaler: StandardScaler, P3: np.ndarray, V: np.ndarray) -> np.ndarray:
    """Standardize, clip and round the scores; invalid months are zero. Returns (N, months * 4)."""
    Z = fsq_round(scaler.transform(P3.reshape(-1, N_COMPONENTS))).reshape(P3.shape)
    Z[~V] = 0.0
    return Z.reshape(len(Z), -1)


def scaler_on(P3_list: list[np.ndarray], V_list: list[np.ndarray]) -> StandardScaler:
    """Standardization fitted on the valid months of the given score arrays together."""
    return StandardScaler().fit(np.concatenate([P[V] for P, V in zip(P3_list, V_list)]))
