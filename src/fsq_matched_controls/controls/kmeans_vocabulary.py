"""The k-means vocabulary: the second control, matched to the tokenizer in vocabulary size (625 codes).

The fit pool's monthly band means are projected onto the four principal components of the pool (as for rounded PCA) and the scores
are standardized on the pool. Mini-batch k-means with 625 centroids (k-means++ initialization, five initializations per fit,
batch 4,096, at most 300 iterations) is fitted on these standardized scores; each month is represented by the four coordinates
of its centroid, so the probe sees four numbers per month, as for the tokens. The vocabulary is fitted five times, with the
seeds below; the letter uses the mean of the five fits. A missing month (EuroCropsML) is zero.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from fsq_matched_controls.controls.rounded_pca import N_COMPONENTS, fit_pca

SEEDS = [1904, 42, 101, 2024, 4242]
N_CODES = 625


@dataclass
class Vocabulary:
    pca: PCA
    scaler: StandardScaler
    kmeans: MiniBatchKMeans

    def codes(self, X3: np.ndarray) -> np.ndarray:
        """(N, months, 10) -> (N, months) code ids."""
        S = self.scaler.transform(self.pca.transform(X3.reshape(-1, X3.shape[-1])))
        return self.kmeans.predict(S).reshape(X3.shape[0], X3.shape[1])

    def represent(self, X3: np.ndarray, V: np.ndarray) -> np.ndarray:
        """Each month as its centroid's four coordinates, invalid months zero. Returns (N, months * 4)."""
        Z = self.kmeans.cluster_centers_[self.codes(X3)].copy()
        Z[~V] = 0.0
        return Z.reshape(len(Z), -1)


def fit_pool_projection(pool_rows: np.ndarray) -> tuple[PCA, StandardScaler, np.ndarray]:
    """PCA of the pool and the standardization of its scores (shared by the five fits)."""
    pca = fit_pca(pool_rows)
    S = pca.transform(pool_rows)
    scaler = StandardScaler().fit(S)
    return pca, scaler, scaler.transform(S)


def fit_vocabulary(pca: PCA, scaler: StandardScaler, pool_std: np.ndarray, seed: int) -> Vocabulary:
    km = MiniBatchKMeans(n_clusters=N_CODES, random_state=seed, batch_size=4096, n_init=5, max_iter=300).fit(pool_std)
    assert km.cluster_centers_.shape == (N_CODES, N_COMPONENTS)
    return Vocabulary(pca, scaler, km)
