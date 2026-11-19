"""Token-space analysis utilities.

Implements the evaluation metrics from the research proposal (§3.3):
  - Codebook utilisation
  - Per-token spectral interpretation (mean spectrum per token)
  - t-SNE / UMAP dimensionality reduction
  - Jensen-Shannon Divergence between crop-conditional token distributions
  - Cross-region edit distance (Levenshtein on token sequences)
  - Reconstruction R² per band

All functions are standalone — they accept plain numpy arrays and return
either numpy arrays, dicts, or matplotlib Figures (return, don't show).
"""

from __future__ import annotations

import logging
from collections import Counter
from typing import Optional

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.spatial.distance import jensenshannon

log = logging.getLogger(__name__)

matplotlib.rcParams["figure.dpi"] = 120


# ── Codebook utilisation ──────────────────────────────────────────────────────

def codebook_utilisation(
    token_ids: np.ndarray,
    codebook_size: int,
) -> dict[str, float | int]:
    """Fraction and count of tokens actually used.

    Parameters
    ----------
    token_ids     : 1-D array of integer token IDs.
    codebook_size : Total possible tokens (n_levels ** n_dims).

    Returns
    -------
    dict with keys: n_used, codebook_size, utilisation (0–1), entropy_bits
    """
    counts = Counter(token_ids.tolist())
    n_used = len(counts)
    total = len(token_ids)

    # Normalised entropy (bits)
    probs = np.array(list(counts.values()), dtype=float) / total
    entropy = float(-np.sum(probs * np.log2(probs + 1e-12)))
    max_entropy = float(np.log2(codebook_size))

    return {
        "n_used": n_used,
        "codebook_size": codebook_size,
        "utilisation": n_used / codebook_size,
        "entropy_bits": entropy,
        "max_entropy_bits": max_entropy,
        "relative_entropy": entropy / (max_entropy + 1e-12),
    }


# ── Per-token spectral mean (phenological interpretation) ─────────────────────

def per_token_spectral_mean(
    token_ids: np.ndarray,            # (N,)
    windows: np.ndarray,              # (N, obs, n_bands) — normalised or raw
    codebook_size: int,
    band_names: Optional[list[str]] = None,
) -> dict[int, np.ndarray]:
    """Compute the mean spectral window for each token code.

    Tokens whose mean spectrum correspond to recognisable phenological stages
    (green-up, peak biomass, senescence, dormancy) validate the tokeniser.

    Returns
    -------
    dict mapping token_id → mean_window array (obs, n_bands), only for
    tokens that appear at least once.
    """
    token_means: dict[int, list[np.ndarray]] = {}
    for tid, win in zip(token_ids.tolist(), windows):
        token_means.setdefault(tid, []).append(win)

    return {tid: np.mean(np.stack(wins), axis=0) for tid, wins in token_means.items()}


def plot_per_token_spectra(
    token_means: dict[int, np.ndarray],
    band_names: Optional[list[str]] = None,
    top_n: int = 16,
    figsize: tuple[int, int] = (16, 10),
) -> plt.Figure:
    """Plot mean spectral signatures for the most frequent tokens.

    Parameters
    ----------
    token_means : Output of ``per_token_spectral_mean``.
    band_names  : Labels for the band axis.
    top_n       : Number of tokens to plot.
    """
    band_names = band_names or [f"B{i}" for i in range(next(iter(token_means.values())).shape[-1])]
    tokens = sorted(token_means.keys())[:top_n]
    n_cols = 4
    n_rows = int(np.ceil(len(tokens) / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, constrained_layout=True)
    axes = np.array(axes).flatten()

    for ax, tid in zip(axes, tokens):
        mean_win = token_means[tid]       # (obs, bands)
        for obs_idx in range(mean_win.shape[0]):
            ax.plot(band_names, mean_win[obs_idx], alpha=0.6, linewidth=0.8)
        ax.plot(band_names, mean_win.mean(axis=0), color="black", linewidth=2, label="mean")
        ax.set_title(f"Token {tid}", fontsize=9)
        ax.set_xlabel("Band", fontsize=7)
        ax.tick_params(labelsize=7)

    for ax in axes[len(tokens):]:
        ax.set_visible(False)

    fig.suptitle("Mean spectral signature per FSQ token", fontsize=12)
    return fig


# ── Dimensionality reduction (t-SNE / UMAP) ──────────────────────────────────

def _maybe_subsample(
    embeddings: np.ndarray,
    labels: np.ndarray,
    max_samples: int,
    rng_seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Subsample for large datasets to keep t-SNE tractable."""
    N = len(embeddings)
    if N <= max_samples:
        return embeddings, labels, np.arange(N)
    rng = np.random.default_rng(rng_seed)
    idx = rng.choice(N, max_samples, replace=False)
    return embeddings[idx], labels[idx], idx


def compute_tsne(
    embeddings: np.ndarray,
    perplexity: int = 30,
    random_state: int = 1904,
    max_samples: int = 10_000,
) -> tuple[np.ndarray, np.ndarray]:
    """Run t-SNE on embeddings, subsample if needed.

    Parameters
    ----------
    embeddings : (N, D) — e.g. z_q reshaped to (N, n_dims) or (N, n_months*n_dims).

    Returns
    -------
    xy    : (N_sub, 2)  2-D embedding
    idx   : (N_sub,)    indices into the original array
    """
    from sklearn.manifold import TSNE

    _, _, idx = _maybe_subsample(embeddings, np.zeros(len(embeddings)), max_samples, random_state)
    sub = embeddings[idx]
    log.info("Running t-SNE on %d samples (D=%d)…", len(sub), sub.shape[-1])
    xy = TSNE(n_components=2, perplexity=perplexity, random_state=random_state).fit_transform(sub)
    return xy, idx


def compute_umap(
    embeddings: np.ndarray,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    random_state: int = 1904,
    max_samples: int = 20_000,
) -> tuple[np.ndarray, np.ndarray]:
    """Run UMAP on embeddings (faster than t-SNE for large N)."""
    try:
        import umap
    except ImportError:
        raise ImportError("Install umap-learn: pip install umap-learn")

    _, _, idx = _maybe_subsample(embeddings, np.zeros(len(embeddings)), max_samples, random_state)
    sub = embeddings[idx]
    log.info("Running UMAP on %d samples (D=%d)…", len(sub), sub.shape[-1])
    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, random_state=random_state)
    xy = reducer.fit_transform(sub)
    return xy, idx


def plot_2d_embedding(
    xy: np.ndarray,
    color_labels: np.ndarray,
    label_names: Optional[dict[int, str]] = None,
    title: str = "",
    figsize: tuple[int, int] = (10, 8),
    alpha: float = 0.4,
    point_size: float = 4.0,
) -> plt.Figure:
    """Scatter plot coloured by integer labels (crop type OR geographic zone).

    Parameters
    ----------
    color_labels : (N,) int array.  Unique values become distinct colours.
    label_names  : Optional mapping int → display name used in the legend.
    """
    unique = np.unique(color_labels)
    cmap = matplotlib.colormaps["tab20"].resampled(len(unique))

    fig, ax = plt.subplots(figsize=figsize)
    for i, uid in enumerate(unique):
        mask = color_labels == uid
        name = label_names.get(int(uid), str(uid)) if label_names else str(uid)
        ax.scatter(
            xy[mask, 0], xy[mask, 1],
            s=point_size, alpha=alpha, color=cmap(i), label=name, linewidths=0,
        )
    ax.legend(markerscale=3, fontsize=8, loc="best")
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("Dim 1")
    ax.set_ylabel("Dim 2")
    ax.axis("off")
    return fig


# ── Jensen-Shannon Divergence ─────────────────────────────────────────────────

def token_distribution(
    token_ids: np.ndarray,
    codebook_size: int,
) -> np.ndarray:
    """Compute normalised token frequency histogram.

    Returns
    -------
    hist : (codebook_size,)  probabilities summing to 1.
    """
    hist = np.bincount(token_ids, minlength=codebook_size).astype(float)
    hist /= hist.sum() + 1e-12
    return hist


def jsd_matrix(
    token_ids_per_group: dict[str, np.ndarray],
    codebook_size: int,
) -> tuple[np.ndarray, list[str]]:
    """Compute pairwise Jensen-Shannon Divergence matrix across groups.

    Useful for:
      - Groups = crop types → JSD between phenological distributions
      - Groups = countries  → geographic divergence

    Parameters
    ----------
    token_ids_per_group : dict mapping group_name → 1-D token_id array.

    Returns
    -------
    matrix : (N_groups, N_groups) symmetric JSD matrix (values in [0, 1]).
    keys   : ordered list of group names (row/column labels).
    """
    keys = sorted(token_ids_per_group.keys())
    hists = {k: token_distribution(token_ids_per_group[k], codebook_size) for k in keys}
    n = len(keys)
    M = np.zeros((n, n))
    for i, k1 in enumerate(keys):
        for j, k2 in enumerate(keys):
            if i <= j:
                d = float(jensenshannon(hists[k1], hists[k2]) ** 2)
                M[i, j] = M[j, i] = d
    return M, keys


def plot_jsd_matrix(
    M: np.ndarray,
    labels: list[str],
    title: str = "Jensen-Shannon Divergence",
    figsize: tuple[int, int] = (8, 7),
) -> plt.Figure:
    """Heatmap of the JSD matrix."""
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(M, vmin=0, vmax=1, cmap="viridis")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(labels, fontsize=9)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center",
                    fontsize=7, color="white" if M[i, j] > 0.5 else "black")
    plt.colorbar(im, ax=ax, label="JSD²")
    ax.set_title(title)
    fig.tight_layout()
    return fig


# ── Edit distance (Levenshtein on token sequences) ───────────────────────────

def _levenshtein(a: np.ndarray, b: np.ndarray) -> int:
    """Standard Levenshtein distance between two integer sequences."""
    m, n = len(a), len(b)
    dp = np.arange(n + 1)
    for i in range(1, m + 1):
        prev = dp.copy()
        dp[0] = i
        for j in range(1, n + 1):
            dp[j] = min(prev[j] + 1, dp[j - 1] + 1, prev[j - 1] + (a[i - 1] != b[j - 1]))
    return int(dp[n])


def normalised_edit_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Levenshtein distance normalised to [0, 1] by max sequence length."""
    raw = _levenshtein(a, b)
    return raw / max(len(a), len(b), 1)


def cross_region_edit_distance(
    token_seqs_A: list[np.ndarray],    # parcels from region A
    token_seqs_B: list[np.ndarray],    # parcels from region B
    labels_A: np.ndarray,
    labels_B: np.ndarray,
    n_pairs: int = 500,
    rng_seed: int = 42,
) -> dict[str, float]:
    """Sample parcel pairs and compute mean edit distances.

    Tests the research claim:
      dist(same crop, region A vs B) < dist(different crops, within region A)

    Returns
    -------
    dict with keys:
      same_crop_cross_region    : mean NED for same-crop pairs across A↔B
      diff_crop_within_region   : mean NED for different-crop pairs within A
      ratio                     : above / below (< 1 is the desired result)
    """
    rng = np.random.default_rng(rng_seed)
    unique_labels = np.intersect1d(labels_A, labels_B)

    # Same crop, cross-region
    same_neds: list[float] = []
    for lbl in unique_labels:
        idx_a = np.where(labels_A == lbl)[0]
        idx_b = np.where(labels_B == lbl)[0]
        if len(idx_a) == 0 or len(idx_b) == 0:
            continue
        n = min(n_pairs // max(len(unique_labels), 1), len(idx_a), len(idx_b))
        for ia, ib in zip(rng.choice(idx_a, n), rng.choice(idx_b, n)):
            same_neds.append(normalised_edit_distance(token_seqs_A[ia], token_seqs_B[ib]))

    # Different crops, within region A
    all_labels_A = np.unique(labels_A)
    diff_neds: list[float] = []
    for _ in range(n_pairs):
        la, lb = rng.choice(all_labels_A, 2, replace=False)
        ia = rng.choice(np.where(labels_A == la)[0])
        ib = rng.choice(np.where(labels_A == lb)[0])
        diff_neds.append(normalised_edit_distance(token_seqs_A[ia], token_seqs_A[ib]))

    mean_same = float(np.mean(same_neds)) if same_neds else float("nan")
    mean_diff = float(np.mean(diff_neds)) if diff_neds else float("nan")
    return {
        "same_crop_cross_region": mean_same,
        "diff_crop_within_region": mean_diff,
        "ratio": mean_same / (mean_diff + 1e-8),
    }

