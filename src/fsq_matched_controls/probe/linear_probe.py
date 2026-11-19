"""Probe evaluation for FSQ token sequences.

Trains a classifier on bag-of-tokens representations of parcels from training
zones, then evaluates it on held-out zone parcels.

Supported probe kinds
---------------------
    "logistic"           — L2-regularised Logistic Regression (default).
                           The standard SSL linear-probe (SimCLR / DINO protocol).
    "linear_svm"         — LinearSVC.  Often slightly better than logistic on
                           high-dimensional BOW features.
    "random_forest"      — RandomForestClassifier (200 trees).  Non-linear,
                           no feature scaling required.
    "gradient_boosting"  — sklearn HistGradientBoostingClassifier.  Fast
                           gradient boosting, no extra install required.
    "xgboost"            — XGBClassifier.  Requires ``pip install xgboost``.
                           Falls back to gradient_boosting with a warning if
                           xgboost is not installed.

Pipeline
--------
1. Harmonise label spaces: classes not shared by train AND test sets are
   remapped to class 6 ("Other") in both splits.
2. For each parcel, build a bag-of-tokens histogram
   (token_id → count, normalised to sum 1).
3. Fit probe on training-zone histograms + harmonised labels.
4. Evaluate on held-out-zone histograms using harmonised held-out labels.

Usage
-----
    from fsq_matched_controls.probe.linear_probe import (
        harmonize_label_sets,
        build_bow_features, train_linear_probe, eval_linear_probe
    )

    train_labels_h, test_labels_h, info = harmonize_label_sets(
        train_labels, test_labels
    )
    train_bow = build_bow_features(train_token_seqs, codebook_size)

    # Default (logistic regression — SSL standard):
    probe = train_linear_probe(train_bow, train_labels_h)

    # Random forest:
    probe = train_linear_probe(train_bow, train_labels_h, kind="random_forest")

    results = eval_linear_probe(probe, held_out_bow, test_labels_h)
    print(results["accuracy"])
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import (
    RandomForestClassifier,
    HistGradientBoostingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt

log = logging.getLogger(__name__)

# ── Probe catalogue ───────────────────────────────────────────────────────────

#: All supported probe kinds.
PROBE_KINDS: list[str] = [
    "logistic",
    "linear_svm",
    "random_forest",
    "gradient_boosting",
    "xgboost",
]


def _build_estimator(kind: str, random_state: int, **kwargs):
    """Return an unfitted sklearn-compatible estimator for the given kind."""
    if kind == "logistic":
        return LogisticRegression(
            max_iter=kwargs.get("max_iter", 1000),
            C=kwargs.get("C", 1.0),
            solver="lbfgs",
            random_state=random_state,
            n_jobs=-1,
        )
    elif kind == "linear_svm":
        # LinearSVC doesn't expose predict_proba; wrap in CalibratedClassifierCV
        # so eval_linear_probe can optionally use probabilities later.
        base = LinearSVC(
            max_iter=kwargs.get("max_iter", 2000),
            C=kwargs.get("C", 1.0),
            random_state=random_state,
        )
        return CalibratedClassifierCV(base, cv=3)
    elif kind == "random_forest":
        return RandomForestClassifier(
            n_estimators=kwargs.get("n_estimators", 200),
            max_depth=kwargs.get("max_depth", None),
            n_jobs=-1,
            random_state=random_state,
        )
    elif kind == "gradient_boosting":
        return HistGradientBoostingClassifier(
            max_iter=kwargs.get("max_iter", 200),
            learning_rate=kwargs.get("learning_rate", 0.1),
            random_state=random_state,
        )
    elif kind == "xgboost":
        try:
            from xgboost import XGBClassifier
            return XGBClassifier(
                n_estimators=kwargs.get("n_estimators", 200),
                learning_rate=kwargs.get("learning_rate", 0.1),
                use_label_encoder=False,
                eval_metric="mlogloss",
                n_jobs=-1,
                random_state=random_state,
            )
        except ImportError:
            log.warning(
                "xgboost not installed — falling back to gradient_boosting. "
                "Install with: pip install xgboost"
            )
            return _build_estimator("gradient_boosting", random_state, **kwargs)
    else:
        raise ValueError(
            f"Unknown probe kind '{kind}'. Choose from: {PROBE_KINDS}"
        )

#: Sentinel for "Other" / unknown class — always 6 in CropGlobe label mapping.
OTHER_CLASS: int = 6


# ── Label harmonisation ───────────────────────────────────────────────────────

def harmonize_label_sets(
    train_labels: np.ndarray,
    test_labels: np.ndarray,
    other_class: int = OTHER_CLASS,
    label_names: Optional[dict[int, str]] = None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Align label spaces between train and test by remapping non-shared
    classes to ``other_class`` (default 6 = "Other").

    According to the paper protocol:
    - Classes present in **both** splits are kept as-is.
    - Classes that exist only in train **or** only in test are remapped to
      ``other_class`` in that split, so the probe is never asked to predict
      a class it could not have learned.

    Parameters
    ----------
    train_labels : (N_train,) integer labels.
    test_labels  : (N_test,)  integer labels.
    other_class  : Target class for non-shared labels. Default 0.
    label_names  : Optional int→name mapping for logging.

    Returns
    -------
    train_labels_h : (N_train,) remapped labels.
    test_labels_h  : (N_test,)  remapped labels.
    info           : dict with keys:
        ``shared_classes``      — set of classes kept in both splits.
        ``train_only_classes``  — classes remapped to other in train.
        ``test_only_classes``   — classes remapped to other in test.
        ``n_classes``           — number of distinct classes after harmonisation.
    """
    train_set = set(np.unique(train_labels).tolist())
    test_set  = set(np.unique(test_labels).tolist())

    shared     = train_set & test_set
    train_only = train_set - test_set
    test_only  = test_set  - train_set

    def _remap(labels: np.ndarray, drop: set) -> np.ndarray:
        out = labels.copy()
        for c in drop:
            out[labels == c] = other_class
        return out

    train_h = _remap(train_labels, train_only)
    test_h  = _remap(test_labels,  test_only)

    def _fmt(classes: set) -> str:
        if not classes:
            return "∅"
        if label_names:
            return ", ".join(f"{label_names.get(c, c)}({c})" for c in sorted(classes))
        return ", ".join(str(c) for c in sorted(classes))

    log.info("Label harmonisation:")
    log.info("  Shared classes    : %s", _fmt(shared))
    log.info("  Train-only → Other: %s", _fmt(train_only))
    log.info("  Test-only  → Other: %s", _fmt(test_only))

    info = {
        "shared_classes":     shared,
        "train_only_classes": train_only,
        "test_only_classes":  test_only,
        "n_classes":          len(set(np.unique(train_h).tolist())),
    }
    return train_h, test_h, info


# ── Feature building ──────────────────────────────────────────────────────────

def build_bow_features(
    token_sequences: list[np.ndarray],
    codebook_size: int,
    normalise: bool = True,
) -> np.ndarray:
    """Build bag-of-tokens count matrix for a list of parcel token sequences.

    Parameters
    ----------
    token_sequences : N × (n_months,) arrays of token IDs.
    codebook_size   : Total number of possible tokens.
    normalise       : If True, normalise each row to sum to 1.

    Returns
    -------
    bow : (N, codebook_size) float32 array.
    """
    N = len(token_sequences)
    bow = np.zeros((N, codebook_size), dtype=np.float32)
    for i, seq in enumerate(token_sequences):
        for tid in seq:
            if 0 <= tid < codebook_size:
                bow[i, tid] += 1.0
    if normalise:
        row_sums = bow.sum(axis=1, keepdims=True) + 1e-12
        bow /= row_sums
    return bow


def build_embedding_features(z_q: np.ndarray) -> np.ndarray:
    """Flatten parcel embeddings (n_months × n_dims) to 1-D feature vector.

    Useful as an alternative to bag-of-tokens — preserves temporal ordering.

    Parameters
    ----------
    z_q : (N, n_months, n_dims)

    Returns
    -------
    features : (N, n_months * n_dims)
    """
    N = z_q.shape[0]
    return z_q.reshape(N, -1).astype(np.float32)


def build_raw_features(dataloader) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Collect mean-per-month spectral features directly from a parcel DataLoader.

    No encoder involved — this is the baseline that tests whether the raw
    normalised reflectance already separates crop types.

    Each parcel window has shape ``(n_months, obs_per_window, n_bands)``.
    We average over ``obs_per_window`` to get ``(n_months, n_bands)`` and
    flatten to ``(n_months * n_bands,)``.

    Parameters
    ----------
    dataloader : DataLoader yielding ``(windows, labels, countries)`` batches
                 as produced by ``CropGlobeParcelDataset`` / ``_parcel_collate``,
                 or ``(windows, labels, countries, valid_mask, obs_mask)`` for
                 datasets with padded, variable-observation windows
                 (EuroCropsML's "_ext" architecture) -- when ``obs_mask`` is
                 present, the mean is computed only over real (non-padded)
                 observations, otherwise padding zeros would bias it.

    Returns
    -------
    features  : (N, n_months * n_bands) float32
    labels    : (N,) int64
    countries : list of country-code strings (length N)
    """
    import torch
    all_feats, all_labels, all_cc = [], [], []
    with torch.no_grad():
        for batch in dataloader:
            windows, labels, countries = batch[0], batch[1], batch[2]
            obs_mask = batch[4] if len(batch) >= 5 else None
            # windows: (B, n_months, obs, bands)
            if obs_mask is None:
                mean_spec = windows.mean(dim=2)           # (B, n_months, bands)
            else:
                m = obs_mask.unsqueeze(-1).to(windows.dtype)          # (B, n_months, obs, 1)
                denom = m.sum(dim=2).clamp(min=1e-6)                   # (B, n_months, 1)
                mean_spec = (windows * m).sum(dim=2) / denom           # (B, n_months, bands)
            flat = mean_spec.reshape(mean_spec.shape[0], -1)  # (B, n_months*bands)
            all_feats.append(flat.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            all_cc.extend(countries)
    features = np.concatenate(all_feats, axis=0).astype(np.float32)
    labels_np = np.concatenate(all_labels, axis=0)
    return features, labels_np, all_cc


def build_pca_features(
    train_feats: np.ndarray,
    test_feats: np.ndarray,
    n_components: Optional[int] = None,
    variance_threshold: float = 0.95,
) -> tuple[np.ndarray, np.ndarray, PCA]:
    """Fit PCA on training features and project both splits.

    Parameters
    ----------
    train_feats       : (N_train, D) raw features.
    test_feats        : (N_test,  D) raw features (same normalisation).
    n_components      : Exact number of PCA components.  If None, choose
                        automatically to explain ``variance_threshold`` variance.
    variance_threshold: Used only when ``n_components`` is None.

    Returns
    -------
    train_pca : (N_train, K) float32
    test_pca  : (N_test,  K) float32
    pca       : fitted ``sklearn.decomposition.PCA`` object
    """
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_feats)
    test_scaled  = scaler.transform(test_feats)

    if n_components is None:
        # Pick K that explains `variance_threshold` of variance
        pca_full = PCA().fit(train_scaled)
        cumvar = np.cumsum(pca_full.explained_variance_ratio_)
        n_components = int(np.searchsorted(cumvar, variance_threshold) + 1)
        log.info("PCA: %d components explain %.1f%% variance", n_components, variance_threshold * 100)

    pca = PCA(n_components=n_components, random_state=1904)
    train_pca = pca.fit_transform(train_scaled).astype(np.float32)
    test_pca  = pca.transform(test_scaled).astype(np.float32)
    log.info(
        "PCA: %d → %d dims, explained variance: %.3f",
        train_feats.shape[1], n_components,
        pca.explained_variance_ratio_.sum(),
    )
    return train_pca, test_pca, pca


# ── Training ──────────────────────────────────────────────────────────────────

#: Probe kinds that require feature scaling (zero-mean, unit-variance).
_NEEDS_SCALING: set[str] = {"logistic", "linear_svm"}


def train_linear_probe(
    features: np.ndarray,
    labels: np.ndarray,
    kind: str = "logistic",
    random_state: int = 1904,
    scale_features: Optional[bool] = None,
    **estimator_kwargs,
) -> dict:
    """Train a probe classifier on frozen token features.

    Parameters
    ----------
    features       : (N_train, D) array — bag-of-tokens or embedding features.
    labels         : (N_train,) int array.
    kind           : Classifier to use.  One of ``PROBE_KINDS``:
                     ``"logistic"`` (default), ``"linear_svm"``,
                     ``"random_forest"``, ``"gradient_boosting"``, ``"xgboost"``.
    random_state   : Random seed for reproducibility.
    scale_features : Whether to apply StandardScaler before fitting.
                     Defaults to ``True`` for linear probes (logistic / linear_svm)
                     and ``False`` for tree-based probes.
    **estimator_kwargs
                   : Extra keyword arguments forwarded to the estimator
                     (e.g. ``C=0.1``, ``n_estimators=300``, ``max_iter=2000``).

    Returns
    -------
    dict with keys:
      ``"probe"``         : fitted estimator
      ``"scaler"``        : fitted StandardScaler (or None)
      ``"train_acc"``     : training accuracy
      ``"train_classes"`` : list of classes the probe knows about
      ``"kind"``          : the probe kind string
    """
    if scale_features is None:
        scale_features = kind in _NEEDS_SCALING

    scaler: Optional[StandardScaler] = None
    if scale_features:
        scaler = StandardScaler()
        features = scaler.fit_transform(features)

    train_classes = np.unique(labels).tolist()

    # XGBoost requires contiguous [0, 1, ..., N-1] class indices.
    # We encode labels internally and decode predictions in eval_linear_probe.
    label_enc: Optional[LabelEncoder] = None
    fit_labels = labels
    if kind == "xgboost":
        label_enc = LabelEncoder()
        fit_labels = label_enc.fit_transform(labels)

    log.info(
        "Training probe [%s]: %d samples, %d features, %d classes",
        kind, features.shape[0], features.shape[1], len(train_classes),
    )

    probe = _build_estimator(kind, random_state, **estimator_kwargs)
    probe.fit(features, fit_labels)
    train_acc = float(accuracy_score(labels, label_enc.inverse_transform(probe.predict(features)) if label_enc else probe.predict(features)))
    log.info("Train accuracy [%s]: %.4f", kind, train_acc)

    return {
        "probe":         probe,
        "scaler":        scaler,
        "label_enc":     label_enc,   # None for all non-XGBoost probes
        "train_acc":     train_acc,
        "train_classes": train_classes,
        "kind":          kind,
    }


# ── Evaluation ────────────────────────────────────────────────────────────────

def eval_linear_probe(
    probe_dict: dict,
    features: np.ndarray,
    labels: np.ndarray,
    label_names: Optional[dict[int, str]] = None,
) -> dict:
    """Evaluate trained probe on a test set.

    Parameters
    ----------
    probe_dict  : Output of ``train_linear_probe``.
    features    : (N_test, D) test features.
    labels      : (N_test,) ground-truth labels — should already be harmonised
                  via ``harmonize_label_sets`` so only shared classes appear.
    label_names : Optional mapping int → crop name for the report.

    Returns
    -------
    dict with keys:
      accuracy, macro_f1, weighted_f1, per_class_report (str), confusion_matrix,
      n_unknown   — samples whose true label was unknown to the probe (should be 0
                    after harmonisation; shown as warning if > 0).
    """
    probe  = probe_dict["probe"]
    scaler = probe_dict.get("scaler")
    label_enc: Optional[LabelEncoder] = probe_dict.get("label_enc")
    train_classes: set = set(probe_dict.get("train_classes", probe.classes_.tolist()))

    if scaler is not None:
        features = scaler.transform(features)

    # Samples whose label the probe has never seen — should = 0 after harmonisation
    unknown_mask = ~np.isin(labels, list(train_classes))
    n_unknown = int(unknown_mask.sum())
    if n_unknown > 0:
        log.warning(
            "%d test samples have labels unknown to the probe. "
            "Call harmonize_label_sets() before training/eval to avoid this.",
            n_unknown,
        )

    raw_preds = probe.predict(features)
    # XGBoost predicts encoded indices — map back to original label space
    preds = label_enc.inverse_transform(raw_preds.astype(int)) if label_enc is not None else raw_preds
    acc = float(accuracy_score(labels, preds))
    macro_f1    = float(f1_score(labels, preds, average="macro",    zero_division=0))
    weighted_f1 = float(f1_score(labels, preds, average="weighted", zero_division=0))

    # Build target_names that covers all labels actually present in test set
    present_labels = sorted(np.unique(labels).tolist())
    target_names = (
        [label_names.get(int(c), str(c)) for c in present_labels]
        if label_names else [str(c) for c in present_labels]
    )
    report = classification_report(
        labels, preds,
        labels=present_labels,
        target_names=target_names,
        zero_division=0,
    )
    cm = confusion_matrix(labels, preds, labels=present_labels)

    log.info(
        "Test accuracy: %.4f  |  Macro-F1: %.4f  |  unknown samples: %d",
        acc, macro_f1, n_unknown,
    )
    return {
        "accuracy":          acc,
        "macro_f1":          macro_f1,
        "weighted_f1":       weighted_f1,
        "per_class_report":  report,
        "confusion_matrix":  cm,
        "present_labels":    present_labels,
        "n_unknown":         n_unknown,
    }


def build_cont_features(
    model,
    dataloader,
    device: str = "cpu",
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Extract pre-quantization continuous encoder embeddings from a parcel DataLoader.

    Works with both ``FSQAutoencoder`` (calls ``model.encoder``) and
    ``FSQBackboneAutoencoder`` (calls ``model.encode``).  The result is the
    continuous bottleneck vector **before** FSQ rounding — this is what the
    encoder actually learned to represent.

    Each parcel has shape ``(n_tokens, obs_per_window, n_bands)``.
    For each token window the encoder produces a ``d`` -dimensional vector,
    so the parcel is flattened to ``(n_tokens * d,)``.

    Parameters
    ----------
    model      : FSQAutoencoder or FSQBackboneAutoencoder (frozen, on any device).
    dataloader : Parcel-level DataLoader yielding ``(windows, labels, countries)``.
    device     : Compute device string (``"cpu"``, ``"cuda"``).

    Returns
    -------
    features  : (N, n_tokens * d) float32
    labels    : (N,) int64
    countries : list of country-code strings (length N)
    """
    import torch
    model = model.to(device).eval()
    all_feats, all_labels, all_cc = [], [], []

    with torch.no_grad():
        for batch in dataloader:
            windows, labels, countries = batch[0], batch[1], batch[2]
            # windows: (B, n_tokens, obs, bands)
            B, W, obs, bands = windows.shape
            flat = windows.view(B * W, obs, bands).to(device)

            # Support both FSQAutoencoder (has .encoder attr) and FSQBackboneAutoencoder (.encode)
            if hasattr(model, "encoder"):
                z = model.encoder(flat)   # (B*W, d)
            else:
                z = model.encode(flat)    # (B*W, d)

            z_np = z.cpu().numpy().reshape(B, W, -1)  # (B, n_tokens, d)
            flat_feats = z_np.reshape(B, -1)           # (B, n_tokens * d)

            all_feats.append(flat_feats.astype(np.float32))
            all_labels.append(labels.cpu().numpy())
            all_cc.extend(countries)

    features = np.concatenate(all_feats, axis=0)
    labels_np = np.concatenate(all_labels, axis=0)
    return features, labels_np, all_cc


def build_pred_features(
    model,
    dataloader,
    device: str = "cpu",
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Extract context transformer representations from ``FSQPredictiveSSL``.

    All token positions are visible (no masking applied).  The context
    transformer produces a ``d_model``-dimensional vector per position,
    capturing the full phenological context.  Parcels are flattened to
    ``(n_tokens * d_model,)``.

    This is the key feature type that captures what the predictive SSL
    context model learned, distinct from plain FSQ embeddings (EMB).

    Parameters
    ----------
    model      : ``FSQPredictiveSSL`` instance (frozen, in eval mode).
    dataloader : Parcel-level DataLoader yielding ``(windows, labels, countries)``.
    device     : Compute device string.

    Returns
    -------
    features  : (N, n_tokens * d_model) float32
    labels    : (N,) int64
    countries : list of country-code strings (length N)
    """
    import torch
    model = model.to(device).eval()
    all_feats, all_labels, all_cc = [], [], []

    with torch.no_grad():
        for batch in dataloader:
            windows, labels, countries = batch[0], batch[1], batch[2]
            windows = windows.to(device)
            ctx = model.encode_sequence(windows)                    # (B, N, d_model)
            flat = ctx.cpu().numpy().reshape(len(windows), -1)      # (B, N*d_model)
            all_feats.append(flat.astype(np.float32))
            all_labels.append(labels.cpu().numpy())
            all_cc.extend(countries)

    features  = np.concatenate(all_feats, axis=0)
    labels_np = np.concatenate(all_labels, axis=0)
    return features, labels_np, all_cc


def compute_indist_cv_score(
    features: np.ndarray,
    labels: np.ndarray,
    kind: str = "logistic",
    cv: int = 5,
    random_state: int = 1904,
    n_jobs: int = -1,
    **estimator_kwargs,
) -> float:
    """Cross-validated in-distribution accuracy — unbiased alternative to train-set accuracy.

    Tree-based probes (random_forest, gradient_boosting, xgboost) memorise the
    training set → training-set accuracy ≈ 1.0 regardless of feature quality.
    Cross-validation gives a realistic estimate of in-distribution generalisation
    that can be meaningfully compared to cross-region accuracy.

    The **transfer gap** should always be:

        gap = compute_indist_cv_score(...) - cross_region_acc

    NOT ``train_acc - cross_region_acc``.

    Parameters
    ----------
    features  : (N_train, D) feature matrix.
    labels    : (N_train,) integer labels (may be non-contiguous, e.g. [0,1,2,3,6]).
    kind      : Probe kind — same values as ``train_linear_probe``.
    cv        : Number of cross-validation folds (default 5).

    Returns
    -------
    float : mean CV accuracy across folds.
    """
    from sklearn.model_selection import cross_val_score
    from sklearn.pipeline import Pipeline

    scale = kind in _NEEDS_SCALING
    estimator = _build_estimator(kind, random_state, **estimator_kwargs)

    if scale:
        from sklearn.pipeline import Pipeline as _Pipeline
        pipe = _Pipeline([("scaler", StandardScaler()), ("clf", estimator)])
    else:
        pipe = estimator

    # XGBoost: encode labels to contiguous range for CV folds
    fit_labels = labels
    if kind == "xgboost":
        le = LabelEncoder()
        fit_labels = le.fit_transform(labels)

    scores = cross_val_score(pipe, features, fit_labels, cv=cv, scoring="accuracy", n_jobs=n_jobs)
    return float(scores.mean())


def plot_confusion_matrix(
    cm: np.ndarray,
    label_names: Optional[list[str]] = None,
    title: str = "Confusion Matrix",
    figsize: tuple[int, int] = (8, 7),
    normalise: bool = True,
) -> plt.Figure:
    """Plot normalised confusion matrix."""
    if normalise:
        cm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-12)

    n = cm.shape[0]
    ticks = label_names or [str(i) for i in range(n)]

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1 if normalise else None)
    plt.colorbar(im, ax=ax)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(ticks, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(ticks, fontsize=9)
    for i in range(n):
        for j in range(n):
            val = cm[i, j]
            fmt = f"{val:.2f}" if normalise else str(int(val))
            ax.text(j, i, fmt, ha="center", va="center",
                    fontsize=7, color="white" if val > 0.5 else "black")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    fig.tight_layout()
    return fig

