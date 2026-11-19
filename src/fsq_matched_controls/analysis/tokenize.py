"""Tokenise a dataset with a frozen FSQ encoder.

Usage
-----
    model = FSQAutoencoder.load("best.ckpt")
    dm = CropGlobeDataModule(...)
    dm.setup()

    results = tokenize_parcel_dataset(model, dm.parcel_dataloader(), device="cpu")
    # results.token_sequences : list of (n_months,) np.ndarray per parcel
    # results.labels          : (N,) labels
    # results.countries       : (N,) country codes
    # results.z_q             : (N, n_months, n_dims) quantised embeddings
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader

log = logging.getLogger(__name__)


@dataclass
class TokenisationResult:
    """Container for tokenisation output."""

    token_sequences: list[np.ndarray]    # N × (n_months,) int arrays
    z_q: np.ndarray                      # (N, n_months, n_dims)  quantised codes
    labels: np.ndarray                   # (N,)  int crop labels
    countries: list[str]                 # (N,)  ISO country code per parcel

    @property
    def n_parcels(self) -> int:
        return len(self.token_sequences)

    def save(self, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            token_sequences=np.array(self.token_sequences, dtype=object),
            z_q=self.z_q,
            labels=self.labels,
            countries=np.array(self.countries),
        )
        log.info("TokenisationResult saved → %s", path)

    @classmethod
    def load(cls, path: Path | str) -> "TokenisationResult":
        data = np.load(path, allow_pickle=True)
        return cls(
            token_sequences=list(data["token_sequences"]),
            z_q=data["z_q"],
            labels=data["labels"],
            countries=list(data["countries"]),
        )


@torch.no_grad()
def tokenize_parcel_dataset(
    model,                    # FSQAutoencoder instance
    dataloader: DataLoader,
    device: str | torch.device = "cpu",
) -> TokenisationResult:
    """Tokenise all parcels in a parcel-level DataLoader.

    Each parcel provides a tensor of shape (n_months, obs_per_window, n_bands).
    The encoder is applied to each month window independently, yielding one
    token per window → a sequence of n_months tokens per parcel.

    Parameters
    ----------
    model      : Frozen FSQAutoencoder (or any object with .encode(x) method).
    dataloader : Parcel-level DataLoader (from CropGlobeDataModule.parcel_dataloader
                 or EuroCropsMLDataModule.parcel_dataloader).
    device     : Compute device.

    Returns
    -------
    TokenisationResult

    Note
    ----
    EuroCropsML batches carry a 4th element (``valid_mask``, marking
    structurally-missing months) that CropGlobe's batches don't have. It is
    accepted here (so tokenisation doesn't crash) but not yet excluded from
    the output — invalid months still get tokenised from their zero-filled
    placeholder, which pollutes codebook-utilisation/t-SNE/BOW stats on
    EuroCropsML's held-out set by whatever a zero-vector happens to quantize
    to. Fine as a known limitation for now; flag if that pollution needs
    fixing (would require carrying valid_mask through TokenisationResult and
    every downstream consumer).

    A 5th element (``obs_mask``, True at real/non-padded observations within
    a window) is used if present -- required for the variable-observation-
    window models (EuroCropsML's "_ext" architecture) so the encoder's
    pooling correctly ignores padding instead of averaging it in.
    """
    model = model.to(device).eval()

    all_token_seqs: list[np.ndarray] = []
    all_z_q: list[np.ndarray] = []
    all_labels: list[int] = []
    all_countries: list[str] = []

    for batch in dataloader:
        windows, labels, countries = batch[0], batch[1], batch[2]
        obs_mask = batch[4] if len(batch) >= 5 else None
        # windows: (B_parcels, n_months, obs, n_bands)
        B, W, obs, bands = windows.shape

        # Flatten parcel × month → one big batch for the encoder
        flat_windows = windows.view(B * W, obs, bands).to(device)
        flat_obs_mask = obs_mask.reshape(B * W, obs).to(device) if obs_mask is not None else None
        z_q, token_ids = model.encode(flat_windows, obs_mask=flat_obs_mask)  # (B*W, n_dims), (B*W,)

        # Reshape back to per-parcel sequences
        z_q_np = z_q.cpu().numpy().reshape(B, W, -1)        # (B, W, n_dims)
        ids_np = token_ids.cpu().numpy().reshape(B, W)       # (B, W)

        for i in range(B):
            all_token_seqs.append(ids_np[i])                 # (W,)
            all_z_q.append(z_q_np[i])                       # (W, n_dims)
            all_labels.append(int(labels[i].item()))
            all_countries.append(countries[i])

    return TokenisationResult(
        token_sequences=all_token_seqs,
        z_q=np.stack(all_z_q, axis=0),                      # (N, W, n_dims)
        labels=np.array(all_labels, dtype=np.int64),
        countries=all_countries,
    )


@torch.no_grad()
def tokenize_window_dataset(
    model,
    dataloader: DataLoader,
    device: str | torch.device = "cpu",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Tokenise a window-level DataLoader (returns flat arrays).

    Returns
    -------
    z_q       : (N_windows, n_dims)
    token_ids : (N_windows,)
    labels    : (N_windows,)
    """
    model = model.to(device).eval()
    z_list, id_list, lbl_list = [], [], []

    for windows, labels in dataloader:
        windows = windows.to(device)
        z_q, token_ids = model.encode(windows)
        z_list.append(z_q.cpu().numpy())
        id_list.append(token_ids.cpu().numpy())
        lbl_list.append(labels.numpy())

    return (
        np.concatenate(z_list),
        np.concatenate(id_list),
        np.concatenate(lbl_list),
    )

