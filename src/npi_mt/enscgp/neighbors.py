# src/npi_mt/enscgp/neighbors.py
from __future__ import annotations
import numpy as np
from scipy.spatial.distance import cdist

def select_neighbors(
    obs_stacked: np.ndarray,     # (2F,)
    ens_stacked: np.ndarray,     # (2F, N)
    n_neighbors: int,
    metric: str = "cosine",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns (indices, distances) for the nearest neighbors in the ensemble.
    """
    if obs_stacked.ndim != 1:
        raise ValueError("obs_stacked must be 1D (2F,)")
    if ens_stacked.ndim != 2:
        raise ValueError("ens_stacked must be 2D (2F, N)")

    d = cdist(obs_stacked[None, :], ens_stacked.T, metric=metric)[0]  # (N,)
    idx = np.argsort(d)[:n_neighbors]
    return idx, d[idx]
