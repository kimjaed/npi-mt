# src/npi_mt/enscgp/update.py
from __future__ import annotations
import numpy as np
from scipy.linalg import solve

def enscgp_update(
    obs_stacked: np.ndarray,      # (2F,)
    data_ens: np.ndarray,         # (2F, N)
    log_rho_ens: np.ndarray,      # (Z, N)
    noise_diag: np.ndarray,       # (2F,) variances
    damping: float = 1.0,
    reg_eps: float = 1e-12,
    perturb_observations: bool = True,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Single ensemble conditioning:

      M^+ = M + C_md (C_dd + R)^{-1} (Y - D)

    where M = log_rho_ens, D = data_ens, Y = obs replicated (optionally perturbed).
    """
    if rng is None:
        rng = np.random.default_rng()

    obs = np.asarray(obs_stacked, float)
    D = np.asarray(data_ens, float)
    M = np.asarray(log_rho_ens, float)
    Rdiag = np.asarray(noise_diag, float)

    n_data, N = D.shape
    Z, N2 = M.shape
    if N != N2:
        raise ValueError("data_ens and log_rho_ens must have same ensemble size N")
    if obs.shape != (n_data,):
        raise ValueError("obs_stacked must have shape (2F,) matching data_ens rows")
    if Rdiag.shape != (n_data,):
        raise ValueError("noise_diag must have shape (2F,)")

    if N < 2:
        raise ValueError("Ensemble size must be >= 2")

    # Build Y
    y = obs[:, None]  # (n_data, 1)
    if perturb_observations:
        std = np.sqrt(np.maximum(Rdiag, 1e-30))[:, None]
        Y = y + std * rng.standard_normal(size=(n_data, N))
    else:
        Y = np.repeat(y, N, axis=1)

    # anomalies
    dM = M - M.mean(axis=1, keepdims=True)   # (Z, N)
    dD = D - D.mean(axis=1, keepdims=True)   # (n_data, N)

    C_md = (dM @ dD.T) / (N - 1)             # (Z, n_data)
    C_dd = (dD @ dD.T) / (N - 1)             # (n_data, n_data)

    A = C_dd.copy()
    A[np.diag_indices_from(A)] += np.maximum(Rdiag, 1e-30) + reg_eps

    innovation = Y - D                       # (n_data, N)
    X = solve(A, innovation)                 # (n_data, N)

    delta = C_md @ X                         # (Z, N)
    return M + damping * delta
