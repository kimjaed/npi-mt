# src/npi_mt/enscgp/types.py
from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class MTObservation:
    """Observed 1D invariant data on observation frequency grid."""
    freqs_hz: np.ndarray          # (F,)
    log_app_res: np.ndarray       # (F,)
    phase_deg: np.ndarray         # (F,)

    def stacked(self) -> np.ndarray:
        return np.concatenate([self.log_app_res, self.phase_deg], axis=0)  # (2F,)

@dataclass(frozen=True)
class MTEnsembleData:
    """
    Ensemble database on *ensemble frequency grid*.
    Store responses in the same units as used in EnsCGP:
      - log_app_res: natural log
      - phase_deg: degrees
      - log_rho: natural log resistivity, (Z, N)
    """
    freqs_hz: np.ndarray          # (Fe,)
    log_app_res: np.ndarray       # (Fe, N)
    phase_deg: np.ndarray         # (Fe, N)
    log_rho: np.ndarray           # (Z, N)

@dataclass(frozen=True)
class NoiseModel:
    """
    Observation noise model in the stacked space [log_app_res; phase_deg].
    Use scalars or vectors (per-frequency).
    """
    app_res_log_sigma: np.ndarray | float   # sigma for log(AppRes) (natural log)
    phase_deg_sigma: np.ndarray | float     # sigma for phase (degrees)

    def diag(self, F: int) -> np.ndarray:
        app = np.asarray(self.app_res_log_sigma, float)
        phs = np.asarray(self.phase_deg_sigma, float)

        if app.ndim == 0:
            app = np.full(F, float(app))
        if phs.ndim == 0:
            phs = np.full(F, float(phs))

        if app.shape != (F,) or phs.shape != (F,):
            raise ValueError("Noise sigmas must be scalar or shape (F,)")

        return np.concatenate([app**2, phs**2], axis=0)  # (2F,)

@dataclass(frozen=True)
class EnsCGPSolverConfig:
    damping: float = 1.0
    reg_eps: float = 1e-12
    perturb_observations: bool = True

@dataclass(frozen=True)
class NeighborConfig:
    metric: str = "cosine"   # "cosine" for now; can add others
    n_neighbors: int = 500

@dataclass(frozen=True)
class EnsCGPResult:
    log_rho_updated: np.ndarray       # (Z, Nn)
    log_app_res_updated: np.ndarray   # (F, Nn)
    phase_deg_updated: np.ndarray     # (F, Nn)
    neighbor_indices: np.ndarray      # (Nn,)
    neighbor_distances: np.ndarray    # (Nn,)
    nrms_mean: float
