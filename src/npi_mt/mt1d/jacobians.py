# src/npi_mt/mt1d/jacobians.py
from __future__ import annotations

import numpy as np

from .constants import MU_0


def apparent_resistivity_from_Q(Q_surf: np.ndarray, omegas: np.ndarray) -> np.ndarray:
    """
    rho_a = |Q|^2 / (omega * mu_0)
    """
    Q = np.asarray(Q_surf, dtype=complex)
    w = np.asarray(omegas, dtype=float)
    if Q.shape != w.shape:
        raise ValueError(f"Q_surf and omegas must have same shape, got {Q.shape} vs {w.shape}")
    return (Q.real * Q.real + Q.imag * Q.imag) / (w * MU_0)


def phase_deg_from_Q(Q_surf: np.ndarray) -> np.ndarray:
    Q = np.asarray(Q_surf, dtype=complex)
    return np.degrees(np.arctan2(Q.imag, Q.real))


def J_appres_wrt_rho(
    Q_surf: np.ndarray,
    dQsurf_drho: np.ndarray,
    omegas: np.ndarray,
) -> np.ndarray:
    """
    Jacobian of rho_a with respect to linear rho (not log).
    Shapes:
      Q_surf: (F,)
      dQsurf_drho: (F,Z) complex
      omegas: (F,)
    Returns:
      (F,Z) float
    """
    Q = np.asarray(Q_surf, dtype=complex)[:, None]
    dQ = np.asarray(dQsurf_drho, dtype=complex)
    w = np.asarray(omegas, dtype=float)[:, None]

    term_1 = 2.0 / (w * MU_0)
    term_2 = Q.real * dQ.real
    term_3 = Q.imag * dQ.imag
    return term_1 * (term_2 + term_3)


def J_phase_deg_wrt_rho(
    Q_surf: np.ndarray,
    dQsurf_drho: np.ndarray,
) -> np.ndarray:
    """
    Jacobian of phase (degrees) with respect to linear rho (not log).
    Shapes:
      Q_surf: (F,)
      dQsurf_drho: (F,Z) complex
    Returns:
      (F,Z) float
    """
    Q = np.asarray(Q_surf, dtype=complex)[:, None]
    dQ = np.asarray(dQsurf_drho, dtype=complex)

    denom = np.square(np.abs(Q))
    denom = np.clip(denom, 1e-32, None)

    term_1 = 1.0 / denom
    term_2 = Q.real * dQ.imag
    term_3 = Q.imag * dQ.real

    J_phase_rad = term_1 * (term_2 - term_3)
    return J_phase_rad * (180.0 / np.pi)


def stack_jacobians_wrt_logrho(
    rho_a: np.ndarray,
    J_appres_wrt_rho_mat: np.ndarray,
    J_phase_deg_wrt_rho_mat: np.ndarray,
    rho_layers: np.ndarray,
    output_appres_in_log: bool = True,
) -> np.ndarray:
    """
    Build stacked Jacobian w.r.t log(rho_layers) for data vector:
      d = [log(rho_a); phase_deg]  (default)
    or
      d = [rho_a; phase_deg]       (if output_appres_in_log=False)

    Using chain rule:
      d/dlogrho = d/drho * rho

    For log(rho_a):
      dlog(rho_a)/drho = (1/rho_a) * drho_a/drho

    Returns
    -------
    J : (2F, Z) float
    """
    rho_a = np.asarray(rho_a, dtype=float)
    Jr = np.asarray(J_appres_wrt_rho_mat, dtype=float)
    Jp = np.asarray(J_phase_deg_wrt_rho_mat, dtype=float)
    rho_layers = np.asarray(rho_layers, dtype=float)

    F = rho_a.size
    Z = rho_layers.size
    if Jr.shape != (F, Z) or Jp.shape != (F, Z):
        raise ValueError(f"Jacobian shapes must be (F,Z)=({F},{Z}), got {Jr.shape} and {Jp.shape}")

    # Convert appres to log-space if requested
    if output_appres_in_log:
        rho_a_clip = np.clip(rho_a, 1e-32, None)
        J_app = Jr / rho_a_clip[:, None]  # dlog(rhoa)/drho
    else:
        J_app = Jr

    # Convert to w.r.t log(rho): multiply by rho per layer
    J_app_logrho = J_app * rho_layers[None, :]
    J_phase_logrho = Jp * rho_layers[None, :]

    return np.vstack([J_app_logrho, J_phase_logrho])
