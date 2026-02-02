# src/npi_mt/mt1d/sensitivities.py
from __future__ import annotations

import cmath
import numpy as np

from .constants import MU_0
from .transfer import compute_Q_interfaces


def compute_local_derivatives(
    rho_ohm_m: np.ndarray,
    thicknesses_m: np.ndarray,
    omegas: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute per-frequency recursion derivatives.

    Returns
    -------
    Q_all : (F, Z) complex
        Q at all interfaces in the internal (bottom->top) recursion order.
        Surface is Q_all[:, -1].
    dQ_drho : (F, Z) complex
        Local derivative dQ_i/drho_i in recursion order.
    dQ_dQprev : (F, Z-1) complex
        Local derivative dQ_i/dQ_{i-1} for i=1..Z-1 in recursion order.
        (No entry for the bottom half-space.)
    """
    rho = np.asarray(rho_ohm_m, dtype=float)
    thk = np.asarray(thicknesses_m, dtype=float)
    w = np.asarray(omegas, dtype=float)

    if w.ndim != 1:
        raise ValueError("omegas must be 1D")
    if np.any(w <= 0):
        raise ValueError("All omegas must be > 0")

    # Flip like the recursion
    rho_b2t = np.flip(rho)
    thk_b2t = np.flip(thk)
    Z = rho.size
    F = w.size

    Q_all = np.empty((F, Z), dtype=complex)
    dQ_drho = np.empty((F, Z), dtype=complex)
    dQ_dQprev = np.empty((F, Z - 1), dtype=complex)

    for fi, omega in enumerate(w):
        Q = compute_Q_interfaces(rho, thk, omega)  # (Z,) bottom->top
        Q_all[fi, :] = Q

        # Bottom layer derivative
        rho0 = rho_b2t[0]
        alpha0 = cmath.sqrt((omega * MU_0 / rho0) * 1j)
        dQ_drho[fi, 0] = 0.5 * alpha0 

        # Upper layers
        for i in range(1, Z):
            rho_i = rho_b2t[i]
            alpha = cmath.sqrt((omega * MU_0 / rho_i) * 1j)
            d = thk_b2t[i - 1]

            Q_prev = Q[i - 1]
            t = cmath.tanh(alpha * d)

            # Define numerator/denominator like legacy
            Q1_j = Q_prev + alpha * rho_i * t
            Q2_j = 1.0 + (Q_prev / (alpha * rho_i)) * t

            # dQ_i/dQ_prev
            dQ1_dQ = 1.0
            dQ2_dQ = t / (alpha * rho_i)
            dQ_dQprev[fi, i - 1] = (dQ1_dQ * Q2_j - dQ2_dQ * Q1_j) / (Q2_j * Q2_j)

            # dQ_i/drho_i
            da_drho = -alpha / (2.0 * rho_i)
            dtanh_drho = (1.0 - (t * t)) * d * da_drho

            dQ1_drho = alpha * t + rho_i * (da_drho * t + alpha * dtanh_drho)

            term_1 = Q_prev * t
            term_2 = alpha * rho_i
            d_term1_drho = Q_prev * dtanh_drho
            d_term2_drho = da_drho * rho_i + alpha
            dQ2_drho = (d_term1_drho * term_2 - d_term2_drho * term_1) / (term_2 * term_2)

            dQ_drho[fi, i] = (dQ1_drho * Q2_j - dQ2_drho * Q1_j) / (Q2_j * Q2_j)

    return Q_all, dQ_drho, dQ_dQprev


def propagate_surface_derivative(
    dQ_drho: np.ndarray,
    dQ_dQprev: np.ndarray,
) -> np.ndarray:
    """
    Propagate local derivatives to surface derivative dQ_surf/drho_k.

    Inputs follow recursion order (bottom->top):
      dQ_drho: (F, Z)
      dQ_dQprev: (F, Z-1) where column j is derivative at layer i=j+1 w.r.t Q_prev

    Returns
    -------
    dQsurf_drho : (F, Z) complex
        Derivatives of surface Q with respect to each rho in recursion order.
    """
    dQ_drho = np.asarray(dQ_drho, dtype=complex)
    dQ_dQprev = np.asarray(dQ_dQprev, dtype=complex)

    dQ_dQprev = np.flip(dQ_dQprev, axis=1)
    dQ_drho = np.flip(dQ_drho, axis=1)

    if dQ_drho.ndim != 2:
        raise ValueError("dQ_drho must be 2D (F,Z)")
    if dQ_dQprev.ndim != 2:
        raise ValueError("dQ_dQprev must be 2D (F,Z-1)")

    F, Z = dQ_drho.shape
    if dQ_dQprev.shape != (F, Z - 1):
        raise ValueError(f"dQ_dQprev must have shape (F,Z-1)={(F,Z-1)}, got {dQ_dQprev.shape}")

    # logic: derivative for layer i is product of dQ_dQprev up to i-1 times local dQ_drho at i
    dQsurf = np.empty((F, Z), dtype=complex)
    dQsurf[:, 0] = dQ_drho[:, 0]
    for i in range(1, Z):
        chain = np.prod(dQ_dQprev[:, :i], axis=1)
        dQsurf[:, i] = chain * dQ_drho[:, i]
    return dQsurf
