# src/npi_mt/mt1d/transfer.py
from __future__ import annotations

import cmath
import numpy as np

from .constants import MU_0


def compute_Q_interfaces(
    rho_ohm_m: np.ndarray,
    thicknesses_m: np.ndarray,
    omega: float,
) -> np.ndarray:
    """
    Compute MT transfer function Q at all interfaces using standard 1D recursion.

    Parameters
    ----------
    rho_ohm_m : (Z,) resistivities in ohm-m (top->bottom)
    thicknesses_m : (Z,) thicknesses in meters for each layer interval.
        If the last layer is a half-space, thicknesses_m[-1] is unused.
    omega : angular frequency (rad/s)

    Returns
    -------
    Q : (Z,) complex
        Q values in the recursion order corresponding to bottom->top in the *internal*
        computation.
        The surface value is Q[-1].
    """
    rho = np.asarray(rho_ohm_m, dtype=float)
    thk = np.asarray(thicknesses_m, dtype=float)

    if rho.ndim != 1:
        raise ValueError("rho_ohm_m must be 1D")
    if thk.ndim != 1:
        raise ValueError("thicknesses_m must be 1D")
    if thk.size != rho.size:
        raise ValueError(
            f"thicknesses_m must have same length as rho. Got {thk.size} vs {rho.size}."
        )
    if omega <= 0:
        raise ValueError("omega must be > 0")
    if np.any(rho <= 0):
        raise ValueError("All resistivities must be > 0")
    if np.any(thk <= 0):
        raise ValueError("All thicknesses must be > 0")

    # flip to recurse from bottom upward
    rho_b2t = np.flip(rho)
    thk_b2t = np.flip(thk)

    Q_list: list[complex] = []
    for k, rho_k in enumerate(rho_b2t):
        alpha = cmath.sqrt(1j * MU_0 * omega / rho_k)
        if k == 0:
            # bottom half-space boundary condition
            Q_k = alpha * rho_k
        else:
            h = thk_b2t[k - 1]
            t = cmath.tanh(alpha * h)
            Q_prev = Q_list[-1]
            Q_k = (Q_prev + alpha * rho_k * t) / (1.0 + (Q_prev / (alpha * rho_k)) * t)

        Q_list.append(Q_k)

    return np.asarray(Q_list, dtype=complex)


def compute_Q_surface(
    rho_ohm_m: np.ndarray,
    thicknesses_m: np.ndarray,
    omega: float,
) -> complex:
    """Convenience wrapper returning only the surface Q."""
    Q_all = compute_Q_interfaces(rho_ohm_m, thicknesses_m, omega)
    return complex(Q_all[-1])
