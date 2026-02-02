# src/npi_mt/mt1d/forward.py
from __future__ import annotations

from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import numpy as np

from .models import LayeredEarthModel
from .transfer import compute_Q_interfaces, compute_Q_surface
from .sensitivities import compute_local_derivatives, propagate_surface_derivative
from .jacobians import (
    apparent_resistivity_from_Q,
    phase_deg_from_Q,
    J_appres_wrt_rho,
    J_phase_deg_wrt_rho,
    stack_jacobians_wrt_logrho,
)


@dataclass(frozen=True)
class MT1DResponse:
    """
    Forward responses for a single model on a frequency grid.

    app_res_ohm_m : (F,)
    log_app_res   : (F,) natural log
    phase_deg     : (F,)
    Q_surf        : (F,) complex
    """
    app_res_ohm_m: np.ndarray
    log_app_res: np.ndarray
    phase_deg: np.ndarray
    Q_surf: np.ndarray


@dataclass(frozen=True)
class MT1DJacobianResponse:
    """
    Response + Jacobian for d = [log(rhoa); phase_deg] w.r.t log(rho_layers).

    J : (2F, Z)
    """
    response: MT1DResponse
    J: np.ndarray


class MT1DForward:
    """
    Public forward-modeling interface used by EnsCGP + NPI.

    - Uses numpy/cmath kernels.
    - Supports optional Jacobians w.r.t log-resistivity for physics loss.
    """

    def __init__(self, model: LayeredEarthModel):
        self.model = model

    def predict(self, freqs_hz: np.ndarray) -> MT1DResponse:
        freqs = np.asarray(freqs_hz, dtype=float)
        if freqs.ndim != 1:
            raise ValueError("freqs_hz must be 1D")
        if np.any(freqs <= 0):
            raise ValueError("All frequencies must be > 0")

        omega = 2.0 * np.pi * freqs

        Q_surf = np.empty(freqs.size, dtype=complex)
        for i, w in enumerate(omega):
            Q_surf[i] = compute_Q_surface(self.model.rho_ohm_m, self.model.thicknesses_m, w)

        app_res = apparent_resistivity_from_Q(Q_surf, omega)
        phase = phase_deg_from_Q(Q_surf)
        log_app = np.log(app_res)

        return MT1DResponse(
            app_res_ohm_m=app_res,
            log_app_res=log_app,
            phase_deg=phase,
            Q_surf=Q_surf,
        )

    def predict_with_jacobian(
        self,
        freqs_hz: np.ndarray,
        output_appres_in_log: bool = True,
    ) -> MT1DJacobianResponse:
        freqs = np.asarray(freqs_hz, dtype=float)
        omega = 2.0 * np.pi * freqs

        # Derivatives in recursion order (bottom->top)
        Q_all, dQ_drho, dQ_dQprev = compute_local_derivatives(
            self.model.rho_ohm_m, self.model.thicknesses_m, omega
        )
        Q_surf = Q_all[:, -1]
        dQsurf_drho = propagate_surface_derivative(dQ_drho, dQ_dQprev)  # (F,Z)

        app_res = apparent_resistivity_from_Q(Q_surf, omega)
        phase = phase_deg_from_Q(Q_surf)
        log_app = np.log(np.clip(app_res, 1e-32, None))

        # Jacobians wrt linear rho 
        J_rhoa_rho = J_appres_wrt_rho(Q_surf, dQsurf_drho, omega)   # (F,Z)
        J_phi_rho = J_phase_deg_wrt_rho(Q_surf, dQsurf_drho)        # (F,Z)

        # Stack Jacobian wrt log(rho_layers)
        J = stack_jacobians_wrt_logrho(
            rho_a=app_res,
            J_appres_wrt_rho_mat=J_rhoa_rho,
            J_phase_deg_wrt_rho_mat=J_phi_rho,
            rho_layers=self.model.rho_ohm_m,
            output_appres_in_log=output_appres_in_log,
        )

        resp = MT1DResponse(
            app_res_ohm_m=app_res,
            log_app_res=log_app,
            phase_deg=phase,
            Q_surf=Q_surf,
        )
        return MT1DJacobianResponse(response=resp, J=J)


def forward_ensemble(
    rho_ens_ohm_m: np.ndarray,
    depth_interfaces_m: np.ndarray,
    freqs_hz: np.ndarray,
    n_workers: int | None = None,
    parallel: str = "thread",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Batch forward for an ensemble of models.

    Parameters
    ----------
    rho_ens_ohm_m : (N, Z)
    depth_interfaces_m : (Z+1,)
    freqs_hz : (F,)
    n_workers : process/thread workers
    parallel : {"process","thread","serial"}

    Returns
    -------
    app_res : (N, F)
    phase   : (N, F)
    """
    rho_ens = np.asarray(rho_ens_ohm_m, dtype=float)
    if rho_ens.ndim != 2:
        raise ValueError("rho_ens_ohm_m must be 2D (N,Z)")
    N, Z = rho_ens.shape

    zif = np.asarray(depth_interfaces_m, dtype=float)
    if zif.ndim != 1 or zif.size != Z + 1:
        raise ValueError("depth_interfaces_m must be shape (Z+1,) matching rho_ens second dim")

    freqs = np.asarray(freqs_hz, dtype=float)
    omega = 2.0 * np.pi * freqs

    thicknesses = np.diff(zif)

    def _one(i: int) -> tuple[int, np.ndarray, np.ndarray]:
        rho = rho_ens[i, :]
        # single-model forward (serial loop over freq)
        Q_s = np.empty(freqs.size, dtype=complex)
        for j, w in enumerate(omega):
            Q_s[j] = compute_Q_interfaces(rho, thicknesses, w)[-1]
        app = apparent_resistivity_from_Q(Q_s, omega)
        ph = phase_deg_from_Q(Q_s)
        return i, app, ph

    app_out = np.empty((N, freqs.size), dtype=float)
    ph_out = np.empty((N, freqs.size), dtype=float)

    if parallel == "serial" or N == 1:
        for i in range(N):
            _, app, ph = _one(i)
            app_out[i, :] = app
            ph_out[i, :] = ph
        return app_out, ph_out

    Executor = ProcessPoolExecutor if parallel == "process" else ThreadPoolExecutor
    with Executor(max_workers=n_workers) as ex:
        futures = {ex.submit(_one, i): i for i in range(N)}
        for fut in as_completed(futures):
            i, app, ph = fut.result()
            app_out[i, :] = app
            ph_out[i, :] = ph

    return app_out, ph_out
