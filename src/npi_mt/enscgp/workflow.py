# src/npi_mt/enscgp/worfklow.py
from __future__ import annotations
import numpy as np

from .types import MTObservation, MTEnsembleData, NoiseModel, EnsCGPSolverConfig, NeighborConfig, EnsCGPResult
from .interp import interp_ensemble_to_freqs
from .neighbors import select_neighbors
from .update import enscgp_update

from npi_mt.mt1d import forward_ensemble 

def nrms_of_mean_prediction(
    obs: MTObservation,
    log_app_res_ens: np.ndarray,   # (F, N)
    phase_deg_ens: np.ndarray,     # (F, N)
    noise: NoiseModel,
) -> float:
    F = obs.freqs_hz.size
    mean_log_app = log_app_res_ens.mean(axis=1)
    mean_phs = phase_deg_ens.mean(axis=1)

    # whitened residuals
    app_sig = np.asarray(noise.app_res_log_sigma, float)
    phs_sig = np.asarray(noise.phase_deg_sigma, float)
    if app_sig.ndim == 0: app_sig = np.full(F, float(app_sig))
    if phs_sig.ndim == 0: phs_sig = np.full(F, float(phs_sig))

    r_app = (obs.log_app_res - mean_log_app) / np.clip(app_sig, 1e-12, None)
    r_phs = (obs.phase_deg - mean_phs) / np.clip(phs_sig, 1e-12, None)

    r = np.concatenate([r_app, r_phs])
    return float(np.sqrt(np.mean(r**2)))

def run_enscgp_update(
    obs: MTObservation,
    ensemble: MTEnsembleData,
    depths_m: np.ndarray,
    noise: NoiseModel,
    solver: EnsCGPSolverConfig = EnsCGPSolverConfig(),
    neighbor_cfg: NeighborConfig = NeighborConfig(),
    interp_kind: str = "cubic",
    fwd_workers: int | None = None,
    rng: np.random.Generator | None = None,
) -> EnsCGPResult:
    """
    End-to-end helper:
      1) interpolate ensemble responses to obs freqs
      2) select nearest neighbors
      3) EnsCGP update in stacked data space
      4) forward model updated ensemble on obs freqs (for outputs)
    """
    F = obs.freqs_hz.size

    # 1) interpolate to obs grid
    log_app_i, phs_i = interp_ensemble_to_freqs(
        ensemble.freqs_hz, obs.freqs_hz, ensemble.log_app_res, ensemble.phase_deg, kind=interp_kind
    )  # (F, N)

    ens_stacked = np.vstack([log_app_i, phs_i])            # (2F, N)
    obs_stacked = obs.stacked()                            # (2F,)

    # 2) neighbors
    idx, dist = select_neighbors(
        obs_stacked=obs_stacked,
        ens_stacked=ens_stacked,
        n_neighbors=neighbor_cfg.n_neighbors,
        metric=neighbor_cfg.metric,
    )

    log_rho_nn = ensemble.log_rho[:, idx]                  # (Z, Nn)
    data_nn = ens_stacked[:, idx]                          # (2F, Nn)

    # 3) update
    noise_diag = noise.diag(F)                             # (2F,)
    log_rho_upd = enscgp_update(
        obs_stacked=obs_stacked,
        data_ens=data_nn,
        log_rho_ens=log_rho_nn,
        noise_diag=noise_diag,
        damping=solver.damping,
        reg_eps=solver.reg_eps,
        perturb_observations=solver.perturb_observations,
        rng=rng,
    )                                                      # (Z, Nn)

    # 4) forward model updated ensemble on obs freqs
    # forward_ensemble expects linear rho with shape (Nn, Z) or (Z, Nn) depending on implementation.
    # In forward module, forward_ensemble returns arrays shaped (F, N) typically.
    rho_linear = np.exp(log_rho_upd).T  # (Nn, Z)

    app_ens, phs_ens = forward_ensemble(
        rho_linear,
        depths_m,
        obs.freqs_hz,
        n_workers=fwd_workers,
        parallel="thread",   
    )

    # forward_ensemble returns (Nn, F);
    log_app_upd = np.log(np.clip(app_ens.T, 1e-32, None))  # (F, Nn)
    phs_upd = phs_ens.T                                 # (F, Nn)

    nrms = nrms_of_mean_prediction(obs, log_app_upd, phs_upd, noise)

    return EnsCGPResult(
        log_rho_updated=log_rho_upd,
        log_app_res_updated=log_app_upd,
        phase_deg_updated=phs_upd,
        neighbor_indices=idx,
        neighbor_distances=dist,
        nrms_mean=nrms,
    )
