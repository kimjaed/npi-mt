# src/npi_mt/npi/refinement.py
from __future__ import annotations

from dataclasses import dataclass
import copy
import numpy as np
import torch

from .normalization import NormalizationStats
from .resnet1d import freeze_bn_stats
from .physics_loss import PhysicsLoss
from ..mt1d.forward import forward_ensemble, MT1DForward
from ..mt1d.models import LayeredEarthModel


# -------------------------
# Public outputs
# -------------------------

@dataclass(frozen=True)
class RefinementOutput:
    """
    Refined resistivity + optional predicted MT responses on a chosen grid.
    """
    log_rho_refined: np.ndarray       # (N,Z)
    rho_refined_ohm_m: np.ndarray     # (N,Z)
    app_res_ohm_m: np.ndarray | None  # (N,F_out) or None
    phase_deg: np.ndarray | None      # (N,F_out) or None


@dataclass(frozen=True)
class RefinementLoopConfig:
    """
    Refinement loop configuration
    """
    max_iter: int = 3000

    # early stopping on nRMS score
    patience: int = 10
    min_delta: float = 1e-4

    # adaptive smoothing to hit target nRMS band
    target: float = 0.1
    band: float = 0.02
    up: float = 1.25
    down: float = 0.8
    gentle_up: float = 1.05
    lambda_smooth_init: float = 1e2
    lambda_smooth_min: float = 1e1
    lambda_smooth_max: float = 1e3

    # smallness (toward EnsCGP baseline)
    lambda_small: float = 1e-2

    # stop when nRMS is within band for K consecutive epochs
    target_hits_needed: int = 10

    # optimization
    grad_clip: float = 1.0


# -------------------------
# Internal utilities
# -------------------------

def _as_batch(x: np.ndarray) -> tuple[np.ndarray, bool]:
    """Return x as (N, ...) and a flag if it was originally unbatched."""
    x = np.asarray(x)
    if x.ndim == 1:
        return x[None, :], True
    return x, False

def smoothness_penalty(rho_log_tensor, alpha=0.3):
    # Second-order (curvature)
    d2 = rho_log_tensor[2:] - 2*rho_log_tensor[1:-1] + rho_log_tensor[:-2]
    s2 = torch.mean(d2**2)
    
    # First-order (slope)
    d1 = rho_log_tensor[1:] - rho_log_tensor[:-1]
    s1 = torch.mean(d1**2)
    
    return (1.0 - alpha) * s2 + alpha * s1


def smallness_penalty(log_rho: torch.Tensor, ref_log_rho: torch.Tensor) -> torch.Tensor:
    """
    log_rho, ref_log_rho: (Z,)
    """
    return torch.mean((log_rho - ref_log_rho) ** 2)


def compute_nrms_logrhoa_phi(
    rho_log_tensor: torch.Tensor,
    depths_m: np.ndarray,
    freqs_hz: np.ndarray,
    real_mt_obs_np: np.ndarray,
    w_app: np.ndarray,
    w_phi: np.ndarray,
    clip: float = 1e-32,
) -> tuple[float, float, float]:
    """
    nRMS metric:
      - forward model on REAL grid
      - residual in stacked space [log_app_res; phase_deg]
      - whiten by w_app and w_phi separately
      - return n_overall, n_app, n_phi
    """
    with torch.no_grad():
        log_rho = rho_log_tensor.detach().cpu().numpy().reshape(-1)

    rho = np.exp(log_rho)
    depths_m = np.asarray(depths_m, float).reshape(-1)
    freqs_hz = np.asarray(freqs_hz, float).reshape(-1)

    model = LayeredEarthModel(rho_ohm_m=rho, depth_interfaces_m=depths_m)
    fwd = MT1DForward(model)
    resp = fwd.predict(freqs_hz)

    log_app = np.asarray(resp.log_app_res, dtype=np.float64)
    phi = np.asarray(resp.phase_deg, dtype=np.float64)

    F = freqs_hz.size
    pred = np.concatenate([log_app, phi], axis=0)

    real_mt_obs_np = np.asarray(real_mt_obs_np, dtype=np.float64).reshape(-1)
    if pred.shape[0] != 2 * F:
        raise ValueError(f"pred length {pred.shape[0]} != 2*F ({2*F})")
    if real_mt_obs_np.shape[0] != 2 * F:
        raise ValueError(f"obs length {real_mt_obs_np.shape[0]} != 2*F ({2*F})")

    w_app = np.asarray(w_app, dtype=np.float64).reshape(-1)
    w_phi = np.asarray(w_phi, dtype=np.float64).reshape(-1)
    if w_app.shape != (F,) or w_phi.shape != (F,):
        raise ValueError("w_app and w_phi must be shape (F,)")

    residual = pred - real_mt_obs_np
    r_app = residual[:F] * w_app
    r_phi = residual[F:] * w_phi

    n_app = float(np.sqrt(np.mean(r_app**2)))
    n_phi = float(np.sqrt(np.mean(r_phi**2)))
    n_all = float(np.sqrt(np.mean(np.concatenate([r_app, r_phi])**2)))

    return n_all, n_app, n_phi


# -------------------------
# Inference API
# -------------------------

def refine_log_rho(
    model: torch.nn.Module,
    stats: NormalizationStats,
    enscgp_log_app_res: np.ndarray,    # (N,F) or (F,)
    enscgp_phase_deg: np.ndarray,      # (N,F) or (F,)
    enscgp_log_rho: np.ndarray,        # (N,Z) or (Z,)
    device: torch.device,
) -> np.ndarray:
    """
    Inference-only:
      log_rho_refined = enscgp_log_rho + (pred_residual * label_std + label_mean)

    Returns: (N,Z)
    """
    log_app, _ = _as_batch(enscgp_log_app_res)
    phi, _ = _as_batch(enscgp_phase_deg)
    logrho0, _ = _as_batch(enscgp_log_rho)

    if log_app.shape != phi.shape:
        raise ValueError(f"log_app shape {log_app.shape} must match phi shape {phi.shape}")

    N, F = log_app.shape
    if logrho0.shape[0] != N:
        raise ValueError(f"enscgp_log_rho first dim {logrho0.shape[0]} must match N={N}")

    x = np.stack([log_app, phi], axis=1).astype(np.float32)  # (N,2,F)

    im = stats.input_mean  # (2,F) or (2,1)
    is_ = stats.input_std

    x_t = torch.as_tensor(x, dtype=torch.float32, device=device)
    x_t = (x_t - im.view(1, 2, -1)) / is_.view(1, 2, -1)

    model.eval()
    with torch.no_grad():
        y = model(x_t)  # (N,Z) normalized residual

    y = y * stats.label_std.view(1, -1) + stats.label_mean.view(1, -1)  # (N,Z)
    logrho_ref = torch.as_tensor(logrho0, dtype=torch.float32, device=device) + y
    return logrho_ref.detach().cpu().numpy()


def refine_and_forward(
    model: torch.nn.Module,
    stats: NormalizationStats,
    enscgp_log_app_res: np.ndarray,   # (N,F_train) or (F_train,)
    enscgp_phase_deg: np.ndarray,     # (N,F_train) or (F_train,)
    enscgp_log_rho: np.ndarray,       # (N,Z) or (Z,)
    depths_m: np.ndarray,             # (Z+1,)
    freqs_out_hz: np.ndarray | None,  # if None, skip forward prediction
    device: torch.device,
    fwd_workers: int | None = None,
) -> RefinementOutput:
    """
    Refine log-rho and optionally forward-model MT responses on freqs_out_hz.
    """
    log_rho_ref = refine_log_rho(
        model=model,
        stats=stats,
        enscgp_log_app_res=enscgp_log_app_res,
        enscgp_phase_deg=enscgp_phase_deg,
        enscgp_log_rho=enscgp_log_rho,
        device=device,
    )  # (N,Z)

    rho_ref = np.exp(log_rho_ref)

    app, phi = None, None
    if freqs_out_hz is not None:
        app, phi = forward_ensemble(
            rho_ref,
            depths_m=np.asarray(depths_m, float),
            freqs_hz=np.asarray(freqs_out_hz, float),
            workers=fwd_workers,
        )

    return RefinementOutput(
        log_rho_refined=log_rho_ref,
        rho_refined_ohm_m=rho_ref,
        app_res_ohm_m=app,
        phase_deg=phi,
    )


# -------------------------
# Fine-tuning API
# -------------------------

def fine_tune_on_real_example(
    model: torch.nn.Module,
    stats: NormalizationStats,
    x_input: torch.Tensor,                 # (1,2,F_train), 
    enscgp_log_rho0: torch.Tensor,         # (Z,), 
    physics_loss: PhysicsLoss,             # uses REAL grid internally
    depths_m: np.ndarray,                  # (Z+1,) 
    freqs_real_hz: np.ndarray,             # (F_real,)
    real_mt_obs_np: np.ndarray,            # (2*F_real,) stacked [log_app; phi]
    w_app: np.ndarray,                     # (F_real,)
    w_phi: np.ndarray,                     # (F_real,)
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau | None = None,
    cfg: RefinementLoopConfig = RefinementLoopConfig(),
    verbose: bool = True,
) -> dict:
    """
    Implements refinement loop:
      - train on total loss (physics + smooth + small)
      - evaluate metric nRMS via forward modeling and whitening
      - adapt lambda_smooth based on nRMS target band
      - early stop on nRMS improvements (min_delta, patience)
      - optional hard stop after K consecutive target hits

    Returns a dict with:
      - best_state_dict
      - history (list of dict per epoch)
      - best_score
    """
    device = next(model.parameters()).device
    x_input = x_input.to(device=device, dtype=torch.float32)
    enscgp_log_rho0 = enscgp_log_rho0.to(device=device, dtype=torch.float32)

    label_mean = stats.label_mean.to(device=device, dtype=torch.float32)
    label_std = stats.label_std.to(device=device, dtype=torch.float32)

    lam_smooth = float(cfg.lambda_smooth_init)
    lam_small = float(cfg.lambda_small)

    best_score = float("inf")
    patience_counter = 0
    target_hits = 0
    best_state = None

    history: list[dict] = []

    for epoch in range(cfg.max_iter):
        # ---- TRAIN ----
        model.train()
        model.apply(freeze_bn_stats)
        optimizer.zero_grad(set_to_none=True)

        pred = model(x_input).squeeze(0)  
        pred_log_rho = pred * label_std + label_mean + enscgp_log_rho0  

        L_phys = physics_loss(pred_log_rho)
        L_smooth = smoothness_penalty(pred_log_rho)
        L_small = smallness_penalty(pred_log_rho, enscgp_log_rho0)
        L = L_phys + lam_smooth * L_smooth + lam_small * L_small

        L.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(cfg.grad_clip))
        optimizer.step()

        if scheduler is not None:
            # step using physics loss
            scheduler.step(float(L_phys.item()))

        # ---- EVAL METRIC (nRMS) ----
        model.eval()
        with torch.no_grad():
            pred_eval = model(x_input).squeeze(0)
            pred_log_rho_eval = pred_eval * label_std + label_mean + enscgp_log_rho0

        n_all, n_app, n_phi = compute_nrms_logrhoa_phi(
            pred_log_rho_eval,
            depths_m=np.asarray(depths_m, float),
            freqs_hz=np.asarray(freqs_real_hz, float),
            real_mt_obs_np=np.asarray(real_mt_obs_np, float),
            w_app=w_app,
            w_phi=w_phi,
        )

        # ---- adaptive lambda_smooth ----
        if n_all > cfg.target + cfg.band:
            lam_smooth *= cfg.down
        elif n_all < cfg.target - cfg.band:
            lam_smooth *= cfg.up
        else:
            lam_smooth *= cfg.gentle_up

        lam_smooth = float(np.clip(lam_smooth, cfg.lambda_smooth_min, cfg.lambda_smooth_max))

        # ---- logging ----
        rec = dict(
            epoch=int(epoch),
            loss_total=float(L.item()),
            loss_phys=float(L_phys.item()),
            loss_smooth=float(L_smooth.item()),
            loss_small=float(L_small.item()),
            nrms_all=float(n_all),
            nrms_app=float(n_app),
            nrms_phi=float(n_phi),
            lambda_smooth=float(lam_smooth),
        )
        history.append(rec)

        if verbose:
            print(
                f"Epoch {epoch+1:4d} | L={rec['loss_total']:.3e} | nRMS={rec['nrms_all']:.3f} "
                f"(app={rec['nrms_app']:.3f}, phi={rec['nrms_phi']:.3f}) | "
                f"λ_smooth={lam_smooth:.3g}"
            )

        # ---- early stopping on nRMS ----
        score = float(n_all)
        improvement = best_score - score
        if improvement > cfg.min_delta:
            best_score = score
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= cfg.patience:
                if verbose:
                    print(f"Early stopping (no improvement > {cfg.min_delta} for {cfg.patience} epochs).")
                break

        # ---- hard stop on target hits ----
        if score <= cfg.target + cfg.band:
            target_hits += 1
            if target_hits >= cfg.target_hits_needed:
                if verbose:
                    print(f"Target nRMS reached for {cfg.target_hits_needed} consecutive epochs. Stopping.")
                break
        else:
            target_hits = 0

    return dict(
        best_state_dict=best_state,
        history=history,
        best_score=float(best_score),
    )
