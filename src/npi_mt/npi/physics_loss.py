# src/npi_mt/npi/physics_loss.py
from __future__ import annotations

import numpy as np
import torch
from torch.autograd import Function

from ..mt1d.models import LayeredEarthModel
from ..mt1d.forward import MT1DForward


def _mt_pred_and_jac_logrho(
    log_rho: np.ndarray,     # (Z,)
    depths_m: np.ndarray,    # (Z+1,)
    freqs_hz: np.ndarray,    # (F,)
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns
    -------
    pred : (2F,)    stacked = [log_app_res; phase_deg]
    J    : (2F, Z)  d pred / d log_rho  (already in this form from MT1DForward)
    """
    log_rho = np.asarray(log_rho, dtype=float).reshape(-1)
    depths_m = np.asarray(depths_m, dtype=float).reshape(-1)
    freqs_hz = np.asarray(freqs_hz, dtype=float).reshape(-1)

    rho = np.exp(log_rho)
    model = LayeredEarthModel(rho_ohm_m=rho, depth_interfaces_m=depths_m)

    fwd = MT1DForward(model)
    out = fwd.predict_with_jacobian(freqs_hz, output_appres_in_log=True)

    # prediction on the given grid
    log_app = np.asarray(out.response.log_app_res, dtype=float)   # (F,)
    phi_deg = np.asarray(out.response.phase_deg, dtype=float)     # (F,)

    pred = np.concatenate([log_app, phi_deg], axis=0)             # (2F,)
    J = np.asarray(out.J, dtype=float)                            # (2F, Z)

    # sanity checks
    F = freqs_hz.size
    if pred.shape != (2 * F,):
        raise ValueError(f"pred must be (2F,), got {pred.shape} for F={F}")
    if J.shape[0] != 2 * F:
        raise ValueError(f"J first dim must be 2F={2*F}, got {J.shape}")
    if J.shape[1] != log_rho.size:
        raise ValueError(f"J second dim must be Z={log_rho.size}, got {J.shape}")

    return pred, J


class MTPhysicsMisfit(Function):
    @staticmethod
    def forward(
        ctx,
        log_rho_t: torch.Tensor,
        depths_m: np.ndarray,
        freqs_hz: np.ndarray,
        obs: np.ndarray,
        w: np.ndarray | None,
    ):
        device = log_rho_t.device
        log_rho = log_rho_t.detach().cpu().numpy()

        pred, J = _mt_pred_and_jac_logrho(log_rho, depths_m, freqs_hz)

        obs = np.asarray(obs, dtype=np.float64).reshape(-1)
        if obs.shape != pred.shape:
            raise ValueError(f"obs shape {obs.shape} must match pred shape {pred.shape}")

        r = (pred - obs).astype(np.float64)

        if w is None:
            wv = np.ones_like(r, dtype=np.float64)
        else:
            wv = np.asarray(w, dtype=np.float64).reshape(-1)
            if wv.shape != r.shape:
                raise ValueError(f"w must have shape {r.shape}, got {wv.shape}")

        r_w = wv * r
        J_w = J.astype(np.float64) * wv[:, None]

        N = float(r.size)
        loss = float(np.dot(r_w, r_w) / N)

        ctx.save_for_backward(
            torch.tensor(J_w, dtype=torch.float64, device=device),
            torch.tensor(r_w, dtype=torch.float64, device=device),
            torch.tensor([N], dtype=torch.float64, device=device),
        )
        return torch.tensor(loss, dtype=log_rho_t.dtype, device=device)

    @staticmethod
    def backward(ctx, grad_out):
        Jw, r_w, N_t = ctx.saved_tensors
        g = (2.0 / N_t.item()) * (Jw.t().mm(r_w.unsqueeze(1))).squeeze(1)  # float64
        g = g.to(dtype=grad_out.dtype)
        return grad_out * g, None, None, None, None


class PhysicsLoss(torch.nn.Module):
    """
    Physics loss on stacked data space [log apparent resistivity; phase(deg)].

    Parameters
    ----------
    depths_m : (Z+1,)
    freqs_hz : (F,)
    obs      : (2F,) stacked [log_app_res; phase_deg]
    w        : (2F,) optional weights (e.g., 1/sigma)
    """
    def __init__(
        self,
        depths_m: np.ndarray,
        freqs_hz: np.ndarray,
        obs: np.ndarray,
        w: np.ndarray | None = None,
    ):
        super().__init__()
        self.depths_m = np.asarray(depths_m, dtype=float).reshape(-1)
        self.freqs_hz = np.asarray(freqs_hz, dtype=float).reshape(-1)
        self.obs = np.asarray(obs, dtype=float).reshape(-1)
        self.w = None if w is None else np.asarray(w, dtype=float).reshape(-1)

        F = self.freqs_hz.size
        if self.obs.size != 2 * F:
            raise ValueError(f"obs must have length 2*F={2*F}, got {self.obs.size}")
        if self.w is not None and self.w.size != 2 * F:
            raise ValueError(f"w must have length 2*F={2*F}, got {self.w.size}")

    def forward(self, log_rho_t: torch.Tensor) -> torch.Tensor:
        return MTPhysicsMisfit.apply(log_rho_t, self.depths_m, self.freqs_hz, self.obs, self.w)
