# src/npi_mt/io/load_data.py
from __future__ import annotations
from pathlib import Path
import numpy as np

from npi_mt.enscgp import MTObservation, MTEnsembleData

def _load_1d_txt(path: Path) -> np.ndarray:
    arr = np.loadtxt(path)
    return np.asarray(arr)

def load_real_example(root: str | Path) -> MTObservation:
    """
    Expected files under root:
      - freqs_hz.txt
      - depths_m.txt (NOT returned here; load separately)
      - app_res_ohm_m.txt   (linear ohm-m)
      - phase_deg.txt       (degrees)
    """
    root = Path(root)
    freqs = _load_1d_txt(root / "freqs_hz.txt")
    app_res = _load_1d_txt(root / "app_res_ohm_m.txt")
    phase = _load_1d_txt(root / "phase_deg.txt")

    log_app_res = np.log(np.clip(app_res, 1e-32, None))
    return MTObservation(freqs_hz=freqs, log_app_res=log_app_res, phase_deg=phase)

def load_depths(root: str | Path) -> np.ndarray:
    """
    depths_m.txt should be interfaces (Z+1,) in meters.
    """
    root = Path(root)
    return _load_1d_txt(root / "depths_m.txt")

def load_ensemble_bank(root: str | Path) -> MTEnsembleData:
    """
    Expected files under root:
      - freqs_hz.txt
      - depths_m.txt (NOT returned here; load separately)
      - rho_ens.txt      linear rho in ohm-m, shape (Z, N) or (N, Z)
      - appres_ens.txt   linear rho_a in ohm-m, shape (F, N) or (N, F)
      - phase_ens.txt    degrees, shape (F, N) or (N, F)

    We convert:
      - log_rho (natural log)
      - log_app_res (natural log)
    and force arrays to (F, N) and (Z, N).
    """
    root = Path(root)

    freqs = _load_1d_txt(root / "freqs_hz.txt")
    rho = _load_1d_txt(root / "rho_ens.txt")
    app = _load_1d_txt(root / "appres_ens.txt")
    phs = _load_1d_txt(root / "phase_ens.txt")

    # Canonicalize shapes
    rho = np.asarray(rho, float)
    app = np.asarray(app, float)
    phs = np.asarray(phs, float)

    # Determine orientation heuristically
    # rho should be (Z, N)
    if rho.ndim != 2:
        raise ValueError("rho_ens.txt must be 2D")
    if rho.shape[0] < rho.shape[1]:
        # assume (Z, N)
        rho_ZN = rho
    else:
        # could be (N, Z) — transpose
        rho_ZN = rho.T

    # app/phs should be (F, N)
    def _to_FN(x: np.ndarray, F: int) -> np.ndarray:
        if x.ndim != 2:
            raise ValueError("ensemble response must be 2D")
        if x.shape[0] == F:
            return x
        if x.shape[1] == F:
            return x.T
        raise ValueError(f"Could not interpret response shape {x.shape} with F={F}")

    F = freqs.size
    app_FN = _to_FN(app, F)
    phs_FN = _to_FN(phs, F)

    log_rho = np.log(np.clip(rho_ZN, 1e-32, None))
    log_app = np.log(np.clip(app_FN, 1e-32, None))

    return MTEnsembleData(freqs_hz=freqs, log_app_res=log_app, phase_deg=phs_FN, log_rho=log_rho)
