# src/npi_mt/enscgp/interp.py
from __future__ import annotations
import numpy as np
from scipy.interpolate import interp1d

def interp_ensemble_to_freqs(
    freqs_src_hz: np.ndarray,
    freqs_tgt_hz: np.ndarray,
    log_appres_src: np.ndarray,   # (Fsrc, N)
    phase_deg_src: np.ndarray,    # (Fsrc, N)
    kind: str = "cubic",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Interpolate ensemble responses from freqs_src -> freqs_tgt.

    - log_appres interpolated over log10(freq)
    - phase interpolated after unwrapping (in radians) over log10(freq)

    Returns
    -------
    log_appres_tgt : (Ftgt, N)
    phase_deg_tgt  : (Ftgt, N)
    """
    freqs_src_hz = np.asarray(freqs_src_hz, float)
    freqs_tgt_hz = np.asarray(freqs_tgt_hz, float)

    log_fs = np.log10(freqs_src_hz)
    log_ft = np.log10(freqs_tgt_hz)

    log_appres_src = np.asarray(log_appres_src, float)
    phase_deg_src  = np.asarray(phase_deg_src, float)

    f_app = interp1d(
        log_fs, log_appres_src, kind=kind, axis=0,
        bounds_error=False, fill_value="extrapolate"
    )
    log_app_tgt = f_app(log_ft)

    phase_rad = np.unwrap(np.deg2rad(phase_deg_src), axis=0)
    f_ph = interp1d(
        log_fs, phase_rad, kind=kind, axis=0,
        bounds_error=False, fill_value="extrapolate"
    )
    phase_tgt = np.rad2deg(f_ph(log_ft))

    return log_app_tgt, phase_tgt
