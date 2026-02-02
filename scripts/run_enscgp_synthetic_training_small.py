#!/usr/bin/env python3
"""
Run EnsCGP (single-shot ensemble conditioning) for every synthetic training observation.

This script is intended to generate the EnsCGP "first-guess" products used by the NPI workflow:
  - mean resistivity model per training sample
  - mean predicted MT responses per training sample (apparent resistivity + phase)
  - (optional) std envelopes

It assumes your repository has the `data/` tree you described, e.g.:

data/
  synthetic/
    training_small/
      noisy_appres_ens.txt
      noisy_phase_ens.txt
      clean_appres_ens.txt          (optional; for QA plots only)
      clean_phase_ens.txt           (optional; for QA plots only)
      rho_ens.txt                   (this is rho_true for training_small)
      depths_m.txt
      freqs_hz.txt
    ensemble_bank_small/
      appres_ens.txt
      phase_ens.txt
      rho_ens.txt
      depths_m.txt
      freqs_hz.txt

Outputs are written into an output directory (default: data/synthetic/training_small/enscgp_out/).

Example
-------
python run_enscgp_synthetic_training_small.py \
  --data-root ./data \
  --training-set synthetic/training_small \
  --ensemble-bank synthetic/ensemble_bank_small \
  --out-dir ./data/synthetic/training_small/enscgp_out \
  --n-neighbors 500 \
  --fwd-workers 0 \
  --parallel thread

Notes
-----
- We compute distance for neighbor selection on the *observation frequency grid*.
- The EnsCGP update is performed in stacked data space d = [log(rhoa); phase_deg].
- Forward modeling of the updated ensemble uses npi_mt.mt1d.forward.forward_ensemble,
  which returns (app_res_ohm_m, phase_deg) as (N, F).
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from npi_mt.enscgp.types import MTObservation, MTEnsembleData, NoiseModel, NeighborConfig, EnsCGPSolverConfig
from npi_mt.enscgp.workflow import run_enscgp_update

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = REPO_ROOT / "data"


def interfaces_to_midpoints(z_if: np.ndarray) -> np.ndarray:
    z_if = np.asarray(z_if, float)
    return 0.5 * (z_if[:-1] + z_if[1:])

def _load_txt(path: Path) -> np.ndarray:
    arr = np.loadtxt(path)
    return np.asarray(arr)


def _ensure_shape_freq_by_sample(x: np.ndarray, name: str) -> np.ndarray:
    """
    Expect x to be (F, S). If user saved (S, F), we transpose automatically.
    """
    x = np.asarray(x)
    if x.ndim != 2:
        raise ValueError(f"{name} must be 2D; got {x.shape}")
    # Heuristic: for MT datasets, number of samples is usually >> number of freqs.
    if x.shape[0] > x.shape[1]:
        x = x.T
    return x


def _save_txt(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, arr)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=str, default=str(DATA_ROOT), help="Path to repo data directory.")
    p.add_argument("--training-set", type=str, default="synthetic/training_small",
                   help="Relative path under data-root for training set.")
    p.add_argument("--ensemble-bank", type=str, default="synthetic/ensemble_bank_small",
                   help="Relative path under data-root for ensemble bank.")
    p.add_argument("--out-dir", type=str, default=None,
                   help="Output directory. Default: <training-set>/enscgp_out under data-root.")
    p.add_argument("--n-neighbors", type=int, default=500)
    p.add_argument("--metric", type=str, default="cosine", choices=["cosine", "euclidean"])
    p.add_argument("--damping", type=float, default=1.0)
    p.add_argument("--reg-eps", type=float, default=1e-12)
    p.add_argument("--perturb-observations", action="store_true", help="Perturb obs in Y (stochastic EnKF style).")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--rel-err", type=float, default=0.05, help="Relative error used to set noise model.")
    p.add_argument("--fwd-workers", type=int, default=0,
                   help="Workers for forward modeling inside ensCGP workflow. 0 means serial/thread safe.")
    p.add_argument("--max-samples", type=int, default=0,
                   help="If >0, only process the first max-samples training samples.")
    p.add_argument("--qa-plot-every", type=int, default=0,
                   help="If >0, save QA plots every K samples (requires matplotlib).")
    p.add_argument("--plot-summary", action="store_true",
               help="After processing all samples, save summary plots (all models / all responses).")
    p.add_argument("--plot-dir", type=str, default=None,
                help="Where to write summary plots. Default: <out-dir>/summary_plots")

    args = p.parse_args()

    rng = np.random.default_rng(args.seed)

    data_root = Path(args.data_root)
    if not data_root.is_absolute():
        data_root = (REPO_ROOT / data_root).resolve()
    if not data_root.exists():
        raise FileNotFoundError(f"data_root not found: {data_root}")
    
    train_dir = (data_root / args.training_set).resolve()
    bank_dir = (data_root / args.ensemble_bank).resolve()

    out_dir = Path(args.out_dir) if args.out_dir is not None else (train_dir / "enscgp_out")
    out_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------
    # Load TRAINING observations (noisy)
    # -----------------------
    freqs_obs = _load_txt(train_dir / "freqs_hz.txt").astype(float).reshape(-1)
    depths_m = _load_txt(train_dir / "depths_m.txt").astype(float).reshape(-1)

    noisy_app = _ensure_shape_freq_by_sample(_load_txt(train_dir / "noisy_appres_ens.txt"), "noisy_appres_ens")
    noisy_phi = _ensure_shape_freq_by_sample(_load_txt(train_dir / "noisy_phase_ens.txt"), "noisy_phase_ens")

    F_obs, S = noisy_app.shape
    if noisy_phi.shape != (F_obs, S):
        raise ValueError(f"noisy_phase_ens shape {noisy_phi.shape} must match noisy_appres_ens shape {noisy_app.shape}")
    if freqs_obs.size != F_obs:
        raise ValueError(f"freqs_hz has {freqs_obs.size} entries but noisy data has F={F_obs}")

    if args.max_samples and args.max_samples > 0:
        S = min(S, args.max_samples)
        noisy_app = noisy_app[:, :S]
        noisy_phi = noisy_phi[:, :S]

    clean_app = clean_phi = None
    clean_app_path = train_dir / "clean_appres_ens.txt"
    clean_phi_path = train_dir / "clean_phase_ens.txt"
    if clean_app_path.exists() and clean_phi_path.exists():
        clean_app = _ensure_shape_freq_by_sample(_load_txt(clean_app_path), "clean_appres_ens")
        clean_phi = _ensure_shape_freq_by_sample(_load_txt(clean_phi_path), "clean_phase_ens")

        if clean_app.shape != (F_obs, noisy_app.shape[1]) or clean_phi.shape != (F_obs, noisy_phi.shape[1]):
            # If max_samples trimmed S, trim clean too
            clean_app = clean_app[:, :S]
            clean_phi = clean_phi[:, :S]

    # training truth (optional; for QA plots only)
    rho_true = None
    true_path = train_dir / "rho_ens.txt"
    if true_path.exists():
        rt = _load_txt(true_path)
        rt = np.asarray(rt, float)
        if rt.ndim == 2 and rt.shape[0] == S:
            rho_true = rt
        elif rt.ndim == 2 and rt.shape[1] == S:
            rho_true = rt.T

    # -----------------------
    # Load ENSEMBLE BANK (prior ensemble)
    # -----------------------
    freqs_bank = _load_txt(bank_dir / "freqs_hz.txt").astype(float).reshape(-1)
    app_bank = _ensure_shape_freq_by_sample(_load_txt(bank_dir / "appres_ens.txt"), "ensemble_bank appres_ens")
    phi_bank = _ensure_shape_freq_by_sample(_load_txt(bank_dir / "phase_ens.txt"), "ensemble_bank phase_ens")
    rho_bank = _load_txt(bank_dir / "rho_ens.txt")

    if app_bank.shape != phi_bank.shape:
        raise ValueError(f"ensemble bank appres {app_bank.shape} and phase {phi_bank.shape} must match")
    F_bank, Nbank = app_bank.shape

    rho_bank = np.asarray(rho_bank, float)
    if rho_bank.ndim != 2:
        raise ValueError(f"ensemble bank rho_ens must be 2D, got {rho_bank.shape}")
    # Prefer (Z, Nbank)
    if rho_bank.shape[0] == Nbank and rho_bank.shape[1] != Nbank:
        rho_bank = rho_bank.T
    if rho_bank.shape[1] != Nbank:
        raise ValueError(f"ensemble bank rho_ens second dim must equal Nbank={Nbank}; got {rho_bank.shape}")

    Z = rho_bank.shape[0]
    if depths_m.size != Z + 1:
        raise ValueError(f"depths_m must be (Z+1,) with Z={Z}; got {depths_m.size}")

    bank = MTEnsembleData(
        freqs_hz=freqs_bank,
        log_app_res=np.log(np.clip(app_bank, 1e-32, None)),
        phase_deg=phi_bank,
        log_rho=np.log(np.clip(rho_bank, 1e-32, None)),
    )

    # -----------------------
    # Noise + configs
    # -----------------------
    sigma_log_app = float(2.0 * args.rel_err)
    sigma_phi_deg = float(np.degrees(np.arctan(args.rel_err)))

    noise = NoiseModel(app_res_log_sigma=sigma_log_app, phase_deg_sigma=sigma_phi_deg)
    neighbor_cfg = NeighborConfig(n_neighbors=int(args.n_neighbors), metric=args.metric)
    solver_cfg = EnsCGPSolverConfig(
        damping=float(args.damping),
        reg_eps=float(args.reg_eps),
        perturb_observations=bool(args.perturb_observations),
    )

    # -----------------------
    # Preallocate outputs
    # -----------------------
    mean_log_rho = np.empty((Z, S), dtype=float)
    std_log_rho = np.empty((Z, S), dtype=float)

    mean_log_app = np.empty((F_obs, S), dtype=float)
    std_log_app = np.empty((F_obs, S), dtype=float)

    mean_phi = np.empty((F_obs, S), dtype=float)
    std_phi = np.empty((F_obs, S), dtype=float)

    do_qa = args.qa_plot_every and args.qa_plot_every > 0
    if do_qa:
        import matplotlib.pyplot as plt
        periods = 1.0 / freqs_obs
        z_mid = 0.5 * (depths_m[:-1] + depths_m[1:])

    # -----------------------
    # Main loop
    # -----------------------
    for i in range(S):
        obs = MTObservation(
            freqs_hz=freqs_obs,
            log_app_res=np.log(np.clip(noisy_app[:, i], 1e-32, None)),
            phase_deg=noisy_phi[:, i],
        )

        result = run_enscgp_update(
            obs=obs,
            ensemble=bank,
            depths_m=depths_m,
            noise=noise,
            solver=solver_cfg,
            neighbor_cfg=neighbor_cfg,
            fwd_workers=int(args.fwd_workers) if args.fwd_workers > 0 else None,
            rng=rng,
        )

        lr = result.log_rho_updated    # (Z, Nn)
        mean_log_rho[:, i] = lr.mean(axis=1)
        std_log_rho[:, i] = lr.std(axis=1)

        app_post = result.log_app_res_updated  # (Nn, F_obs) linear
        phi_post = result.phase_deg_updated    # (Nn, F_obs)

        # Ensure shape is (Nn, F_obs)
        if app_post.shape[0] == F_obs and app_post.shape[1] != F_obs:
            # currently (F, Nn) -> transpose
            app_post = app_post.T
            phi_post = phi_post.T
        elif app_post.shape[1] != F_obs:
            raise ValueError(f"Unexpected app_post shape {app_post.shape}, expected (*, {F_obs}) or ({F_obs}, *)")

        mean_log_app[:, i] = app_post.mean(axis=0)
        std_log_app[:, i] = app_post.std(axis=0)

        mean_phi[:, i] = phi_post.mean(axis=0)
        std_phi[:, i] = phi_post.std(axis=0)

        if (i == 0) or ((i + 1) % max(1, min(50, S)) == 0):
            print(f"[{i+1:5d}/{S}] nRMS(mean on obs grid) = {result.nrms_mean:.4f}")

        if do_qa and ((i % args.qa_plot_every) == 0):
            fig = plt.figure(figsize=(10, 8))
            gs = fig.add_gridspec(2, 2, width_ratios=[1, 1.5])

            ax0 = fig.add_subplot(gs[:, 0])
            ax0.plot(np.exp(mean_log_rho[:, i]), z_mid, lw=2, label="EnsCGP mean")
            ax0.fill_betweenx(
                z_mid,
                np.exp(mean_log_rho[:, i] - std_log_rho[:, i]),
                np.exp(mean_log_rho[:, i] + std_log_rho[:, i]),
                alpha=0.25,
                label="±1sigma",
            )
            if rho_true is not None and rho_true.shape[0] >= S and rho_true.shape[1] == Z:
                ax0.plot(rho_true[i, :], z_mid, lw=2, ls="--", label="true")
            ax0.set_xscale("log")
            ax0.invert_yaxis()
            ax0.set_xlabel("Resistivity (ohm-m)")
            ax0.set_ylabel("Depth (m)")
            ax0.grid(True, which="both", ls=":")
            ax0.legend(loc="best")

            ax1 = fig.add_subplot(gs[0, 1])
            ax1.plot(periods, noisy_app[:, i], lw=1.0, label="noisy obs")
            ax1.plot(periods, np.exp(mean_log_app[:, i]), lw=2.0, label="EnsCGP mean")
            ax1.set_xscale("log")
            ax1.set_yscale("log")
            ax1.set_ylabel("App. resistivity (ohm-m)")
            ax1.grid(True, which="both", ls=":")
            ax1.legend(loc="best")

            ax2 = fig.add_subplot(gs[1, 1])
            ax2.plot(periods, noisy_phi[:, i], lw=1.0, label="noisy obs")
            ax2.plot(periods, mean_phi[:, i], lw=2.0, label="EnsCGP mean")
            ax2.set_xscale("log")
            ax2.set_xlabel("Period (s)")
            ax2.set_ylabel("Phase (deg)")
            ax2.grid(True, which="both", ls=":")
            ax2.legend(loc="best")

            fig.suptitle(f"EnsCGP training sample {i} | nRMS={result.nrms_mean:.3f}")
            fig.tight_layout()

            qa_path = out_dir / "qa_plots" / f"sample_{i:05d}.png"
            qa_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(qa_path, dpi=150)
            plt.close(fig)

    # -----------------------
    # Save outputs
    # -----------------------
    _save_txt(out_dir / "enscgp_mean_rho_ohm_m.txt", np.exp(mean_log_rho).T)  # (S, Z)
    _save_txt(out_dir / "enscgp_std_log_rho.txt", std_log_rho.T)             # (S, Z) in log-space
    _save_txt(out_dir / "enscgp_mean_appres_ohm_m.txt", np.exp(mean_log_app).T)  # (S, F)
    _save_txt(out_dir / "enscgp_std_log_appres.txt", std_log_app.T)              # (S, F) in log-space
    _save_txt(out_dir / "enscgp_mean_phase_deg.txt", mean_phi.T)                 # (S, F)
    _save_txt(out_dir / "enscgp_std_phase_deg.txt", std_phi.T)                   # (S, F)

    meta = {
        "data_root": str(data_root),
        "training_set": str(args.training_set),
        "ensemble_bank": str(args.ensemble_bank),
        "n_samples": int(S),
        "n_bank": int(Nbank),
        "n_layers": int(Z),
        "n_freqs_obs": int(F_obs),
        "n_freqs_bank": int(F_bank),
        "neighbor_cfg": asdict(neighbor_cfg),
        "solver_cfg": asdict(solver_cfg),
        "noise": asdict(noise),
        "seed": int(args.seed),
    }
    (out_dir / "enscgp_run_metadata.json").write_text(json.dumps(meta, indent=2))
    _save_txt(out_dir / "freqs_hz.txt", freqs_obs)
    _save_txt(out_dir / "depths_m.txt", depths_m)

    print(f"Done. Wrote EnsCGP products to: {out_dir}")

    # -----------------------
    # Summary plots (all samples)
    # -----------------------
    if args.plot_summary:
        import matplotlib.pyplot as plt

        plot_dir_arg = getattr(args, "plot_dir", None)
        plot_dir = Path(plot_dir_arg) if plot_dir_arg is not None else (out_dir / "summary_plots")
        plot_dir.mkdir(parents=True, exist_ok=True)

        periods = 1.0 / freqs_obs
        z_mid = interfaces_to_midpoints(depths_m)

        # ---- (A) Resistivity models: True / EnsCGP mean / % error ----
        if rho_true is not None:
            # rho_true expected (S, Z); convert to log-space for error consistency
            true_log_rho = np.log(np.clip(rho_true[:, :Z].T, 1e-32, None))  # (Z, S)
            est_log_rho = mean_log_rho                                     # (Z, S)

            # log10 ratio misfit
            log_misfit = np.log10(np.exp(est_log_rho)/np.exp(true_log_rho))

            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 7), sharey=True)

            ax1.plot(np.exp(true_log_rho), z_mid)
            ax1.invert_yaxis()
            ax1.set_xlabel("Resistivity (ohm-m)")
            ax1.set_ylabel("Depth (m)")
            ax1.set_xscale("log")
            ax1.set_title("True resistivity models")
            ax1.grid(True, which="both", ls=":")

            ax2.plot(np.exp(est_log_rho), z_mid)
            ax2.invert_yaxis()
            ax2.set_xlabel("Resistivity (ohm-m)")
            ax2.set_xscale("log")
            ax2.set_title("EnsCGP mean models")
            ax2.grid(True, which="both", ls=":")

            ax3.plot(np.mean(log_misfit, axis=1), z_mid)
            ax3.invert_yaxis()
            ax3.set_xlabel(r'Log$_{10}$ Misfit: $\log_{10}(\rho_{\mathrm{est}}/\rho_{\mathrm{true}})$')
            ax3.set_title("Log-Ratio Misfit")
            ax3.grid(True, which="both", ls=":")

            fig.tight_layout()
            fig.savefig(plot_dir / "resistivity_enscgp_summary.png", bbox_inches="tight", dpi=200)
            plt.close(fig)

        # ---- (B) MT responses: Clean / Noisy / Denoised + mean % error ----
        # We want to mimic your reference: 4 stacked panels for appres and for phase.
        # Denoised = exp(mean_log_app) for appres, and mean_phi for phase.

        # Apparent resistivity summary
        if clean_app is not None:
            # clean_app is linear appres (F,S). mean_log_app is log(appres) mean (F,S)
            denoised_app = np.exp(mean_log_app)   # (F,S)
            noisy_app_lin = noisy_app             # (F,S) already linear
            clean_app_lin = clean_app             # (F,S) linear

            err_app = clean_app_lin - denoised_app
            pct_err_app = (np.abs(err_app) / np.clip(clean_app_lin, 1e-32, None)) * 100.0
            mean_pct_err_app = np.mean(pct_err_app, axis=1)  # (F,)

            fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(10, 13), sharex=True)

            ax1.plot(periods, clean_app_lin, "-")
            ax1.set_xscale("log")
            ax1.set_yscale("log")
            ax1.set_ylabel("App. resistivity (ohm-m)")
            ax1.set_title("Clean")

            ax2.plot(periods, noisy_app_lin, "-")
            ax2.set_xscale("log")
            ax2.set_yscale("log")
            ax2.set_ylabel("App. resistivity (ohm-m)")
            ax2.set_title("Noisy")

            ax3.plot(periods, denoised_app, "-")
            ax3.set_xscale("log")
            ax3.set_yscale("log")
            ax3.set_ylabel("App. resistivity (ohm-m)")
            ax3.set_title("Denoised (EnsCGP mean)")

            ax4.plot(periods, mean_pct_err_app, "-")
            ax4.set_xscale("log")
            ax4.set_ylabel("Percentage Error (%)")
            ax4.set_xlabel("Period (s)")
            ax4.set_title("Mean Percentage Error")

            fig.tight_layout()
            fig.savefig(plot_dir / "appres_enscgp_summary.png", bbox_inches="tight", dpi=200)
            plt.close(fig)

            # Phase summary
            denoised_phi = mean_phi    # (F,S)
            noisy_phi_deg = noisy_phi  # (F,S)
            clean_phi_deg = clean_phi  # (F,S)

            err_phi = clean_phi_deg - denoised_phi
            pct_err_phi = (np.abs(err_phi) / np.clip(clean_phi_deg, 1e-32, None)) * 100.0
            mean_pct_err_phi = np.mean(pct_err_phi, axis=1)  # (F,)

            fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(10, 13), sharex=True)

            ax1.plot(periods, clean_phi_deg, "-")
            ax1.set_xscale("log")
            ax1.set_ylabel("Phase (deg)")
            ax1.set_title("Clean")

            ax2.plot(periods, noisy_phi_deg, "-")
            ax2.set_xscale("log")
            ax2.set_ylabel("Phase (deg)")
            ax2.set_title("Noisy")

            ax3.plot(periods, denoised_phi, "-")
            ax3.set_xscale("log")
            ax3.set_ylabel("Phase (deg)")
            ax3.set_title("Denoised (EnsCGP mean)")

            ax4.plot(periods, mean_pct_err_phi, "-")
            ax4.set_xscale("log")
            ax4.set_ylabel("Percentage Error (%)")
            ax4.set_xlabel("Period (s)")
            ax4.set_title("Mean Percentage Error")

            fig.tight_layout()
            fig.savefig(plot_dir / "phase_enscgp_summary.png", bbox_inches="tight", dpi=200)
            plt.close(fig)



if __name__ == "__main__":
    main()
