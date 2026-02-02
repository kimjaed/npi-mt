#!/usr/bin/env python3
"""
Train NPI residual network on synthetic data using EnsCGP mean products,
and (optionally) generate summary plots similar to EnsCGP.

Defaults
--------
Inputs:
  data/synthetic/training_small/enscgp_out/
    - enscgp_mean_appres_ohm_m.txt   (S, F) linear apparent resistivity
    - enscgp_mean_phase_deg.txt     (S, F) phase in degrees
    - enscgp_mean_rho_ohm_m.txt     (S, Z) linear resistivity
  data/synthetic/training_small/
    - rho_ens.txt                   (S, Z) true linear resistivity
    - depths_m.txt                  (Z+1,)
    - freqs_hz.txt                  (F,)

Outputs:
  data/synthetic/training_small/npi_out/
    - best_model_fold{k}.pth
    - train_losses_fold{k}.txt
    - val_losses_fold{k}.txt
    - input_mean.pth, input_std.pth, label_mean.pth, label_std.pth
    - training_metadata.json
    - best_fold.json
    - (optional) summary_plots/*
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, Subset
from sklearn.model_selection import KFold

from npi_mt.npi import ResNetResidual1D


REPO_ROOT = Path(__file__).resolve().parents[1]


# -------------------------
# Utilities
# -------------------------

def interfaces_to_midpoints(z_if: np.ndarray) -> np.ndarray:
    z_if = np.asarray(z_if, float)
    return 0.5 * (z_if[:-1] + z_if[1:])

def _load_txt(path: Path) -> np.ndarray:
    return np.asarray(np.loadtxt(path))

def _ensure_2d_samples_by_features(x: np.ndarray, name: str) -> np.ndarray:
    """
    Expect x to be (S, F/Z). If stored as (F/Z, S), transpose.
    Heuristic: transpose only if second dim is clearly "sample-like".
    """
    x = np.asarray(x)
    if x.ndim != 2:
        raise ValueError(f"{name} must be 2D; got shape {x.shape}")

    # Typical: S >> F or S >> Z -> keep (S, F/Z)
    # If the array is (F/Z, S) then S is the second dim and much larger.
    if x.shape[1] >= 2 * x.shape[0]:
        return x.T
    return x

def _standardize_inputs(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    x: (S, 2, F)
    Returns: x_norm, input_mean(1,2,F), input_std(1,2,F)
    """
    mean = x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, keepdims=True) + 1e-8
    return (x - mean) / std, mean, std

def _standardize_labels(y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    y: (S, Z)
    Returns: y_norm, label_mean(Z,), label_std(Z,)
    """
    mean = y.mean(axis=0)
    std = y.std(axis=0) + 1e-8
    return (y - mean) / std, mean, std

def _save_stats(out_dir: Path, input_mean, input_std, label_mean, label_std) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"input_mean": torch.as_tensor(input_mean, dtype=torch.float32)}, out_dir / "input_mean.pth")
    torch.save({"input_std": torch.as_tensor(input_std, dtype=torch.float32)}, out_dir / "input_std.pth")
    torch.save({"label_mean": torch.as_tensor(label_mean, dtype=torch.float32)}, out_dir / "label_mean.pth")
    torch.save({"label_std": torch.as_tensor(label_std, dtype=torch.float32)}, out_dir / "label_std.pth")

def _load_stats(out_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    im = torch.load(out_dir / "input_mean.pth", map_location="cpu", weights_only=False)["input_mean"].numpy()
    is_ = torch.load(out_dir / "input_std.pth", map_location="cpu", weights_only=False)["input_std"].numpy()
    lm = torch.load(out_dir / "label_mean.pth", map_location="cpu", weights_only=False)["label_mean"].numpy()
    ls = torch.load(out_dir / "label_std.pth", map_location="cpu", weights_only=False)["label_std"].numpy()
    return im, is_, lm, ls


# -------------------------
# Data loading
# -------------------------

def load_training_arrays(enscgp_dir: Path, truth_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns
    -------
    log_app : (S,F)
    phi_deg : (S,F)
    log_rho0: (S,Z)  EnsCGP baseline
    log_rhoT: (S,Z)  truth
    """
    eps = 1e-32

    app = _ensure_2d_samples_by_features(
        _load_txt(enscgp_dir / "enscgp_mean_appres_ohm_m.txt"),
        "enscgp_mean_appres_ohm_m",
    )
    phi = _ensure_2d_samples_by_features(
        _load_txt(enscgp_dir / "enscgp_mean_phase_deg.txt"),
        "enscgp_mean_phase_deg",
    )
    rho0 = _ensure_2d_samples_by_features(
        _load_txt(enscgp_dir / "enscgp_mean_rho_ohm_m.txt"),
        "enscgp_mean_rho_ohm_m",
    )
    rhoT = _ensure_2d_samples_by_features(
        _load_txt(truth_dir / "rho_ens.txt"),
        "rho_ens (truth)",
    )

    if app.shape != phi.shape:
        raise ValueError(f"AppRes {app.shape} must match Phase {phi.shape}")
    if rho0.shape != rhoT.shape:
        raise ValueError(f"EnsCGP rho {rho0.shape} must match truth rho {rhoT.shape}")
    if app.shape[0] != rho0.shape[0]:
        raise ValueError(f"Sample count mismatch: app has {app.shape[0]} samples but rho has {rho0.shape[0]}")

    log_app = np.log(np.clip(app, eps, None)).astype(np.float32)
    phi_deg = phi.astype(np.float32)
    log_rho0 = np.log(np.clip(rho0, eps, None)).astype(np.float32)
    log_rhoT = np.log(np.clip(rhoT, eps, None)).astype(np.float32)

    return log_app, phi_deg, log_rho0, log_rhoT


# -------------------------
# Inference helper (for plots)
# -------------------------

@torch.no_grad()
def predict_refined_log_rho(
    model: torch.nn.Module,
    device: torch.device,
    x_inputs: np.ndarray,          # (S,2,F) UN-normalized (log_app, phi)
    log_rho0: np.ndarray,          # (S,Z)
    input_mean: np.ndarray,        # (1,2,F)
    input_std: np.ndarray,         # (1,2,F)
    label_mean: np.ndarray,        # (Z,)
    label_std: np.ndarray,         # (Z,)
    batch_size: int = 256,
) -> np.ndarray:
    """
    Returns refined log-rho: log_rho0 + denorm(pred_residual_norm).
    """
    model.eval()

    x_norm = (x_inputs - input_mean) / input_std
    S = x_norm.shape[0]
    Z = log_rho0.shape[1]

    out = np.empty((S, Z), dtype=np.float32)

    for i0 in range(0, S, batch_size):
        i1 = min(S, i0 + batch_size)
        xb = torch.from_numpy(x_norm[i0:i1]).to(device=device, dtype=torch.float32)  # (B,2,F)

        pred_norm = model(xb).cpu().numpy().astype(np.float32)  # (B,Z)
        pred = pred_norm * label_std[None, :] + label_mean[None, :]  # denorm residual
        out[i0:i1, :] = log_rho0[i0:i1, :] + pred

    return out


# -------------------------
# Summary plotting
# -------------------------

def make_summary_plots(
    plot_dir: Path,
    depths_m: np.ndarray,
    freqs_hz: np.ndarray,
    rho_true_lin: np.ndarray,          # (S,Z) linear
    rho_enscgp_lin: np.ndarray,        # (S,Z) linear
    rho_npi_lin: np.ndarray,           # (S,Z) linear
    *,
    # Optional: response plot ingredients (all (F,S) in linear/deg)
    clean_app: np.ndarray | None = None,
    noisy_app: np.ndarray | None = None,
    clean_phi: np.ndarray | None = None,
    noisy_phi: np.ndarray | None = None,
    enscgp_app: np.ndarray | None = None,
    enscgp_phi: np.ndarray | None = None,
    npi_app: np.ndarray | None = None,
    npi_phi: np.ndarray | None = None,
) -> None:
    import matplotlib.pyplot as plt

    plot_dir.mkdir(parents=True, exist_ok=True)

    periods = 1.0 / np.asarray(freqs_hz, float).reshape(-1)
    z_mid = interfaces_to_midpoints(depths_m)
    print("z_mid: ", z_mid)

    eps = 1e-30
    S, Z = rho_true_lin.shape

    # ---- (A) Resistivity: True / EnsCGP / NPI / mean log-ratio misfit ----
    rho_true_lin = rho_true_lin.T
    rho_enscgp_lin = rho_enscgp_lin.T
    rho_npi_lin = rho_npi_lin.T

    # log-ratio misfit (more stable than % error across decades)
    log_misfit_enscgp = np.log10(rho_enscgp_lin/rho_true_lin)
    log_misfit_npi = np.log10(rho_npi_lin/rho_true_lin)
    mean_abs_log_misfit_enscgp = np.mean(np.abs(log_misfit_enscgp), axis=1)  # (Z,)
    mean_abs_log_misfit_npi = np.mean(np.abs(log_misfit_npi), axis=1)  # (Z,)

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(14, 7), sharey=True)

    ax1.plot(rho_true_lin, z_mid)
    ax1.set_xscale("log")
    ax1.set_xlabel("Resistivity (ohm-m)")
    ax1.set_ylabel("Depth (m)")
    ax1.set_title("True models")
    ax1.grid(True, which="both", ls=":")

    ax2.plot(rho_enscgp_lin, z_mid)
    ax2.set_xscale("log")
    ax2.set_xlabel("Resistivity (ohm-m)")
    ax2.set_title("EnsCGP mean")
    ax2.grid(True, which="both", ls=":")

    ax3.plot(rho_npi_lin, z_mid)
    ax3.set_xscale("log")
    ax3.set_xlabel("Resistivity (ohm-m)")
    ax3.set_title("NPI refined")
    ax3.grid(True, which="both", ls=":")

    ax4.plot(mean_abs_log_misfit_enscgp, z_mid, label="EnsCGP")
    ax4.plot(mean_abs_log_misfit_npi, z_mid, label="NPI")
    ax4.set_xlabel(r'Log$_{10}$ Misfit: $\log_{10}(\rho_{\mathrm{est}}/\rho_{\mathrm{true}})$')
    ax4.set_title("Log-Ratio Misfit")
    ax4.grid(True, which="both", ls=":")
    ax4.legend(loc="best")

    ax1.invert_yaxis()

    fig.tight_layout()
    fig.savefig(plot_dir / "resistivity_npi_summary.png", bbox_inches="tight", dpi=200)
    plt.close(fig)

    # ---- (B) MT responses summary (optional) ----
    # Mimics your EnsCGP 4-panel figures, but "Denoised" becomes:
    #   - EnsCGP mean (direct from file)
    #   - NPI forward response (if provided)
    if clean_app is not None and noisy_app is not None and enscgp_app is not None and npi_app is not None:
        # all should be (F,S)
        clean_app = np.asarray(clean_app)
        noisy_app = np.asarray(noisy_app)
        enscgp_app = np.asarray(enscgp_app)
        npi_app = np.asarray(npi_app)

        # mean % error relative to clean
        pct_err_ens = np.mean(np.abs((enscgp_app - clean_app) / np.clip(clean_app, 1e-32, None)) * 100.0, axis=1)
        pct_err_npi = np.mean(np.abs((npi_app - clean_app) / np.clip(clean_app, 1e-32, None)) * 100.0, axis=1)

        fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, 1, figsize=(10, 16), sharex=True)

        ax1.plot(periods, clean_app, "-")
        ax1.set_xscale("log")
        ax1.set_yscale("log")
        ax1.set_ylabel("App. resistivity (ohm-m)")
        ax1.set_title("Clean")
        ax1.grid(True, which="both", ls=":")

        ax2.plot(periods, noisy_app, "-")
        ax2.set_xscale("log")
        ax2.set_yscale("log")
        ax2.set_ylabel("App. resistivity (ohm-m)")
        ax2.set_title("Noisy")
        ax2.grid(True, which="both", ls=":")

        ax3.plot(periods, enscgp_app, "-")
        ax3.set_xscale("log")
        ax3.set_yscale("log")
        ax3.set_ylabel("App. resistivity (ohm-m)")
        ax3.set_title("EnsCGP mean")
        ax3.grid(True, which="both", ls=":")

        ax4.plot(periods, npi_app, "-")
        ax4.set_xscale("log")
        ax4.set_yscale("log")
        ax4.set_ylabel("App. resistivity (ohm-m)")
        ax4.set_title("NPI forward (from refined rho)")
        ax4.grid(True, which="both", ls=":")

        ax5.plot(periods, pct_err_ens, "-", label="EnsCGP")
        ax5.plot(periods, pct_err_npi, "-", label="NPI")
        ax5.set_xscale("log")
        ax5.set_ylabel("Mean % error")
        ax5.set_xlabel("Period (s)")
        ax5.set_title("Mean percentage error vs clean")
        ax5.grid(True, which="both", ls=":")
        ax5.legend(loc="best")

        fig.tight_layout()
        fig.savefig(plot_dir / "appres_npi_summary.png", bbox_inches="tight", dpi=200)
        plt.close(fig)

    if clean_phi is not None and noisy_phi is not None and enscgp_phi is not None and npi_phi is not None:
        clean_phi = np.asarray(clean_phi)
        noisy_phi = np.asarray(noisy_phi)
        enscgp_phi = np.asarray(enscgp_phi)
        npi_phi = np.asarray(npi_phi)

        pct_err_ens = np.mean(np.abs((enscgp_phi - clean_phi) / np.clip(clean_phi, 1e-32, None)) * 100.0, axis=1)
        pct_err_npi = np.mean(np.abs((npi_phi - clean_phi) / np.clip(clean_phi, 1e-32, None)) * 100.0, axis=1)

        fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, 1, figsize=(10, 16), sharex=True)

        ax1.plot(periods, clean_phi, "-")
        ax1.set_xscale("log")
        ax1.set_ylabel("Phase (deg)")
        ax1.set_title("Clean")
        ax1.grid(True, which="both", ls=":")

        ax2.plot(periods, noisy_phi, "-")
        ax2.set_xscale("log")
        ax2.set_ylabel("Phase (deg)")
        ax2.set_title("Noisy")
        ax2.grid(True, which="both", ls=":")

        ax3.plot(periods, enscgp_phi, "-")
        ax3.set_xscale("log")
        ax3.set_ylabel("Phase (deg)")
        ax3.set_title("EnsCGP mean")
        ax3.grid(True, which="both", ls=":")

        ax4.plot(periods, npi_phi, "-")
        ax4.set_xscale("log")
        ax4.set_ylabel("Phase (deg)")
        ax4.set_title("NPI forward (from refined rho)")
        ax4.grid(True, which="both", ls=":")

        ax5.plot(periods, pct_err_ens, "-", label="EnsCGP")
        ax5.plot(periods, pct_err_npi, "-", label="NPI")
        ax5.set_xscale("log")
        ax5.set_ylabel("Mean % error")
        ax5.set_xlabel("Period (s)")
        ax5.set_title("Mean percentage error vs clean")
        ax5.grid(True, which="both", ls=":")
        ax5.legend(loc="best")

        fig.tight_layout()
        fig.savefig(plot_dir / "phase_npi_summary.png", bbox_inches="tight", dpi=200)
        plt.close(fig)


# -------------------------
# Training
# -------------------------

def train_kfold(args: argparse.Namespace) -> None:
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if (torch.cuda.is_available() and not args.cpu) else "cpu")
    print("device:", device)

    # Resolve dirs
    data_root = Path(args.data_root)
    if not data_root.is_absolute():
        data_root = (REPO_ROOT / data_root).resolve()

    training_dir = (data_root / args.training_set).resolve()
    enscgp_dir = Path(args.enscgp_dir).resolve() if args.enscgp_dir else (training_dir / "enscgp_out")
    out_dir = Path(args.out_dir).resolve() if args.out_dir else (training_dir / "npi_out")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load grids (for plotting / forward later)
    depths_m = _load_txt(training_dir / "depths_m.txt").astype(float).reshape(-1)
    freqs_hz = _load_txt(training_dir / "freqs_hz.txt").astype(float).reshape(-1)

    # Load arrays
    log_app, phi_deg, log_rho0, log_rhoT = load_training_arrays(enscgp_dir=enscgp_dir, truth_dir=training_dir)

    S, F = log_app.shape
    Z = log_rho0.shape[1]
    print(f"Loaded: S={S}, F={F}, Z={Z}")

    if args.max_samples > 0:
        S2 = min(S, args.max_samples)
        log_app = log_app[:S2]
        phi_deg = phi_deg[:S2]
        log_rho0 = log_rho0[:S2]
        log_rhoT = log_rhoT[:S2]
        S = S2
        print(f"Trimmed training to max_samples={S}")

    # Build unnormalized x for later plots/inference
    x_raw = np.stack([log_app, phi_deg], axis=1).astype(np.float32)  # (S,2,F)
    y_raw = (log_rhoT - log_rho0).astype(np.float32)                # (S,Z)

    # Standardize on full dataset (matches your legacy script)
    x_norm, input_mean, input_std = _standardize_inputs(x_raw)
    y_norm, label_mean, label_std = _standardize_labels(y_raw)

    _save_stats(out_dir, input_mean, input_std, label_mean, label_std)

    dataset = TensorDataset(torch.from_numpy(x_norm), torch.from_numpy(y_norm))
    kfold = KFold(n_splits=args.k_folds, shuffle=True, random_state=args.seed)

    # Metadata
    meta = {
        "data_root": str(data_root),
        "training_set": str(args.training_set),
        "enscgp_dir": str(enscgp_dir),
        "out_dir": str(out_dir),
        "seed": int(args.seed),
        "S": int(S),
        "F": int(F),
        "Z": int(Z),
        "k_folds": int(args.k_folds),
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "patience": int(args.patience),
        "min_delta": float(args.min_delta),
        "huber_delta": float(args.huber_delta),
        "dropout": float(args.dropout),
        "strict_freqs": bool(args.strict_freqs),
    }
    (out_dir / "training_metadata.json").write_text(json.dumps(meta, indent=2))

    fold_best_vals: list[float] = []

    # Train folds
    for fold, (train_idx, val_idx) in enumerate(kfold.split(np.arange(len(dataset))), start=1):
        print(f"\nFold {fold}/{args.k_folds}")

        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)

        train_loader = DataLoader(
            train_subset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
        )
        val_loader = DataLoader(
            val_subset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
        )

        model = ResNetResidual1D(
            n_freqs=(F if args.strict_freqs else None),
            n_layers=Z,
            dropout=args.dropout,
        ).to(device)

        criterion = torch.nn.HuberLoss(delta=args.huber_delta)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", patience=args.lr_patience, factor=args.lr_factor, min_lr=args.min_lr
        )

        best_val = float("inf")
        best_state = None
        trigger = 0

        train_losses: list[float] = []
        val_losses: list[float] = []

        for epoch in range(1, args.epochs + 1):
            # ---- train ----
            model.train()
            run = 0.0
            for xb, yb in train_loader:
                xb = xb.to(device=device, dtype=torch.float32)
                yb = yb.to(device=device, dtype=torch.float32)

                optimizer.zero_grad(set_to_none=True)
                pred = model(xb)
                loss = criterion(pred, yb)
                loss.backward()

                if args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(args.grad_clip))

                optimizer.step()
                run += float(loss.item())

            train_loss = run / max(1, len(train_loader))
            train_losses.append(train_loss)

            # ---- val ----
            model.eval()
            run = 0.0
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb = xb.to(device=device, dtype=torch.float32)
                    yb = yb.to(device=device, dtype=torch.float32)
                    pred = model(xb)
                    loss = criterion(pred, yb)
                    run += float(loss.item())

            val_loss = run / max(1, len(val_loader))
            val_losses.append(val_loss)

            scheduler.step(val_loss)

            if (epoch == 1) or (epoch % args.print_every == 0):
                print(
                    f"Epoch {epoch:4d}/{args.epochs} | "
                    f"train={train_loss:.6e} | val={val_loss:.6e} | "
                    f"lr={optimizer.param_groups[0]['lr']:.2e}"
                )

            # ---- early stopping ----
            improved = (best_val - val_loss) > args.min_delta
            if improved:
                best_val = val_loss
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                trigger = 0
            else:
                trigger += 1
                if trigger >= args.patience:
                    print(f"Early stopping at epoch {epoch} (no improvement > {args.min_delta} for {args.patience} epochs).")
                    break

        fold_best_vals.append(float(best_val))

        # Save fold artifacts
        fold_model_path = out_dir / f"best_model_fold{fold}.pth"
        torch.save(best_state if best_state is not None else model.state_dict(), fold_model_path)

        np.savetxt(out_dir / f"train_losses_fold{fold}.txt", np.asarray(train_losses))
        np.savetxt(out_dir / f"val_losses_fold{fold}.txt", np.asarray(val_losses))

        print(f"Fold {fold}: best val = {best_val:.6e} | saved {fold_model_path.name}")

    # Record best fold
    best_fold = int(np.argmin(np.asarray(fold_best_vals)) + 1)
    best_info = {"best_fold": best_fold, "best_val": float(np.min(fold_best_vals)), "fold_best_vals": fold_best_vals}
    (out_dir / "best_fold.json").write_text(json.dumps(best_info, indent=2))
    print(f"\nBest fold: {best_fold} (val={best_info['best_val']:.6e})")

    # -----------------------
    # Summary plots (all samples)
    # -----------------------
    if args.plot_summary:
        plot_dir = Path(args.plot_dir).resolve() if args.plot_dir else (out_dir / "summary_plots")
        plot_dir.mkdir(parents=True, exist_ok=True)

        # Load stats (ensures plot inference matches what you saved)
        input_mean, input_std, label_mean, label_std = _load_stats(out_dir)

        # Load best fold model
        model = ResNetResidual1D(
            n_freqs=(F if args.strict_freqs else None),
            n_layers=Z,
            dropout=args.dropout,
        ).to(device)

        state = torch.load(out_dir / f"best_model_fold{best_fold}.pth", map_location="cpu", weights_only=False)
        model.load_state_dict(state, strict=True)

        # Predict refined log-rho for all samples (or a subset for plotting)
        Sp = S
        if args.plot_max_samples > 0:
            Sp = min(S, args.plot_max_samples)

        log_rho_npi = predict_refined_log_rho(
            model=model,
            device=device,
            x_inputs=x_raw[:Sp],            # (Sp,2,F) raw (log_app,phi)
            log_rho0=log_rho0[:Sp],         # (Sp,Z)
            input_mean=input_mean,
            input_std=input_std,
            label_mean=label_mean,
            label_std=label_std,
            batch_size=args.plot_batch_size,
        )  # (Sp,Z)

        rho_true_lin = np.exp(log_rhoT[:Sp]).astype(float)
        rho_enscgp_lin = np.exp(log_rho0[:Sp]).astype(float)
        rho_npi_lin = np.exp(log_rho_npi).astype(float)

        # Optional response plots if clean/noisy exist
        clean_app = clean_phi = noisy_app = noisy_phi = None
        clean_app_path = training_dir / "clean_appres_ens.txt"
        clean_phi_path = training_dir / "clean_phase_ens.txt"
        noisy_app_path = training_dir / "noisy_appres_ens.txt"
        noisy_phi_path = training_dir / "noisy_phase_ens.txt"

        # EnsCGP mean responses (these exist because they’re inputs)
        enscgp_app = _ensure_2d_samples_by_features(
            _load_txt(enscgp_dir / "enscgp_mean_appres_ohm_m.txt"),
            "enscgp_mean_appres_ohm_m",
        ).T  # -> (F,S) for plotting style consistency
        enscgp_phi = _ensure_2d_samples_by_features(
            _load_txt(enscgp_dir / "enscgp_mean_phase_deg.txt"),
            "enscgp_mean_phase_deg",
        ).T

        # Trim to plotting subset and ensure (F,Sp)
        enscgp_app = enscgp_app[:, :Sp]
        enscgp_phi = enscgp_phi[:, :Sp]

        # Forward model NPI refined rho to get predicted responses for plotting
        npi_app = npi_phi = None
        if args.plot_forward_responses:
            from npi_mt.mt1d.forward import forward_ensemble

            npi_app, npi_phi = forward_ensemble(
                rho_ens_ohm_m=rho_npi_lin,
                depth_interfaces_m=depths_m,
                freqs_hz=freqs_hz,
                n_workers=(args.plot_fwd_workers if args.plot_fwd_workers > 0 else None),
                parallel=args.plot_parallel,
            )
            # forward_ensemble returns (Sp,F) -> transpose to (F,Sp) for plotting
            npi_app = npi_app.T
            npi_phi = npi_phi.T

        # Load clean/noisy if they exist
        if clean_app_path.exists() and clean_phi_path.exists() and noisy_app_path.exists() and noisy_phi_path.exists():
            clean_app = _ensure_2d_samples_by_features(_load_txt(clean_app_path), "clean_appres_ens").T[:, :Sp]
            clean_phi = _ensure_2d_samples_by_features(_load_txt(clean_phi_path), "clean_phase_ens").T[:, :Sp]
            noisy_app = _ensure_2d_samples_by_features(_load_txt(noisy_app_path), "noisy_appres_ens").T[:, :Sp]
            noisy_phi = _ensure_2d_samples_by_features(_load_txt(noisy_phi_path), "noisy_phase_ens").T[:, :Sp]

        make_summary_plots(
            plot_dir=plot_dir,
            depths_m=depths_m,
            freqs_hz=freqs_hz,
            rho_true_lin=rho_true_lin,
            rho_enscgp_lin=rho_enscgp_lin,
            rho_npi_lin=rho_npi_lin,
            clean_app=clean_app,
            noisy_app=noisy_app,
            clean_phi=clean_phi,
            noisy_phi=noisy_phi,
            enscgp_app=enscgp_app,
            enscgp_phi=enscgp_phi,
            npi_app=npi_app,
            npi_phi=npi_phi,
        )

        print(f"Summary plots written to: {plot_dir}")

    print(f"\nDone. Outputs written to: {out_dir}")


# -------------------------
# CLI
# -------------------------

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train NPI residual network on synthetic EnsCGP products.")
    p.add_argument("--data-root", type=str, default="data", help="Repo data directory (relative or absolute).")
    p.add_argument("--training-set", type=str, default="synthetic/training_small",
                   help="Relative path under data-root containing rho_ens.txt, depths_m.txt, freqs_hz.txt.")

    p.add_argument("--enscgp-dir", type=str, default=None,
                   help="Directory containing EnsCGP mean products. Default: <training-set>/enscgp_out")
    p.add_argument("--out-dir", type=str, default=None,
                   help="Output directory. Default: <training-set>/npi_out")

    # Training hyperparameters
    p.add_argument("--k-folds", type=int, default=5)
    p.add_argument("--epochs", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--huber-delta", type=float, default=1.0)

    # Early stopping + scheduler
    p.add_argument("--patience", type=int, default=200)
    p.add_argument("--min-delta", type=float, default=1e-6)
    p.add_argument("--lr-patience", type=int, default=50)
    p.add_argument("--lr-factor", type=float, default=0.5)
    p.add_argument("--min-lr", type=float, default=1e-6)

    # Runtime
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cpu", action="store_true", help="Force CPU even if CUDA is available.")
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--grad-clip", type=float, default=0.8)
    p.add_argument("--print-every", type=int, default=10)
    p.add_argument("--max-samples", type=int, default=0, help="If >0, train only on the first max-samples.")
    p.add_argument("--strict-freqs", action="store_true",
                   help="If set, enforce model input F matches training F exactly.")

    # Plotting
    p.add_argument("--plot-summary", action="store_true", help="Write summary plots after training.")
    p.add_argument("--plot-dir", type=str, default=None, help="Plot directory. Default: <out-dir>/summary_plots")
    p.add_argument("--plot-max-samples", type=int, default=0,
                   help="If >0, only use first K samples for summary plots.")
    p.add_argument("--plot-batch-size", type=int, default=256, help="Batch size for NPI inference used in plots.")

    # Optional forward modeling for response plots
    p.add_argument("--plot-forward-responses", action="store_true",
                   help="If set, forward model refined rho to plot MT responses (more expensive).")
    p.add_argument("--plot-fwd-workers", type=int, default=0, help="Workers for forward_ensemble in plots.")
    p.add_argument("--plot-parallel", type=str, default="thread", choices=["serial", "thread", "process"],
                   help="Parallel backend for forward_ensemble in plots.")
    return p


def main() -> None:
    args = build_argparser().parse_args()
    train_kfold(args)


if __name__ == "__main__":
    main()
