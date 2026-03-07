from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from phase4a.array_simulator import simulate_identical_array_spectra
from phase4a.common import load_config, resolve_project_path
from phase4a.local_window import apply_window_shift, extract_local_distorted_spectrum


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate multiple Phase4 simulation output figures")
    p.add_argument("--config", type=str, default="config/phase4_array.yaml")
    p.add_argument("--sample-index", type=int, default=-1, help="-1 means first test sample")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    data_path = resolve_project_path(cfg["phase4a"]["dataset_path"])
    out_dir = resolve_project_path(cfg["phase4a"]["results_dir"]) / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not data_path.exists():
        raise FileNotFoundError(f"Missing dataset: {data_path}")

    d = np.load(data_path)
    wavelengths = d["wavelengths"].astype(np.float64)
    x_local = d["X_local"].astype(np.float64)
    x_total = d["X_total"].astype(np.float64)
    y = d["Y_dlambda_target"].astype(np.float64)
    idx_test = d["idx_test"].astype(np.int64)
    centers = d["centers_nm"].astype(np.float64)
    target_index = int(d["target_index"][0])

    leakage_weights = d["leakage_weights"].astype(np.float64) if "leakage_weights" in d else None
    neighbor_mode = d["neighbor_mode"].astype(np.int64) if "neighbor_mode" in d else None
    window_shift_nm = d["window_shift_nm"].astype(np.float64) if "window_shift_nm" in d else None
    amp_scales = d["amplitude_scales"].astype(np.float64) if "amplitude_scales" in d else None
    lw_scales = d["linewidth_scales"].astype(np.float64) if "linewidth_scales" in d else None

    sample_idx = int(idx_test[0]) if args.sample_index < 0 else int(args.sample_index)
    if not (0 <= sample_idx < len(x_local)):
        raise ValueError(f"sample-index out of range: {sample_idx}")

    # Rebuild per-grating for selected sample
    dlam = float(y[sample_idx])
    amps = amp_scales[sample_idx] if amp_scales is not None else None
    sigs = lw_scales[sample_idx] if lw_scales is not None else None
    per_grating, total_rebuilt, _ = simulate_identical_array_spectra(
        wavelengths,
        cfg,
        dlam,
        amplitude_scales=amps,
        linewidth_scales=sigs,
    )
    w = leakage_weights[sample_idx] if leakage_weights is not None else None
    local_clean, used_weights = extract_local_distorted_spectrum(per_grating, target_index, cfg["local_window"], weights=w)
    shift_nm = float(window_shift_nm[sample_idx]) if window_shift_nm is not None else 0.0
    local_shifted = apply_window_shift(local_clean, wavelengths, shift_nm)

    # Figure 1: Array components + total
    fig, ax = plt.subplots(figsize=(10.5, 4.5), constrained_layout=True)
    for i in range(per_grating.shape[0]):
        ax.plot(wavelengths, per_grating[i], linewidth=1.2, label=f"FBG{i+1}")
    ax.plot(wavelengths, total_rebuilt, "k-", linewidth=2.0, label="Array total")
    ax.set_title("Phase4 Array Components and Total Spectrum")
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Reflectivity (a.u.)")
    ax.grid(True, alpha=0.25)
    ax.legend(ncol=3, fontsize=8)
    fig.savefig(out_dir / "fig1_array_components.png", dpi=300)
    plt.close(fig)

    # Figure 2: Local clean/shifted/noisy
    fig, ax = plt.subplots(figsize=(10.5, 4.5), constrained_layout=True)
    ax.plot(wavelengths, local_clean, color="#1f77b4", linewidth=1.6, label="Local clean (weighted sum)")
    ax.plot(wavelengths, local_shifted, color="#ff7f0e", linewidth=1.4, label=f"After random shift ({shift_nm:+.4f} nm)")
    ax.plot(wavelengths, x_local[sample_idx], color="#d62728", linewidth=1.2, alpha=0.9, label="Final local distorted/noisy")
    ax.plot(wavelengths, x_total[sample_idx], color="#2ca02c", linewidth=1.1, alpha=0.6, label="Array total (saved)")
    ax.set_title(f"Phase4 Local Window Generation (Sample {sample_idx}, Δλ={dlam:.4f} nm)")
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Intensity (a.u.)")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    fig.savefig(out_dir / "fig2_local_clean_shift_noisy.png", dpi=300)
    plt.close(fig)

    # Figure 3: Multiple samples sorted by label
    order = np.argsort(y[idx_test])
    pick = np.linspace(0, len(order) - 1, 10, dtype=int)
    chosen = idx_test[order[pick]]
    fig, ax = plt.subplots(figsize=(10.5, 4.8), constrained_layout=True)
    cmap = plt.get_cmap("viridis")
    for i, idx in enumerate(chosen):
        ax.plot(wavelengths, x_local[idx], color=cmap(i / max(1, len(chosen) - 1)), linewidth=1.3, label=f"Δλ={y[idx]:.3f}")
    ax.set_title("Phase4 Local Spectra Across Different Labels (Test)")
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Input intensity")
    ax.grid(True, alpha=0.25)
    ax.legend(ncol=2, fontsize=8)
    fig.savefig(out_dir / "fig3_multi_samples_sorted_by_label.png", dpi=300)
    plt.close(fig)

    # Figure 4: Leakage mode + weight distribution
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), constrained_layout=True)
    if neighbor_mode is not None:
        modes, counts = np.unique(neighbor_mode, return_counts=True)
        axes[0].bar([str(int(m)) for m in modes], counts, color=["#4e79a7", "#e15759"])
        axes[0].set_title("Neighbor Mode Distribution")
        axes[0].set_xlabel("Number of neighbors used")
        axes[0].set_ylabel("Count")
    else:
        axes[0].text(0.5, 0.5, "neighbor_mode not found", ha="center", va="center")
        axes[0].axis("off")

    if leakage_weights is not None:
        axes[1].hist(leakage_weights[:, 1], bins=40, alpha=0.55, label="left-1", color="#4e79a7")
        axes[1].hist(leakage_weights[:, 3], bins=40, alpha=0.55, label="right-1", color="#f28e2b")
        axes[1].hist(leakage_weights[:, 0], bins=40, alpha=0.4, label="left-2", color="#59a14f")
        axes[1].hist(leakage_weights[:, 4], bins=40, alpha=0.4, label="right-2", color="#e15759")
        axes[1].set_title("Random Leakage Weights")
        axes[1].set_xlabel("Weight")
        axes[1].set_ylabel("Count")
        axes[1].legend(fontsize=8)
        axes[1].grid(True, alpha=0.25)
    else:
        axes[1].text(0.5, 0.5, "leakage_weights not found", ha="center", va="center")
        axes[1].axis("off")
    fig.savefig(out_dir / "fig4_leakage_modes_and_weights.png", dpi=300)
    plt.close(fig)

    # Figure 5: Randomized grating parameter distribution
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.0), constrained_layout=True)
    if amp_scales is not None:
        axes[0].hist(amp_scales.reshape(-1), bins=50, color="#4e79a7", alpha=0.75)
        axes[0].set_title("Amplitude Scale Distribution (All Gratings)")
        axes[0].set_xlabel("Amplitude scale")
        axes[0].set_ylabel("Count")
        axes[0].grid(True, alpha=0.25)
    else:
        axes[0].text(0.5, 0.5, "amplitude_scales not found", ha="center", va="center")
        axes[0].axis("off")

    if lw_scales is not None:
        axes[1].hist(lw_scales.reshape(-1), bins=50, color="#e15759", alpha=0.75)
        axes[1].set_title("Linewidth Scale Distribution (All Gratings)")
        axes[1].set_xlabel("Linewidth scale")
        axes[1].set_ylabel("Count")
        axes[1].grid(True, alpha=0.25)
    else:
        axes[1].text(0.5, 0.5, "linewidth_scales not found", ha="center", va="center")
        axes[1].axis("off")
    fig.savefig(out_dir / "fig5_random_grating_params.png", dpi=300)
    plt.close(fig)

    print(f"Saved figures to: {out_dir}")


if __name__ == "__main__":
    main()

