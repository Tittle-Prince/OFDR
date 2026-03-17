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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot current phase4a sample spectra vs ideal no-neighbor-shift spectra")
    parser.add_argument("--config", type=str, default="config/phase4a_shift004_linewidth_l3.yaml")
    parser.add_argument("--sample-index", type=int, default=-1, help="-1 means first test sample from dataset")
    return parser.parse_args()


def _pick_sample_index(dataset: np.lib.npyio.NpzFile, sample_index: int) -> int:
    if sample_index >= 0:
        return int(sample_index)
    idx_test = dataset["idx_test"].astype(np.int64)
    return int(idx_test[0])


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    dataset_path = resolve_project_path(cfg["phase4a"]["dataset_path"])
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    d = np.load(dataset_path)
    sample_idx = _pick_sample_index(d, args.sample_index)

    wavelengths = d["wavelengths"].astype(np.float64)
    y = d["Y_dlambda_target"].astype(np.float64)
    amp_scales = d["amplitude_scales"].astype(np.float64)
    lw_scales = d["linewidth_scales"].astype(np.float64)
    neighbor_deltas = d["neighbor_delta_lambdas_nm"].astype(np.float64)
    target_index = int(d["target_index"][0])

    dlam = float(y[sample_idx])
    amps = amp_scales[sample_idx]
    sigs = lw_scales[sample_idx]
    neigh = neighbor_deltas[sample_idx]

    per_current, total_current, centers_current = simulate_identical_array_spectra(
        wavelengths,
        cfg,
        dlam,
        amplitude_scales=amps,
        linewidth_scales=sigs,
        neighbor_delta_lambdas_nm=neigh,
    )

    ideal_neigh = np.zeros_like(neigh, dtype=np.float64)
    per_ideal, total_ideal, centers_ideal = simulate_identical_array_spectra(
        wavelengths,
        cfg,
        dlam,
        amplitude_scales=amps,
        linewidth_scales=sigs,
        neighbor_delta_lambdas_nm=ideal_neigh,
    )

    out_dir = resolve_project_path(cfg["phase4a"]["results_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "neighbor_shift_vs_ideal_5gratings.png"

    colors = ["#4e79a7", "#f28e2b", "#59a14f", "#e15759", "#76b7b2"]
    fig, axes = plt.subplots(2, 1, figsize=(12.0, 8.0), constrained_layout=True, sharex=True)

    for i in range(per_current.shape[0]):
        axes[0].plot(wavelengths, per_current[i], color=colors[i], linewidth=1.5, label=f"FBG{i+1}")
    axes[0].plot(wavelengths, total_current, color="black", linewidth=2.0, alpha=0.8, linestyle="--", label="Total")
    axes[0].set_title("Current simulated spectra with neighbor shift")
    axes[0].set_ylabel("Reflectivity (a.u.)")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(ncol=3, fontsize=9)

    for i in range(per_ideal.shape[0]):
        axes[1].plot(wavelengths, per_ideal[i], color=colors[i], linewidth=1.5, label=f"FBG{i+1}")
    axes[1].plot(wavelengths, total_ideal, color="black", linewidth=2.0, alpha=0.8, linestyle="--", label="Total")
    axes[1].set_title("Ideal simulated spectra without neighbor shift")
    axes[1].set_xlabel("Wavelength (nm)")
    axes[1].set_ylabel("Reflectivity (a.u.)")
    axes[1].grid(True, alpha=0.25)

    neigh_text = ", ".join(f"{v:+.4f}" for v in neigh)
    ideal_text = ", ".join(f"{v:+.4f}" for v in ideal_neigh)
    fig.suptitle(
        f"Sample {sample_idx} | target FBG={target_index + 1} | target dlambda={dlam:+.4f} nm\n"
        f"Neighbor deltas current=[{neigh_text}] | ideal=[{ideal_text}]",
        fontsize=11,
    )

    fig.savefig(out_path, dpi=300)
    plt.close(fig)

    cur_centers = ", ".join(f"{c:.4f}" for c in centers_current)
    ideal_centers_str = ", ".join(f"{c:.4f}" for c in centers_ideal)
    print(f"Saved figure: {out_path}")
    print(f"Current centers (nm): [{cur_centers}]")
    print(f"Ideal centers (nm): [{ideal_centers_str}]")


if __name__ == "__main__":
    main()
