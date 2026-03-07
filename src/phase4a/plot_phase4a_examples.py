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
from phase4a.local_window import extract_local_distorted_spectrum


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot Phase4-A array and local distorted spectrum example")
    parser.add_argument("--config", type=str, default="config/phase4a.yaml")
    parser.add_argument("--sample-index", type=int, default=-1, help="Use one sample index from saved dataset; -1 means first test sample")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    dataset_path = resolve_project_path(cfg["phase4a"]["dataset_path"])
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {dataset_path}\n"
            "Run `python src\\phase4a\\generate_dataset_phase4a.py --config config\\phase4a.yaml` first."
        )

    data = np.load(dataset_path)
    wavelengths = data["wavelengths"].astype(np.float64)
    y = data["Y_dlambda_target"].astype(np.float64)
    x_local = data["X_local"].astype(np.float64)
    x_total = data["X_total"].astype(np.float64)
    idx_test = data["idx_test"].astype(np.int64)
    target_index = int(data["target_index"][0])

    if args.sample_index >= 0:
        idx = int(args.sample_index)
    else:
        idx = int(idx_test[0])
    dlam = float(y[idx])

    per_grating, total_rebuilt, centers = simulate_identical_array_spectra(wavelengths, cfg, dlam)
    local_clean, weights = extract_local_distorted_spectrum(per_grating, target_index, cfg["local_window"])
    local_noisy = x_local[idx]
    total_saved = x_total[idx]

    out_dir = resolve_project_path(cfg["phase4a"]["results_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "phase4a_example_spectra.png"

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.6), constrained_layout=True)

    for i in range(per_grating.shape[0]):
        axes[0].plot(wavelengths, per_grating[i], linewidth=1.2, label=f"FBG{i+1} (w={weights[i]:.2f})")
    axes[0].plot(wavelengths, total_rebuilt, "k-", linewidth=2.0, alpha=0.8, label="Array total")
    axes[0].set_title("Identical UWBFG Array Spectra")
    axes[0].set_xlabel("Wavelength (nm)")
    axes[0].set_ylabel("Reflectivity (a.u.)")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(fontsize=8)

    axes[1].plot(wavelengths, local_clean, color="#1f77b4", linewidth=1.8, label="Local window clean")
    axes[1].plot(wavelengths, local_noisy, color="#d62728", linewidth=1.2, alpha=0.9, label="Local distorted/noisy")
    axes[1].plot(wavelengths, total_saved, color="#2ca02c", linewidth=1.2, alpha=0.65, label="Array total (saved)")
    axes[1].set_title("Target Local Window Extraction")
    axes[1].set_xlabel("Wavelength (nm)")
    axes[1].set_ylabel("Intensity (a.u.)")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(fontsize=8)
    axes[1].text(
        0.03,
        0.97,
        f"Sample={idx}\nTarget=FBG{target_index+1}\nDelta-lambda3={dlam:.4f} nm",
        transform=axes[1].transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.9},
    )

    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Saved figure: {out_path}")

    centers_str = ", ".join(f"{c:.4f}" for c in centers)
    print(f"Centers (nm): [{centers_str}]")


if __name__ == "__main__":
    main()

