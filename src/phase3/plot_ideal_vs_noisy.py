from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml


def gaussian_spectrum(wavelengths: np.ndarray, center_nm: float, sigma_nm: float) -> np.ndarray:
    return np.exp(-0.5 * ((wavelengths - center_nm) / sigma_nm) ** 2)


def normalize_minmax(x: np.ndarray) -> np.ndarray:
    x_min = float(np.min(x))
    x_max = float(np.max(x))
    return (x - x_min) / (x_max - x_min + 1e-12)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot ideal spectrum vs noisy spectrum from Dataset_B")
    parser.add_argument("--config", type=str, default="config/phase3.yaml")
    parser.add_argument("--sample-index", type=int, default=-1, help="Global sample index in Dataset_B; -1 means first test sample")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[2]

    cfg_path = project_root / args.config
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    data_path = project_root / cfg["phase3"]["dataset_b_path"]
    out_path = project_root / "results" / "phase3b" / "ideal_vs_noisy_spectrum.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    data = np.load(data_path)
    x = data["X"].astype(np.float64)
    y_dlambda = data["Y_dlambda"].astype(np.float64)
    wavelengths = data["wavelengths"].astype(np.float64)
    idx_test = data["idx_test"].astype(np.int64)

    if args.sample_index >= 0:
        idx = int(args.sample_index)
    else:
        idx = int(idx_test[0])

    lambda_b0 = float(cfg["dataset_b"]["lambda_b0_nm"])
    sigma = float(cfg["dataset_b"]["base_linewidth_nm"])
    normalize_mode = str(cfg["dataset_b"]["normalize"])

    dlam = float(y_dlambda[idx])
    center = lambda_b0 + dlam
    ideal = gaussian_spectrum(wavelengths, center_nm=center, sigma_nm=sigma)
    if normalize_mode == "minmax_per_sample":
        ideal = normalize_minmax(ideal)

    noisy = x[idx]

    plt.figure(figsize=(9.2, 4.6), constrained_layout=True)
    plt.plot(wavelengths, ideal, linewidth=2.0, color="#1f77b4", label="Ideal spectrum")
    plt.plot(wavelengths, noisy, linewidth=1.2, color="#d62728", alpha=0.90, label="Noisy/distorted spectrum (Dataset_B)")
    plt.title("Ideal vs Noisy Spectrum (Dataset_B)")
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Normalized Reflectivity")
    plt.grid(True, alpha=0.25)
    plt.legend(loc="best")
    plt.text(
        0.02,
        0.96,
        f"Sample index={idx}\nDelta-lambda={dlam:.4f} nm",
        transform=plt.gca().transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.9},
    )
    plt.savefig(out_path, dpi=300)
    plt.close()

    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()

