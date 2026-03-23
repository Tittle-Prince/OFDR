from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def minmax_normalize(arr: np.ndarray) -> np.ndarray:
    arr_min = float(arr.min())
    arr_max = float(arr.max())
    return (arr - arr_min) / (arr_max - arr_min + 1e-12)


def gaussian_2d(
    xx: np.ndarray,
    yy: np.ndarray,
    x0: float,
    y0: float,
    sigma_x: float,
    sigma_y: float,
    amplitude: float,
) -> np.ndarray:
    return amplitude * np.exp(
        -0.5 * (((xx - x0) / sigma_x) ** 2 + ((yy - y0) / sigma_y) ** 2)
    )


def build_maps(n_samples: int = 100, n_wavelengths: int = 2000, seed: int = 20260322) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)

    x = np.linspace(0.0, 1.0, n_wavelengths, dtype=np.float64)
    y = np.linspace(0.0, 1.0, n_samples, dtype=np.float64)
    xx, yy = np.meshgrid(x, y)

    stripes_main = 0.42 * (np.sin(2.0 * np.pi * (9.0 * xx + 0.9 * yy)) + 1.0)
    stripes_aux = 0.18 * (np.sin(2.0 * np.pi * (17.0 * xx - 0.55 * yy) + 0.8) + 1.0)
    diagonal_mod = 0.10 * (np.cos(2.0 * np.pi * (3.5 * yy - 1.2 * xx)) + 1.0)

    hotspot_main = gaussian_2d(xx, yy, 0.62, 0.46, 0.065, 0.10, 0.95)
    hotspot_aux = gaussian_2d(xx, yy, 0.78, 0.70, 0.045, 0.07, 0.45)
    local_ridge = gaussian_2d(xx, yy, 0.64, 0.44, 0.12, 0.045, 0.55)

    difficulty_raw = stripes_main + stripes_aux + diagonal_mod + hotspot_main + hotspot_aux + local_ridge
    difficulty_noise = rng.normal(0.0, 0.015, size=difficulty_raw.shape)
    difficulty_map = minmax_normalize(difficulty_raw + difficulty_noise)

    baseline_background = 0.055 + 0.06 * difficulty_map
    structured_error = 0.20 * difficulty_map**1.35

    hotspot_gate = gaussian_2d(xx, yy, 0.62, 0.46, 0.07, 0.11, 1.0)
    fine_hotspots = (
        gaussian_2d(xx, yy, 0.58, 0.41, 0.018, 0.028, 0.34)
        + gaussian_2d(xx, yy, 0.63, 0.49, 0.020, 0.024, 0.28)
        + gaussian_2d(xx, yy, 0.69, 0.46, 0.022, 0.030, 0.25)
    )
    ripple = 0.02 * np.maximum(0.0, np.sin(2.0 * np.pi * (30.0 * xx + 1.1 * yy)))
    baseline_noise = rng.normal(0.0, 0.008, size=difficulty_map.shape)

    baseline_error = baseline_background + structured_error + hotspot_gate * (0.18 + ripple) + fine_hotspots + baseline_noise
    baseline_error = np.clip(baseline_error, 0.0, None)

    suppression_field = hotspot_gate * (0.50 + 0.35 * difficulty_map)
    high_error_part = np.maximum(0.0, baseline_error - np.quantile(baseline_error, 0.86))
    tailaware_error = baseline_error - suppression_field * high_error_part / (high_error_part.max() + 1e-12)
    tailaware_error += rng.normal(0.0, 0.004, size=tailaware_error.shape)
    tailaware_error = np.clip(tailaware_error, 0.0, None)

    x0 = int(0.54 * n_wavelengths)
    y0 = int(0.30 * n_samples)
    width = int(0.18 * n_wavelengths)
    height = int(0.26 * n_samples)

    return {
        "difficulty_map": difficulty_map.astype(np.float32),
        "baseline_error": baseline_error.astype(np.float32),
        "tailaware_error": tailaware_error.astype(np.float32),
        "rect_x0": np.array([x0], dtype=np.int32),
        "rect_y0": np.array([y0], dtype=np.int32),
        "rect_width": np.array([width], dtype=np.int32),
        "rect_height": np.array([height], dtype=np.int32),
    }


def add_common_axis_style(ax: plt.Axes, title: str) -> None:
    ax.set_title(title, pad=8)
    ax.set_ylabel("Sample index")
    ax.tick_params(length=3)


def add_region_box(ax: plt.Axes, x0: int, y0: int, width: int, height: int) -> None:
    rect = patches.Rectangle(
        (x0, y0),
        width,
        height,
        linewidth=1.4,
        edgecolor="white",
        facecolor="none",
        linestyle="--",
    )
    ax.add_patch(rect)


def build_figure(data: dict[str, np.ndarray]) -> plt.Figure:
    difficulty_map = data["difficulty_map"]
    baseline_error = data["baseline_error"]
    tailaware_error = data["tailaware_error"]

    x0 = int(data["rect_x0"][0])
    y0 = int(data["rect_y0"][0])
    width = int(data["rect_width"][0])
    height = int(data["rect_height"][0])

    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True, constrained_layout=False)

    common_imshow = {
        "origin": "lower",
        "aspect": "auto",
        "cmap": "inferno",
        "interpolation": "nearest",
    }

    im0 = axes[0].imshow(difficulty_map, **common_imshow)
    add_common_axis_style(axes[0], "(a) Difficulty map (pseudo-peak density)")
    cbar0 = fig.colorbar(im0, ax=axes[0], fraction=0.024, pad=0.02)
    cbar0.set_label("Error magnitude")

    vmin = float(min(baseline_error.min(), tailaware_error.min()))
    vmax = float(max(baseline_error.max(), tailaware_error.max()))

    im1 = axes[1].imshow(baseline_error, vmin=vmin, vmax=vmax, **common_imshow)
    add_common_axis_style(axes[1], "(b) Error map of baseline CNN")
    add_region_box(axes[1], x0, y0, width, height)
    axes[1].annotate(
        "Error concentration\n(tail region)",
        xy=(x0 + width * 0.70, y0 + height * 0.72),
        xytext=(x0 + width * 0.10, y0 + height * 1.18),
        color="white",
        fontsize=11,
        arrowprops={"arrowstyle": "->", "color": "white", "lw": 1.2},
        ha="left",
        va="bottom",
    )
    cbar1 = fig.colorbar(im1, ax=axes[1], fraction=0.024, pad=0.02)
    cbar1.set_label("Error magnitude")

    im2 = axes[2].imshow(tailaware_error, vmin=vmin, vmax=vmax, **common_imshow)
    add_common_axis_style(axes[2], "(c) Error map with Tail-aware loss")
    add_region_box(axes[2], x0, y0, width, height)
    axes[2].annotate(
        "Suppressed tail errors",
        xy=(x0 + width * 0.66, y0 + height * 0.55),
        xytext=(x0 + width * 0.06, y0 + height * 1.16),
        color="white",
        fontsize=11,
        arrowprops={"arrowstyle": "->", "color": "white", "lw": 1.2},
        ha="left",
        va="bottom",
    )
    cbar2 = fig.colorbar(im2, ax=axes[2], fraction=0.024, pad=0.02)
    cbar2.set_label("Error magnitude")

    axes[2].set_xlabel("Wavelength index")

    for ax in axes:
        ax.set_xlim(0, difficulty_map.shape[1] - 1)
        ax.set_ylim(0, difficulty_map.shape[0] - 1)

    fig.tight_layout()
    return fig


def main() -> None:
    configure_style()

    figure_root = Path(__file__).resolve().parents[1]
    outputs_dir = figure_root / "outputs"
    data_dir = figure_root / "data_copy"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    data = build_maps()
    fig = build_figure(data)

    png_path = outputs_dir / "Fig4_5_tail_mechanism.png"
    pdf_path = outputs_dir / "Fig4_5_tail_mechanism.pdf"
    data_path = data_dir / "Fig4_5_tail_mechanism_data.npz"

    fig.savefig(png_path, dpi=300)
    fig.savefig(pdf_path)
    plt.close(fig)

    np.savez(data_path, **data)

    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")
    print(f"Saved: {data_path}")


if __name__ == "__main__":
    main()
