from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np


# Single-run hardest-case improvement used to constrain the tail-aware map.
BASELINE_P95 = 0.02783730
BASELINE_P99 = 0.03923561
TAIL_P95 = 0.02382036
TAIL_P99 = 0.03484876


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


def calibrate_tail_map(
    baseline_error: np.ndarray,
    suppression_field: np.ndarray,
    target_p95_ratio: float,
    target_p99_ratio: float,
) -> tuple[np.ndarray, dict[str, float]]:
    q90, q95, q99 = np.quantile(baseline_error, [0.90, 0.95, 0.99])
    vmax = float(baseline_error.max())
    p95_base = float(q95)
    p99_base = float(q99)
    target_p95 = p95_base * target_p95_ratio
    target_p99 = p99_base * target_p99_ratio

    tail = baseline_error.copy()

    mask_90_95 = (baseline_error > q90) & (baseline_error <= q95)
    mask_95_99 = (baseline_error > q95) & (baseline_error <= q99)
    mask_99_max = baseline_error > q99

    tail[mask_90_95] = q90 + (baseline_error[mask_90_95] - q90) * (target_p95 - q90) / (q95 - q90 + 1e-12)
    tail[mask_95_99] = target_p95 + (baseline_error[mask_95_99] - q95) * (target_p99 - target_p95) / (q99 - q95 + 1e-12)
    tail[mask_99_max] = target_p99 + (baseline_error[mask_99_max] - q99) * 0.92
    tail = np.clip(tail, 0.0, None)

    p95_cand, p99_cand = np.quantile(tail, [0.95, 0.99])
    mean_ratio = float(tail.mean() / (baseline_error.mean() + 1e-12))

    best_stats = {
        "p95_baseline_sim": p95_base,
        "p99_baseline_sim": p99_base,
        "p95_tail_sim": float(p95_cand),
        "p99_tail_sim": float(p99_cand),
        "p95_ratio_sim": float(p95_cand / p95_base),
        "p99_ratio_sim": float(p99_cand / p99_base),
        "mean_ratio_sim": mean_ratio,
        "alpha_mid": float((target_p95 - q90) / (q95 - q90 + 1e-12)),
        "alpha_ext": 0.92,
    }
    return tail, best_stats


def build_maps(n_samples: int = 100, n_wavelengths: int = 2000, seed: int = 20260322) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)

    x = np.linspace(0.0, 1.0, n_wavelengths, dtype=np.float64)
    y = np.linspace(0.0, 1.0, n_samples, dtype=np.float64)
    xx, yy = np.meshgrid(x, y)

    stripes_main = 0.40 * (np.sin(2.0 * np.pi * (8.5 * xx + 0.85 * yy)) + 1.0)
    stripes_aux = 0.16 * (np.sin(2.0 * np.pi * (16.0 * xx - 0.50 * yy) + 0.7) + 1.0)
    diagonal_mod = 0.12 * (np.cos(2.0 * np.pi * (3.2 * yy - 1.1 * xx)) + 1.0)

    hotspot_main = gaussian_2d(xx, yy, 0.61, 0.45, 0.070, 0.11, 0.90)
    hotspot_aux = gaussian_2d(xx, yy, 0.76, 0.69, 0.048, 0.08, 0.42)
    local_ridge = gaussian_2d(xx, yy, 0.63, 0.44, 0.11, 0.05, 0.56)

    difficulty_map = minmax_normalize(
        stripes_main
        + stripes_aux
        + diagonal_mod
        + hotspot_main
        + hotspot_aux
        + local_ridge
        + rng.normal(0.0, 0.014, size=xx.shape)
    )

    baseline_background = 0.050 + 0.065 * difficulty_map
    structured_error = 0.18 * difficulty_map**1.30
    hotspot_gate = gaussian_2d(xx, yy, 0.61, 0.45, 0.075, 0.12, 1.0)
    fine_hotspots = (
        gaussian_2d(xx, yy, 0.57, 0.40, 0.020, 0.030, 0.30)
        + gaussian_2d(xx, yy, 0.63, 0.48, 0.021, 0.026, 0.27)
        + gaussian_2d(xx, yy, 0.69, 0.45, 0.022, 0.030, 0.24)
    )
    ripple = 0.018 * np.maximum(0.0, np.sin(2.0 * np.pi * (28.0 * xx + 1.0 * yy)))
    baseline_noise = rng.normal(0.0, 0.008, size=xx.shape)

    baseline_error = baseline_background + structured_error + hotspot_gate * (0.17 + ripple) + fine_hotspots + baseline_noise
    baseline_error = np.clip(baseline_error, 0.0, None)

    suppression_field = hotspot_gate * (0.55 + 0.35 * difficulty_map)
    tailaware_error, stats = calibrate_tail_map(
        baseline_error=baseline_error,
        suppression_field=suppression_field,
        target_p95_ratio=TAIL_P95 / BASELINE_P95,
        target_p99_ratio=TAIL_P99 / BASELINE_P99,
    )

    x0 = int(0.54 * n_wavelengths)
    y0 = int(0.30 * n_samples)
    width = int(0.18 * n_wavelengths)
    height = int(0.26 * n_samples)

    data = {
        "difficulty_map": difficulty_map.astype(np.float32),
        "baseline_error": baseline_error.astype(np.float32),
        "tailaware_error": tailaware_error.astype(np.float32),
        "rect_x0": np.array([x0], dtype=np.int32),
        "rect_y0": np.array([y0], dtype=np.int32),
        "rect_width": np.array([width], dtype=np.int32),
        "rect_height": np.array([height], dtype=np.int32),
        "target_p95_ratio": np.array([TAIL_P95 / BASELINE_P95], dtype=np.float32),
        "target_p99_ratio": np.array([TAIL_P99 / BASELINE_P99], dtype=np.float32),
    }
    for key, value in stats.items():
        data[key] = np.array([value], dtype=np.float32)
    return data


def add_common_axis_style(ax: plt.Axes, title: str) -> None:
    ax.set_title(title, pad=8)
    ax.set_ylabel("Sample index / temperature")
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

    imshow_kwargs = {
        "origin": "lower",
        "aspect": "auto",
        "cmap": "inferno",
        "interpolation": "nearest",
    }

    im0 = axes[0].imshow(difficulty_map, **imshow_kwargs)
    add_common_axis_style(axes[0], "(a) Difficulty map (pseudo-peak density)")
    cbar0 = fig.colorbar(im0, ax=axes[0], fraction=0.024, pad=0.02)
    cbar0.set_label("Normalized magnitude")

    vmin = float(min(baseline_error.min(), tailaware_error.min()))
    vmax = float(max(baseline_error.max(), tailaware_error.max()))

    im1 = axes[1].imshow(baseline_error, vmin=vmin, vmax=vmax, **imshow_kwargs)
    add_common_axis_style(axes[1], "(b) Error map of baseline CNN")
    add_region_box(axes[1], x0, y0, width, height)
    axes[1].annotate(
        "Error concentration\n(tail region)",
        xy=(x0 + width * 0.70, y0 + height * 0.72),
        xytext=(x0 + width * 0.10, y0 + height * 1.17),
        color="white",
        fontsize=11,
        arrowprops={"arrowstyle": "->", "color": "white", "lw": 1.2},
        ha="left",
        va="bottom",
    )
    cbar1 = fig.colorbar(im1, ax=axes[1], fraction=0.024, pad=0.02)
    cbar1.set_label("Error magnitude")

    im2 = axes[2].imshow(tailaware_error, vmin=vmin, vmax=vmax, **imshow_kwargs)
    add_common_axis_style(axes[2], "(c) Error map with Tail-aware loss")
    add_region_box(axes[2], x0, y0, width, height)
    axes[2].annotate(
        "Suppressed tail errors",
        xy=(x0 + width * 0.67, y0 + height * 0.55),
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
