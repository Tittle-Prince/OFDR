"""
This figure is a simulated preview for layout/design only, not experimental data.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


@dataclass
class PreviewConfig:
    random_seed: int = 20260330
    num_points: int = 512
    figsize: tuple[float, float] = (10.0, 8.0)
    dpi: int = 300


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.linewidth": 0.9,
        }
    )


def ensure_dirs(figure_root: Path) -> tuple[Path, Path, Path]:
    scripts_dir = figure_root / "scripts"
    outputs_dir = figure_root / "outputs"
    data_dir = figure_root / "data_copy"
    scripts_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    return scripts_dir, outputs_dir, data_dir


def gaussian_peak(x: np.ndarray, center: float, sigma: float, amplitude: float = 1.0) -> np.ndarray:
    return amplitude * np.exp(-0.5 * ((x - center) / sigma) ** 2)


def smooth_1d(y: np.ndarray, sigma: float) -> np.ndarray:
    radius = max(1, int(round(4.0 * sigma)))
    grid = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (grid / sigma) ** 2)
    kernel /= np.sum(kernel)
    return np.convolve(y, kernel, mode="same")


def correlated_noise(rng: np.random.Generator, n: int, sigma: float, scale: float) -> np.ndarray:
    white = rng.normal(0.0, 1.0, size=n)
    return scale * smooth_1d(white, sigma=sigma)


def minmax_normalize(y: np.ndarray) -> np.ndarray:
    lo = float(np.percentile(y, 0.5))
    hi = float(np.percentile(y, 99.7))
    y_norm = np.clip((y - lo) / (hi - lo + 1e-12), 0.0, None)
    y_norm /= np.max(y_norm) + 1e-12
    return np.clip(y_norm, 0.0, 1.02)


def add_texture(
    x: np.ndarray,
    y: np.ndarray,
    rng: np.random.Generator,
    *,
    baseline_scale: float,
    ripple_scale: float,
    noise_scale: float,
) -> np.ndarray:
    grid = np.linspace(0.0, 1.0, x.size)
    baseline = baseline_scale * (0.6 + 0.9 * grid)
    baseline += 0.7 * baseline_scale * np.sin(2.0 * np.pi * (0.85 * grid + rng.uniform(-0.08, 0.08)))

    ripple = ripple_scale * np.sin(2.0 * np.pi * (4.2 * grid + rng.uniform(-0.15, 0.15)))
    ripple += 0.65 * ripple_scale * np.cos(2.0 * np.pi * (7.8 * grid + rng.uniform(-0.18, 0.18)))

    corr = correlated_noise(rng, x.size, sigma=4.5, scale=noise_scale)
    hf = 0.35 * noise_scale * rng.normal(0.0, 1.0, size=x.size)
    return np.clip(y + baseline + ripple + corr + hf, 0.0, None)


def generate_near_ideal(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    target = gaussian_peak(x, center=256.0, sigma=21.0, amplitude=1.0)
    weak_left = gaussian_peak(x, center=204.0, sigma=24.0, amplitude=0.10)
    weak_right = gaussian_peak(x, center=309.0, sigma=23.0, amplitude=0.08)
    pedestal = 0.018 * gaussian_peak(x, center=256.0, sigma=58.0, amplitude=1.0)
    y = target + weak_left + weak_right + pedestal
    y = add_texture(x, y, rng, baseline_scale=0.004, ripple_scale=0.0018, noise_scale=0.0020)
    y = 0.92 * y + 0.08 * smooth_1d(y, sigma=1.5)
    return minmax_normalize(y)


def generate_skewed(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    target = gaussian_peak(x, center=260.0, sigma=20.0, amplitude=1.0)
    shoulder_neighbor = gaussian_peak(x, center=236.0, sigma=11.5, amplitude=0.34)
    broad_left_tail = gaussian_peak(x, center=226.0, sigma=28.0, amplitude=0.18)
    weak_right = gaussian_peak(x, center=309.0, sigma=24.0, amplitude=0.07)
    local_skew = 0.030 * gaussian_peak(x, center=246.0, sigma=8.0, amplitude=1.0)
    y = target + shoulder_neighbor + broad_left_tail + weak_right + local_skew
    y = add_texture(x, y, rng, baseline_scale=0.007, ripple_scale=0.0028, noise_scale=0.0030)
    y = 0.80 * y + 0.20 * smooth_1d(y, sigma=2.0)
    return minmax_normalize(y)


def generate_broadened(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    broad_core = gaussian_peak(x, center=257.0, sigma=34.0, amplitude=0.84)
    flat_top = gaussian_peak(x, center=257.0, sigma=16.0, amplitude=0.22)
    pedestal = gaussian_peak(x, center=257.0, sigma=58.0, amplitude=0.11)
    weak_left = gaussian_peak(x, center=207.0, sigma=26.0, amplitude=0.07)
    weak_right = gaussian_peak(x, center=307.0, sigma=25.0, amplitude=0.07)
    y = broad_core + flat_top + pedestal + weak_left + weak_right
    y = add_texture(x, y, rng, baseline_scale=0.007, ripple_scale=0.0020, noise_scale=0.0026)
    y = 0.76 * y + 0.24 * smooth_1d(y, sigma=3.2)
    return minmax_normalize(y)


def generate_disturbed(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    target = gaussian_peak(x, center=256.0, sigma=22.0, amplitude=1.0)
    weak_left = gaussian_peak(x, center=209.0, sigma=25.0, amplitude=0.12)
    weak_right = gaussian_peak(x, center=303.0, sigma=23.0, amplitude=0.11)
    local_bump = 0.085 * gaussian_peak(x, center=281.0, sigma=7.5, amplitude=1.0)
    narrow_spike = 0.060 * gaussian_peak(x, center=233.0, sigma=2.2, amplitude=1.0)
    floor_lift = 0.035 * gaussian_peak(x, center=267.0, sigma=18.0, amplitude=1.0)
    y = target + weak_left + weak_right + local_bump + narrow_spike + floor_lift
    y = add_texture(x, y, rng, baseline_scale=0.012, ripple_scale=0.0065, noise_scale=0.0048)
    local_ripple = 0.020 * np.exp(-0.5 * ((x - 260.0) / 45.0) ** 2) * np.sin(2.0 * np.pi * (x - 214.0) / 17.0)
    y = np.clip(y + local_ripple, 0.0, None)
    return minmax_normalize(y)


def generate_preview_data(cfg: PreviewConfig) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    rng = np.random.default_rng(cfg.random_seed)
    x = np.arange(cfg.num_points, dtype=np.float64)
    spectra = {
        "near_ideal": generate_near_ideal(x, rng),
        "skewed": generate_skewed(x, rng),
        "broadened": generate_broadened(x, rng),
        "disturbed": generate_disturbed(x, rng),
    }
    return x, spectra


def plot_case(ax: plt.Axes, x: np.ndarray, y: np.ndarray, title: str, panel: str) -> None:
    ax.plot(x, y, color="#1f1f1f", linewidth=1.45)
    ax.set_xlim(float(x.min()), float(x.max()))
    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel("Local sample index")
    ax.set_ylabel("Normalized intensity")
    ax.set_title(f"({panel}) {title}", loc="left", pad=5)
    ax.grid(True, linestyle="--", alpha=0.18)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def build_figure(x: np.ndarray, spectra: dict[str, np.ndarray], cfg: PreviewConfig) -> plt.Figure:
    fig, axes = plt.subplots(2, 2, figsize=cfg.figsize, constrained_layout=True)

    plot_case(axes[0, 0], x, spectra["near_ideal"], "Near-ideal", "a")
    plot_case(axes[0, 1], x, spectra["skewed"], "Skewed / shouldered", "b")
    plot_case(axes[1, 0], x, spectra["broadened"], "Broadened", "c")
    plot_case(axes[1, 1], x, spectra["disturbed"], "Disturbed", "d")

    return fig


def main() -> None:
    configure_style()
    cfg = PreviewConfig()

    figure_root = Path(__file__).resolve().parents[1]
    _, outputs_dir, data_dir = ensure_dirs(figure_root)

    x, spectra = generate_preview_data(cfg)
    fig = build_figure(x, spectra, cfg)

    png_path = outputs_dir / "fig41_simulated_preview_v2.png"
    pdf_path = outputs_dir / "fig41_simulated_preview_v2.pdf"
    fig.savefig(png_path, dpi=cfg.dpi, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    np.savez(
        data_dir / "fig41_simulated_preview_v2_data.npz",
        local_index=x.astype(np.int32),
        near_ideal=spectra["near_ideal"].astype(np.float32),
        skewed=spectra["skewed"].astype(np.float32),
        broadened=spectra["broadened"].astype(np.float32),
        disturbed=spectra["disturbed"].astype(np.float32),
    )

    print(f"Saved PNG: {png_path}")
    print(f"Saved PDF: {pdf_path}")
    print("Data mode: simulated preview only (not experimental data)")


if __name__ == "__main__":
    main()
