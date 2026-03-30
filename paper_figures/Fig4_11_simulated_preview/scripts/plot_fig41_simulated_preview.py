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
            "axes.linewidth": 1.0,
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


def gaussian_peak(x: np.ndarray, center: float, sigma: float, amplitude: float) -> np.ndarray:
    return amplitude * np.exp(-0.5 * ((x - center) / sigma) ** 2)


def smooth_1d(x: np.ndarray, sigma: float) -> np.ndarray:
    radius = max(1, int(round(4.0 * sigma)))
    grid = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (grid / sigma) ** 2)
    kernel /= np.sum(kernel)
    return np.convolve(x, kernel, mode="same")


def correlated_noise(rng: np.random.Generator, n: int, sigma: float, scale: float) -> np.ndarray:
    noise = rng.normal(0.0, 1.0, size=n)
    return scale * smooth_1d(noise, sigma=sigma)


def minmax_normalize(y: np.ndarray) -> np.ndarray:
    lo = float(np.percentile(y, 0.5))
    hi = float(np.percentile(y, 99.6))
    out = np.clip((y - lo) / (hi - lo + 1e-12), 0.0, None)
    out /= np.max(out) + 1e-12
    return np.clip(out, 0.0, 1.02)


def build_base_components(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    target = gaussian_peak(x, center=256.0, sigma=22.0, amplitude=1.0)
    left = gaussian_peak(x, center=210.0, sigma=26.0, amplitude=0.28)
    right = gaussian_peak(x, center=302.0, sigma=25.0, amplitude=0.26)
    return left, target, right


def add_measurement_texture(x: np.ndarray, y: np.ndarray, rng: np.random.Generator, *,
                            base_scale: float, ripple_scale: float, noise_scale: float) -> np.ndarray:
    grid = np.linspace(0.0, 1.0, x.size)
    baseline = base_scale * (0.7 + 0.6 * grid)
    baseline += 0.5 * base_scale * np.sin(2.0 * np.pi * (1.05 * grid + rng.uniform(-0.12, 0.12)))
    ripple = ripple_scale * np.sin(2.0 * np.pi * (5.0 * grid + rng.uniform(-0.25, 0.25)))
    ripple += 0.6 * ripple_scale * np.cos(2.0 * np.pi * (9.0 * grid + rng.uniform(-0.20, 0.20)))
    noise = correlated_noise(rng, x.size, sigma=5.0, scale=noise_scale)
    hf = 0.25 * noise_scale * rng.normal(0.0, 1.0, size=x.size)
    y = y + baseline + ripple + noise + hf
    return np.clip(y, 0.0, None)


def generate_near_ideal(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    left, target, right = build_base_components(x)
    y = target + 0.85 * left + 0.80 * right
    y += 0.025 * gaussian_peak(x, center=255.0, sigma=52.0, amplitude=1.0)
    y = add_measurement_texture(x, y, rng, base_scale=0.010, ripple_scale=0.004, noise_scale=0.006)
    return minmax_normalize(y)


def generate_skewed(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    left = gaussian_peak(x, center=224.0, sigma=28.0, amplitude=0.46)
    target = gaussian_peak(x, center=257.0, sigma=21.0, amplitude=1.0)
    right = gaussian_peak(x, center=304.0, sigma=25.0, amplitude=0.20)
    shoulder = 0.085 * gaussian_peak(x, center=239.0, sigma=13.0, amplitude=1.0)
    y = left + target + right + shoulder
    y = 0.88 * y + 0.12 * smooth_1d(y, sigma=8.0)
    y = add_measurement_texture(x, y, rng, base_scale=0.014, ripple_scale=0.006, noise_scale=0.008)
    return minmax_normalize(y)


def generate_broadened(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    left = gaussian_peak(x, center=212.0, sigma=27.0, amplitude=0.26)
    target_core = gaussian_peak(x, center=256.0, sigma=27.0, amplitude=0.82)
    target_top = gaussian_peak(x, center=257.5, sigma=17.0, amplitude=0.26)
    right = gaussian_peak(x, center=304.0, sigma=28.0, amplitude=0.24)
    broad_pedestal = 0.090 * gaussian_peak(x, center=257.0, sigma=50.0, amplitude=1.0)
    y = left + target_core + target_top + right + broad_pedestal
    y = add_measurement_texture(x, y, rng, base_scale=0.018, ripple_scale=0.005, noise_scale=0.008)
    return minmax_normalize(y)


def generate_disturbed(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    left, target, right = build_base_components(x)
    y = target + 0.95 * left + 0.88 * right
    pseudo = 0.060 * gaussian_peak(x, center=282.0, sigma=8.0, amplitude=1.0)
    spike = 0.050 * gaussian_peak(x, center=238.0, sigma=2.6, amplitude=1.0)
    local_bump = 0.030 * gaussian_peak(x, center=268.0, sigma=16.0, amplitude=1.0)
    y = y + pseudo + spike + local_bump
    y = add_measurement_texture(x, y, rng, base_scale=0.020, ripple_scale=0.010, noise_scale=0.010)
    local_ripple = 0.018 * np.exp(-0.5 * ((x - 260.0) / 45.0) ** 2) * np.sin(2.0 * np.pi * (x - 220.0) / 18.0)
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
    ax.plot(x, y, color="#1f1f1f", linewidth=1.5)
    ax.set_xlim(float(x.min()), float(x.max()))
    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel("Local sample index")
    ax.set_ylabel("Normalized intensity")
    ax.set_title(f"({panel}) {title}", loc="left", pad=6)
    ax.grid(True, linestyle="--", alpha=0.22)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def build_figure(x: np.ndarray, spectra: dict[str, np.ndarray], cfg: PreviewConfig) -> plt.Figure:
    fig, axes = plt.subplots(2, 2, figsize=cfg.figsize, constrained_layout=True)
    fig.suptitle("Simulated preview of typical local spectral shapes", fontsize=13, y=0.995)

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

    png_path = outputs_dir / "fig41_simulated_preview.png"
    pdf_path = outputs_dir / "fig41_simulated_preview.pdf"
    fig.savefig(png_path, dpi=cfg.dpi, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    np.savez(
        data_dir / "fig41_simulated_preview_data.npz",
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
