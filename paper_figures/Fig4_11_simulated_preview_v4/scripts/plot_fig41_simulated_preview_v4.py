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


def ensure_dirs(root: Path) -> tuple[Path, Path, Path]:
    scripts_dir = root / "scripts"
    outputs_dir = root / "outputs"
    data_dir = root / "data_copy"
    scripts_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    return scripts_dir, outputs_dir, data_dir


def gaussian_peak(x: np.ndarray, center: float, sigma: float, amplitude: float = 1.0) -> np.ndarray:
    return amplitude * np.exp(-0.5 * ((x - center) / sigma) ** 2)


def asymmetric_gaussian(
    x: np.ndarray,
    center: float,
    sigma_left: float,
    sigma_right: float,
    amplitude: float = 1.0,
) -> np.ndarray:
    sigma = np.where(x <= center, sigma_left, sigma_right)
    return amplitude * np.exp(-0.5 * ((x - center) / sigma) ** 2)


def broad_ofdr_peak(
    x: np.ndarray,
    center: float,
    sigma_left: float,
    sigma_right: float,
    cap_sigma: float,
    skirt_sigma: float,
    amplitude: float,
) -> np.ndarray:
    core = asymmetric_gaussian(x, center, sigma_left, sigma_right, amplitude=0.78 * amplitude)
    cap = gaussian_peak(x, center + 0.6, cap_sigma, amplitude=0.17 * amplitude)
    skirt = gaussian_peak(x, center, skirt_sigma, amplitude=0.12 * amplitude)
    return core + cap + skirt


def smooth_1d(y: np.ndarray, sigma: float) -> np.ndarray:
    radius = max(1, int(round(4.0 * sigma)))
    grid = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (grid / sigma) ** 2)
    kernel /= np.sum(kernel)
    return np.convolve(y, kernel, mode="same")


def colored_noise(rng: np.random.Generator, n: int, sigma: float, scale: float) -> np.ndarray:
    white = rng.normal(0.0, 1.0, size=n)
    return scale * smooth_1d(white, sigma=sigma)


def baseline_texture(x: np.ndarray, rng: np.random.Generator, drift: float, ripple: float) -> np.ndarray:
    grid = np.linspace(0.0, 1.0, x.size)
    low = drift * (0.7 + 0.9 * grid)
    low += 0.85 * drift * np.sin(2.0 * np.pi * (0.65 * grid + rng.uniform(-0.05, 0.05)))
    mid = ripple * np.sin(2.0 * np.pi * (3.8 * grid + rng.uniform(-0.16, 0.16)))
    mid += 0.65 * ripple * np.cos(2.0 * np.pi * (6.6 * grid + rng.uniform(-0.14, 0.14)))
    return low + mid


def measurement_texture(
    x: np.ndarray,
    y: np.ndarray,
    rng: np.random.Generator,
    *,
    drift: float,
    ripple: float,
    corr_noise: float,
    hf_noise: float,
) -> np.ndarray:
    tex = baseline_texture(x, rng, drift=drift, ripple=ripple)
    tex += colored_noise(rng, x.size, sigma=7.0, scale=corr_noise)
    tex += 0.45 * colored_noise(rng, x.size, sigma=2.8, scale=corr_noise)
    tex += hf_noise * rng.normal(0.0, 1.0, size=x.size)
    return np.clip(y + tex, 0.0, None)


def normalize_trace(y: np.ndarray) -> np.ndarray:
    lo = float(np.percentile(y, 0.8))
    hi = float(np.percentile(y, 99.5))
    out = np.clip((y - lo) / (hi - lo + 1e-12), 0.0, None)
    out /= np.max(out) + 1e-12
    return np.clip(out, 0.0, 1.02)


def generate_near_ideal(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    main = broad_ofdr_peak(x, 256.0, 24.0, 25.0, cap_sigma=13.0, skirt_sigma=62.0, amplitude=1.0)
    weak_left = gaussian_peak(x, 208.0, 24.0, 0.055)
    weak_right = gaussian_peak(x, 305.0, 25.0, 0.050)
    base_lift = gaussian_peak(x, 255.0, 95.0, 0.020)
    y = main + weak_left + weak_right + base_lift
    y = measurement_texture(x, y, rng, drift=0.0055, ripple=0.0018, corr_noise=0.0020, hf_noise=0.0012)
    y = 0.90 * y + 0.10 * smooth_1d(y, sigma=1.8)
    return normalize_trace(y)


def generate_skewed(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    main = broad_ofdr_peak(x, 259.0, 19.0, 28.5, cap_sigma=14.0, skirt_sigma=70.0, amplitude=0.97)
    shoulder = gaussian_peak(x, 235.0, 14.0, 0.30)
    drag = gaussian_peak(x, 223.0, 35.0, 0.17)
    faint_right = gaussian_peak(x, 307.0, 26.0, 0.050)
    y = main + shoulder + drag + faint_right + gaussian_peak(x, 252.0, 22.0, 0.035)
    y = measurement_texture(x, y, rng, drift=0.0065, ripple=0.0027, corr_noise=0.0028, hf_noise=0.0018)
    y = 0.82 * y + 0.18 * smooth_1d(y, sigma=2.6)
    return normalize_trace(y)


def generate_broadened(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    main = broad_ofdr_peak(x, 257.0, 38.0, 41.0, cap_sigma=17.0, skirt_sigma=88.0, amplitude=0.96)
    flat_table = gaussian_peak(x, 257.0, 23.0, 0.12)
    pedestal = gaussian_peak(x, 257.0, 108.0, 0.055)
    weak_left = gaussian_peak(x, 210.0, 27.0, 0.040)
    weak_right = gaussian_peak(x, 303.0, 28.0, 0.045)
    y = main + flat_table + pedestal + weak_left + weak_right
    y = measurement_texture(x, y, rng, drift=0.0065, ripple=0.0022, corr_noise=0.0025, hf_noise=0.0015)
    y = 0.72 * y + 0.28 * smooth_1d(y, sigma=4.4)
    return normalize_trace(y)


def generate_disturbed(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    main = broad_ofdr_peak(x, 255.5, 24.0, 26.0, cap_sigma=13.5, skirt_sigma=68.0, amplitude=0.98)
    weak_left = gaussian_peak(x, 208.0, 24.0, 0.075)
    weak_right = gaussian_peak(x, 303.5, 24.0, 0.070)
    local_bump = gaussian_peak(x, 280.0, 8.5, 0.070)
    narrow_spike = gaussian_peak(x, 232.0, 2.3, 0.050)
    floor_anomaly = gaussian_peak(x, 268.0, 18.0, 0.038)
    y = main + weak_left + weak_right + local_bump + narrow_spike + floor_anomaly
    y = measurement_texture(x, y, rng, drift=0.0105, ripple=0.0058, corr_noise=0.0036, hf_noise=0.0026)
    local_window = np.exp(-0.5 * ((x - 260.0) / 50.0) ** 2)
    y += 0.018 * local_window * np.sin(2.0 * np.pi * (x - 208.0) / 18.0)
    y = np.clip(y, 0.0, None)
    return normalize_trace(y)


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
    ax.grid(True, linestyle="--", alpha=0.15)
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

    root = Path(__file__).resolve().parents[1]
    _, outputs_dir, data_dir = ensure_dirs(root)

    x, spectra = generate_preview_data(cfg)
    fig = build_figure(x, spectra, cfg)

    png_path = outputs_dir / "fig41_simulated_preview_v4.png"
    pdf_path = outputs_dir / "fig41_simulated_preview_v4.pdf"
    fig.savefig(png_path, dpi=cfg.dpi, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    np.savez(
        data_dir / "fig41_simulated_preview_v4_data.npz",
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
