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


def smooth_1d(y: np.ndarray, sigma: float) -> np.ndarray:
    radius = max(1, int(round(4.0 * sigma)))
    grid = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (grid / sigma) ** 2)
    kernel /= np.sum(kernel)
    return np.convolve(y, kernel, mode="same")


def colored_noise(rng: np.random.Generator, n: int, sigma: float, scale: float) -> np.ndarray:
    white = rng.normal(0.0, 1.0, size=n)
    return scale * smooth_1d(white, sigma=sigma)


def normalize_trace(y: np.ndarray) -> np.ndarray:
    y = np.clip(y, 0.0, None)
    lo = float(np.percentile(y, 0.8))
    hi = float(np.percentile(y, 99.5))
    out = np.clip((y - lo) / (hi - lo + 1e-12), 0.0, None)
    out /= np.max(out) + 1e-12
    return np.clip(out, 0.0, 1.02)


def baseline_texture(
    x: np.ndarray,
    rng: np.random.Generator,
    *,
    drift: float,
    ripple: float,
    corr_noise: float,
    hf_noise: float,
    slope_sign: float = 1.0,
) -> np.ndarray:
    grid = np.linspace(0.0, 1.0, x.size)
    base = drift * (0.55 + 0.95 * grid)
    base += 0.85 * drift * np.sin(2.0 * np.pi * (0.58 * grid + rng.uniform(-0.08, 0.08)))
    base += 0.40 * drift * slope_sign * (grid - 0.45)

    ripple_term = ripple * np.sin(2.0 * np.pi * (3.4 * grid + rng.uniform(-0.16, 0.16)))
    ripple_term += 0.70 * ripple * np.cos(2.0 * np.pi * (6.9 * grid + rng.uniform(-0.18, 0.18)))

    corr = colored_noise(rng, x.size, sigma=7.0, scale=corr_noise)
    corr += 0.55 * colored_noise(rng, x.size, sigma=2.5, scale=corr_noise)
    hf = hf_noise * rng.normal(0.0, 1.0, size=x.size)
    return base + ripple_term + corr + hf


def build_near_ideal(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    center = 242.0
    main = asymmetric_gaussian(x, center, 23.0, 24.0, 0.82)
    cap = gaussian_peak(x, center + 0.5, 10.5, 0.18)
    wide_skirt = gaussian_peak(x, center, 74.0, 0.085)
    faint_side = gaussian_peak(x, 305.0, 31.0, 0.035)
    floor = gaussian_peak(x, 242.0, 130.0, 0.018)
    y = main + cap + wide_skirt + faint_side + floor
    y += baseline_texture(x, rng, drift=0.0045, ripple=0.0012, corr_noise=0.0018, hf_noise=0.0011, slope_sign=0.4)
    y = 0.93 * y + 0.07 * smooth_1d(y, sigma=1.8)
    return normalize_trace(y)


def build_skewed(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    center = 275.0
    main = asymmetric_gaussian(x, center, 17.0, 23.5, 0.68)
    cap = gaussian_peak(x, center + 0.8, 10.0, 0.11)
    broad_support = gaussian_peak(x, center, 82.0, 0.10)
    shoulder = gaussian_peak(x, 249.0, 18.0, 0.34)
    left_drag = gaussian_peak(x, 229.0, 38.0, 0.15)
    kink = gaussian_peak(x, 261.0, 7.0, 0.055)
    y = main + cap + broad_support + shoulder + left_drag + kink
    y += baseline_texture(x, rng, drift=0.0065, ripple=0.0024, corr_noise=0.0026, hf_noise=0.0017, slope_sign=-0.8)
    y = 0.80 * y + 0.20 * smooth_1d(y, sigma=2.4)
    return normalize_trace(y)


def build_broadened(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    center = 252.0
    main = asymmetric_gaussian(x, center, 39.0, 47.0, 0.58)
    cap = gaussian_peak(x, center, 22.0, 0.17)
    pedestal = gaussian_peak(x, center - 3.0, 108.0, 0.13)
    base_lift = gaussian_peak(x, center + 8.0, 148.0, 0.030)
    shallow_tail = gaussian_peak(x, 314.0, 36.0, 0.032)
    y = main + cap + pedestal + base_lift + shallow_tail
    y += baseline_texture(x, rng, drift=0.0060, ripple=0.0018, corr_noise=0.0022, hf_noise=0.0013, slope_sign=0.2)
    y = 0.72 * y + 0.28 * smooth_1d(y, sigma=4.8)
    return normalize_trace(y)


def build_disturbed(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    center = 263.0
    main = asymmetric_gaussian(x, center, 24.0, 28.0, 0.72)
    cap = gaussian_peak(x, center + 0.5, 11.0, 0.12)
    wide_base = gaussian_peak(x, center, 86.0, 0.11)
    local_bump = gaussian_peak(x, 288.0, 7.5, 0.082)
    narrow_spike = gaussian_peak(x, 236.0, 2.2, 0.050)
    side_lift = gaussian_peak(x, 274.0, 18.0, 0.045)
    y = main + cap + wide_base + local_bump + narrow_spike + side_lift
    y += baseline_texture(x, rng, drift=0.0105, ripple=0.0065, corr_noise=0.0039, hf_noise=0.0027, slope_sign=-0.4)
    window1 = np.exp(-0.5 * ((x - 269.0) / 44.0) ** 2)
    window2 = np.exp(-0.5 * ((x - 305.0) / 16.0) ** 2)
    y += 0.020 * window1 * np.sin(2.0 * np.pi * (x - 212.0) / 17.0)
    y += 0.012 * window2 * np.cos(2.0 * np.pi * x / 6.1)
    y = np.clip(y, 0.0, None)
    return normalize_trace(y)


def generate_preview_data(cfg: PreviewConfig) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    rng = np.random.default_rng(cfg.random_seed)
    x = np.arange(cfg.num_points, dtype=np.float64)
    spectra = {
        "near_ideal": build_near_ideal(x, rng),
        "skewed": build_skewed(x, rng),
        "broadened": build_broadened(x, rng),
        "disturbed": build_disturbed(x, rng),
    }
    return x, spectra


def plot_case(ax: plt.Axes, x: np.ndarray, y: np.ndarray, title: str, panel: str) -> None:
    ax.plot(x, y, color="#1f1f1f", linewidth=1.45)
    ax.set_xlim(float(x.min()), float(x.max()))
    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel("Local sample index")
    ax.set_ylabel("Normalized intensity")
    ax.set_title(f"({panel}) {title}", loc="left", pad=5)
    ax.grid(True, linestyle="--", alpha=0.14)
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

    png_path = outputs_dir / "fig41_simulated_preview_v5.png"
    pdf_path = outputs_dir / "fig41_simulated_preview_v5.pdf"
    fig.savefig(png_path, dpi=cfg.dpi, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    np.savez(
        data_dir / "fig41_simulated_preview_v5_data.npz",
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
