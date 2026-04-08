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


def pseudo_voigt_like(
    x: np.ndarray,
    center: float,
    sigma_left: float,
    sigma_right: float,
    gamma: float,
    amplitude: float,
) -> np.ndarray:
    g = asymmetric_gaussian(x, center, sigma_left, sigma_right, amplitude=1.0)
    lor = amplitude / (1.0 + ((x - center) / gamma) ** 2)
    return 0.72 * amplitude * g + 0.28 * lor


def smooth_1d(y: np.ndarray, sigma: float) -> np.ndarray:
    radius = max(1, int(round(4.0 * sigma)))
    grid = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (grid / sigma) ** 2)
    kernel /= np.sum(kernel)
    return np.convolve(y, kernel, mode="same")


def colored_noise(rng: np.random.Generator, n: int, sigma: float, scale: float) -> np.ndarray:
    white = rng.normal(0.0, 1.0, size=n)
    return scale * smooth_1d(white, sigma=sigma)


def baseline_components(x: np.ndarray, rng: np.random.Generator, *, drift: float, ripple: float) -> np.ndarray:
    grid = np.linspace(0.0, 1.0, x.size)
    low = drift * (0.55 + 0.85 * grid)
    low += 0.9 * drift * np.sin(2.0 * np.pi * (0.72 * grid + rng.uniform(-0.06, 0.06)))
    mid = ripple * np.sin(2.0 * np.pi * (4.6 * grid + rng.uniform(-0.18, 0.18)))
    mid += 0.75 * ripple * np.cos(2.0 * np.pi * (8.2 * grid + rng.uniform(-0.20, 0.20)))
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
    texture = baseline_components(x, rng, drift=drift, ripple=ripple)
    texture += colored_noise(rng, x.size, sigma=6.0, scale=corr_noise)
    texture += 0.6 * colored_noise(rng, x.size, sigma=2.4, scale=0.7 * corr_noise)
    texture += hf_noise * rng.normal(0.0, 1.0, size=x.size)
    return np.clip(y + texture, 0.0, None)


def normalize_trace(y: np.ndarray) -> np.ndarray:
    y = np.clip(y, 0.0, None)
    lo = float(np.percentile(y, 0.7))
    hi = float(np.percentile(y, 99.5))
    out = np.clip((y - lo) / (hi - lo + 1e-12), 0.0, None)
    out /= np.max(out) + 1e-12
    return np.clip(out, 0.0, 1.02)


def near_ideal_trace(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    main = pseudo_voigt_like(x, 256.0, 20.5, 21.5, gamma=18.0, amplitude=0.92)
    skirt = gaussian_peak(x, 256.0, 48.0, 0.065)
    weak_left = gaussian_peak(x, 207.0, 22.0, 0.055)
    weak_right = gaussian_peak(x, 307.0, 23.0, 0.050)
    y = main + skirt + weak_left + weak_right
    y = measurement_texture(x, y, rng, drift=0.005, ripple=0.0018, corr_noise=0.0022, hf_noise=0.0015)
    y = 0.94 * y + 0.06 * smooth_1d(y, sigma=1.2)
    return normalize_trace(y)


def skewed_trace(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    main = pseudo_voigt_like(x, 260.0, 18.5, 24.5, gamma=18.0, amplitude=0.90)
    shoulder = gaussian_peak(x, 236.5, 11.0, 0.34)
    left_drag = gaussian_peak(x, 225.0, 30.0, 0.16)
    weak_right = gaussian_peak(x, 309.0, 24.0, 0.060)
    small_coupling = gaussian_peak(x, 246.5, 7.5, 0.060)
    y = main + shoulder + left_drag + weak_right + small_coupling
    y = measurement_texture(x, y, rng, drift=0.007, ripple=0.0035, corr_noise=0.0030, hf_noise=0.0022)
    y = 0.82 * y + 0.18 * smooth_1d(y, sigma=2.2)
    return normalize_trace(y)


def broadened_trace(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    broad_main = pseudo_voigt_like(x, 257.0, 34.0, 36.0, gamma=28.0, amplitude=0.78)
    flat_cap = gaussian_peak(x, 257.0, 15.0, 0.20)
    wide_base = gaussian_peak(x, 257.0, 62.0, 0.11)
    faint_left = gaussian_peak(x, 208.0, 25.0, 0.055)
    faint_right = gaussian_peak(x, 306.0, 27.0, 0.060)
    y = broad_main + flat_cap + wide_base + faint_left + faint_right
    y = measurement_texture(x, y, rng, drift=0.0065, ripple=0.0023, corr_noise=0.0028, hf_noise=0.0018)
    y = 0.74 * y + 0.26 * smooth_1d(y, sigma=3.8)
    return normalize_trace(y)


def disturbed_trace(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    main = pseudo_voigt_like(x, 255.5, 21.0, 22.0, gamma=18.5, amplitude=0.90)
    weak_left = gaussian_peak(x, 208.0, 24.0, 0.10)
    weak_right = gaussian_peak(x, 304.0, 23.0, 0.085)
    local_bump = gaussian_peak(x, 279.0, 7.0, 0.090)
    narrow_spike = gaussian_peak(x, 232.5, 2.1, 0.065)
    under_peak_wobble = gaussian_peak(x, 267.0, 16.0, 0.040)
    y = main + weak_left + weak_right + local_bump + narrow_spike + under_peak_wobble
    y = measurement_texture(x, y, rng, drift=0.011, ripple=0.0068, corr_noise=0.0042, hf_noise=0.0030)
    local_window = np.exp(-0.5 * ((x - 260.0) / 46.0) ** 2)
    y += 0.022 * local_window * np.sin(2.0 * np.pi * (x - 210.0) / 16.0)
    y += 0.012 * np.exp(-0.5 * ((x - 305.0) / 14.0) ** 2) * np.cos(2.0 * np.pi * x / 5.8)
    y = np.clip(y, 0.0, None)
    return normalize_trace(y)


def generate_preview_data(cfg: PreviewConfig) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    rng = np.random.default_rng(cfg.random_seed)
    x = np.arange(cfg.num_points, dtype=np.float64)
    spectra = {
        "near_ideal": near_ideal_trace(x, rng),
        "skewed": skewed_trace(x, rng),
        "broadened": broadened_trace(x, rng),
        "disturbed": disturbed_trace(x, rng),
    }
    return x, spectra


def plot_case(ax: plt.Axes, x: np.ndarray, y: np.ndarray, title: str, panel: str) -> None:
    ax.plot(x, y, color="#1e1e1e", linewidth=1.45)
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

    png_path = outputs_dir / "fig41_simulated_preview_v3.png"
    pdf_path = outputs_dir / "fig41_simulated_preview_v3.pdf"
    fig.savefig(png_path, dpi=cfg.dpi, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    np.savez(
        data_dir / "fig41_simulated_preview_v3_data.npz",
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
