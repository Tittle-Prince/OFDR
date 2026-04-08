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


def sigmoid(x: np.ndarray, center: float, width: float) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-(x - center) / width))


def local_shelf(x: np.ndarray, left: float, right: float, height: float, edge: float) -> np.ndarray:
    return height * (sigmoid(x, left, edge) - sigmoid(x, right, edge))


def normalize_trace(y: np.ndarray) -> np.ndarray:
    y = np.clip(y, 0.0, None)
    lo = float(np.percentile(y, 0.8))
    hi = float(np.percentile(y, 99.5))
    out = np.clip((y - lo) / (hi - lo + 1e-12), 0.0, None)
    out /= np.max(out) + 1e-12
    return np.clip(out, 0.0, 1.02)


def baseline_family(
    x: np.ndarray,
    rng: np.random.Generator,
    *,
    drift: float,
    ripple: float,
    corr_noise: float,
    hf_noise: float,
    phase_shift: float,
    slope_bias: float,
    rough_sigma: float,
) -> np.ndarray:
    grid = np.linspace(0.0, 1.0, x.size)
    base = drift * (0.45 + 0.9 * grid + slope_bias * (grid - 0.5))
    base += 0.75 * drift * np.sin(2.0 * np.pi * (0.52 * grid + phase_shift))
    base += ripple * np.sin(2.0 * np.pi * (3.2 * grid + 0.5 * phase_shift))
    base += 0.65 * ripple * np.cos(2.0 * np.pi * (6.4 * grid - 0.3 * phase_shift))
    base += colored_noise(rng, x.size, sigma=rough_sigma, scale=corr_noise)
    base += 0.55 * colored_noise(rng, x.size, sigma=max(1.3, rough_sigma / 2.5), scale=0.9 * corr_noise)
    base += hf_noise * rng.normal(0.0, 1.0, size=x.size)
    return base


def build_near_ideal(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    center = 236.0
    main = asymmetric_gaussian(x, center, 22.0, 23.0, 0.83)
    cap = gaussian_peak(x, center + 0.5, 10.2, 0.16)
    skirt = gaussian_peak(x, center, 68.0, 0.09)
    faint_side = gaussian_peak(x, 314.0, 35.0, 0.025)
    floor = gaussian_peak(x, center - 5.0, 148.0, 0.012)
    y = main + cap + skirt + faint_side + floor
    y += baseline_family(
        x,
        rng,
        drift=0.0042,
        ripple=0.0011,
        corr_noise=0.0016,
        hf_noise=0.0010,
        phase_shift=0.08,
        slope_bias=0.20,
        rough_sigma=7.5,
    )
    y = 0.94 * y + 0.06 * smooth_1d(y, sigma=1.4)
    return normalize_trace(y)


def build_skewed(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    center = 282.0
    main = asymmetric_gaussian(x, center, 16.5, 22.0, 0.63)
    broad_base = gaussian_peak(x, center + 1.0, 76.0, 0.12)
    shoulder_core = gaussian_peak(x, 257.0, 12.5, 0.24)
    shoulder_wing = gaussian_peak(x, 247.0, 26.0, 0.14)
    local_kink = local_shelf(x, 260.0, 273.0, 0.030, 2.0)
    tiny_notch_fill = gaussian_peak(x, 269.0, 5.0, 0.030)
    y = main + broad_base + shoulder_core + shoulder_wing + local_kink + tiny_notch_fill
    y += baseline_family(
        x,
        rng,
        drift=0.0062,
        ripple=0.0024,
        corr_noise=0.0024,
        hf_noise=0.0016,
        phase_shift=0.27,
        slope_bias=-0.55,
        rough_sigma=6.0,
    )
    y = 0.84 * y + 0.16 * smooth_1d(y, sigma=2.0)
    return normalize_trace(y)


def build_broadened(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    center = 248.0
    broad_left = gaussian_peak(x, center - 2.0, 42.0, 0.34)
    broad_right = gaussian_peak(x, center + 2.0, 51.0, 0.30)
    top_fill = gaussian_peak(x, center + 1.0, 18.0, 0.12)
    pedestal = gaussian_peak(x, center - 8.0, 118.0, 0.15)
    asym_floor = local_shelf(x, 205.0, 310.0, 0.020, 9.0)
    weak_far = gaussian_peak(x, 330.0, 42.0, 0.020)
    y = broad_left + broad_right + top_fill + pedestal + asym_floor + weak_far
    y += baseline_family(
        x,
        rng,
        drift=0.0068,
        ripple=0.0018,
        corr_noise=0.0020,
        hf_noise=0.0012,
        phase_shift=-0.14,
        slope_bias=0.12,
        rough_sigma=9.5,
    )
    y += 0.010 * gaussian_peak(x, 225.0, 14.0, 1.0)
    y = 0.76 * y + 0.24 * smooth_1d(y, sigma=4.8)
    return normalize_trace(y)


def build_disturbed(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    center = 266.0
    main = asymmetric_gaussian(x, center, 23.0, 26.0, 0.70)
    cap = gaussian_peak(x, center, 11.2, 0.11)
    wide_skirt = gaussian_peak(x, center + 1.0, 82.0, 0.12)
    anomaly_peak = gaussian_peak(x, 292.0, 6.0, 0.080)
    anomaly_shelf = local_shelf(x, 273.0, 287.0, 0.032, 1.7)
    narrow_spike = gaussian_peak(x, 238.0, 1.8, 0.050)
    right_floor = gaussian_peak(x, 309.0, 17.0, 0.028)
    y = main + cap + wide_skirt + anomaly_peak + anomaly_shelf + narrow_spike + right_floor
    y += baseline_family(
        x,
        rng,
        drift=0.0108,
        ripple=0.0068,
        corr_noise=0.0038,
        hf_noise=0.0028,
        phase_shift=0.41,
        slope_bias=-0.20,
        rough_sigma=4.8,
    )
    y += 0.018 * np.exp(-0.5 * ((x - 270.0) / 43.0) ** 2) * np.sin(2.0 * np.pi * (x - 210.0) / 16.5)
    y += 0.009 * np.exp(-0.5 * ((x - 303.0) / 13.0) ** 2) * np.cos(2.0 * np.pi * x / 5.4)
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

    png_path = outputs_dir / "fig41_simulated_preview_v6.png"
    pdf_path = outputs_dir / "fig41_simulated_preview_v6.pdf"
    fig.savefig(png_path, dpi=cfg.dpi, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    np.savez(
        data_dir / "fig41_simulated_preview_v6_data.npz",
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
