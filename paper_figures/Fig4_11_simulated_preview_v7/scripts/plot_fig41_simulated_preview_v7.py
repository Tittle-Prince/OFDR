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
    reference_path: Path = Path(r"F:\54\4lowerplota.txt")


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


def smooth_1d(y: np.ndarray, sigma: float) -> np.ndarray:
    radius = max(1, int(round(4.0 * sigma)))
    grid = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (grid / sigma) ** 2)
    kernel /= np.sum(kernel)
    return np.convolve(y, kernel, mode="same")


def normalize_trace(y: np.ndarray) -> np.ndarray:
    y = np.clip(y, 0.0, None)
    lo = float(np.percentile(y, 0.6))
    hi = float(np.percentile(y, 99.6))
    out = np.clip((y - lo) / (hi - lo + 1e-12), 0.0, None)
    out /= np.max(out) + 1e-12
    return np.clip(out, 0.0, 1.02)


def interp_shift(y: np.ndarray, shift_samples: float) -> np.ndarray:
    x = np.arange(y.size, dtype=np.float64)
    return np.interp(x - shift_samples, x, y, left=y[0], right=y[-1])


def interp_stretch(y: np.ndarray, factor: float, center: float) -> np.ndarray:
    x = np.arange(y.size, dtype=np.float64)
    src = center + (x - center) / factor
    return np.interp(src, x, y, left=y[0], right=y[-1])


def colored_texture(rng: np.random.Generator, n: int, sigma: float, scale: float) -> np.ndarray:
    noise = rng.normal(0.0, 1.0, size=n)
    return scale * smooth_1d(noise, sigma=sigma)


def local_shelf(x: np.ndarray, left: float, right: float, height: float, edge: float) -> np.ndarray:
    sig = lambda z: 1.0 / (1.0 + np.exp(-z / edge))
    return height * (sig(x - left) - sig(x - right))


def load_reference_template(cfg: PreviewConfig) -> tuple[np.ndarray, str]:
    x_local = np.arange(cfg.num_points, dtype=np.float64)
    if cfg.reference_path.exists():
        raw = np.loadtxt(cfg.reference_path)
        y_db = raw[:, 1]
        peak_idx = int(np.argmax(y_db))
        half = cfg.num_points // 2
        lo = max(0, peak_idx - half)
        hi = min(len(y_db), peak_idx + half)
        seg = y_db[lo:hi]
        if seg.size < cfg.num_points:
            pad_left = max(0, half - peak_idx)
            pad_right = cfg.num_points - seg.size - pad_left
            seg = np.pad(seg, (pad_left, pad_right), mode="edge")
        # Convert dB-like trace to relative linear intensity and preserve broad flat-top character.
        lin = 10.0 ** ((seg - np.max(seg)) / 10.0)
        baseline = np.percentile(lin, 1.0)
        lin = np.clip(lin - 0.65 * baseline, 0.0, None)
        ref = normalize_trace(lin)
        ref = 0.92 * ref + 0.08 * smooth_1d(ref, sigma=1.2)
        return ref, "reference-informed from F:\\54\\4lowerplota.txt"

    # Fallback if the reference file is absent.
    fallback = (
        0.72 * np.exp(-0.5 * ((x_local - 250.0) / 24.0) ** 2)
        + 0.16 * np.exp(-0.5 * ((x_local - 250.5) / 11.0) ** 2)
        + 0.12 * np.exp(-0.5 * ((x_local - 250.0) / 78.0) ** 2)
        + 0.015 * np.exp(-0.5 * ((x_local - 320.0) / 40.0) ** 2)
    )
    return normalize_trace(fallback), "synthetic fallback"


def add_background_family(
    x: np.ndarray,
    rng: np.random.Generator,
    *,
    drift: float,
    ripple: float,
    corr: float,
    hf: float,
    slope: float,
    phase: float,
) -> np.ndarray:
    grid = np.linspace(0.0, 1.0, x.size)
    baseline = drift * (0.55 + slope * (grid - 0.45))
    baseline += 0.85 * drift * np.sin(2.0 * np.pi * (0.58 * grid + phase))
    ripple_term = ripple * np.sin(2.0 * np.pi * (3.2 * grid + 0.6 * phase))
    ripple_term += 0.70 * ripple * np.cos(2.0 * np.pi * (6.7 * grid - 0.3 * phase))
    texture = colored_texture(rng, x.size, sigma=6.8, scale=corr)
    texture += 0.55 * colored_texture(rng, x.size, sigma=2.5, scale=0.9 * corr)
    texture += hf * rng.normal(0.0, 1.0, size=x.size)
    return baseline + ripple_term + texture


def build_near_ideal(ref: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    x = np.arange(ref.size, dtype=np.float64)
    y = interp_shift(ref, -18.0)
    y = 0.96 * y + 0.04 * smooth_1d(y, sigma=1.6)
    y += 0.010 * gaussian_peak(x, 332.0, 36.0, 1.0)
    y += add_background_family(x, rng, drift=0.0038, ripple=0.0011, corr=0.0015, hf=0.0009, slope=0.22, phase=0.08)
    return normalize_trace(y)


def build_skewed(ref: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    x = np.arange(ref.size, dtype=np.float64)
    base = interp_shift(ref, +24.0)
    base = interp_stretch(base, factor=0.96, center=282.0)
    shoulder = 0.30 * interp_shift(ref, -2.0)
    shoulder = interp_shift(shoulder, -28.0)
    shoulder = interp_stretch(shoulder, factor=0.70, center=250.0)
    drag = 0.10 * gaussian_peak(x, 238.0, 32.0, 1.0)
    kink = local_shelf(x, 258.0, 273.0, 0.030, 1.9)
    y = 0.78 * base + shoulder + drag + kink
    y += add_background_family(x, rng, drift=0.0058, ripple=0.0023, corr=0.0022, hf=0.0015, slope=-0.48, phase=0.22)
    y = 0.88 * y + 0.12 * smooth_1d(y, sigma=1.9)
    return normalize_trace(y)


def build_broadened(ref: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    x = np.arange(ref.size, dtype=np.float64)
    wide = interp_shift(ref, -6.0)
    wide = interp_stretch(wide, factor=1.42, center=244.0)
    pedestal = 0.23 * interp_stretch(ref, factor=1.85, center=240.0)
    asym_lift = 0.020 * gaussian_peak(x, 214.0, 58.0, 1.0)
    right_floor = 0.018 * gaussian_peak(x, 296.0, 70.0, 1.0)
    y = 0.72 * wide + pedestal + asym_lift + right_floor
    y += add_background_family(x, rng, drift=0.0064, ripple=0.0017, corr=0.0020, hf=0.0012, slope=0.16, phase=-0.12)
    y += 0.008 * local_shelf(x, 228.0, 304.0, 1.0, 8.5)
    y = 0.83 * y + 0.17 * smooth_1d(y, sigma=4.6)
    return normalize_trace(y)


def build_disturbed(ref: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    x = np.arange(ref.size, dtype=np.float64)
    base = interp_shift(ref, +10.0)
    base = interp_stretch(base, factor=1.08, center=261.0)
    anomaly_peak = 0.070 * gaussian_peak(x, 291.0, 6.0, 1.0)
    anomaly_shelf = local_shelf(x, 274.0, 289.0, 0.030, 1.6)
    narrow_spike = 0.050 * gaussian_peak(x, 236.0, 1.9, 1.0)
    floor_raise = 0.030 * gaussian_peak(x, 304.0, 15.0, 1.0)
    y = 0.86 * base + anomaly_peak + anomaly_shelf + narrow_spike + floor_raise
    y += add_background_family(x, rng, drift=0.0100, ripple=0.0065, corr=0.0037, hf=0.0027, slope=-0.14, phase=0.37)
    local_window = np.exp(-0.5 * ((x - 270.0) / 42.0) ** 2)
    y += 0.018 * local_window * np.sin(2.0 * np.pi * (x - 212.0) / 16.8)
    y += 0.009 * np.exp(-0.5 * ((x - 304.0) / 14.0) ** 2) * np.cos(2.0 * np.pi * x / 5.6)
    return normalize_trace(y)


def generate_preview_data(cfg: PreviewConfig) -> tuple[np.ndarray, dict[str, np.ndarray], str]:
    rng = np.random.default_rng(cfg.random_seed)
    x = np.arange(cfg.num_points, dtype=np.float64)
    ref, mode = load_reference_template(cfg)
    spectra = {
        "near_ideal": build_near_ideal(ref, rng),
        "skewed": build_skewed(ref, rng),
        "broadened": build_broadened(ref, rng),
        "disturbed": build_disturbed(ref, rng),
        "reference_template": ref,
    }
    return x, spectra, mode


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

    x, spectra, mode = generate_preview_data(cfg)
    fig = build_figure(x, spectra, cfg)

    png_path = outputs_dir / "fig41_simulated_preview_v7.png"
    pdf_path = outputs_dir / "fig41_simulated_preview_v7.pdf"
    fig.savefig(png_path, dpi=cfg.dpi, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    np.savez(
        data_dir / "fig41_simulated_preview_v7_data.npz",
        local_index=x.astype(np.int32),
        reference_template=spectra["reference_template"].astype(np.float32),
        near_ideal=spectra["near_ideal"].astype(np.float32),
        skewed=spectra["skewed"].astype(np.float32),
        broadened=spectra["broadened"].astype(np.float32),
        disturbed=spectra["disturbed"].astype(np.float32),
    )

    print(f"Saved PNG: {png_path}")
    print(f"Saved PDF: {pdf_path}")
    print(f"Reference mode: {mode}")
    print("Data mode: simulated preview only (not experimental data)")


if __name__ == "__main__":
    main()
