from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def gaussian_peak(x: np.ndarray, center: float, sigma: float, amplitude: float = 1.0) -> np.ndarray:
    return amplitude * np.exp(-0.5 * ((x - center) / sigma) ** 2)


def add_target_artifact(x: np.ndarray, center: float, amplitude: float = 0.075, width: float = 0.018) -> np.ndarray:
    return amplitude * np.exp(-0.5 * ((x - center) / width) ** 2)


def add_spike(x: np.ndarray, center: float, amplitude: float = 0.11, width: float = 0.0035) -> np.ndarray:
    return amplitude * np.exp(-0.5 * ((x - center) / width) ** 2)


def add_baseline_ripple(x: np.ndarray) -> np.ndarray:
    span = x.max() - x.min()
    baseline = 0.025 + 0.015 * (x - x.min()) / span
    ripple = 0.012 * np.sin(2.0 * np.pi * 9.0 * (x - x.min()) / span + 0.4)
    return baseline + ripple


def make_clean_components(x: np.ndarray) -> dict[str, np.ndarray]:
    target = gaussian_peak(x, 1550.00, 0.055, 1.00)
    left = gaussian_peak(x, 1549.69, 0.055, 0.82)
    right = gaussian_peak(x, 1550.31, 0.055, 0.80)
    total = target + left + right
    return {"target": target, "left": left, "right": right, "total": total}


def make_neighbor_shift(x: np.ndarray) -> dict[str, np.ndarray]:
    target = gaussian_peak(x, 1550.00, 0.055, 1.00)
    left_before = gaussian_peak(x, 1549.69, 0.055, 0.82)
    right_before = gaussian_peak(x, 1550.31, 0.055, 0.80)
    left_after = gaussian_peak(x, 1549.64, 0.055, 0.82)
    right_after = gaussian_peak(x, 1550.36, 0.055, 0.80)
    total_before = target + left_before + right_before
    total_after = target + left_after + right_after
    return {
        "target": target,
        "left_before": left_before,
        "right_before": right_before,
        "left_after": left_after,
        "right_after": right_after,
        "total_before": total_before,
        "total_after": total_after,
    }


def make_linewidth_variation(x: np.ndarray) -> dict[str, np.ndarray]:
    before = make_clean_components(x)["total"]
    target = gaussian_peak(x, 1550.00, 0.075, 1.00)
    left = gaussian_peak(x, 1549.69, 0.070, 0.82)
    right = gaussian_peak(x, 1550.31, 0.072, 0.80)
    after = target + left + right
    return {"before": before, "after": after}


def make_target_artifact_case(x: np.ndarray) -> dict[str, np.ndarray]:
    clean = make_clean_components(x)["total"]
    artifact = add_target_artifact(x, center=1549.955)
    distorted = clean + artifact
    return {"clean": clean, "artifact": artifact, "distorted": distorted}


def make_spike_case(x: np.ndarray) -> dict[str, np.ndarray]:
    clean = make_clean_components(x)["total"]
    spike = add_spike(x, center=1550.38)
    distorted = clean + spike
    return {"clean": clean, "spike": spike, "distorted": distorted}


def make_hardest_case(x: np.ndarray, rng: np.random.Generator) -> dict[str, np.ndarray]:
    target = gaussian_peak(x, 1550.00, 0.073, 1.00)
    left = gaussian_peak(x, 1549.63, 0.069, 0.82)
    right = gaussian_peak(x, 1550.36, 0.071, 0.80)
    structural = target + left + right
    baseline_ripple = add_baseline_ripple(x)
    artifact = add_target_artifact(x, center=1549.95, amplitude=0.070, width=0.016)
    spike = add_spike(x, center=1550.39, amplitude=0.090, width=0.003)
    noise = rng.normal(0.0, 0.0018, size=x.shape)
    hardest = structural + baseline_ripple + artifact + spike + noise
    return {
        "clean_reference": make_clean_components(x)["total"],
        "structural": structural,
        "baseline_ripple": baseline_ripple,
        "artifact": artifact,
        "spike": spike,
        "hardest": hardest,
    }


def style_axis(ax: plt.Axes, title: str, x: np.ndarray, show_xlabel: bool, show_ylabel: bool) -> None:
    ax.set_title(title, fontsize=10, pad=4)
    ax.set_xlim(float(x.min()), float(x.max()))
    ax.set_ylim(-0.02, 1.42)
    ax.set_xlabel("Wavelength (nm)" if show_xlabel else "")
    ax.set_ylabel("Normalized intensity (a.u.)" if show_ylabel else "")
    ax.tick_params(labelsize=8)
    ax.grid(False)


def build_figure(data: dict[str, np.ndarray]) -> plt.Figure:
    x = data["x"]
    fig, axes = plt.subplots(2, 3, figsize=(13.2, 7.6), constrained_layout=True)

    main_color = "#111111"
    ref_color = "#b0b0b0"
    accent_color = "#4f6d8a"

    ax = axes[0, 0]
    ax.plot(x, data["clean_total"], color=main_color, linewidth=1.9)
    ax.axvline(1550.0, color=accent_color, linewidth=1.1, linestyle="--", alpha=0.8)
    style_axis(ax, "(a) Clean", x, show_xlabel=False, show_ylabel=True)

    ax = axes[0, 1]
    ax.plot(x, data["clean_total"], color=ref_color, linewidth=1.3, linestyle="--")
    ax.plot(x, data["shift_total"], color=main_color, linewidth=1.9)
    style_axis(ax, "(b) Neighbor shift", x, show_xlabel=False, show_ylabel=False)
    ax.annotate("shifted neighbor", xy=(1549.64, 0.63), xytext=(1549.28, 0.95), arrowprops={"arrowstyle": "->", "lw": 0.9}, fontsize=8)

    ax = axes[0, 2]
    ax.plot(x, data["clean_total"], color=ref_color, linewidth=1.3, linestyle="--")
    ax.plot(x, data["linewidth_total"], color=main_color, linewidth=1.9)
    style_axis(ax, "(c) Linewidth variation", x, show_xlabel=False, show_ylabel=False)
    ax.annotate("broadened peak", xy=(1550.00, 1.00), xytext=(1549.76, 1.18), arrowprops={"arrowstyle": "->", "lw": 0.9}, fontsize=8)

    ax = axes[1, 0]
    ax.plot(x, data["clean_total"], color=ref_color, linewidth=1.3, linestyle="--")
    ax.plot(x, data["artifact_total"], color=main_color, linewidth=1.9)
    style_axis(ax, "(d) Target-nearby artifact", x, show_xlabel=True, show_ylabel=True)
    ax.annotate("local artifact", xy=(1549.955, 1.03), xytext=(1549.70, 1.18), arrowprops={"arrowstyle": "->", "lw": 0.9}, fontsize=8)

    ax = axes[1, 1]
    ax.plot(x, data["clean_total"], color=ref_color, linewidth=1.3, linestyle="--")
    ax.plot(x, data["spike_total"], color=main_color, linewidth=1.9)
    style_axis(ax, "(e) Spike disturbance", x, show_xlabel=True, show_ylabel=False)
    ax.annotate("spike", xy=(1550.38, 0.54), xytext=(1550.16, 0.94), arrowprops={"arrowstyle": "->", "lw": 0.9}, fontsize=8)

    ax = axes[1, 2]
    ax.plot(x, data["clean_total"], color=ref_color, linewidth=1.2, linestyle="--")
    ax.plot(x, data["hardest_total"], color=main_color, linewidth=1.95)
    style_axis(ax, "(f) Hardest case", x, show_xlabel=True, show_ylabel=False)
    ax.annotate("artifact", xy=(1549.95, 0.95), xytext=(1549.66, 1.15), arrowprops={"arrowstyle": "->", "lw": 0.9}, fontsize=8)
    ax.annotate("spike", xy=(1550.39, 0.60), xytext=(1550.17, 1.03), arrowprops={"arrowstyle": "->", "lw": 0.9}, fontsize=8)

    return fig


def build_data() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(20260315)
    x = np.linspace(1549.0, 1551.0, 512, dtype=np.float64)

    clean = make_clean_components(x)
    shift = make_neighbor_shift(x)
    linewidth = make_linewidth_variation(x)
    artifact_case = make_target_artifact_case(x)
    spike_case = make_spike_case(x)
    hardest = make_hardest_case(x, rng)

    return {
        "x": x.astype(np.float32),
        "clean_total": clean["total"].astype(np.float32),
        "shift_total": shift["total_after"].astype(np.float32),
        "linewidth_total": linewidth["after"].astype(np.float32),
        "artifact_total": artifact_case["distorted"].astype(np.float32),
        "spike_total": spike_case["distorted"].astype(np.float32),
        "hardest_total": hardest["hardest"].astype(np.float32),
        "artifact_term": artifact_case["artifact"].astype(np.float32),
        "spike_term": spike_case["spike"].astype(np.float32),
        "hardest_baseline_ripple": hardest["baseline_ripple"].astype(np.float32),
        "hardest_artifact": hardest["artifact"].astype(np.float32),
        "hardest_spike": hardest["spike"].astype(np.float32),
    }


def main() -> None:
    configure_style()
    figure_root = Path(__file__).resolve().parents[1]
    outputs_dir = figure_root / "outputs"
    data_dir = figure_root / "data_copy"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    data = build_data()
    fig = build_figure(data)

    png_path = outputs_dir / "Fig3_2_typical_spectra.png"
    pdf_path = outputs_dir / "Fig3_2_typical_spectra.pdf"
    data_path = data_dir / "Fig3_2_typical_spectra_data.npz"

    fig.savefig(png_path, dpi=300)
    fig.savefig(pdf_path)
    plt.close(fig)
    np.savez(data_path, **data)

    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")
    print(f"Saved: {data_path}")


if __name__ == "__main__":
    main()
