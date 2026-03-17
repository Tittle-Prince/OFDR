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


def add_baseline(x: np.ndarray) -> np.ndarray:
    return 0.03 + 0.02 * (x - x.min()) / (x.max() - x.min()) + 0.01 * np.cos(2.0 * np.pi * (x - x.min()) / (x.max() - x.min()))


def add_ripple(x: np.ndarray) -> np.ndarray:
    span = x.max() - x.min()
    return 0.015 * np.sin(2.0 * np.pi * 10.0 * (x - x.min()) / span)


def add_artifact(x: np.ndarray, center: float, width: float, amplitude: float) -> np.ndarray:
    return amplitude * np.exp(-0.5 * ((x - center) / width) ** 2)


def add_spike(x: np.ndarray, center: float, width: float, amplitude: float) -> np.ndarray:
    return amplitude * np.exp(-0.5 * ((x - center) / width) ** 2)


def build_pipeline_data() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(20260315)
    x = np.linspace(1549.0, 1551.0, 512, dtype=np.float64)

    target_center = 1550.0
    left_center = 1549.68
    right_center = 1550.32
    target_sigma = 0.055
    neighbor_sigma = 0.055

    ideal_target = gaussian_peak(x, target_center, target_sigma, amplitude=1.0)

    neighbors_nominal = (
        gaussian_peak(x, left_center, neighbor_sigma, amplitude=0.80)
        + gaussian_peak(x, right_center, neighbor_sigma, amplitude=0.78)
    )
    superposed_nominal = ideal_target + neighbors_nominal

    left_center_shifted = left_center - 0.045
    right_center_shifted = right_center + 0.035
    neighbors_shifted = (
        gaussian_peak(x, left_center_shifted, neighbor_sigma, amplitude=0.80)
        + gaussian_peak(x, right_center_shifted, neighbor_sigma, amplitude=0.78)
    )
    superposed_shifted = ideal_target + neighbors_shifted

    target_wide = gaussian_peak(x, target_center, 0.075, amplitude=1.0)
    left_wide = gaussian_peak(x, left_center_shifted, 0.070, amplitude=0.80)
    right_wide = gaussian_peak(x, right_center_shifted, 0.072, amplitude=0.78)
    linewidth_before = superposed_shifted
    linewidth_after = target_wide + left_wide + right_wide

    baseline = add_baseline(x)
    ripple = add_ripple(x)
    artifact = add_artifact(x, center=1549.93, width=0.018, amplitude=0.085)
    spike = add_spike(x, center=1550.41, width=0.003, amplitude=0.10)
    local_noise = rng.normal(0.0, 0.002, size=x.shape)

    distortions = baseline + ripple + artifact + spike
    final_spectrum = linewidth_after + distortions + local_noise

    return {
        "x": x.astype(np.float32),
        "ideal_target": ideal_target.astype(np.float32),
        "target_component": ideal_target.astype(np.float32),
        "left_nominal": gaussian_peak(x, left_center, neighbor_sigma, amplitude=0.80).astype(np.float32),
        "right_nominal": gaussian_peak(x, right_center, neighbor_sigma, amplitude=0.78).astype(np.float32),
        "left_shift_before": gaussian_peak(x, left_center, neighbor_sigma, amplitude=0.80).astype(np.float32),
        "right_shift_before": gaussian_peak(x, right_center, neighbor_sigma, amplitude=0.78).astype(np.float32),
        "left_shift_after": gaussian_peak(x, left_center_shifted, neighbor_sigma, amplitude=0.80).astype(np.float32),
        "right_shift_after": gaussian_peak(x, right_center_shifted, neighbor_sigma, amplitude=0.78).astype(np.float32),
        "superposed_nominal": superposed_nominal.astype(np.float32),
        "superposed_shifted": superposed_shifted.astype(np.float32),
        "linewidth_before": linewidth_before.astype(np.float32),
        "linewidth_after": linewidth_after.astype(np.float32),
        "baseline": baseline.astype(np.float32),
        "ripple": ripple.astype(np.float32),
        "artifact": artifact.astype(np.float32),
        "spike": spike.astype(np.float32),
        "distortions": distortions.astype(np.float32),
        "final_spectrum": final_spectrum.astype(np.float32),
        "target_center_nm": np.array([target_center], dtype=np.float32),
        "true_delta_nm": np.array([target_center - 1550.0], dtype=np.float32),
        "true_position_nm": np.array([target_center], dtype=np.float32),
    }


def style_axis(ax: plt.Axes, title: str, x: np.ndarray, show_xlabel: bool, show_ylabel: bool) -> None:
    ax.set_title(title, fontsize=10, pad=4)
    ax.set_xlim(float(x.min()), float(x.max()))
    ax.set_ylim(-0.02, 1.45)
    ax.set_xlabel("Wavelength (nm)" if show_xlabel else "", fontsize=9)
    ax.set_ylabel("Intensity (a.u.)" if show_ylabel else "", fontsize=9)
    ax.tick_params(labelsize=8)
    ax.grid(False)


def build_pipeline_figure(data: dict[str, np.ndarray]) -> plt.Figure:
    x = data["x"]
    lambda0 = 1550.0
    fig, axes = plt.subplots(2, 4, figsize=(15.2, 7.6), constrained_layout=True)
    colors = {
        "target": "#1f77b4",
        "neighbor": "#6c757d",
        "shifted": "#d62728",
        "distort": "#9467bd",
        "final": "#111111",
        "light": "#bdbdbd",
    }

    ax = axes[0, 0]
    ax.plot(x, data["ideal_target"], color=colors["target"], linewidth=2.0)
    style_axis(ax, "(a) Ideal target peak", x, show_xlabel=False, show_ylabel=True)

    ax = axes[0, 1]
    ax.plot(x, data["target_component"], color=colors["target"], linewidth=1.8, label="Target")
    ax.plot(x, data["left_nominal"], color=colors["neighbor"], linewidth=1.5, linestyle="--", label="Neighbor L")
    ax.plot(x, data["right_nominal"], color=colors["neighbor"], linewidth=1.5, linestyle="-.", label="Neighbor R")
    ax.plot(x, data["superposed_nominal"], color=colors["final"], linewidth=2.0, alpha=0.9, label="Combined")
    style_axis(ax, "(b) Neighbor superposition", x, show_xlabel=False, show_ylabel=False)
    ax.legend(fontsize=8, frameon=False)

    ax = axes[0, 2]
    ax.plot(x, data["left_shift_before"], color=colors["neighbor"], linewidth=1.2, linestyle="--", alpha=0.8)
    ax.plot(x, data["right_shift_before"], color=colors["neighbor"], linewidth=1.2, linestyle="--", alpha=0.8)
    ax.plot(x, data["left_shift_after"], color=colors["shifted"], linewidth=1.8)
    ax.plot(x, data["right_shift_after"], color=colors["shifted"], linewidth=1.8)
    ax.plot(x, data["target_component"], color=colors["target"], linewidth=1.6)
    style_axis(ax, "(c) Neighbor shift", x, show_xlabel=False, show_ylabel=False)
    ax.annotate("shifted", xy=(1549.63, 0.58), xytext=(1549.47, 0.88), arrowprops={"arrowstyle": "->", "lw": 1.0}, fontsize=8)

    ax = axes[0, 3]
    ax.plot(x, data["linewidth_before"], color=colors["neighbor"], linewidth=1.4, linestyle="--", label="Before")
    ax.plot(x, data["linewidth_after"], color=colors["shifted"], linewidth=2.0, label="After")
    style_axis(ax, "(d) Linewidth variation", x, show_xlabel=False, show_ylabel=False)
    ax.legend(fontsize=8, frameon=False)

    ax = axes[1, 0]
    ax.plot(x, data["linewidth_after"], color=colors["light"], linewidth=1.5, linestyle="--", label="Before distortion")
    ax.plot(x, data["distortions"], color=colors["distort"], linewidth=1.2, alpha=0.95, label="Distortion term")
    ax.plot(x, data["linewidth_after"] + data["distortions"], color=colors["final"], linewidth=2.0, label="After distortion")
    style_axis(ax, "(e) Background and local distortions", x, show_xlabel=True, show_ylabel=True)
    ax.legend(frameon=False, loc="upper left")
    ax.annotate("artifact", xy=(1549.93, 0.17), xytext=(1549.73, 0.37), arrowprops={"arrowstyle": "->", "lw": 0.9}, fontsize=8)
    ax.annotate("spike", xy=(1550.41, 0.13), xytext=(1550.23, 0.33), arrowprops={"arrowstyle": "->", "lw": 0.9}, fontsize=8)

    ax = axes[1, 1]
    ax.plot(x, data["final_spectrum"], color=colors["final"], linewidth=2.1)
    ax.plot(x, data["target_component"], color=colors["target"], linewidth=1.0, linestyle=":", alpha=0.8)
    style_axis(ax, "(f) Final composite spectrum", x, show_xlabel=True, show_ylabel=False)
    ax.annotate("target", xy=(1550.0, 1.0), xytext=(1549.84, 1.23), arrowprops={"arrowstyle": "->", "lw": 1.0}, fontsize=8)
    ax.annotate("shifted neighbor", xy=(1549.63, 0.62), xytext=(1549.25, 0.98), arrowprops={"arrowstyle": "->", "lw": 1.0}, fontsize=8)
    ax.annotate("artifact / spike", xy=(1550.41, 0.57), xytext=(1550.14, 1.18), arrowprops={"arrowstyle": "->", "lw": 1.0}, fontsize=8)

    ax = axes[1, 2]
    ax.plot(x, data["final_spectrum"], color="#d0d0d0", linewidth=1.0)
    ax.axvline(lambda0, color=colors["neighbor"], linewidth=1.4, linestyle="--")
    ax.axvline(float(data["target_center_nm"][0]), color=colors["target"], linewidth=2.0)
    style_axis(ax, "(g) Label definition", x, show_xlabel=True, show_ylabel=False)
    ax.text(
        0.06,
        0.92,
        "dashed: $\\lambda_0$\nsolid: $\\lambda_{target}$\n$y = \\Delta\\lambda_{target} = \\lambda_{target} - \\lambda_0$",
        transform=ax.transAxes,
        fontsize=9,
        va="top",
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "#777777", "alpha": 0.95},
    )

    ax = axes[1, 3]
    ax.axis("off")
    ax.set_title("(h) Pipeline summary", fontsize=10, pad=4)
    ax.text(
        0.04,
        0.88,
        "$S_{final}(x) =$" "\n"
        "$\\;S_{target}(x) + S_{neighbors}(x)$" "\n"
        "$\\;+ B(x) + R(x) + A(x) + P(x)$" "\n\n"
        "$S_{neighbors}(x)$ : shifted adjacent peaks\n"
        "$B(x)$ : baseline drift\n"
        "$R(x)$ : ripple perturbation\n"
        "$A(x)$ : local artifact\n"
        "$P(x)$ : spike impulse\n\n"
        "$y = \\Delta\\lambda_{target}$",
        fontsize=10,
        va="top",
        linespacing=1.35,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "#f7f7f7", "edgecolor": "#b0b0b0", "alpha": 0.98},
    )

    return fig


def main() -> None:
    configure_style()
    figure_root = Path(__file__).resolve().parents[1]
    outputs_dir = figure_root / "outputs"
    data_dir = figure_root / "data_copy"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    data = build_pipeline_data()
    fig = build_pipeline_figure(data)

    png_path = outputs_dir / "Fig3_1_simulation_pipeline.png"
    pdf_path = outputs_dir / "Fig3_1_simulation_pipeline.pdf"
    data_path = data_dir / "Fig3_1_simulation_pipeline_data.npz"

    fig.savefig(png_path, dpi=300)
    fig.savefig(pdf_path)
    plt.close(fig)
    np.savez(data_path, **data)

    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")
    print(f"Saved: {data_path}")


if __name__ == "__main__":
    main()
