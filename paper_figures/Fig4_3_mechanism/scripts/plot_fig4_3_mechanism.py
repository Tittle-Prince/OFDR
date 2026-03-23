from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def gaussian_peak(x: np.ndarray, center: float, sigma: float, amplitude: float) -> np.ndarray:
    return amplitude * np.exp(-0.5 * ((x - center) / sigma) ** 2)


def generate_two_peak_spectrum(
    x: np.ndarray,
    true_center: float,
    neighbor_center: float,
    neighbor_amplitude: float,
    true_sigma: float = 0.015,
    neighbor_sigma: float = 0.016,
    true_amplitude: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    true_peak = gaussian_peak(x, true_center, true_sigma, true_amplitude)
    neighbor_peak = gaussian_peak(x, neighbor_center, neighbor_sigma, neighbor_amplitude)
    spectrum = true_peak + neighbor_peak
    return spectrum, true_peak, neighbor_peak


def build_case_data() -> dict[str, np.ndarray]:
    x = np.linspace(1549.85, 1550.18, 1000, dtype=np.float64)
    true_center = 1550.0

    easy_spectrum, easy_true, easy_neighbor = generate_two_peak_spectrum(
        x=x,
        true_center=true_center,
        neighbor_center=1550.135,
        neighbor_amplitude=0.10,
        neighbor_sigma=0.018,
    )
    medium_spectrum, medium_true, medium_neighbor = generate_two_peak_spectrum(
        x=x,
        true_center=true_center,
        neighbor_center=1550.068,
        neighbor_amplitude=0.48,
        neighbor_sigma=0.017,
    )
    hard_spectrum, hard_true, hard_neighbor = generate_two_peak_spectrum(
        x=x,
        true_center=true_center,
        neighbor_center=1550.035,
        neighbor_amplitude=1.20,
        neighbor_sigma=0.018,
    )

    pseudo_peak = float(x[np.argmax(hard_spectrum)])

    cnn_easy = 1550.0015
    cnn_medium = 1550.0035
    cnn_hard = 1550.0260

    difficulty_labels = np.array(["Easy", "Medium", "Hard"])
    p99_values = np.array([0.010, 0.020, 0.040], dtype=np.float64)

    return {
        "x": x,
        "true_center": np.array([true_center], dtype=np.float64),
        "easy_spectrum": easy_spectrum,
        "easy_true": easy_true,
        "easy_neighbor": easy_neighbor,
        "medium_spectrum": medium_spectrum,
        "medium_true": medium_true,
        "medium_neighbor": medium_neighbor,
        "hard_spectrum": hard_spectrum,
        "hard_true": hard_true,
        "hard_neighbor": hard_neighbor,
        "pseudo_peak": np.array([pseudo_peak], dtype=np.float64),
        "cnn_easy": np.array([cnn_easy], dtype=np.float64),
        "cnn_medium": np.array([cnn_medium], dtype=np.float64),
        "cnn_hard": np.array([cnn_hard], dtype=np.float64),
        "difficulty_labels": difficulty_labels,
        "p99_values": p99_values,
    }


def style_axis(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, linestyle="--", alpha=0.3, linewidth=0.6)
    ax.tick_params(direction="out", length=3, width=0.8)


def build_figure(data: dict[str, np.ndarray]) -> plt.Figure:
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    x = data["x"]
    true_center = float(data["true_center"][0])
    base_color = "#1f1f1f"
    ref_color = "#6b7280"
    pred_color = "#4c78a8"
    bar_colors = ["#d9d9d9", "#9fb6cc", "#4c78a8"]

    ax = axes[0, 0]
    ax.plot(x, data["easy_spectrum"], color=base_color, linewidth=2.0)
    ax.axvline(true_center, color=ref_color, linestyle="--", linewidth=1.5)
    ax.axvline(float(data["cnn_easy"][0]), color=pred_color, linestyle="-", linewidth=2.0)
    ax.set_title("(a) Easy case (no interference)")
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Intensity (a.u.)")
    ax.annotate("True peak", xy=(true_center, data["easy_spectrum"].max() * 0.98), xytext=(1549.955, 0.96),
                arrowprops={"arrowstyle": "->", "lw": 1.0}, fontsize=9)
    ax.annotate("CNN prediction", xy=(float(data["cnn_easy"][0]), data["easy_spectrum"].max() * 0.90), xytext=(1550.045, 0.82),
                arrowprops={"arrowstyle": "->", "lw": 1.0}, fontsize=9)
    style_axis(ax)

    ax = axes[0, 1]
    ax.plot(x, data["medium_spectrum"], color=base_color, linewidth=2.0)
    ax.axvline(true_center, color=ref_color, linestyle="--", linewidth=1.5)
    ax.axvline(float(data["cnn_medium"][0]), color=pred_color, linestyle="-", linewidth=2.0)
    ax.set_title("(b) Medium case (moderate overlap)")
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Intensity (a.u.)")
    ax.annotate("True peak", xy=(true_center, data["medium_spectrum"].max() * 0.97), xytext=(1549.948, 1.02),
                arrowprops={"arrowstyle": "->", "lw": 1.0}, fontsize=9)
    ax.annotate("CNN prediction", xy=(float(data["cnn_medium"][0]), data["medium_spectrum"].max() * 0.88), xytext=(1550.050, 0.88),
                arrowprops={"arrowstyle": "->", "lw": 1.0}, fontsize=9)
    style_axis(ax)

    ax = axes[1, 0]
    ax.plot(x, data["hard_spectrum"], color=base_color, linewidth=2.0)
    ax.axvline(true_center, color=ref_color, linestyle="--", linewidth=1.5)
    ax.axvline(float(data["cnn_hard"][0]), color=pred_color, linestyle="-", linewidth=2.0)
    ax.set_title("(c) Hardest case (pseudo peak interference)")
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Intensity (a.u.)")
    ax.annotate(
        "Pseudo peak",
        xy=(float(data["pseudo_peak"][0]), data["hard_spectrum"].max()),
        xytext=(1550.070, 1.32),
        arrowprops={"arrowstyle": "->", "lw": 1.0},
        fontsize=9,
    )
    ax.annotate(
        "True peak",
        xy=(true_center, np.interp(true_center, x, data["hard_spectrum"])),
        xytext=(1549.930, 1.08),
        arrowprops={"arrowstyle": "->", "lw": 1.0},
        fontsize=9,
    )
    ax.annotate(
        "CNN prediction",
        xy=(float(data["cnn_hard"][0]), np.interp(float(data["cnn_hard"][0]), x, data["hard_spectrum"]) * 0.88),
        xytext=(1550.080, 0.92),
        arrowprops={"arrowstyle": "->", "lw": 1.0},
        fontsize=9,
    )
    style_axis(ax)

    ax = axes[1, 1]
    xpos = np.arange(len(data["difficulty_labels"]))
    bars = ax.bar(xpos, data["p99_values"], color=bar_colors, edgecolor="#222222", linewidth=0.7, width=0.62)
    ax.set_xticks(xpos, data["difficulty_labels"])
    ax.set_xlabel("Difficulty level")
    ax.set_ylabel("P99 Error (nm)")
    ax.set_title("(d) Error vs difficulty (P99)")
    for bar, value in zip(bars, data["p99_values"]):
        ax.text(bar.get_x() + bar.get_width() / 2.0, float(value) + 0.001, f"{value:.3f}",
                ha="center", va="bottom", fontsize=9)
    style_axis(ax)

    for ax in axes.ravel():
        ax.set_xlim(float(x.min()), float(x.max())) if ax is not axes[1, 1] else None

    fig.tight_layout()
    return fig


def main() -> None:
    configure_style()

    figure_root = Path(__file__).resolve().parents[1]
    outputs_dir = figure_root / "outputs"
    data_dir = figure_root / "data_copy"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    data = build_case_data()
    fig = build_figure(data)

    png_path = outputs_dir / "fig4_3_mechanism.png"
    pdf_path = outputs_dir / "fig4_3_mechanism.pdf"
    data_path = data_dir / "fig4_3_mechanism_data.npz"

    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    np.savez(data_path, **data)

    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")
    print(f"Saved: {data_path}")


if __name__ == "__main__":
    main()
