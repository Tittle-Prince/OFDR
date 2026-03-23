from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.size": 12,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def gaussian_peak(x: np.ndarray, center: float, sigma: float, amplitude: float) -> np.ndarray:
    return amplitude * np.exp(-0.5 * ((x - center) / sigma) ** 2)


def build_spectrum_data() -> dict[str, np.ndarray]:
    x = np.linspace(-0.08, 0.08, 1601, dtype=np.float64)
    true_center = 0.0
    pseudo_center = 0.02

    true_peak = gaussian_peak(x, true_center, sigma=0.015, amplitude=1.0)
    pseudo_peak_component = gaussian_peak(x, pseudo_center, sigma=0.02, amplitude=0.9)
    spectrum = true_peak + pseudo_peak_component

    # Predicted positions for the explanatory illustration.
    pred_mlp = 0.014
    pred_xcorr = 0.018
    pred_cnn = 0.0015

    # Build a mirrored local profile around the true peak to emphasize asymmetry.
    x_zoom = np.linspace(-0.05, 0.05, 1001, dtype=np.float64)
    y_zoom = np.interp(x_zoom, x, spectrum)
    y_mirror = np.interp(-x_zoom, x, spectrum)

    p99_levels = np.array([0.01, 0.02, 0.04], dtype=np.float64)
    difficulty = np.array(["Easy", "Medium", "Hard"])

    return {
        "x": x,
        "true_peak_component": true_peak,
        "pseudo_peak_component": pseudo_peak_component,
        "spectrum": spectrum,
        "true_center": np.array([true_center], dtype=np.float64),
        "pseudo_center": np.array([pseudo_center], dtype=np.float64),
        "pred_mlp": np.array([pred_mlp], dtype=np.float64),
        "pred_xcorr": np.array([pred_xcorr], dtype=np.float64),
        "pred_cnn": np.array([pred_cnn], dtype=np.float64),
        "x_zoom": x_zoom,
        "y_zoom": y_zoom,
        "y_mirror": y_mirror,
        "difficulty": difficulty,
        "p99_levels": p99_levels,
    }


def load_metric_bars(figure_root: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    metric_npz = (
        figure_root.parents[1]
        / "Fig4_1_baseline_comparison"
        / "data_copy"
        / "fig4_1_baseline_comparison_data.npz"
    )
    if metric_npz.exists():
        p = np.load(metric_npz)
        return p["methods"], p["mae"].astype(np.float64), p["rmse"].astype(np.float64)

    methods = np.array(["Cross-correlation", "MLP", "CNN"])
    mae = np.array([0.14247526, 0.104, 0.01070953], dtype=np.float64)
    rmse = np.array([0.26818001, 0.252, 0.01388817], dtype=np.float64)
    return methods, mae, rmse


def style_axis(ax: plt.Axes) -> None:
    ax.grid(True, linestyle="--", alpha=0.3, linewidth=0.7)
    ax.tick_params(direction="out", length=3.5, width=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def annotate_bars(ax: plt.Axes, bars) -> None:
    for bar in bars:
        height = float(bar.get_height())
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + max(0.003, height * 0.03),
            f"{height:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )


def build_figure(data: dict[str, np.ndarray], methods: np.ndarray, mae: np.ndarray, rmse: np.ndarray) -> plt.Figure:
    fig, axes = plt.subplots(3, 2, figsize=(12, 11))

    x = data["x"]
    y = data["spectrum"]
    true_component = data["true_peak_component"]
    pseudo_component = data["pseudo_peak_component"]
    true_center = float(data["true_center"][0])
    pseudo_center = float(data["pseudo_center"][0])
    pred_mlp = float(data["pred_mlp"][0])
    pred_xcorr = float(data["pred_xcorr"][0])
    pred_cnn = float(data["pred_cnn"][0])

    black = "#111111"
    gray = "#7a7a7a"
    blue = "#4c78a8"
    red = "#b24a4a"
    orange = "#d28b36"
    light_gray = "#d9d9d9"

    # (a) MAE comparison
    ax = axes[0, 0]
    bar_colors = [light_gray, "#9fb6cc", blue]
    bars = ax.bar(methods, mae, color=bar_colors, edgecolor="#222222", linewidth=0.7, width=0.62)
    annotate_bars(ax, bars)
    ax.set_title("(a) MAE comparison")
    ax.set_ylabel("MAE (nm)")
    ax.set_ylim(0.0, float(mae.max()) * 1.22)
    ax.tick_params(axis="x", rotation=10)
    style_axis(ax)

    # (b) RMSE comparison
    ax = axes[0, 1]
    bars = ax.bar(methods, rmse, color=bar_colors, edgecolor="#222222", linewidth=0.7, width=0.62)
    annotate_bars(ax, bars)
    ax.set_title("(b) RMSE comparison")
    ax.set_ylabel("RMSE (nm)")
    ax.set_ylim(0.0, float(rmse.max()) * 1.22)
    ax.tick_params(axis="x", rotation=10)
    style_axis(ax)

    # (c) Raw spectrum with pseudo-peak structure
    ax = axes[1, 0]
    ax.plot(x, y, color=black, linewidth=2.2)
    ax.plot(x, true_component, color=gray, linewidth=1.3, linestyle="--")
    ax.plot(x, pseudo_component, color=gray, linewidth=1.3, linestyle=":")
    ax.axvline(true_center, color=black, linestyle="--", linewidth=1.4)
    ax.axvline(pseudo_center, color=gray, linestyle="--", linewidth=1.2)
    ax.annotate(
        "True peak",
        xy=(true_center, np.interp(true_center, x, y)),
        xytext=(-0.046, 1.70),
        arrowprops={"arrowstyle": "->", "lw": 1.0, "color": black},
        fontsize=11,
        color=black,
    )
    ax.annotate(
        "Pseudo peak",
        xy=(pseudo_center, np.interp(pseudo_center, x, y)),
        xytext=(0.026, 1.56),
        arrowprops={"arrowstyle": "->", "lw": 1.0, "color": gray},
        fontsize=11,
        color=gray,
    )
    rect = patches.Rectangle(
        (-0.05, 0.0),
        0.10,
        float(y.max()) * 1.03,
        linewidth=1.2,
        edgecolor="#666666",
        facecolor="none",
        linestyle="--",
    )
    ax.add_patch(rect)
    ax.set_title("(c) Spectrum with pseudo-peak interference")
    ax.set_xlabel("Wavelength Offset (nm)")
    ax.set_ylabel("Intensity (a.u.)")
    ax.set_xlim(-0.08, 0.08)
    ax.set_ylim(0.0, float(y.max()) * 1.12)
    style_axis(ax)

    # (d) Zoomed local structural differences
    ax = axes[1, 1]
    x_zoom = data["x_zoom"]
    y_zoom = data["y_zoom"]
    y_mirror = data["y_mirror"]
    ax.plot(x_zoom, y_zoom, color=black, linewidth=2.2)
    ax.plot(x_zoom, y_mirror, color=blue, linewidth=1.6, linestyle="--")
    ax.axvline(true_center, color=black, linestyle="--", linewidth=1.2)
    ax.annotate(
        "Slope difference",
        xy=(-0.018, np.interp(-0.018, x_zoom, y_zoom)),
        xytext=(-0.043, 1.60),
        arrowprops={"arrowstyle": "->", "lw": 1.0, "color": black},
        fontsize=11,
        color=black,
    )
    ax.annotate(
        "",
        xy=(0.018, np.interp(0.018, x_zoom, y_zoom)),
        xytext=(0.042, 1.58),
        arrowprops={"arrowstyle": "->", "lw": 1.0, "color": black},
    )
    ax.text(0.008, 1.66, "Asymmetric shape", fontsize=11, color=black)
    ax.text(0.015, 1.28, "Broader shoulder", fontsize=11, color=black)
    ax.set_title("(d) Zoomed local structural differences")
    ax.set_xlabel("Wavelength Offset (nm)")
    ax.set_ylabel("Intensity (a.u.)")
    ax.set_xlim(-0.05, 0.05)
    ax.set_ylim(0.3, float(y_zoom.max()) * 1.08)
    style_axis(ax)

    # (e) MLP / cross-correlation are attracted by the pseudo peak
    ax = axes[2, 0]
    ax.plot(x, y, color=black, linewidth=2.2)
    ax.axvline(true_center, color=black, linestyle="--", linewidth=1.4)
    ax.axvline(pred_mlp, color=red, linestyle="--", linewidth=2.0)
    ax.axvline(pred_xcorr, color=orange, linestyle="--", linewidth=2.0)
    ax.annotate(
        "MLP prediction",
        xy=(pred_mlp, np.interp(pred_mlp, x, y)),
        xytext=(-0.055, 1.50),
        arrowprops={"arrowstyle": "->", "lw": 1.0, "color": red},
        fontsize=11,
        color=red,
    )
    ax.annotate(
        "Cross-correlation",
        xy=(pred_xcorr, np.interp(pred_xcorr, x, y) * 0.98),
        xytext=(0.030, 1.72),
        arrowprops={"arrowstyle": "->", "lw": 1.0, "color": orange},
        fontsize=11,
        color=orange,
    )
    ax.text(0.010, 1.18, "Attracted by amplitude similarity", fontsize=11, color="#333333")
    ax.set_title("(e) MLP / cross-correlation prediction")
    ax.set_xlabel("Wavelength Offset (nm)")
    ax.set_ylabel("Intensity (a.u.)")
    ax.set_xlim(-0.08, 0.08)
    ax.set_ylim(0.0, float(y.max()) * 1.12)
    style_axis(ax)

    # (f) CNN remains near the true peak
    ax = axes[2, 1]
    ax.plot(x, y, color=black, linewidth=2.2)
    ax.axvline(true_center, color=black, linestyle="--", linewidth=1.4)
    ax.axvline(pred_cnn, color=blue, linestyle="-", linewidth=2.4)
    ax.annotate(
        "CNN prediction",
        xy=(pred_cnn, np.interp(pred_cnn, x, y)),
        xytext=(0.024, 1.66),
        arrowprops={"arrowstyle": "->", "lw": 1.0, "color": blue},
        fontsize=11,
        color=blue,
    )
    ax.text(-0.056, 1.42, "Captured local structural features", fontsize=11, color="#333333")
    ax.text(-0.056, 1.24, "Robust to pseudo-peak interference", fontsize=11, color="#333333")
    ax.set_title("(f) CNN prediction under pseudo-peak interference")
    ax.set_xlabel("Wavelength Offset (nm)")
    ax.set_ylabel("Intensity (a.u.)")
    ax.set_xlim(-0.08, 0.08)
    ax.set_ylim(0.0, float(y.max()) * 1.12)
    style_axis(ax)

    fig.tight_layout()
    return fig


def main() -> None:
    configure_style()

    figure_root = Path(__file__).resolve().parents[1]
    outputs_dir = figure_root / "outputs"
    data_dir = figure_root / "data_copy"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    data = build_spectrum_data()
    methods, mae, rmse = load_metric_bars(figure_root)
    fig = build_figure(data, methods, mae, rmse)

    png_path = outputs_dir / "fig_structure_comparison.png"
    pdf_path = outputs_dir / "fig_structure_comparison.pdf"
    data_path = data_dir / "fig_structure_comparison_data.npz"

    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    np.savez(data_path, **data, methods=methods, mae=mae, rmse=rmse)

    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")
    print(f"Saved: {data_path}")


if __name__ == "__main__":
    main()
