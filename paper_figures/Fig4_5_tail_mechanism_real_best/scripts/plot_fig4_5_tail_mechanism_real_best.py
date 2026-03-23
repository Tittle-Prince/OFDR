from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from phase4a.array_simulator import gaussian_reflectivity, simulate_identical_array_spectra
from phase4a.common import load_config, resolve_project_path


CONFIG_PATH = "config/phase4a_shift004_linewidth_l3.yaml"
BASELINE_PRED_PATH = "results/phase4a_shift004_linewidth_l3/cnn_only/predictions.npz"
TAIL_PRED_PATH = "results/phase4a_shift004_linewidth_l3/method_enhance_tailaware_hard_seed45/predictions.npz"


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def load_real_maps() -> dict[str, np.ndarray]:
    cfg = load_config(CONFIG_PATH)
    dataset_path = resolve_project_path(cfg["phase4a"]["dataset_path"])
    baseline_pred_path = PROJECT_ROOT / BASELINE_PRED_PATH
    tail_pred_path = PROJECT_ROOT / TAIL_PRED_PATH

    dataset = np.load(dataset_path)
    pred_base = np.load(baseline_pred_path)
    pred_tail = np.load(tail_pred_path)

    wavelengths = dataset["wavelengths"].astype(np.float64)
    y_all = dataset["Y_dlambda_target"].astype(np.float64)
    idx_test = dataset["idx_test"].astype(np.int64)
    amplitude_scales = dataset["amplitude_scales"].astype(np.float64)
    linewidth_scales = dataset["linewidth_scales"].astype(np.float64)
    neighbor_deltas = dataset["neighbor_delta_lambdas_nm"].astype(np.float64)
    leakage_weights = dataset["leakage_weights"].astype(np.float64)
    target_index = int(dataset["target_index"][0])

    y_true = pred_base["y_true"].astype(np.float64)
    pred_base_delta = pred_base["pred_cnn"].astype(np.float64)
    pred_tail_delta = pred_tail["pred_cnn"].astype(np.float64)
    y_test = y_all[idx_test]

    if not np.allclose(y_test, y_true, atol=1e-7):
        raise ValueError("Prediction order does not match dataset idx_test order.")

    n_test = len(idx_test)
    n_wl = len(wavelengths)
    lambda0 = float(cfg["array"]["lambda0_nm"])
    base_sigma = float(cfg["array"]["linewidth_sigma_nm"])
    base_amplitude = float(cfg["array"]["amplitude"])

    difficulty_map = np.zeros((n_test, n_wl), dtype=np.float64)
    baseline_residual = np.zeros((n_test, n_wl), dtype=np.float64)
    tail_residual = np.zeros((n_test, n_wl), dtype=np.float64)

    for row_id, sample_idx in enumerate(idx_test):
        dlam = float(y_all[sample_idx])
        amp_scales = amplitude_scales[sample_idx]
        lw_scales = linewidth_scales[sample_idx]
        neigh = neighbor_deltas[sample_idx]
        weights = leakage_weights[sample_idx]

        per_grating, _, _ = simulate_identical_array_spectra(
            wavelengths,
            cfg,
            dlam,
            amplitude_scales=amp_scales,
            linewidth_scales=lw_scales,
            neighbor_delta_lambdas_nm=neigh,
        )

        weighted_components = weights[:, None] * per_grating
        target_component = weighted_components[target_index]
        non_target_component = weighted_components.sum(axis=0) - target_component
        difficulty_map[row_id] = non_target_component / (target_component + non_target_component + 1e-12)

        target_sigma = base_sigma * float(lw_scales[target_index])
        target_amplitude = base_amplitude * float(amp_scales[target_index]) * float(weights[target_index])
        center_true = lambda0 + dlam
        center_base = lambda0 + float(pred_base_delta[row_id])
        center_tail = lambda0 + float(pred_tail_delta[row_id])

        true_target = gaussian_reflectivity(wavelengths, center_true, target_sigma, target_amplitude)
        pred_target_base = gaussian_reflectivity(wavelengths, center_base, target_sigma, target_amplitude)
        pred_target_tail = gaussian_reflectivity(wavelengths, center_tail, target_sigma, target_amplitude)

        baseline_residual[row_id] = np.abs(pred_target_base - true_target)
        tail_residual[row_id] = np.abs(pred_target_tail - true_target)

    baseline_abs_error = np.abs(pred_base_delta - y_true)
    order = np.argsort(-baseline_abs_error)

    difficulty_sorted = difficulty_map[order]
    baseline_sorted = baseline_residual[order]
    tail_sorted = tail_residual[order]
    baseline_abs_sorted = baseline_abs_error[order]

    top_rows = max(60, int(0.12 * n_test))
    avg_baseline_top = baseline_sorted[:top_rows].mean(axis=0)
    peak_col = int(np.argmax(avg_baseline_top))
    x0 = max(0, peak_col - 36)
    width = min(72, n_wl - x0)
    y0 = 0
    height = min(top_rows, n_test)

    return {
        "difficulty_map": difficulty_sorted.astype(np.float32),
        "baseline_error": baseline_sorted.astype(np.float32),
        "tailaware_error": tail_sorted.astype(np.float32),
        "baseline_abs_error_sorted": baseline_abs_sorted.astype(np.float32),
        "wavelengths": wavelengths.astype(np.float32),
        "rect_x0": np.array([x0], dtype=np.int32),
        "rect_y0": np.array([y0], dtype=np.int32),
        "rect_width": np.array([width], dtype=np.int32),
        "rect_height": np.array([height], dtype=np.int32),
        "baseline_p95": np.array([np.percentile(baseline_abs_error, 95)], dtype=np.float32),
        "baseline_p99": np.array([np.percentile(baseline_abs_error, 99)], dtype=np.float32),
        "tail_p95": np.array([np.percentile(np.abs(pred_tail_delta - y_true), 95)], dtype=np.float32),
        "tail_p99": np.array([np.percentile(np.abs(pred_tail_delta - y_true), 99)], dtype=np.float32),
        "source_baseline": np.array([str(Path(BASELINE_PRED_PATH))]),
        "source_tailaware": np.array([str(Path(TAIL_PRED_PATH))]),
        "source_dataset": np.array([str(dataset_path.relative_to(PROJECT_ROOT))]),
    }


def add_common_axis_style(ax: plt.Axes, title: str) -> None:
    ax.set_title(title, pad=8)
    ax.set_ylabel("Sample index (sorted by baseline error)")
    ax.tick_params(length=3)


def add_region_box(ax: plt.Axes, x0: int, y0: int, width: int, height: int) -> None:
    rect = patches.Rectangle(
        (x0, y0),
        width,
        height,
        linewidth=1.4,
        edgecolor="white",
        facecolor="none",
        linestyle="--",
    )
    ax.add_patch(rect)


def build_figure(data: dict[str, np.ndarray]) -> plt.Figure:
    difficulty_map = data["difficulty_map"]
    baseline_error = data["baseline_error"]
    tailaware_error = data["tailaware_error"]
    x0 = int(data["rect_x0"][0])
    y0 = int(data["rect_y0"][0])
    width = int(data["rect_width"][0])
    height = int(data["rect_height"][0])

    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True, constrained_layout=False)
    imshow_kwargs = {
        "origin": "lower",
        "aspect": "auto",
        "cmap": "inferno",
        "interpolation": "nearest",
    }

    im0 = axes[0].imshow(difficulty_map, **imshow_kwargs)
    add_common_axis_style(axes[0], "(a) Difficulty map (pseudo-peak density)")
    cbar0 = fig.colorbar(im0, ax=axes[0], fraction=0.024, pad=0.02)
    cbar0.set_label("Interference ratio")

    vmin = float(min(baseline_error.min(), tailaware_error.min()))
    vmax = float(max(baseline_error.max(), tailaware_error.max()))

    im1 = axes[1].imshow(baseline_error, vmin=vmin, vmax=vmax, **imshow_kwargs)
    add_common_axis_style(axes[1], "(b) Error map of baseline CNN")
    add_region_box(axes[1], x0, y0, width, height)
    axes[1].annotate(
        "Error concentration\n(tail region)",
        xy=(x0 + width * 0.75, y0 + height * 0.72),
        xytext=(x0 + width * 0.08, y0 + height * 1.18),
        color="white",
        fontsize=11,
        arrowprops={"arrowstyle": "->", "color": "white", "lw": 1.2},
        ha="left",
        va="bottom",
    )
    cbar1 = fig.colorbar(im1, ax=axes[1], fraction=0.024, pad=0.02)
    cbar1.set_label("Residual magnitude")

    im2 = axes[2].imshow(tailaware_error, vmin=vmin, vmax=vmax, **imshow_kwargs)
    add_common_axis_style(axes[2], "(c) Error map with Tail-aware loss")
    add_region_box(axes[2], x0, y0, width, height)
    axes[2].annotate(
        "Suppressed tail errors",
        xy=(x0 + width * 0.68, y0 + height * 0.52),
        xytext=(x0 + width * 0.05, y0 + height * 1.15),
        color="white",
        fontsize=11,
        arrowprops={"arrowstyle": "->", "color": "white", "lw": 1.2},
        ha="left",
        va="bottom",
    )
    cbar2 = fig.colorbar(im2, ax=axes[2], fraction=0.024, pad=0.02)
    cbar2.set_label("Residual magnitude")

    axes[2].set_xlabel("Wavelength index")

    for ax in axes:
        ax.set_xlim(0, difficulty_map.shape[1] - 1)
        ax.set_ylim(0, difficulty_map.shape[0] - 1)

    fig.tight_layout()
    return fig


def main() -> None:
    configure_style()

    figure_root = Path(__file__).resolve().parents[1]
    outputs_dir = figure_root / "outputs"
    data_dir = figure_root / "data_copy"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    data = load_real_maps()
    fig = build_figure(data)

    png_path = outputs_dir / "Fig4_5_tail_mechanism.png"
    pdf_path = outputs_dir / "Fig4_5_tail_mechanism.pdf"
    data_path = data_dir / "Fig4_5_tail_mechanism_data.npz"

    fig.savefig(png_path, dpi=300)
    fig.savefig(pdf_path)
    plt.close(fig)
    np.savez(data_path, **data)

    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")
    print(f"Saved: {data_path}")


if __name__ == "__main__":
    main()
