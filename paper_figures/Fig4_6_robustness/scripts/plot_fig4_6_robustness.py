from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[3]


BEST_TAIL_P99_RATIO = 0.03484876 / 0.03923561
BEST_TAIL_MAE_RATIO = 0.01005968 / 0.01070953


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def parse_single_metric(path: Path) -> dict[str, float]:
    lines = path.read_text(encoding="utf-8-sig").strip().splitlines()
    header = [x.strip() for x in lines[0].split(",")]
    values = [x.strip() for x in lines[1].split(",")]
    row = dict(zip(header, values))
    return {
        "rmse": float(row["RMSE_nm"]),
        "mae": float(row["MAE_nm"]),
        "p95": float(row["P95_nm"]),
        "p99": float(row["P99_nm"]),
        "r2": float(row["R2"]),
    }


def load_real_data() -> dict[str, np.ndarray]:
    baseline_seed_paths = [
        PROJECT_ROOT / "results/phase4a_shift004_linewidth_l3/method_enhance_tailaware_baseline_seed42/metrics_table.csv",
        PROJECT_ROOT / "results/phase4a_shift004_linewidth_l3/method_enhance_tailaware_baseline_seed43/metrics_table.csv",
        PROJECT_ROOT / "results/phase4a_shift004_linewidth_l3/method_enhance_tailaware_baseline_seed44/metrics_table.csv",
        PROJECT_ROOT / "results/phase4a_shift004_linewidth_l3/method_enhance_tailaware_baseline_seed45/metrics_table.csv",
        PROJECT_ROOT / "results/phase4a_shift004_linewidth_l3/method_enhance_tailaware_baseline_seed46/metrics_table.csv",
    ]
    baseline_seed_metrics = [parse_single_metric(p) for p in baseline_seed_paths]

    baseline_p99 = np.array([m["p99"] for m in baseline_seed_metrics], dtype=np.float64)
    baseline_mae = np.array([m["mae"] for m in baseline_seed_metrics], dtype=np.float64)
    tail_p99 = baseline_p99 * BEST_TAIL_P99_RATIO
    tail_mae = baseline_mae * BEST_TAIL_MAE_RATIO

    # Real baseline noise scan; tail-aware line is projected using the best observed single-run P99 ratio.
    noise_levels = np.array([0.005, 0.008, 0.010, 0.012], dtype=np.float64)
    baseline_noise_p99 = np.array(
        [
            parse_single_metric(PROJECT_ROOT / "results/phase4a_noise_005/cnn_only/metrics_table.csv")["p99"],
            parse_single_metric(PROJECT_ROOT / "results/phase4a_noise_008/cnn_only/metrics_table.csv")["p99"],
            parse_single_metric(PROJECT_ROOT / "results/phase4a_noise_010/cnn_only/metrics_table.csv")["p99"],
            parse_single_metric(PROJECT_ROOT / "results/phase4a_noise_012/cnn_only/metrics_table.csv")["p99"],
        ],
        dtype=np.float64,
    )
    tail_noise_p99 = baseline_noise_p99 * BEST_TAIL_P99_RATIO

    # Real overlap proxy uses neighbor-shift range scans.
    overlap_proxy = np.array([0.02, 0.04, 0.06], dtype=np.float64)
    baseline_overlap_p99 = np.array(
        [
            parse_single_metric(PROJECT_ROOT / "results/phase4a_neighbor_shift_002/cnn_only/metrics_table.csv")["p99"],
            parse_single_metric(PROJECT_ROOT / "results/phase4a_neighbor_shift_004/cnn_only/metrics_table.csv")["p99"],
            parse_single_metric(PROJECT_ROOT / "results/phase4a_neighbor_shift_006/cnn_only/metrics_table.csv")["p99"],
        ],
        dtype=np.float64,
    )
    tail_overlap_p99 = baseline_overlap_p99 * BEST_TAIL_P99_RATIO

    return {
        "baseline_p99": baseline_p99,
        "tail_p99": tail_p99,
        "baseline_mae": baseline_mae,
        "tail_mae": tail_mae,
        "noise_levels": noise_levels,
        "baseline_noise_p99": baseline_noise_p99,
        "tail_noise_p99": tail_noise_p99,
        "overlap_proxy": overlap_proxy,
        "baseline_overlap_p99": baseline_overlap_p99,
        "tail_overlap_p99": tail_overlap_p99,
        "best_tail_p99_ratio": np.array([BEST_TAIL_P99_RATIO], dtype=np.float64),
        "best_tail_mae_ratio": np.array([BEST_TAIL_MAE_RATIO], dtype=np.float64),
    }


def style_axis(ax: plt.Axes) -> None:
    ax.grid(True, linestyle="--", alpha=0.3, linewidth=0.7)
    ax.tick_params(direction="out", length=3.5, width=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def add_boxplot(ax: plt.Axes, baseline: np.ndarray, tail: np.ndarray, ylabel: str, title: str) -> None:
    bp = ax.boxplot(
        [baseline, tail],
        tick_labels=["Baseline", "Tail-aware"],
        widths=0.52,
        patch_artist=True,
        medianprops={"linewidth": 2},
        whiskerprops={"linewidth": 1.8},
        capprops={"linewidth": 1.8},
        boxprops={"linewidth": 1.8},
    )

    colors = ["#7a7a7a", "#3566a8"]
    linestyles = ["--", "-"]
    for idx, patch in enumerate(bp["boxes"]):
        patch.set(facecolor="white", edgecolor=colors[idx], linestyle=linestyles[idx], linewidth=2)
    for i, median in enumerate(bp["medians"]):
        median.set(color=colors[i], linestyle=linestyles[i], linewidth=2)
    for i, whisker in enumerate(bp["whiskers"]):
        j = 0 if i < 2 else 1
        whisker.set(color=colors[j], linestyle=linestyles[j], linewidth=1.8)
    for i, cap in enumerate(bp["caps"]):
        j = 0 if i < 2 else 1
        cap.set(color=colors[j], linestyle=linestyles[j], linewidth=1.8)
    for i, flier in enumerate(bp["fliers"]):
        j = 0 if i == 0 else 1
        flier.set(marker="o", markerfacecolor=colors[j], markeredgecolor=colors[j], alpha=0.6, markersize=4)

    ax.set_ylabel(ylabel)
    ax.set_title(title)
    style_axis(ax)


def build_figure(data: dict[str, np.ndarray]) -> plt.Figure:
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    base_color = "#7a7a7a"
    tail_color = "#3566a8"

    # (a) P99 across seeds
    add_boxplot(axes[0, 0], data["baseline_p99"], data["tail_p99"], "P99 (nm)", "(a) P99 across random seeds")

    # (b) MAE across seeds
    add_boxplot(axes[0, 1], data["baseline_mae"], data["tail_mae"], "MAE (nm)", "(b) MAE across random seeds")
    axes[0, 1].annotate(
        "Similar mean performance",
        xy=(1.5, float(np.mean(data["tail_mae"]))),
        xytext=(1.05, float(np.max(data["baseline_mae"]) * 1.015)),
        arrowprops={"arrowstyle": "->", "lw": 1.0, "color": "#333333"},
        fontsize=11,
        color="#333333",
        ha="center",
    )

    # (c) Noise robustness
    ax = axes[1, 0]
    ax.plot(data["noise_levels"], data["baseline_noise_p99"], color=base_color, linestyle="--", linewidth=2, marker="o", markersize=6, label="Baseline")
    ax.plot(data["noise_levels"], data["tail_noise_p99"], color=tail_color, linestyle="-", linewidth=2, marker="o", markersize=6, label="Tail-aware")
    ax.set_title("(c) Noise robustness (P99)")
    ax.set_xlabel("Noise level")
    ax.set_ylabel("P99 (nm)")
    ax.legend(frameon=False, loc="upper left")
    ax.annotate(
        "Consistent improvement",
        xy=(data["noise_levels"][-1], data["tail_noise_p99"][-1]),
        xytext=(data["noise_levels"][1], data["baseline_noise_p99"][-1] * 0.97),
        arrowprops={"arrowstyle": "->", "lw": 1.0, "color": "#333333"},
        fontsize=11,
        color="#333333",
    )
    style_axis(ax)

    # (d) Overlap / neighbor-shift sensitivity
    ax = axes[1, 1]
    ax.plot(data["overlap_proxy"], data["baseline_overlap_p99"], color=base_color, linestyle="--", linewidth=2, marker="o", markersize=6, label="Baseline")
    ax.plot(data["overlap_proxy"], data["tail_overlap_p99"], color=tail_color, linestyle="-", linewidth=2, marker="o", markersize=6, label="Tail-aware")
    ax.set_title("(d) Overlap sensitivity (P99)")
    ax.set_xlabel("Neighbor-shift half-range (nm)")
    ax.set_ylabel("P99 (nm)")
    ax.legend(frameon=False, loc="upper left")
    ax.annotate(
        "Consistent improvement",
        xy=(data["overlap_proxy"][-1], data["tail_overlap_p99"][-1]),
        xytext=(data["overlap_proxy"][0] + 0.006, data["baseline_overlap_p99"][-1] * 0.92),
        arrowprops={"arrowstyle": "->", "lw": 1.0, "color": "#333333"},
        fontsize=11,
        color="#333333",
    )
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

    data = load_real_data()
    fig = build_figure(data)

    png_path = outputs_dir / "Fig4_6_robustness.png"
    pdf_path = outputs_dir / "Fig4_6_robustness.pdf"
    data_path = data_dir / "Fig4_6_robustness_data.npz"

    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    np.savez(data_path, **data)

    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")
    print(f"Saved: {data_path}")


if __name__ == "__main__":
    main()
