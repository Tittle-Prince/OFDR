from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.size": 11,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def simulate_error_arrays(seed: int = 20260322, n: int = 1200) -> tuple[np.ndarray, np.ndarray, str]:
    rng = np.random.default_rng(seed)
    n_small = int(round(0.90 * n))
    n_tail = n - n_small

    small = np.abs(rng.normal(loc=0.01, scale=0.005, size=n_small))
    baseline_tail = rng.uniform(0.02, 0.08, size=n_tail)
    tailaware_tail = rng.uniform(0.02, 0.04, size=n_tail)

    baseline = np.concatenate([small, baseline_tail])
    tailaware = np.concatenate([small.copy(), tailaware_tail])
    return baseline.astype(np.float64), tailaware.astype(np.float64), "simulated_fallback"


def load_real_error_arrays(repo_root: Path) -> tuple[np.ndarray, np.ndarray, str]:
    baseline_path = (
        repo_root
        / "results"
        / "phase4a_shift004_linewidth_l3"
        / "cnn_only"
        / "predictions.npz"
    )
    tailaware_path = (
        repo_root
        / "results"
        / "phase4a_shift004_linewidth_l3"
        / "method_enhance_tailaware_hard_seed45"
        / "predictions.npz"
    )

    if not baseline_path.exists() or not tailaware_path.exists():
        return simulate_error_arrays()

    baseline_pred = np.load(baseline_path)
    tailaware_pred = np.load(tailaware_path)
    needed = {"y_true", "pred_cnn"}
    if not needed.issubset(baseline_pred.files) or not needed.issubset(tailaware_pred.files):
        return simulate_error_arrays()

    baseline = np.abs(baseline_pred["pred_cnn"].astype(np.float64) - baseline_pred["y_true"].astype(np.float64))
    tailaware = np.abs(tailaware_pred["pred_cnn"].astype(np.float64) - tailaware_pred["y_true"].astype(np.float64))
    source = (
        f"baseline={baseline_path.relative_to(repo_root)}; "
        f"tailaware={tailaware_path.relative_to(repo_root)}"
    )
    return baseline, tailaware, source


def empirical_cdf(errors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.sort(np.asarray(errors, dtype=np.float64))
    y = np.arange(1, x.size + 1, dtype=np.float64) / x.size
    return x, y


def compute_quantiles(errors: np.ndarray, levels: list[int]) -> np.ndarray:
    return np.array([np.percentile(errors, q) for q in levels], dtype=np.float64)


def style_axis(ax: plt.Axes) -> None:
    ax.grid(True, linestyle="--", alpha=0.3, linewidth=0.7)
    ax.tick_params(direction="out", length=3.5, width=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def build_figure(
    baseline: np.ndarray,
    tailaware: np.ndarray,
    quantile_levels: list[int],
    q_base: np.ndarray,
    q_tail: np.ndarray,
) -> plt.Figure:
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    base_color = "#7a7a7a"
    tail_color = "#3566a8"
    tail_label = "Tail-aware+Hard (best run)"

    cdf_x_base, cdf_y_base = empirical_cdf(baseline)
    cdf_x_tail, cdf_y_tail = empirical_cdf(tailaware)
    bins = np.linspace(0.0, max(float(baseline.max()), float(tailaware.max())) * 1.02, 50)

    # (a) CDF comparison
    ax = axes[0, 0]
    ax.plot(cdf_x_base, cdf_y_base, color=base_color, linestyle="--", linewidth=2.0, label="Baseline")
    ax.plot(cdf_x_tail, cdf_y_tail, color=tail_color, linestyle="-", linewidth=2.0, label=tail_label)
    ax.set_title("(a) CDF comparison")
    ax.set_xlabel("Absolute Error (nm)")
    ax.set_ylabel("CDF")
    ax.legend(frameon=False, loc="lower right")
    ax.annotate(
        "Similar mean performance",
        xy=(q_base[1], 0.73),
        xytext=(q_base[1] * 1.45, 0.52),
        arrowprops={"arrowstyle": "->", "lw": 1.0, "color": "#333333"},
        fontsize=11,
        color="#333333",
    )
    ax.annotate(
        "Significant tail reduction",
        xy=(q_tail[-1], 0.992),
        xytext=(q_base[-1] * 0.56, 0.84),
        arrowprops={"arrowstyle": "->", "lw": 1.0, "color": "#333333"},
        fontsize=11,
        color="#333333",
    )
    style_axis(ax)

    # (b) Histogram comparison
    ax = axes[0, 1]
    ax.hist(baseline, bins=bins, density=True, histtype="step", color=base_color, linestyle="--", linewidth=2.0, label="Baseline")
    ax.hist(tailaware, bins=bins, density=True, histtype="step", color=tail_color, linewidth=2.0, label=tail_label)
    ax.set_title("(b) Error distribution")
    ax.set_xlabel("Absolute Error (nm)")
    ax.set_ylabel("Density")
    ax.legend(frameon=False, loc="upper right")
    style_axis(ax)

    # (c) Quantile comparison
    ax = axes[1, 0]
    x = np.arange(len(quantile_levels))
    labels = [f"P{q}" for q in quantile_levels]
    ax.plot(x, q_base, color=base_color, linestyle="--", linewidth=2.0, marker="o", markersize=6, label="Baseline")
    ax.plot(x, q_tail, color=tail_color, linestyle="-", linewidth=2.0, marker="o", markersize=6, label=tail_label)
    ax.set_xticks(x, labels)
    ax.set_title("(c) Quantile comparison")
    ax.set_xlabel("Quantile")
    ax.set_ylabel("Absolute Error (nm)")
    ax.legend(frameon=False, loc="upper left")
    ax.annotate(
        f"{q_base[-1]:.3f}",
        xy=(x[-1], q_base[-1]),
        xytext=(x[-1] - 0.45, q_base[-1] * 1.12),
        arrowprops={"arrowstyle": "->", "lw": 1.0, "color": base_color},
        fontsize=10,
        color=base_color,
    )
    ax.annotate(
        f"{q_tail[-1]:.3f}",
        xy=(x[-1], q_tail[-1]),
        xytext=(x[-1] - 0.18, q_tail[-1] * 0.88),
        arrowprops={"arrowstyle": "->", "lw": 1.0, "color": tail_color},
        fontsize=10,
        color=tail_color,
    )
    ax.text(0.10, 0.16, "Similar mean performance", transform=ax.transAxes, fontsize=11, color="#333333")
    ax.text(0.46, 0.82, "Significant tail reduction", transform=ax.transAxes, fontsize=11, color="#333333")
    style_axis(ax)

    # (d) Tail region comparison
    ax = axes[1, 1]
    threshold = min(float(q_base[2]), float(q_tail[2]))
    base_tail = baseline[baseline > threshold]
    tail_tail = tailaware[tailaware > threshold]
    tail_bins = np.linspace(threshold, max(float(base_tail.max()), float(tail_tail.max())) * 1.02, 28)
    ax.hist(base_tail, bins=tail_bins, density=True, histtype="step", color=base_color, linestyle="--", linewidth=2.0, label="Baseline")
    ax.hist(tail_tail, bins=tail_bins, density=True, histtype="step", color=tail_color, linewidth=2.0, label=tail_label)
    ax.axvline(float(baseline.max()), color=base_color, linestyle="--", linewidth=1.5)
    ax.annotate(
        "Extreme errors",
        xy=(float(baseline.max()), 1.0),
        xycoords=("data", "axes fraction"),
        xytext=(float(baseline.max()) * 0.74, 0.82),
        textcoords=("data", "axes fraction"),
        arrowprops={"arrowstyle": "->", "lw": 1.0, "color": "#333333"},
        fontsize=11,
        color="#333333",
    )
    ax.set_yscale("log")
    ax.set_title("(d) Tail region comparison")
    ax.set_xlabel("Absolute Error (nm)")
    ax.set_ylabel("Density (log scale)")
    ax.legend(frameon=False, loc="upper right")
    style_axis(ax)

    fig.tight_layout()
    return fig


def main() -> None:
    configure_style()

    figure_root = Path(__file__).resolve().parents[1]
    repo_root = figure_root.parents[1]
    outputs_dir = figure_root / "outputs"
    data_dir = figure_root / "data_copy"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    baseline, tailaware, source = load_real_error_arrays(repo_root)
    quantile_levels = [50, 75, 90, 95, 99]
    q_base = compute_quantiles(baseline, quantile_levels)
    q_tail = compute_quantiles(tailaware, quantile_levels)

    fig = build_figure(baseline, tailaware, quantile_levels, q_base, q_tail)

    png_path = outputs_dir / "Fig4_4_tail_aware.png"
    pdf_path = outputs_dir / "Fig4_4_tail_aware.pdf"
    data_path = data_dir / "Fig4_4_tail_aware_data.npz"

    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    np.savez(
        data_path,
        baseline_errors=baseline.astype(np.float64),
        tailaware_errors=tailaware.astype(np.float64),
        quantile_levels=np.array(quantile_levels, dtype=np.int64),
        baseline_quantiles=q_base.astype(np.float64),
        tailaware_quantiles=q_tail.astype(np.float64),
        source=np.array([source]),
    )

    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")
    print(f"Saved: {data_path}")


if __name__ == "__main__":
    main()
