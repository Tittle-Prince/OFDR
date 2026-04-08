from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
import sys

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Real.stage1_three_grating_ideal_ofdr_bridge import BridgeConfig, load_local_grid


@dataclass
class FigureConfig:
    dpi: int = 300
    figsize: tuple[float, float] = (11.0, 8.0)
    paired_line_count_each_side: int = 12
    failure_case_count: int = 4
    jitter_seed: int = 20260331


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.size": 10,
            "axes.labelsize": 10,
            "axes.titlesize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 8.5,
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


def load_predictions(repo_root: Path) -> dict[str, np.ndarray]:
    npz_path = repo_root / "Real" / "stage2_ofdr_preexperiment_outputs" / "combined_hard" / "predictions.npz"
    if not npz_path.exists():
        raise FileNotFoundError(
            f"Missing real combined_hard prediction file: {npz_path}\n"
            "Expected arrays: y_true_nm, pred_baseline_nm, pred_tail_nm, X_bridge."
        )

    with np.load(npz_path) as data:
        return {k: data[k] for k in data.files}


def compute_metrics(errors: np.ndarray) -> dict[str, float]:
    return {
        "MAE_nm": float(np.mean(errors)),
        "RMSE_nm": float(np.sqrt(np.mean(errors**2))),
        "P95_nm": float(np.percentile(errors, 95)),
        "P99_nm": float(np.percentile(errors, 99)),
        "Median_nm": float(np.median(errors)),
    }


def build_result_dataframe(data: dict[str, np.ndarray]) -> pd.DataFrame:
    ref = data["y_true_nm"].astype(np.float64)
    baseline = data["pred_baseline_nm"].astype(np.float64)
    proposed = data["pred_tail_nm"].astype(np.float64)
    err_baseline = np.abs(baseline - ref)
    err_proposed = np.abs(proposed - ref)
    df = pd.DataFrame(
        {
            "sample_id": np.arange(ref.size, dtype=int),
            "reference_shift_nm": ref,
            "baseline_shift_nm": baseline,
            "proposed_shift_nm": proposed,
            "baseline_abs_error_nm": err_baseline,
            "proposed_abs_error_nm": err_proposed,
            "max_abs_error_nm": np.maximum(err_baseline, err_proposed),
            "delta_error_nm": err_proposed - err_baseline,
        }
    )
    return df


def select_paired_subset(df: pd.DataFrame, n_each_side: int) -> np.ndarray:
    better_for_baseline = df[df["delta_error_nm"] > 0.0].sort_values("max_abs_error_nm", ascending=False)
    better_for_proposed = df[df["delta_error_nm"] < 0.0].sort_values("max_abs_error_nm", ascending=False)

    idx = list(better_for_baseline["sample_id"].head(n_each_side))
    idx += list(better_for_proposed["sample_id"].head(n_each_side))
    if not idx:
        return np.array([], dtype=int)
    return np.array(idx[: 2 * n_each_side], dtype=int)


def select_failure_cases(df: pd.DataFrame, n_cases: int) -> np.ndarray:
    high_tail = df.sort_values("max_abs_error_nm", ascending=False)
    selected: list[int] = []

    # 1) top two globally worst cases
    for idx in high_tail["sample_id"].tolist():
        if idx not in selected:
            selected.append(int(idx))
        if len(selected) >= min(2, n_cases):
            break

    # 2) strongest cases where proposed is better
    prop_better = df[df["delta_error_nm"] < 0.0].sort_values("max_abs_error_nm", ascending=False)
    for idx in prop_better["sample_id"].tolist():
        if idx not in selected:
            selected.append(int(idx))
        if len(selected) >= min(3, n_cases):
            break

    # 3) additional diverse high-error cases
    for idx in high_tail["sample_id"].tolist():
        if idx not in selected:
            selected.append(int(idx))
        if len(selected) >= n_cases:
            break

    return np.array(selected[:n_cases], dtype=int)


def export_data_copy(
    data_dir: Path,
    df: pd.DataFrame,
    paired_idx: np.ndarray,
    failure_idx: np.ndarray,
    X_bridge: np.ndarray,
) -> None:
    df.to_csv(data_dir / "combined_hard_predictions.csv", index=False)
    df[["baseline_abs_error_nm"]].rename(columns={"baseline_abs_error_nm": "abs_error_nm"}).to_csv(
        data_dir / "combined_hard_errors_baseline.csv", index=False
    )
    df[["proposed_abs_error_nm"]].rename(columns={"proposed_abs_error_nm": "abs_error_nm"}).to_csv(
        data_dir / "combined_hard_errors_proposed.csv", index=False
    )
    df[df["sample_id"].isin(failure_idx)][
        [
            "sample_id",
            "reference_shift_nm",
            "baseline_shift_nm",
            "proposed_shift_nm",
            "baseline_abs_error_nm",
            "proposed_abs_error_nm",
        ]
    ].to_csv(data_dir / "representative_failure_cases.csv", index=False)

    rows: list[dict[str, float]] = []
    for sample_id in failure_idx:
        spectrum = X_bridge[sample_id]
        for sample_index, value in enumerate(spectrum):
            rows.append(
                {
                    "sample_id": int(sample_id),
                    "local_sample_index": int(sample_index),
                    "normalized_intensity": float(value),
                }
            )
    pd.DataFrame(rows).to_csv(data_dir / "representative_failure_spectra.csv", index=False)

    pd.DataFrame({"sample_id": paired_idx.astype(int)}).to_csv(data_dir / "paired_subset_indices.csv", index=False)


def style_axes(ax: plt.Axes, grid_axis: str = "both") -> None:
    ax.grid(True, axis=grid_axis, linestyle="--", alpha=0.18)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_global_boxplot(ax: plt.Axes, df: pd.DataFrame) -> None:
    baseline_color = "#9a9a9a"
    proposed_color = "#1f4e79"

    baseline_err = df["baseline_abs_error_nm"].to_numpy()
    proposed_err = df["proposed_abs_error_nm"].to_numpy()
    bp = ax.boxplot(
        [baseline_err, proposed_err],
        tick_labels=["Baseline CNN", "Proposed CNN"],
        patch_artist=True,
        widths=0.58,
        medianprops={"color": "black", "linewidth": 1.7},
        whiskerprops={"color": "#555555", "linewidth": 1.0},
        capprops={"color": "#555555", "linewidth": 1.0},
        flierprops={
            "marker": "o",
            "markersize": 3.0,
            "markerfacecolor": "#666666",
            "markeredgecolor": "#666666",
            "alpha": 0.5,
        },
    )
    for patch, color in zip(bp["boxes"], [baseline_color, proposed_color]):
        patch.set_facecolor(color)
        patch.set_edgecolor(color)
        patch.set_alpha(0.35)
        patch.set_linewidth(1.1)

    mb = compute_metrics(baseline_err)
    mp = compute_metrics(proposed_err)
    ymax = float(max(np.max(baseline_err), np.max(proposed_err)))
    ax.set_ylim(0.0, ymax * 1.32)

    ax.text(
        0.02,
        0.98,
        (
            f"Baseline: RMSE={mb['RMSE_nm']:.4f} nm, MAE={mb['MAE_nm']:.4f} nm\n"
            f"Proposed: RMSE={mp['RMSE_nm']:.4f} nm, MAE={mp['MAE_nm']:.4f} nm"
        ),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.2,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "#d0d0d0", "alpha": 0.92},
    )

    ax.set_title("(a) Global error distribution", loc="left", pad=4)
    ax.set_ylabel("Absolute error (nm)")
    style_axes(ax, grid_axis="y")


def plot_cdf(ax: plt.Axes, df: pd.DataFrame) -> None:
    baseline_err = np.sort(df["baseline_abs_error_nm"].to_numpy())
    proposed_err = np.sort(df["proposed_abs_error_nm"].to_numpy())
    y = np.arange(1, baseline_err.size + 1, dtype=np.float64) / baseline_err.size

    ax.plot(baseline_err, y, color="#8a8a8a", linestyle="--", linewidth=1.4, label="Baseline CNN")
    ax.plot(proposed_err, y, color="#1f4e79", linestyle="-", linewidth=1.5, label="Proposed CNN")

    mb = compute_metrics(baseline_err)
    mp = compute_metrics(proposed_err)
    ax.axvline(mb["P95_nm"], color="#b5b5b5", linestyle="--", linewidth=0.9, alpha=0.9)
    ax.axvline(mp["P95_nm"], color="#5f88b1", linestyle="--", linewidth=0.9, alpha=0.9)
    ax.axvline(mb["P99_nm"], color="#8f8f8f", linestyle=":", linewidth=0.95, alpha=0.9)
    ax.axvline(mp["P99_nm"], color="#1f4e79", linestyle=":", linewidth=0.95, alpha=0.9)

    ax.text(
        0.03,
        0.06,
        (
            f"Baseline P95/P99 = {mb['P95_nm']:.4f}/{mb['P99_nm']:.4f} nm\n"
            f"Proposed P95/P99 = {mp['P95_nm']:.4f}/{mp['P99_nm']:.4f} nm"
        ),
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.0,
        bbox={"boxstyle": "round,pad=0.22", "facecolor": "white", "edgecolor": "#d0d0d0", "alpha": 0.9},
    )

    ax.set_title("(b) Cumulative error distribution", loc="left", pad=4)
    ax.set_xlabel("Absolute error (nm)")
    ax.set_ylabel("Cumulative probability")
    ax.set_xlim(0.0, max(float(baseline_err.max()), float(proposed_err.max())) * 1.05)
    ax.set_ylim(0.0, 1.0)
    style_axes(ax, grid_axis="both")
    ax.legend(frameon=False, loc="lower right")


def plot_paired_comparison(ax: plt.Axes, df: pd.DataFrame, paired_idx: np.ndarray, cfg: FigureConfig) -> None:
    rng = np.random.default_rng(cfg.jitter_seed)
    baseline_err = df["baseline_abs_error_nm"].to_numpy()
    proposed_err = df["proposed_abs_error_nm"].to_numpy()

    x1 = 1.0 + rng.uniform(-0.06, 0.06, size=baseline_err.size)
    x2 = 2.0 + rng.uniform(-0.06, 0.06, size=proposed_err.size)
    ax.scatter(
        x1,
        baseline_err,
        s=18,
        facecolors="white",
        edgecolors="#8c8c8c",
        linewidths=0.8,
        alpha=0.85,
        label="Baseline CNN",
        zorder=3,
    )
    ax.scatter(
        x2,
        proposed_err,
        s=20,
        facecolors="#1f4e79",
        edgecolors="#1f4e79",
        linewidths=0.3,
        alpha=0.9,
        label="Proposed CNN",
        zorder=4,
    )

    subset = df[df["sample_id"].isin(paired_idx)].copy()
    for _, row in subset.iterrows():
        ax.plot([1, 2], [row["baseline_abs_error_nm"], row["proposed_abs_error_nm"]], color="#c6c6c6", linewidth=0.75, alpha=0.4, zorder=1)

    med_b = float(np.median(baseline_err))
    med_p = float(np.median(proposed_err))
    ax.hlines(med_b, xmin=0.78, xmax=1.22, colors="#555555", linewidth=1.6, zorder=5)
    ax.hlines(med_p, xmin=1.78, xmax=2.22, colors="#0f3760", linewidth=1.6, zorder=5)

    ax.text(
        0.98,
        0.96,
        f"Median = {med_b:.4f} nm",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8.0,
        color="#555555",
    )
    ax.text(
        0.98,
        0.88,
        f"Median = {med_p:.4f} nm",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8.0,
        color="#1f4e79",
    )

    ax.set_title("(c) Paired sample-wise comparison", loc="left", pad=4)
    ax.set_ylabel("Absolute error (nm)")
    ax.set_xticks([1, 2])
    ax.set_xticklabels(["Baseline CNN", "Proposed CNN"])
    style_axes(ax, grid_axis="y")
    ax.legend(frameon=False, loc="upper left")


def shift_to_local_index(shift_nm: float, local_grid_nm: np.ndarray, nominal_center_nm: float = 1550.0) -> float:
    center_nm = nominal_center_nm + shift_nm
    return float(np.interp(center_nm, local_grid_nm, np.arange(local_grid_nm.size, dtype=np.float64)))


def plot_failure_cases(ax: plt.Axes, df: pd.DataFrame, X_bridge: np.ndarray, failure_idx: np.ndarray, local_grid_nm: np.ndarray) -> None:
    offsets = np.linspace(0.0, 1.8, failure_idx.size)
    legend_done = False

    for order, (sample_id, offset) in enumerate(zip(failure_idx, offsets), start=1):
        row = df.loc[df["sample_id"] == sample_id].iloc[0]
        x = np.arange(X_bridge.shape[1], dtype=np.int32)
        y = X_bridge[sample_id] + offset

        ax.plot(x, y, color="#202020", linewidth=1.15, zorder=2)

        idx_ref = shift_to_local_index(float(row["reference_shift_nm"]), local_grid_nm)
        idx_baseline = shift_to_local_index(float(row["baseline_shift_nm"]), local_grid_nm)
        idx_proposed = shift_to_local_index(float(row["proposed_shift_nm"]), local_grid_nm)

        y0 = offset - 0.02
        y1 = offset + 1.02
        ax.vlines(idx_ref, y0, y1, colors="black", linestyles="--", linewidth=0.95, label="Reference" if not legend_done else None)
        ax.vlines(idx_baseline, y0, y1, colors="#8c8c8c", linestyles="-.", linewidth=0.95, label="Baseline CNN" if not legend_done else None)
        ax.vlines(idx_proposed, y0, y1, colors="#1f4e79", linestyles="--", linewidth=1.0, label="Proposed CNN" if not legend_done else None)
        legend_done = True

        ax.text(
            x[-1] + 8,
            offset + 0.76,
            f"Case {order}\nEb={row['baseline_abs_error_nm']:.3f}\nEp={row['proposed_abs_error_nm']:.3f}",
            ha="left",
            va="center",
            fontsize=7.8,
            color="#333333",
        )

    ax.set_title("(d) Representative failure cases", loc="left", pad=4)
    ax.set_xlabel("Local sample index")
    ax.set_ylabel("Normalized intensity + offset")
    ax.set_xlim(0, X_bridge.shape[1] - 1)
    ax.set_ylim(-0.05, offsets[-1] + 1.15)
    style_axes(ax, grid_axis="both")
    ax.legend(frameon=False, loc="upper left")


def main() -> None:
    configure_style()
    cfg = FigureConfig()

    repo_root = Path(__file__).resolve().parents[3]
    figure_root = repo_root / "paper_figures" / "Fig11_combined_hard"
    _, outputs_dir, data_dir = ensure_dirs(figure_root)

    raw = load_predictions(repo_root)
    df = build_result_dataframe(raw)
    paired_idx = select_paired_subset(df, cfg.paired_line_count_each_side)
    failure_idx = select_failure_cases(df, cfg.failure_case_count)

    local_grid_nm = load_local_grid(BridgeConfig()) * 1e9
    X_bridge = raw["X_bridge"][:, 0, :].astype(np.float64)

    export_data_copy(data_dir, df, paired_idx, failure_idx, X_bridge)

    fig, axes = plt.subplots(2, 2, figsize=cfg.figsize, constrained_layout=True)
    plot_global_boxplot(axes[0, 0], df)
    plot_cdf(axes[0, 1], df)
    plot_paired_comparison(axes[1, 0], df, paired_idx, cfg)
    plot_failure_cases(axes[1, 1], df, X_bridge, failure_idx, local_grid_nm)

    png_path = outputs_dir / "fig11_combined_hard.png"
    pdf_path = outputs_dir / "fig11_combined_hard.pdf"
    fig.savefig(png_path, dpi=cfg.dpi, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    print("Figure 11 saved successfully.")


if __name__ == "__main__":
    main()
