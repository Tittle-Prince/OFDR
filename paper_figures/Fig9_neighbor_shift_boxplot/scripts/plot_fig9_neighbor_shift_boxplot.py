from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


@dataclass
class FigureConfig:
    dpi: int = 300
    figsize: tuple[float, float] = (9.6, 4.4)
    random_seed: int = 20260323
    num_cases_per_scenario: int = 120


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.size": 10,
            "axes.labelsize": 10,
            "axes.titlesize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 8.4,
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


def load_calibration_fit(repo_root: Path) -> tuple[float, float]:
    calibration_path = (
        repo_root
        / "paper_figures"
        / "Fig4_7_real_experiment_reference_figure"
        / "data_copy"
        / "calibration_curve.csv"
    )
    if not calibration_path.exists():
        return 4.90076, -97.71426

    calibration_df = pd.read_csv(calibration_path)
    grouped = calibration_df.groupby("load_value", as_index=False)["delta_lambda_pm"].mean()
    slope_pm_per_c, intercept_pm = np.polyfit(grouped["load_value"], grouped["delta_lambda_pm"], 1)
    return float(slope_pm_per_c), float(intercept_pm)


def reconstruct_neighbor_shift_temperatures(repo_root: Path, cfg: FigureConfig) -> pd.DataFrame:
    slope_pm_per_c, intercept_pm = load_calibration_fit(repo_root)
    rng = np.random.default_rng(cfg.random_seed)

    for _ in range(cfg.num_cases_per_scenario):
        rng.uniform(-30.0, 30.0)

    rows: list[dict[str, float]] = []
    for exp_idx in range(1, cfg.num_cases_per_scenario + 1):
        target_pm = float(rng.uniform(-30.0, 30.0))
        left_pm = float(rng.uniform(-35.0, -6.0))
        right_pm = float(rng.uniform(6.0, 35.0))

        rng.uniform(0.95, 1.05)
        rng.uniform(0.98, 1.02)
        rng.uniform(0.95, 1.05)
        rng.uniform(4.0, 9.0)

        rows.append(
            {
                "exp_idx": exp_idx,
                "temp_left": (left_pm - intercept_pm) / slope_pm_per_c,
                "temp_target": (target_pm - intercept_pm) / slope_pm_per_c,
                "temp_right": (right_pm - intercept_pm) / slope_pm_per_c,
            }
        )

    return pd.DataFrame(rows)


def load_neighbor_shift_results(repo_root: Path) -> pd.DataFrame:
    npz_path = repo_root / "Real" / "stage2_ofdr_preexperiment_outputs" / "neighbor_shift" / "predictions.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"Missing real prediction file: {npz_path}")

    with np.load(npz_path) as data:
        df = pd.DataFrame(
            {
                "exp_idx": np.arange(1, data["y_true_nm"].shape[0] + 1),
                "reference_shift_nm": data["y_true_nm"],
                "baseline_shift_nm": data["pred_baseline_nm"],
                "proposed_shift_nm": data["pred_tail_nm"],
            }
        )

    df["baseline_abs_error_nm"] = np.abs(df["baseline_shift_nm"] - df["reference_shift_nm"])
    df["proposed_abs_error_nm"] = np.abs(df["proposed_shift_nm"] - df["reference_shift_nm"])
    return df


def export_data_copy(data_dir: Path, combo_df: pd.DataFrame, result_df: pd.DataFrame) -> None:
    combo_df.to_csv(data_dir / "neighbor_shift_temperature_combinations.csv", index=False)
    result_df.to_csv(data_dir / "neighbor_shift_error_results.csv", index=False)


def compute_metrics(errors: np.ndarray) -> dict[str, float]:
    return {
        "MAE_nm": float(np.mean(errors)),
        "RMSE_nm": float(np.sqrt(np.mean(errors**2))),
        "P95_nm": float(np.percentile(errors, 95)),
    }


def plot_temperature_combinations(ax: plt.Axes, combo_df: pd.DataFrame) -> None:
    x = combo_df["exp_idx"].to_numpy()
    ax.plot(
        x,
        combo_df["temp_left"].to_numpy(),
        color="#8c8c8c",
        linewidth=0.95,
        marker="s",
        markersize=2.8,
        markerfacecolor="white",
        markeredgewidth=0.8,
        markevery=8,
        label="Left neighbor",
    )
    ax.plot(
        x,
        combo_df["temp_target"].to_numpy(),
        color="#1f4e79",
        linewidth=1.05,
        marker="o",
        markersize=2.6,
        markerfacecolor="#1f4e79",
        markeredgewidth=0.0,
        markevery=8,
        label="Target",
    )
    ax.plot(
        x,
        combo_df["temp_right"].to_numpy(),
        color="#5d5d5d",
        linewidth=0.95,
        linestyle="--",
        marker="^",
        markersize=3.0,
        markerfacecolor="white",
        markeredgewidth=0.8,
        markevery=8,
        label="Right neighbor",
    )
    ax.set_title("(a) Experimental temperature combinations", loc="left", pad=4)
    ax.set_xlabel("Experiment index")
    ax.set_ylabel("Temperature (°C)")
    ax.grid(True, linestyle="--", alpha=0.18)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, loc="upper right")


def plot_error_distribution(ax: plt.Axes, result_df: pd.DataFrame) -> None:
    baseline_err = result_df["baseline_abs_error_nm"].to_numpy()
    proposed_err = result_df["proposed_abs_error_nm"].to_numpy()

    baseline_color = "#9a9a9a"
    proposed_color = "#1f4e79"

    bp = ax.boxplot(
        [baseline_err, proposed_err],
        tick_labels=["Baseline CNN", "Proposed CNN"],
        patch_artist=True,
        widths=0.55,
        medianprops={"color": "#111111", "linewidth": 1.5},
        whiskerprops={"color": "#555555", "linewidth": 1.0},
        capprops={"color": "#555555", "linewidth": 1.0},
        flierprops={
            "marker": "o",
            "markersize": 3.0,
            "markerfacecolor": "#666666",
            "markeredgecolor": "#666666",
            "alpha": 0.65,
        },
    )
    for patch, color in zip(bp["boxes"], [baseline_color, proposed_color]):
        patch.set_facecolor(color)
        patch.set_alpha(0.35)
        patch.set_edgecolor(color)
        patch.set_linewidth(1.1)

    metrics_baseline = compute_metrics(baseline_err)
    metrics_proposed = compute_metrics(proposed_err)

    ymax = float(max(np.max(baseline_err), np.max(proposed_err)))
    ax.set_ylim(0.0, ymax * 1.22)

    for xpos, metrics in zip([1, 2], [metrics_baseline, metrics_proposed]):
        text = (
            f"MAE = {metrics['MAE_nm']:.4f} nm\n"
            f"RMSE = {metrics['RMSE_nm']:.4f} nm\n"
            f"P95 = {metrics['P95_nm']:.4f} nm"
        )
        ax.text(
            xpos,
            ymax * 1.03,
            text,
            ha="center",
            va="bottom",
            fontsize=8.0,
        )

    ax.set_title("(b) Error distribution under neighbor shift", loc="left", pad=4)
    ax.set_ylabel("Absolute error (nm)")
    ax.grid(True, axis="y", linestyle="--", alpha=0.18)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def main() -> None:
    configure_style()
    cfg = FigureConfig()

    repo_root = Path(__file__).resolve().parents[3]
    figure_root = repo_root / "paper_figures" / "Fig9_neighbor_shift_boxplot"
    _, outputs_dir, data_dir = ensure_dirs(figure_root)

    combo_df = reconstruct_neighbor_shift_temperatures(repo_root, cfg)
    result_df = load_neighbor_shift_results(repo_root)
    export_data_copy(data_dir, combo_df, result_df)

    fig, axes = plt.subplots(
        1,
        2,
        figsize=cfg.figsize,
        constrained_layout=True,
        gridspec_kw={"width_ratios": [1.0, 1.1]},
    )
    plot_temperature_combinations(axes[0], combo_df)
    plot_error_distribution(axes[1], result_df)

    png_path = outputs_dir / "fig9_neighbor_shift_boxplot.png"
    pdf_path = outputs_dir / "fig9_neighbor_shift_boxplot.pdf"
    fig.savefig(png_path, dpi=cfg.dpi, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    print("Figure 9 (boxplot version) saved successfully.")


if __name__ == "__main__":
    main()
