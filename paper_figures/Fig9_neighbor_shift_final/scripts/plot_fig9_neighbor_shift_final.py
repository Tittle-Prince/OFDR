from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


@dataclass
class FigureConfig:
    dpi: int = 300
    figsize: tuple[float, float] = (10.2, 4.9)
    random_seed: int = 20260323
    num_cases_per_scenario: int = 120
    downsample_step: int = 3


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.size": 10,
            "axes.labelsize": 10,
            "axes.titlesize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 8.6,
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

    # Advance through reference_clean first, following the original stage-2 generation order.
    for _ in range(cfg.num_cases_per_scenario):
        rng.uniform(-30.0, 30.0)

    rows: list[dict[str, float]] = []
    for exp_idx in range(1, cfg.num_cases_per_scenario + 1):
        target_pm = float(rng.uniform(-30.0, 30.0))
        left_pm = float(rng.uniform(-35.0, -6.0))
        right_pm = float(rng.uniform(6.0, 35.0))

        # Consume the remaining random draws used in sample_neighbor_shift_case.
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


def downsample_for_display(combo_df: pd.DataFrame, result_df: pd.DataFrame, cfg: FigureConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    idx = np.arange(0, len(result_df), cfg.downsample_step)
    return combo_df.iloc[idx].reset_index(drop=True), result_df.iloc[idx].reset_index(drop=True)


def export_data_copy(data_dir: Path, combo_df: pd.DataFrame, result_df: pd.DataFrame) -> None:
    combo_df.to_csv(data_dir / "neighbor_shift_temperature_combinations_display.csv", index=False)
    result_df.to_csv(data_dir / "neighbor_shift_results_display.csv", index=False)


def plot_temperature_combinations(ax: plt.Axes, combo_df: pd.DataFrame) -> None:
    x = combo_df["exp_idx"].to_numpy()
    ax.plot(
        x,
        combo_df["temp_left"].to_numpy(),
        color="#8c8c8c",
        linewidth=0.95,
        marker="s",
        markersize=3.0,
        markerfacecolor="white",
        markeredgewidth=0.8,
        markevery=3,
        label="Left neighbor",
    )
    ax.plot(
        x,
        combo_df["temp_target"].to_numpy(),
        color="#1f4e79",
        linewidth=1.05,
        marker="o",
        markersize=2.8,
        markerfacecolor="#1f4e79",
        markeredgewidth=0.0,
        markevery=3,
        label="Target",
    )
    ax.plot(
        x,
        combo_df["temp_right"].to_numpy(),
        color="#5d5d5d",
        linewidth=0.95,
        linestyle="--",
        marker="^",
        markersize=3.2,
        markerfacecolor="white",
        markeredgewidth=0.8,
        markevery=3,
        label="Right neighbor",
    )
    ax.set_title("(a) Experimental temperature combinations", loc="left", pad=4)
    ax.set_xlabel("Experiment index")
    ax.set_ylabel("Temperature (°C)")
    ax.grid(True, linestyle="--", alpha=0.18)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, loc="upper right")


def plot_demod_results(ax_top: plt.Axes, ax_bottom: plt.Axes, result_df: pd.DataFrame) -> None:
    x = result_df["exp_idx"].to_numpy()
    ref = result_df["reference_shift_nm"].to_numpy()
    baseline = result_df["baseline_shift_nm"].to_numpy()
    proposed = result_df["proposed_shift_nm"].to_numpy()
    baseline_err = result_df["baseline_abs_error_nm"].to_numpy()
    proposed_err = result_df["proposed_abs_error_nm"].to_numpy()

    ax_top.plot(
        x,
        ref,
        color="black",
        linewidth=1.35,
        marker="o",
        markersize=2.8,
        markerfacecolor="black",
        markeredgewidth=0.0,
        markevery=4,
        label="Reference",
        zorder=4,
    )
    ax_top.plot(
        x,
        baseline,
        color="#8c8c8c",
        linewidth=1.05,
        linestyle="--",
        marker="s",
        markersize=3.0,
        markerfacecolor="white",
        markeredgewidth=0.8,
        markevery=4,
        label="Baseline CNN",
        zorder=3,
    )
    ax_top.plot(
        x,
        proposed,
        color="#1f4e79",
        linewidth=1.25,
        linestyle="-",
        marker="^",
        markersize=3.2,
        markerfacecolor="#1f4e79",
        markeredgewidth=0.0,
        markevery=4,
        label="Proposed CNN",
        zorder=5,
    )
    ax_top.set_title("(b) Demodulation results under neighbor shift", loc="left", pad=4)
    ax_top.set_ylabel("Target wavelength shift (nm)")
    ax_top.grid(True, linestyle="--", alpha=0.18)
    ax_top.spines["top"].set_visible(False)
    ax_top.spines["right"].set_visible(False)
    ax_top.legend(frameon=False, loc="upper right")
    ax_top.tick_params(labelbottom=False)

    ax_bottom.plot(x, baseline_err, color="#8c8c8c", linewidth=1.15, linestyle="--", label="|Baseline - Reference|")
    ax_bottom.plot(x, proposed_err, color="#1f4e79", linewidth=1.25, linestyle="-", label="|Proposed - Reference|")
    ax_bottom.set_xlabel("Experiment index")
    ax_bottom.set_ylabel("Abs. error (nm)")
    ax_bottom.grid(True, linestyle="--", alpha=0.18)
    ax_bottom.spines["top"].set_visible(False)
    ax_bottom.spines["right"].set_visible(False)


def main() -> None:
    configure_style()
    cfg = FigureConfig()

    repo_root = Path(__file__).resolve().parents[3]
    figure_root = repo_root / "paper_figures" / "Fig9_neighbor_shift_final"
    _, outputs_dir, data_dir = ensure_dirs(figure_root)

    combo_full = reconstruct_neighbor_shift_temperatures(repo_root, cfg)
    result_full = load_neighbor_shift_results(repo_root)
    combo_df, result_df = downsample_for_display(combo_full, result_full, cfg)
    export_data_copy(data_dir, combo_df, result_df)

    fig = plt.figure(figsize=cfg.figsize, constrained_layout=True)
    outer = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.55])
    ax_a = fig.add_subplot(outer[0, 0])
    right = outer[0, 1].subgridspec(2, 1, height_ratios=[2.1, 1.0], hspace=0.03)
    ax_b_top = fig.add_subplot(right[0, 0])
    ax_b_bottom = fig.add_subplot(right[1, 0], sharex=ax_b_top)

    plot_temperature_combinations(ax_a, combo_df)
    plot_demod_results(ax_b_top, ax_b_bottom, result_df)

    png_path = outputs_dir / "fig9_neighbor_shift_final.png"
    pdf_path = outputs_dir / "fig9_neighbor_shift_final.pdf"
    fig.savefig(png_path, dpi=cfg.dpi, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    print("Figure 9 saved successfully.")


if __name__ == "__main__":
    main()
