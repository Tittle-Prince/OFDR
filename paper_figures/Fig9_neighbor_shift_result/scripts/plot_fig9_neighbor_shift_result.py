from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


@dataclass
class FigureConfig:
    dpi: int = 300
    figsize: tuple[float, float] = (10.2, 4.4)
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
            "legend.fontsize": 8.8,
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


def load_calibration_fit(repo_root: Path) -> tuple[float, float, pd.DataFrame | None]:
    calibration_path = (
        repo_root
        / "paper_figures"
        / "Fig4_7_real_experiment_reference_figure"
        / "data_copy"
        / "calibration_curve.csv"
    )
    if not calibration_path.exists():
        return 4.90076, -97.71426, None

    calibration_df = pd.read_csv(calibration_path)
    grouped = calibration_df.groupby("load_value", as_index=False)["delta_lambda_pm"].mean()
    slope_pm_per_c, intercept_pm = np.polyfit(grouped["load_value"], grouped["delta_lambda_pm"], 1)
    return float(slope_pm_per_c), float(intercept_pm), calibration_df


def load_temperature_combinations(repo_root: Path, cfg: FigureConfig) -> tuple[pd.DataFrame, str]:
    custom_path = repo_root / "paper_figures" / "Fig9_neighbor_shift_result" / "data_copy" / "neighbor_shift_temperature_combinations.csv"
    if custom_path.exists():
        return pd.read_csv(custom_path), "real data file"

    slope_pm_per_c, intercept_pm, _ = load_calibration_fit(repo_root)
    rng = np.random.default_rng(cfg.random_seed)

    # Advance the RNG through the reference_clean scenario to reproduce the exact neighbor_shift sampling order.
    for _ in range(cfg.num_cases_per_scenario):
        rng.uniform(-30.0, 30.0)

    rows: list[dict[str, float]] = []
    for exp_idx in range(1, cfg.num_cases_per_scenario + 1):
        target_pm = float(rng.uniform(-30.0, 30.0))
        left_pm = float(rng.uniform(-35.0, -6.0))
        right_pm = float(rng.uniform(6.0, 35.0))

        # Consume the remaining random draws used in sample_neighbor_shift_case so the sequence stays exact.
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
                "left_shift_pm": left_pm,
                "target_shift_pm": target_pm,
                "right_shift_pm": right_pm,
            }
        )

    return pd.DataFrame(rows), "reconstructed from real scenario generator"


def load_demod_results(repo_root: Path, cfg: FigureConfig) -> tuple[pd.DataFrame, str]:
    csv_path = repo_root / "paper_figures" / "Fig9_neighbor_shift_result" / "data_copy" / "neighbor_shift_results.csv"
    npz_path = repo_root / "Real" / "stage2_ofdr_preexperiment_outputs" / "neighbor_shift" / "predictions.npz"

    if csv_path.exists():
        return pd.read_csv(csv_path), "real data file"

    if npz_path.exists():
        with np.load(npz_path) as data:
            df = pd.DataFrame(
                {
                    "exp_idx": np.arange(1, data["y_true_nm"].shape[0] + 1),
                    "reference_shift_nm": data["y_true_nm"],
                    "baseline_shift_nm": data["pred_baseline_nm"],
                    "proposed_shift_nm": data["pred_tail_nm"],
                }
            )
        return df, "real predictions"

    # Placeholder template: fill these arrays with real values if no data file is available.
    exp_idx = np.arange(1, cfg.num_cases_per_scenario + 1)
    reference_shift_nm = np.zeros_like(exp_idx, dtype=np.float64)
    baseline_shift_nm = np.zeros_like(exp_idx, dtype=np.float64)
    proposed_shift_nm = np.zeros_like(exp_idx, dtype=np.float64)
    df = pd.DataFrame(
        {
            "exp_idx": exp_idx,
            "reference_shift_nm": reference_shift_nm,
            "baseline_shift_nm": baseline_shift_nm,
            "proposed_shift_nm": proposed_shift_nm,
        }
    )
    return df, "placeholder template"


def export_data_copy(data_dir: Path, combo_df: pd.DataFrame, result_df: pd.DataFrame) -> None:
    combo_df.to_csv(data_dir / "neighbor_shift_temperature_combinations.csv", index=False)
    result_df.to_csv(data_dir / "neighbor_shift_results.csv", index=False)


def plot_temperature_combinations(ax: plt.Axes, combo_df: pd.DataFrame) -> None:
    x = combo_df["exp_idx"].to_numpy()
    ax.plot(
        x,
        combo_df["temp_left"].to_numpy(),
        color="#8a8a8a",
        linewidth=1.0,
        marker="s",
        markersize=3.4,
        markerfacecolor="white",
        markeredgewidth=0.8,
        markevery=8,
        label="Left neighbor",
    )
    ax.plot(
        x,
        combo_df["temp_target"].to_numpy(),
        color="#1f4e79",
        linewidth=1.15,
        marker="o",
        markersize=3.2,
        markerfacecolor="#1f4e79",
        markeredgewidth=0.0,
        markevery=8,
        label="Target",
    )
    ax.plot(
        x,
        combo_df["temp_right"].to_numpy(),
        color="#5a5a5a",
        linewidth=1.0,
        linestyle="--",
        marker="^",
        markersize=3.6,
        markerfacecolor="white",
        markeredgewidth=0.8,
        markevery=8,
        label="Right neighbor",
    )
    ax.set_title("(a) Experimental temperature combinations", loc="left", pad=4)
    ax.set_xlabel("Experiment index")
    ax.set_ylabel("Temperature (°C)")
    ax.grid(True, linestyle="--", alpha=0.22)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, loc="upper right")


def plot_demod_results(ax: plt.Axes, result_df: pd.DataFrame) -> None:
    x = result_df["exp_idx"].to_numpy()
    ref = result_df["reference_shift_nm"].to_numpy()
    baseline = result_df["baseline_shift_nm"].to_numpy()
    proposed = result_df["proposed_shift_nm"].to_numpy()

    ax.plot(
        x,
        ref,
        color="black",
        linewidth=1.1,
        marker="o",
        markersize=3.0,
        markerfacecolor="black",
        markeredgewidth=0.0,
        markevery=7,
        alpha=0.90,
        label="Reference",
    )
    ax.plot(
        x,
        baseline,
        color="#8a8a8a",
        linewidth=1.1,
        linestyle="--",
        marker="s",
        markersize=3.6,
        markerfacecolor="white",
        markeredgewidth=0.8,
        markevery=7,
        alpha=0.95,
        label="Baseline CNN",
    )
    ax.plot(
        x,
        proposed,
        color="#1f4e79",
        linewidth=1.25,
        linestyle="-",
        marker="^",
        markersize=3.8,
        markerfacecolor="#1f4e79",
        markeredgewidth=0.0,
        markevery=7,
        alpha=0.95,
        label="Proposed CNN",
    )

    ax.set_title("(b) Demodulation results under neighbor shift", loc="left", pad=4)
    ax.set_xlabel("Experiment index")
    ax.set_ylabel("Target wavelength shift (nm)")
    ax.grid(True, linestyle="--", alpha=0.22)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, loc="upper right")


def main() -> None:
    configure_style()
    cfg = FigureConfig()

    repo_root = Path(__file__).resolve().parents[3]
    figure_root = repo_root / "paper_figures" / "Fig9_neighbor_shift_result"
    _, outputs_dir, data_dir = ensure_dirs(figure_root)

    combo_df, combo_mode = load_temperature_combinations(repo_root, cfg)
    result_df, result_mode = load_demod_results(repo_root, cfg)
    export_data_copy(data_dir, combo_df, result_df)

    fig, axes = plt.subplots(
        1,
        2,
        figsize=cfg.figsize,
        constrained_layout=True,
        gridspec_kw={"width_ratios": [1.0, 1.35]},
    )
    plot_temperature_combinations(axes[0], combo_df)
    plot_demod_results(axes[1], result_df)

    png_path = outputs_dir / "fig9_neighbor_shift_result.png"
    pdf_path = outputs_dir / "fig9_neighbor_shift_result.pdf"
    fig.savefig(png_path, dpi=cfg.dpi, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    print(f"Temperature data mode: {combo_mode}")
    print(f"Result data mode: {result_mode}")
    print(f"Saved PNG: {png_path}")
    print(f"Saved PDF: {pdf_path}")
    print("Figure 9 saved successfully.")


if __name__ == "__main__":
    main()
