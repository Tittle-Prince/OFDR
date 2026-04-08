from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


@dataclass
class FigureConfig:
    random_seed: int = 20260330
    dpi: int = 300
    figsize: tuple[float, float] = (7.2, 4.8)
    n_bins: int = 7


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.size": 10,
            "axes.labelsize": 10,
            "axes.titlesize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
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


def load_real_inputs(repo_root: Path) -> tuple[pd.DataFrame | None, dict[str, np.ndarray] | None]:
    calibration_path = (
        repo_root
        / "paper_figures"
        / "Fig4_7_real_experiment_reference_figure"
        / "data_copy"
        / "calibration_curve.csv"
    )
    prediction_path = (
        repo_root
        / "Real"
        / "stage2_ofdr_preexperiment_outputs"
        / "reference_clean"
        / "predictions.npz"
    )

    calibration_df = pd.read_csv(calibration_path) if calibration_path.exists() else None
    pred_dict = None
    if prediction_path.exists():
        with np.load(prediction_path) as data:
            pred_dict = {k: data[k] for k in data.files}

    return calibration_df, pred_dict


def build_placeholder_data(cfg: FigureConfig) -> tuple[pd.DataFrame, str]:
    rng = np.random.default_rng(cfg.random_seed)
    temperature = np.linspace(20.0, 40.0, 7)
    reference_nm = 0.00485 * (temperature - 20.0)
    baseline_mean = reference_nm + rng.normal(0.0, 0.00055, size=temperature.size)
    proposed_mean = reference_nm + rng.normal(0.0, 0.00035, size=temperature.size)
    baseline_std = np.linspace(0.00045, 0.00070, temperature.size)
    proposed_std = np.linspace(0.00035, 0.00055, temperature.size)
    reference_scatter = pd.DataFrame(
        {
            "temperature_C": np.repeat(temperature, 3),
            "reference_shift_nm": np.repeat(reference_nm, 3)
            + rng.normal(0.0, 0.00022, size=temperature.size * 3),
        }
    )

    df = pd.DataFrame(
        {
            "temperature_C": temperature,
            "reference_shift_nm": reference_nm,
            "reference_std_nm": np.full_like(reference_nm, 0.00022),
            "baseline_mean_nm": baseline_mean,
            "baseline_std_nm": baseline_std,
            "proposed_mean_nm": proposed_mean,
            "proposed_std_nm": proposed_std,
        }
    )
    df.attrs["reference_scatter"] = reference_scatter
    return df, "placeholder template"


def prepare_plot_dataframe(
    calibration_df: pd.DataFrame | None,
    pred_dict: dict[str, np.ndarray] | None,
    cfg: FigureConfig,
) -> tuple[pd.DataFrame, dict[str, float], str]:
    if calibration_df is None or pred_dict is None:
        df, mode = build_placeholder_data(cfg)
        summary = {
            "slope_nm_per_C": 0.00485,
            "intercept_nm": -0.0970,
        }
        return df, summary, mode

    calibration_nm = calibration_df.copy()
    calibration_nm["delta_lambda_nm"] = calibration_nm["delta_lambda_pm"] / 1000.0

    grouped_cal = calibration_nm.groupby("load_value", as_index=False).agg(
        reference_shift_nm=("delta_lambda_nm", "mean"),
        reference_std_nm=("delta_lambda_nm", "std"),
        reference_count=("delta_lambda_nm", "count"),
    )
    slope, intercept = np.polyfit(grouped_cal["load_value"], grouped_cal["reference_shift_nm"], 1)

    y_true = pred_dict["y_true_nm"].astype(np.float64)
    pred_baseline = pred_dict["pred_baseline_nm"].astype(np.float64)
    pred_proposed = pred_dict["pred_tail_nm"].astype(np.float64)

    # Use the real clean-scene prediction residuals, but anchor everything on the real calibration temperatures.
    baseline_residual = pred_baseline - y_true
    proposed_residual = pred_proposed - y_true
    sort_idx = np.argsort(y_true)
    group_indices = np.array_split(sort_idx, len(grouped_cal))

    rows: list[dict[str, float]] = []
    for i, (_, cal_row) in enumerate(grouped_cal.iterrows()):
        idx = group_indices[i]
        b_res = baseline_residual[idx]
        p_res = proposed_residual[idx]
        ref_mean = float(cal_row["reference_shift_nm"])
        rows.append(
            {
                "temperature_C": float(cal_row["load_value"]),
                "reference_shift_nm": ref_mean,
                "reference_std_nm": float(cal_row["reference_std_nm"]) if pd.notna(cal_row["reference_std_nm"]) else 0.0,
                "baseline_mean_nm": ref_mean + float(np.mean(b_res)),
                "baseline_std_nm": float(np.std(b_res, ddof=1)) if len(b_res) > 1 else 0.0,
                "proposed_mean_nm": ref_mean + float(np.mean(p_res)),
                "proposed_std_nm": float(np.std(p_res, ddof=1)) if len(p_res) > 1 else 0.0,
                "count": int(len(idx)),
            }
        )

    df = pd.DataFrame(rows)
    df.attrs["reference_scatter"] = calibration_nm.rename(
        columns={"load_value": "temperature_C", "delta_lambda_nm": "reference_shift_nm"}
    )[["temperature_C", "reference_shift_nm"]].copy()
    summary = {
        "slope_nm_per_C": float(slope),
        "intercept_nm": float(intercept),
    }
    return df, summary, "real data"


def export_data_copy(
    data_dir: Path,
    plot_df: pd.DataFrame,
    calibration_df: pd.DataFrame | None,
    pred_dict: dict[str, np.ndarray] | None,
) -> None:
    plot_df.to_csv(data_dir / "fig8_clean_grouped_curve.csv", index=False)

    if calibration_df is not None:
        calibration_df.to_csv(data_dir / "calibration_curve_source.csv", index=False)

    if pred_dict is not None:
        pred_export = pd.DataFrame(
            {
                "true_shift_nm": pred_dict["y_true_nm"],
                "baseline_shift_nm": pred_dict["pred_baseline_nm"],
                "proposed_shift_nm": pred_dict["pred_tail_nm"],
                "baseline_abs_error_nm": pred_dict["abs_err_baseline_nm"],
                "proposed_abs_error_nm": pred_dict["abs_err_tail_nm"],
            }
        )
        pred_export.to_csv(data_dir / "reference_clean_predictions_source.csv", index=False)


def plot_clean_result(
    ax: plt.Axes,
    df: pd.DataFrame,
    calibration_summary: dict[str, float],
) -> None:
    temp = df["temperature_C"].to_numpy()
    ref = df["reference_shift_nm"].to_numpy()
    ref_std = df["reference_std_nm"].to_numpy()
    base = df["baseline_mean_nm"].to_numpy()
    prop = df["proposed_mean_nm"].to_numpy()
    base_std = df["baseline_std_nm"].to_numpy()
    prop_std = df["proposed_std_nm"].to_numpy()
    reference_scatter = df.attrs.get("reference_scatter")

    fit_x = np.linspace(float(temp.min()) - 0.5, float(temp.max()) + 0.5, 300)
    ref_line = calibration_summary["slope_nm_per_C"] * fit_x + calibration_summary["intercept_nm"]

    if reference_scatter is not None and len(reference_scatter) > 0:
        ax.scatter(
            reference_scatter["temperature_C"].to_numpy(),
            reference_scatter["reference_shift_nm"].to_numpy(),
            color="black",
            s=13,
            alpha=0.40,
            zorder=2,
        )
    ax.plot(fit_x, ref_line, color="black", linewidth=1.1, linestyle="-", alpha=0.95, label="Reference")
    ax.errorbar(
        temp,
        ref,
        yerr=ref_std,
        color="black",
        linewidth=0.9,
        linestyle="none",
        marker="o",
        markersize=4.0,
        markerfacecolor="black",
        markeredgewidth=0.0,
        elinewidth=0.9,
        alpha=0.90,
        capsize=2.2,
        zorder=4,
    )

    ax.fill_between(temp, base - base_std, base + base_std, color="#7a7a7a", alpha=0.12, linewidth=0)
    ax.fill_between(temp, prop - prop_std, prop + prop_std, color="#1f4e79", alpha=0.10, linewidth=0)

    ax.errorbar(
        temp,
        base,
        yerr=base_std,
        color="#7a7a7a",
        linewidth=1.15,
        linestyle="--",
        marker="s",
        markersize=4.5,
        markerfacecolor="white",
        markeredgewidth=1.0,
        elinewidth=0.9,
        capsize=2.2,
        label="Baseline CNN",
        zorder=5,
    )
    ax.errorbar(
        temp,
        prop,
        yerr=prop_std,
        color="#1f4e79",
        linewidth=1.25,
        linestyle="-",
        marker="^",
        markersize=5.0,
        markerfacecolor="#1f4e79",
        markeredgewidth=0.8,
        elinewidth=0.9,
        capsize=2.2,
        label="Proposed CNN",
        zorder=6,
    )

    ax.set_xlabel("Temperature (°C)")
    ax.set_ylabel("Target wavelength shift (nm)")
    ax.grid(True, linestyle="--", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, loc="upper left")


def main() -> None:
    configure_style()
    cfg = FigureConfig()

    repo_root = Path(__file__).resolve().parents[3]
    figure_root = repo_root / "paper_figures" / "Fig8_clean_result"
    _, outputs_dir, data_dir = ensure_dirs(figure_root)

    calibration_df, pred_dict = load_real_inputs(repo_root)
    plot_df, calibration_summary, mode = prepare_plot_dataframe(calibration_df, pred_dict, cfg)
    export_data_copy(data_dir, plot_df, calibration_df, pred_dict)

    fig, ax = plt.subplots(figsize=cfg.figsize, constrained_layout=True)
    plot_clean_result(ax, plot_df, calibration_summary)

    png_path = outputs_dir / "fig8_clean_result.png"
    pdf_path = outputs_dir / "fig8_clean_result.pdf"
    fig.savefig(png_path, dpi=cfg.dpi, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    print(f"Data mode: {mode}")
    print(f"Saved PNG: {png_path}")
    print(f"Saved PDF: {pdf_path}")
    print("Figure 8 saved successfully.")


if __name__ == "__main__":
    main()
