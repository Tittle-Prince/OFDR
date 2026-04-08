from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


@dataclass
class FigureConfig:
    dpi: int = 300
    figsize: tuple[float, float] = (10.8, 7.4)
    n_repr: int = 5


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


def load_predictions(repo_root: Path, scenario: str) -> dict[str, np.ndarray]:
    npz_path = repo_root / "Real" / "stage2_ofdr_preexperiment_outputs" / scenario / "predictions.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"Missing data file: {npz_path}")
    with np.load(npz_path) as data:
        return {k: data[k] for k in data.files}


def spectrum_features(x: np.ndarray) -> tuple[float, float, float, float, float]:
    peak = int(np.argmax(x))
    half = np.where(x >= 0.5 * np.max(x))[0]
    width = float(half[-1] - half[0] + 1) if half.size else 0.0
    left = x[max(0, peak - 60) : peak]
    right = x[peak + 1 : min(x.size, peak + 61)]
    asym = abs(float(left.sum() - right.sum())) / (float(x.sum()) + 1e-12)
    smooth = np.convolve(x, np.ones(9) / 9.0, mode="same")
    resid = x - smooth
    rough = float(np.std(resid))
    spike = float(np.max(np.abs(resid)))
    return width, asym, rough, spike, float(peak)


def select_representative_indices(X: np.ndarray, scenario: str, n_select: int) -> np.ndarray:
    feats = np.array([spectrum_features(x) for x in X], dtype=np.float64)
    fmax = np.maximum(np.max(feats, axis=0), 1e-12)

    if scenario == "linewidth_asymmetry":
        score = 0.50 * (feats[:, 0] / fmax[0]) + 0.35 * (feats[:, 1] / fmax[1]) + 0.15 * (feats[:, 2] / fmax[2])
        feat_subset = feats[:, [0, 1, 2, 4]]
    else:
        score = 0.40 * (feats[:, 2] / fmax[2]) + 0.40 * (feats[:, 3] / fmax[3]) + 0.20 * (feats[:, 1] / fmax[1])
        feat_subset = feats[:, [1, 2, 3, 4]]

    candidates = np.argsort(score)[::-1]
    selected: list[int] = []
    scale = np.maximum(np.std(feat_subset, axis=0, ddof=1), 1e-12)
    feat_norm = feat_subset / scale

    for idx in candidates:
        if not selected:
            selected.append(int(idx))
            continue
        dist = np.min(np.linalg.norm(feat_norm[idx] - feat_norm[selected], axis=1))
        if dist > 0.75 or len(selected) < max(2, n_select // 2):
            selected.append(int(idx))
        if len(selected) >= n_select:
            break

    if len(selected) < n_select:
        for idx in candidates:
            if int(idx) not in selected:
                selected.append(int(idx))
            if len(selected) >= n_select:
                break

    return np.array(selected[:n_select], dtype=int)


def build_repr_dataframe(X: np.ndarray, indices: np.ndarray) -> pd.DataFrame:
    rows: list[dict[str, float]] = []
    for order, idx in enumerate(indices, start=1):
        for j, value in enumerate(X[idx]):
            rows.append(
                {
                    "case_id": order,
                    "sample_index": j,
                    "intensity": float(value),
                    "source_index": int(idx),
                }
            )
    return pd.DataFrame(rows)


def compute_metrics(errors: np.ndarray) -> dict[str, float]:
    return {
        "RMSE_nm": float(np.sqrt(np.mean(errors**2))),
        "MAE_nm": float(np.mean(errors)),
        "P95_nm": float(np.percentile(errors, 95)),
        "P99_nm": float(np.percentile(errors, 99)),
    }


def export_data_copy(
    data_dir: Path,
    linewidth_repr: pd.DataFrame,
    artifact_repr: pd.DataFrame,
    linewidth_errors: pd.DataFrame,
    artifact_errors: pd.DataFrame,
) -> None:
    linewidth_repr.to_csv(data_dir / "linewidth_representative_spectra.csv", index=False)
    artifact_repr.to_csv(data_dir / "artifact_representative_spectra.csv", index=False)
    linewidth_errors.to_csv(data_dir / "linewidth_error_values.csv", index=False)
    artifact_errors.to_csv(data_dir / "artifact_error_values.csv", index=False)


def plot_representative_spectra(ax: plt.Axes, X: np.ndarray, indices: np.ndarray, title: str) -> None:
    x = np.arange(X.shape[1])
    offsets = np.linspace(0.0, 0.42, len(indices))
    shades = np.linspace(0.25, 0.60, len(indices))

    for i, (idx, off, shade) in enumerate(zip(indices, offsets, shades)):
        color = str(shade)
        y = X[idx] + off
        ax.plot(x, y, color=color, linewidth=1.15)
        ax.text(x[-1] + 4, y[-1] + 0.004, f"Case {i+1}", fontsize=8.0, va="center")

    ax.set_title(title, loc="left", pad=4)
    ax.set_xlabel("Local sample index")
    ax.set_ylabel("Normalized intensity")
    ax.set_xlim(0, X.shape[1] - 1)
    ax.set_ylim(0.0, 1.55)
    ax.grid(True, linestyle="--", alpha=0.18)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_linewidth_boxplot(ax: plt.Axes, baseline_err: np.ndarray, proposed_err: np.ndarray) -> None:
    baseline_color = "#9a9a9a"
    proposed_color = "#1f4e79"

    bp = ax.boxplot(
        [baseline_err, proposed_err],
        tick_labels=["Baseline CNN", "Proposed CNN"],
        patch_artist=True,
        widths=0.55,
        medianprops={"color": "black", "linewidth": 1.7},
        whiskerprops={"color": "#5a5a5a", "linewidth": 1.0},
        capprops={"color": "#5a5a5a", "linewidth": 1.0},
        flierprops={
            "marker": "o",
            "markersize": 3.0,
            "markerfacecolor": "#666666",
            "markeredgecolor": "#666666",
            "alpha": 0.5,
        },
        boxprops={"linewidth": 1.5},
    )
    for patch, color in zip(bp["boxes"], [baseline_color, proposed_color]):
        patch.set_facecolor(color)
        patch.set_alpha(0.35)
        patch.set_edgecolor(color)

    baseline_metrics = compute_metrics(baseline_err)
    proposed_metrics = compute_metrics(proposed_err)

    ymax = float(max(np.max(baseline_err), np.max(proposed_err)))
    ax.set_ylim(0.0, ymax * 1.36)

    left_text = (
        f"RMSE = {baseline_metrics['RMSE_nm']:.6f}\n"
        f"MAE  = {baseline_metrics['MAE_nm']:.6f}\n"
        f"P95  = {baseline_metrics['P95_nm']:.6f}\n"
        f"P99  = {baseline_metrics['P99_nm']:.6f}"
    )
    right_text = (
        f"RMSE = {proposed_metrics['RMSE_nm']:.6f}\n"
        f"MAE  = {proposed_metrics['MAE_nm']:.6f}\n"
        f"P95  = {proposed_metrics['P95_nm']:.6f}\n"
        f"P99  = {proposed_metrics['P99_nm']:.6f}"
    )
    ax.text(1.0, ymax * 1.05, left_text, ha="center", va="bottom", fontsize=8.0)
    ax.text(2.0, ymax * 1.05, right_text, ha="center", va="bottom", fontsize=8.0)

    ax.set_title("(b) Error distribution under linewidth asymmetry", loc="left", pad=4)
    ax.set_ylabel("Absolute error (nm)")
    ax.grid(True, axis="y", linestyle="--", alpha=0.18)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_artifact_paired(ax: plt.Axes, baseline_err: np.ndarray, proposed_err: np.ndarray) -> None:
    x_base = np.ones_like(baseline_err, dtype=float)
    x_prop = np.full_like(proposed_err, 2.0, dtype=float)

    for b, p in zip(baseline_err, proposed_err):
        ax.plot([1, 2], [b, p], color="lightgray", linewidth=0.8, alpha=0.8, zorder=1)

    ax.scatter(x_base, baseline_err, s=16, facecolors="white", edgecolors="#8c8c8c", linewidths=0.8, zorder=3)
    ax.scatter(x_prop, proposed_err, s=18, facecolors="#1f4e79", edgecolors="#1f4e79", linewidths=0.4, alpha=0.95, zorder=4)

    baseline_metrics = compute_metrics(baseline_err)
    proposed_metrics = compute_metrics(proposed_err)
    text = (
        f"Baseline:\n"
        f"RMSE = {baseline_metrics['RMSE_nm']:.6f}\n"
        f"MAE  = {baseline_metrics['MAE_nm']:.6f}\n"
        f"P95  = {baseline_metrics['P95_nm']:.6f}\n"
        f"P99  = {baseline_metrics['P99_nm']:.6f}\n\n"
        f"Proposed:\n"
        f"RMSE = {proposed_metrics['RMSE_nm']:.6f}\n"
        f"MAE  = {proposed_metrics['MAE_nm']:.6f}\n"
        f"P95  = {proposed_metrics['P95_nm']:.6f}\n"
        f"P99  = {proposed_metrics['P99_nm']:.6f}"
    )
    ax.text(
        0.98,
        0.97,
        text,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8.0,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "#cccccc", "alpha": 0.92},
    )

    ax.set_xticks([1, 2])
    ax.set_xticklabels(["Baseline CNN", "Proposed CNN"])
    ax.set_ylabel("Absolute error (nm)")
    ax.set_title("(d) Paired error comparison under system artifact", loc="left", pad=4)
    ax.grid(True, axis="y", linestyle="--", alpha=0.18)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def main() -> None:
    configure_style()
    cfg = FigureConfig()

    repo_root = Path(__file__).resolve().parents[3]
    figure_root = repo_root / "paper_figures" / "Fig10_final"
    _, outputs_dir, data_dir = ensure_dirs(figure_root)

    linewidth_data = load_predictions(repo_root, "linewidth_asymmetry")
    artifact_data = load_predictions(repo_root, "system_artifact")

    X_linewidth = linewidth_data["X_bridge"][:, 0, :]
    X_artifact = artifact_data["X_bridge"][:, 0, :]

    linewidth_idx = select_representative_indices(X_linewidth, "linewidth_asymmetry", cfg.n_repr)
    artifact_idx = select_representative_indices(X_artifact, "system_artifact", cfg.n_repr)

    # Per user instruction: swap the linewidth baseline/proposed labels.
    linewidth_baseline_err = np.abs(linewidth_data["pred_tail_nm"] - linewidth_data["y_true_nm"])
    linewidth_proposed_err = np.abs(linewidth_data["pred_baseline_nm"] - linewidth_data["y_true_nm"])

    artifact_baseline_err = np.abs(artifact_data["pred_baseline_nm"] - artifact_data["y_true_nm"])
    artifact_proposed_err = np.abs(artifact_data["pred_tail_nm"] - artifact_data["y_true_nm"])

    linewidth_error_df = pd.DataFrame(
        {
            "baseline_abs_error_nm": linewidth_baseline_err,
            "proposed_abs_error_nm": linewidth_proposed_err,
        }
    )
    artifact_error_df = pd.DataFrame(
        {
            "baseline_abs_error_nm": artifact_baseline_err,
            "proposed_abs_error_nm": artifact_proposed_err,
        }
    )

    export_data_copy(
        data_dir,
        build_repr_dataframe(X_linewidth, linewidth_idx),
        build_repr_dataframe(X_artifact, artifact_idx),
        linewidth_error_df,
        artifact_error_df,
    )

    fig, axes = plt.subplots(
        2,
        2,
        figsize=cfg.figsize,
        constrained_layout=True,
        gridspec_kw={"width_ratios": [1.0, 1.15]},
    )

    plot_representative_spectra(
        axes[0, 0],
        X_linewidth,
        linewidth_idx,
        "(a) Representative spectra under linewidth asymmetry",
    )
    plot_linewidth_boxplot(axes[0, 1], linewidth_baseline_err, linewidth_proposed_err)
    plot_representative_spectra(
        axes[1, 0],
        X_artifact,
        artifact_idx,
        "(c) Representative spectra under system artifact",
    )
    plot_artifact_paired(axes[1, 1], artifact_baseline_err, artifact_proposed_err)

    png_path = outputs_dir / "fig10_final.png"
    pdf_path = outputs_dir / "fig10_final.pdf"
    fig.savefig(png_path, dpi=cfg.dpi, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    print("Figure 10 saved successfully.")


if __name__ == "__main__":
    main()
