from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


@dataclass
class FigureConfig:
    dpi: int = 300
    figsize: tuple[float, float] = (10.6, 7.2)
    n_repr: int = 5
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


def spectrum_features(x: np.ndarray) -> tuple[float, float, float, float, float, float]:
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
    shoulder = float(np.max(np.abs(np.diff(np.diff(smooth)))))
    return width, asym, rough, spike, shoulder, float(peak)


def select_representative_indices(X: np.ndarray, scenario: str, n_select: int) -> np.ndarray:
    feats = np.array([spectrum_features(x) for x in X], dtype=np.float64)
    fmax = np.maximum(np.max(feats, axis=0), 1e-12)

    if scenario == "linewidth_asymmetry":
        score = 0.45 * (feats[:, 0] / fmax[0]) + 0.40 * (feats[:, 1] / fmax[1]) + 0.15 * (feats[:, 4] / fmax[4])
        feat_subset = feats[:, [0, 1, 4, 5]]
    else:
        score = 0.45 * (feats[:, 2] / fmax[2]) + 0.40 * (feats[:, 3] / fmax[3]) + 0.15 * (feats[:, 4] / fmax[4])
        feat_subset = feats[:, [2, 3, 4, 5]]

    candidates = np.argsort(score)[::-1]
    selected: list[int] = []
    feat_norm = feat_subset / np.maximum(np.std(feat_subset, axis=0, ddof=1), 1e-12)

    for idx in candidates:
        if not selected:
            selected.append(int(idx))
            continue
        dist = np.min(np.linalg.norm(feat_norm[idx] - feat_norm[selected], axis=1))
        if dist > 0.85 or len(selected) < max(2, n_select // 2):
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


def prepare_result_dataframe(data: dict[str, np.ndarray], downsample_step: int) -> pd.DataFrame:
    idx = np.arange(0, data["y_true_nm"].shape[0], downsample_step)
    df = pd.DataFrame(
        {
            "exp_idx": idx + 1,
            "reference_shift_nm": data["y_true_nm"][idx],
            "baseline_shift_nm": data["pred_baseline_nm"][idx],
            "proposed_shift_nm": data["pred_tail_nm"][idx],
        }
    )
    df["baseline_abs_error_nm"] = np.abs(df["baseline_shift_nm"] - df["reference_shift_nm"])
    df["proposed_abs_error_nm"] = np.abs(df["proposed_shift_nm"] - df["reference_shift_nm"])
    return df


def export_data_copy(
    data_dir: Path,
    linewidth_repr: pd.DataFrame,
    artifact_repr: pd.DataFrame,
    linewidth_results: pd.DataFrame,
    artifact_results: pd.DataFrame,
) -> None:
    linewidth_repr.to_csv(data_dir / "linewidth_representative_spectra.csv", index=False)
    artifact_repr.to_csv(data_dir / "artifact_representative_spectra.csv", index=False)
    linewidth_results.to_csv(data_dir / "linewidth_results_display.csv", index=False)
    artifact_results.to_csv(data_dir / "artifact_results_display.csv", index=False)


def build_repr_dataframe(X: np.ndarray, indices: np.ndarray) -> pd.DataFrame:
    records = []
    for order, idx in enumerate(indices, start=1):
        for j, value in enumerate(X[idx]):
            records.append({"case_id": order, "sample_index": j, "intensity": float(value), "source_index": int(idx)})
    return pd.DataFrame(records)


def plot_representative_spectra(ax: plt.Axes, X: np.ndarray, indices: np.ndarray, title: str, color_base: str) -> None:
    x = np.arange(X.shape[1])
    offsets = np.linspace(0.0, 0.44, len(indices))
    for i, (idx, off) in enumerate(zip(indices, offsets)):
        y = X[idx] + off
        alpha = 0.95 - 0.10 * i
        ax.plot(x, y, color=color_base, linewidth=1.15, alpha=alpha)
        ax.text(x[-1] + 4, y[-1] + 0.005, f"Case {i+1}", fontsize=8.0, va="center")

    ax.set_title(title, loc="left", pad=4)
    ax.set_xlabel("Local sample index")
    ax.set_ylabel("Normalized intensity")
    ax.set_xlim(0, X.shape[1] - 1)
    ax.set_ylim(0.0, 1.55)
    ax.grid(True, linestyle="--", alpha=0.18)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_demod_results(ax: plt.Axes, df: pd.DataFrame, title: str, show_legend: bool) -> None:
    x = df["exp_idx"].to_numpy()
    ref = df["reference_shift_nm"].to_numpy()
    baseline = df["baseline_shift_nm"].to_numpy()
    proposed = df["proposed_shift_nm"].to_numpy()

    ax.plot(
        x,
        ref,
        color="black",
        linewidth=1.15,
        marker="o",
        markersize=2.8,
        markerfacecolor="black",
        markeredgewidth=0.0,
        markevery=4,
        alpha=0.90,
        label="Reference",
    )
    ax.plot(
        x,
        baseline,
        color="#8c8c8c",
        linewidth=1.05,
        linestyle="--",
        marker="s",
        markersize=3.1,
        markerfacecolor="white",
        markeredgewidth=0.8,
        markevery=4,
        alpha=0.95,
        label="Baseline CNN",
    )
    ax.plot(
        x,
        proposed,
        color="#1f4e79",
        linewidth=1.20,
        linestyle="-",
        marker="^",
        markersize=3.3,
        markerfacecolor="#1f4e79",
        markeredgewidth=0.0,
        markevery=4,
        alpha=0.95,
        label="Proposed CNN",
    )
    ax.set_title(title, loc="left", pad=4)
    ax.set_xlabel("Experiment index")
    ax.set_ylabel("Target wavelength shift (nm)")
    ax.grid(True, linestyle="--", alpha=0.18)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if show_legend:
        ax.legend(frameon=False, loc="upper right")


def main() -> None:
    configure_style()
    cfg = FigureConfig()

    repo_root = Path(__file__).resolve().parents[3]
    figure_root = repo_root / "paper_figures" / "Fig10_linewidth_artifact"
    _, outputs_dir, data_dir = ensure_dirs(figure_root)

    linewidth_data = load_predictions(repo_root, "linewidth_asymmetry")
    artifact_data = load_predictions(repo_root, "system_artifact")

    X_linewidth = linewidth_data["X_bridge"][:, 0, :]
    X_artifact = artifact_data["X_bridge"][:, 0, :]

    linewidth_idx = select_representative_indices(X_linewidth, "linewidth_asymmetry", cfg.n_repr)
    artifact_idx = select_representative_indices(X_artifact, "system_artifact", cfg.n_repr)

    linewidth_results = prepare_result_dataframe(linewidth_data, cfg.downsample_step)
    artifact_results = prepare_result_dataframe(artifact_data, cfg.downsample_step)

    export_data_copy(
        data_dir,
        build_repr_dataframe(X_linewidth, linewidth_idx),
        build_repr_dataframe(X_artifact, artifact_idx),
        linewidth_results,
        artifact_results,
    )

    fig, axes = plt.subplots(
        2,
        2,
        figsize=cfg.figsize,
        constrained_layout=True,
        gridspec_kw={"width_ratios": [1.0, 1.2]},
    )

    plot_representative_spectra(
        axes[0, 0],
        X_linewidth,
        linewidth_idx,
        "(a) Representative spectra under linewidth asymmetry",
        color_base="#3a3a3a",
    )
    plot_demod_results(
        axes[0, 1],
        linewidth_results,
        "(b) Demodulation results under linewidth asymmetry",
        show_legend=True,
    )
    plot_representative_spectra(
        axes[1, 0],
        X_artifact,
        artifact_idx,
        "(c) Representative spectra under system artifact",
        color_base="#404040",
    )
    plot_demod_results(
        axes[1, 1],
        artifact_results,
        "(d) Demodulation results under system artifact",
        show_legend=False,
    )

    png_path = outputs_dir / "fig10_linewidth_artifact.png"
    pdf_path = outputs_dir / "fig10_linewidth_artifact.pdf"
    fig.savefig(png_path, dpi=cfg.dpi, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    print("Figure 10 saved successfully.")


if __name__ == "__main__":
    main()
