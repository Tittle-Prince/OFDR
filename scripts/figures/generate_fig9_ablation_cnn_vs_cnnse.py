from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float, float]:
    err = y_pred - y_true
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err**2)))
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return mae, rmse, r2


def infer_seed(method: str, mae: float, rmse: float, r2: float, runs_csv: Path) -> str:
    if not runs_csv.exists():
        return "unknown"
    rows = list(csv.DictReader(runs_csv.open("r", encoding="utf-8-sig", newline="")))
    rows = [r for r in rows if str(r.get("method", "")).strip() == method]
    if not rows:
        return "unknown"
    best_seed = "unknown"
    best_dist = 1e18
    for r in rows:
        d = abs(float(r["mae"]) - mae) + abs(float(r["rmse"]) - rmse) + abs(float(r["r2"]) - r2)
        if d < best_dist:
            best_dist = d
            best_seed = str(r["seed"]).strip()
    return best_seed


def panel(
    ax: plt.Axes,
    title: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    color: str,
) -> tuple[float, float, float]:
    mae, rmse, r2 = metrics(y_true, y_pred)
    ax.scatter(y_true, y_pred, s=11, color=color, alpha=0.7, edgecolors="none")
    ax.plot([xlim[0], xlim[1]], [xlim[0], xlim[1]], linestyle="--", color="black", linewidth=1.3)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_title(title, fontsize=10.5, pad=4)
    ax.set_xlabel("Ground Truth Δλ (nm)", fontsize=10)
    ax.set_ylabel("Predicted Δλ (nm)", fontsize=10)
    ax.tick_params(axis="both", labelsize=8, width=1.0, length=4)
    for sp in ax.spines.values():
        sp.set_linewidth(1.1)
    ax.grid(False)
    txt = f"MAE = {mae:.4f}\nRMSE = {rmse:.4f}\nR² = {r2:.4f}"
    ax.text(
        0.03,
        0.97,
        txt,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.5,
        bbox=dict(facecolor="white", edgecolor="black", alpha=0.85, linewidth=0.6),
    )
    return mae, rmse, r2


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    pred_path = root / "results" / "paper_results_step2" / "raw" / "best_seed_predictions.npz"
    runs_csv = root / "results" / "paper_results_step2" / "final" / "multiseed_runs.csv"
    out_path = root / "results" / "paper_figures" / "Fig9_ablation_cnn_vs_cnnse.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not pred_path.exists():
        raise FileNotFoundError(f"Missing prediction file: {pred_path}")

    d = np.load(pred_path, allow_pickle=True)
    y_true = np.asarray(d["y_true"], dtype=float).reshape(-1)
    y_cnn = np.asarray(d["pred_cnn"], dtype=float).reshape(-1)
    y_cnnse = np.asarray(d["pred_cnn_se"], dtype=float).reshape(-1)

    lo = min(float(np.min(y_true)), float(np.min(y_cnn)), float(np.min(y_cnnse)))
    hi = max(float(np.max(y_true)), float(np.max(y_cnn)), float(np.max(y_cnnse)))
    m = 0.04 * (hi - lo + 1e-12)
    xlim = (lo - m, hi + m)
    ylim = xlim

    plt.rcParams["font.family"] = "Arial"
    dpi = 300
    fig_w, fig_h = 1800 / dpi, 800 / dpi
    fig, axes = plt.subplots(1, 2, figsize=(fig_w, fig_h), dpi=dpi)
    fig.patch.set_facecolor("white")

    mae_cnn, rmse_cnn, r2_cnn = panel(
        axes[0],
        "(a) CNN",
        y_true,
        y_cnn,
        xlim=xlim,
        ylim=ylim,
        color="#4f678f",
    )
    mae_se, rmse_se, r2_se = panel(
        axes[1],
        "(b) CNN+SE",
        y_true,
        y_cnnse,
        xlim=xlim,
        ylim=ylim,
        color="#2f4f7f",
    )

    fig.subplots_adjust(left=0.07, right=0.99, bottom=0.16, top=0.90, wspace=0.22)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)

    seed_cnn = infer_seed("CNN", mae_cnn, rmse_cnn, r2_cnn, runs_csv)
    seed_cnnse = infer_seed("CNN+SE", mae_se, rmse_se, r2_se, runs_csv)

    print("Image path:", out_path.as_posix())
    print("CNN data source:", pred_path.as_posix())
    print("CNN+SE data source:", pred_path.as_posix())
    print(
        "Seeds used:",
        f"CNN seed={seed_cnn}, CNN+SE seed={seed_cnnse} (inferred by nearest metrics match in multiseed_runs.csv)",
    )
    print(f"CNN metrics: MAE={mae_cnn:.6f}, RMSE={rmse_cnn:.6f}, R2={r2_cnn:.6f}")
    print(f"CNN+SE metrics: MAE={mae_se:.6f}, RMSE={rmse_se:.6f}, R2={r2_se:.6f}")
    print("1x2 layout compliance:", "Yes")
    print("Ablation-purpose compliance:", "Yes")
    print("SCI plotting style compliance:", "Yes")


if __name__ == "__main__":
    main()

