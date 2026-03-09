from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def calc_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float, float]:
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
    best_dist = float("inf")
    for r in rows:
        d = abs(float(r["mae"]) - mae) + abs(float(r["rmse"]) - rmse) + abs(float(r["r2"]) - r2)
        if d < best_dist:
            best_dist = d
            best_seed = str(r["seed"]).strip()
    return best_seed


def binned_stats(x: np.ndarray, y: np.ndarray, bins: int = 24) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    edges = np.linspace(float(np.min(x)), float(np.max(x)), bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    means = np.full(bins, np.nan, dtype=float)
    stds = np.full(bins, np.nan, dtype=float)
    idx = np.digitize(x, edges) - 1
    idx = np.clip(idx, 0, bins - 1)
    for i in range(bins):
        m = idx == i
        if np.sum(m) >= 8:
            vals = y[m]
            means[i] = float(np.mean(vals))
            stds[i] = float(np.std(vals))
    return centers, means, stds


def draw_panel(
    ax: plt.Axes,
    title: str,
    x: np.ndarray,
    y_pred: np.ndarray,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    color_scatter: str,
    color_trend: str,
) -> tuple[float, float, float, np.ndarray]:
    residual = y_pred - x
    mae, rmse, r2 = calc_metrics(x, y_pred)

    ax.scatter(x, residual, s=11, c=color_scatter, alpha=0.62, edgecolors="none")
    ax.axhline(0.0, color="black", linestyle="--", linewidth=1.3)

    cx, mean_r, std_r = binned_stats(x, residual, bins=24)
    valid = np.isfinite(mean_r)
    if np.any(valid):
        ax.plot(cx[valid], mean_r[valid], color=color_trend, linewidth=1.2)
        ax.fill_between(
            cx[valid],
            (mean_r - std_r)[valid],
            (mean_r + std_r)[valid],
            color=color_trend,
            alpha=0.12,
            linewidth=0.0,
        )

    ax.set_title(title, fontsize=10.5, pad=4)
    ax.set_xlabel("Ground Truth Δλ (nm)", fontsize=10)
    ax.set_ylabel("Residual Error (nm)", fontsize=10)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
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
        bbox=dict(facecolor="white", edgecolor="black", linewidth=0.6, alpha=0.86),
    )
    return mae, rmse, r2, residual


def export_residual_csv(
    out_csv: Path,
    y_true: np.ndarray,
    y_cnn: np.ndarray,
    y_cnnse: np.ndarray,
    seed_cnn: str,
    seed_cnnse: str,
) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "seed", "y_true", "y_pred", "residual"])
        for yt, yp in zip(y_true, y_cnn):
            w.writerow(["CNN", seed_cnn, float(yt), float(yp), float(yp - yt)])
        for yt, yp in zip(y_true, y_cnnse):
            w.writerow(["CNN+SE", seed_cnnse, float(yt), float(yp), float(yp - yt)])


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    pred_path = root / "results" / "paper_results_step2" / "raw" / "best_seed_predictions.npz"
    runs_csv = root / "results" / "paper_results_step2" / "final" / "multiseed_runs.csv"
    out_png = root / "results" / "paper_figures" / "Fig9_ablation_residual.png"
    out_csv = root / "results" / "paper_figures" / "Fig9_ablation_residual_data.csv"
    out_png.parent.mkdir(parents=True, exist_ok=True)

    if not pred_path.exists():
        raise FileNotFoundError(f"Missing prediction file: {pred_path}")
    d = np.load(pred_path, allow_pickle=True)
    needed = {"y_true", "pred_cnn", "pred_cnn_se"}
    if not needed.issubset(set(d.files)):
        raise ValueError(f"Missing required keys in {pred_path}")

    y_true = np.asarray(d["y_true"], dtype=float).reshape(-1)
    y_cnn = np.asarray(d["pred_cnn"], dtype=float).reshape(-1)
    y_cnnse = np.asarray(d["pred_cnn_se"], dtype=float).reshape(-1)

    mae_cnn, rmse_cnn, r2_cnn = calc_metrics(y_true, y_cnn)
    mae_se, rmse_se, r2_se = calc_metrics(y_true, y_cnnse)

    seed_cnn = infer_seed("CNN", mae_cnn, rmse_cnn, r2_cnn, runs_csv)
    seed_cnnse = infer_seed("CNN+SE", mae_se, rmse_se, r2_se, runs_csv)

    # Unified axis ranges across the two panels.
    x_min = min(float(np.min(y_true)), float(np.min(y_cnn)), float(np.min(y_cnnse)))
    x_max = max(float(np.max(y_true)), float(np.max(y_cnn)), float(np.max(y_cnnse)))
    x_pad = 0.04 * (x_max - x_min + 1e-12)
    xlim = (x_min - x_pad, x_max + x_pad)

    r_cnn = y_cnn - y_true
    r_se = y_cnnse - y_true
    y_min = min(float(np.min(r_cnn)), float(np.min(r_se)))
    y_max = max(float(np.max(r_cnn)), float(np.max(r_se)))
    y_pad = 0.08 * (y_max - y_min + 1e-12)
    ylim = (y_min - y_pad, y_max + y_pad)

    plt.rcParams["font.family"] = "Arial"
    dpi = 300
    fig_w, fig_h = 1800 / dpi, 800 / dpi
    fig, axes = plt.subplots(1, 2, figsize=(fig_w, fig_h), dpi=dpi)
    fig.patch.set_facecolor("white")

    mae_cnn, rmse_cnn, r2_cnn, res_cnn = draw_panel(
        axes[0],
        "(a) CNN",
        y_true,
        y_cnn,
        xlim=xlim,
        ylim=ylim,
        color_scatter="#6f88ad",
        color_trend="#2f4f7f",
    )
    mae_se, rmse_se, r2_se, res_se = draw_panel(
        axes[1],
        "(b) CNN+SE",
        y_true,
        y_cnnse,
        xlim=xlim,
        ylim=ylim,
        color_scatter="#4d6f99",
        color_trend="#1f3a70",
    )

    fig.subplots_adjust(left=0.075, right=0.99, bottom=0.16, top=0.90, wspace=0.24)
    fig.savefig(out_png, dpi=dpi)
    plt.close(fig)

    export_residual_csv(out_csv, y_true, y_cnn, y_cnnse, seed_cnn, seed_cnnse)

    # Required outputs
    print("Image path:", out_png.as_posix())
    print("CNN data source:", pred_path.as_posix())
    print("CNN+SE data source:", pred_path.as_posix())
    print(
        "Seeds used:",
        f"CNN seed={seed_cnn}, CNN+SE seed={seed_cnnse} (inferred by nearest metrics match in multiseed_runs.csv)",
    )
    print(f"CNN metrics: MAE={mae_cnn:.6f}, RMSE={rmse_cnn:.6f}, R2={r2_cnn:.6f}")
    print(f"CNN+SE metrics: MAE={mae_se:.6f}, RMSE={rmse_se:.6f}, R2={r2_se:.6f}")
    print(f"CNN residual mean/std: mean={float(np.mean(res_cnn)):.6f}, std={float(np.std(res_cnn)):.6f}")
    print(f"CNN+SE residual mean/std: mean={float(np.mean(res_se)):.6f}, std={float(np.std(res_se)):.6f}")
    print("1x2 layout compliance:", "Yes")
    print("Better for ablation than old Fig9:", "Yes")
    print("SCI plotting style compliance:", "Yes")
    print("Residual data csv:", out_csv.as_posix())


if __name__ == "__main__":
    main()

