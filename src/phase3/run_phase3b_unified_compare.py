from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from phase2.baselines import (
    estimate_center_by_parametric_fit,
    estimate_shift_by_cross_correlation,
    gaussian_spectrum,
    normalize_minmax,
)
from phase2.nn_models import MLPRegressor
from phase3.common import load_config, load_dataset_b, metrics_dict, set_seed
from phase3.models import build_model
from phase3.train_utils import make_loaders, predict, train_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase3b unified comparison on Dataset_B")
    parser.add_argument("--config", type=str, default="config/phase3.yaml")
    return parser.parse_args()


def run_cross_correlation(data, cfg: dict) -> np.ndarray:
    step_nm = float(data.wavelengths[1] - data.wavelengths[0])
    lambda_b0 = float(cfg["dataset_b"]["lambda_b0_nm"])
    sigma_nm = float(cfg["dataset_b"]["base_linewidth_nm"])

    ref = gaussian_spectrum(data.wavelengths, center_nm=lambda_b0, sigma_nm=sigma_nm, amplitude=1.0, baseline=0.0)
    if str(cfg["dataset_b"]["normalize"]) == "minmax_per_sample":
        ref = normalize_minmax(ref)

    y_pred = np.zeros(len(data.idx_test), dtype=np.float32)
    for i, idx in enumerate(data.idx_test):
        y_pred[i] = estimate_shift_by_cross_correlation(ref, data.x[idx], step_nm)
    return y_pred


def run_parametric_fitting(data, cfg: dict) -> np.ndarray:
    lambda_b0 = float(cfg["dataset_b"]["lambda_b0_nm"])
    y_pred = np.zeros(len(data.idx_test), dtype=np.float32)
    for i, idx in enumerate(data.idx_test):
        center = estimate_center_by_parametric_fit(
            data.wavelengths,
            data.x[idx],
            fit_window_points=61,
            baseline_percentile=10.0,
        )
        y_pred[i] = float(center - lambda_b0)
    return y_pred


def run_neural_model(data, cfg: dict, model_kind: str, seed_offset: int) -> np.ndarray:
    set_seed(int(cfg["phase3"]["seed"]) + seed_offset)
    train_loader, val_loader = make_loaders(
        data.x,
        data.y_dlambda,
        data.idx_train,
        data.idx_val,
        batch_size=int(cfg["train"]["batch_size"]),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if model_kind == "mlp":
        model = MLPRegressor(input_dim=data.x.shape[1]).to(device)
    elif model_kind in {"cnn_baseline", "cnn_se"}:
        model = build_model(method_key=model_kind, input_dim=data.x.shape[1]).to(device)
    else:
        raise ValueError(f"Unsupported model_kind: {model_kind}")

    model = train_model(model, train_loader, val_loader, cfg["train"], device)
    return predict(model, data.x, data.idx_test, device)


def metric_row(method_name: str, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    m = metrics_dict(y_true, y_pred)
    return {
        "Method": method_name,
        "RMSE_nm": m["rmse"],
        "MAE_nm": m["mae"],
        "R2": m["r2"],
    }


def plot_parity(out_dir: Path, y_true: np.ndarray, pred_map: dict[str, np.ndarray]) -> None:
    names = list(pred_map.keys())
    colors = ["#4e79a7", "#f28e2b", "#59a14f", "#e15759", "#76b7b2"]

    fig, axes = plt.subplots(2, 3, figsize=(13.0, 7.2), constrained_layout=True)
    flat_axes = axes.flatten()

    y_min = float(min(y_true.min(), min(v.min() for v in pred_map.values())))
    y_max = float(max(y_true.max(), max(v.max() for v in pred_map.values())))
    pad = 0.03 * (y_max - y_min + 1e-12)
    lo = y_min - pad
    hi = y_max + pad

    for i, name in enumerate(names):
        ax = flat_axes[i]
        pred = pred_map[name]
        m = metrics_dict(y_true, pred)
        ax.scatter(y_true, pred, s=10, alpha=0.35, c=colors[i], edgecolors="none")
        ax.plot([lo, hi], [lo, hi], "k--", linewidth=1.0)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(name)
        ax.set_xlabel("True Δλ (nm)")
        ax.set_ylabel("Pred Δλ (nm)")
        ax.text(
            0.03,
            0.97,
            f"RMSE={m['rmse']:.6f}\nMAE={m['mae']:.6f}\nR2={m['r2']:.6f}",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=8,
            bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "alpha": 0.9},
        )

    flat_axes[-1].axis("off")
    fig.suptitle("Dataset_B Parity Plots (All Methods)")
    fig.savefig(out_dir / "parity_plot.png", dpi=300)
    plt.close(fig)


def plot_residual(out_dir: Path, y_true: np.ndarray, pred_map: dict[str, np.ndarray]) -> None:
    names = list(pred_map.keys())
    colors = ["#4e79a7", "#f28e2b", "#59a14f", "#e15759", "#76b7b2"]
    residuals = [pred_map[n] - y_true for n in names]

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.3), constrained_layout=True)

    bp = axes[0].boxplot(residuals, tick_labels=names, showfliers=False, patch_artist=True)
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.65)
    axes[0].axhline(0.0, color="k", linestyle="--", linewidth=1.0)
    axes[0].set_ylabel("Residual (Pred - True) in nm")
    axes[0].set_title("Residual Boxplot")
    axes[0].tick_params(axis="x", rotation=16)

    all_res = np.concatenate(residuals)
    bins = np.linspace(float(all_res.min()), float(all_res.max()), 70)
    for r, n, c in zip(residuals, names, colors):
        axes[1].hist(r, bins=bins, density=True, alpha=0.35, color=c, label=n)
    axes[1].axvline(0.0, color="k", linestyle="--", linewidth=1.0)
    axes[1].set_xlabel("Residual (nm)")
    axes[1].set_ylabel("Density")
    axes[1].set_title("Residual Distribution")
    axes[1].legend(fontsize=8)

    fig.suptitle("Dataset_B Residual Analysis (All Methods)")
    fig.savefig(out_dir / "residual_plot.png", dpi=300)
    plt.close(fig)


def write_outputs(out_dir: Path, rows: list[dict], y_true: np.ndarray, pred_map: dict[str, np.ndarray]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics_json = {}
    for row in rows:
        metrics_json[row["Method"]] = {
            "RMSE_nm": row["RMSE_nm"],
            "MAE_nm": row["MAE_nm"],
            "R2": row["R2"],
        }
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_json, f, indent=2)

    with open(out_dir / "metrics_table.csv", "w", encoding="utf-8") as f:
        f.write("Method, RMSE_nm, MAE_nm, R2\n")
        for row in rows:
            f.write(f"{row['Method']},{row['RMSE_nm']:.8f},{row['MAE_nm']:.8f},{row['R2']:.8f}\n")

    labels = [r["Method"] for r in rows]
    rmse_values = [r["RMSE_nm"] for r in rows]
    colors = ["#4e79a7", "#f28e2b", "#59a14f", "#e15759", "#76b7b2"]
    x = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(9.5, 4.2), constrained_layout=True)
    bars = ax.bar(x, rmse_values, color=colors, width=0.62)
    ax.set_title("Dataset_B RMSE Comparison")
    ax.set_ylabel("RMSE (nm)")
    ax.set_xticks(x, labels, rotation=15)
    ax.grid(True, axis="y", alpha=0.25)
    for b, v in zip(bars, rmse_values):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.6f}", ha="center", va="bottom", fontsize=8)
    fig.savefig(out_dir / "comparison_plot.png", dpi=300)
    plt.close(fig)

    np.savez(
        out_dir / "predictions.npz",
        y_true=y_true.astype(np.float32),
        pred_cross_correlation=pred_map["Cross-correlation"].astype(np.float32),
        pred_parametric_fitting=pred_map["Parametric fitting"].astype(np.float32),
        pred_mlp=pred_map["MLP"].astype(np.float32),
        pred_cnn=pred_map["CNN"].astype(np.float32),
        pred_cnn_se=pred_map["CNN+SE"].astype(np.float32),
    )

    plot_parity(out_dir, y_true, pred_map)
    plot_residual(out_dir, y_true, pred_map)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    data = load_dataset_b(cfg)
    y_true = data.y_dlambda[data.idx_test]

    set_seed(int(cfg["phase3"]["seed"]))
    pred_cc = run_cross_correlation(data, cfg)
    pred_pf = run_parametric_fitting(data, cfg)
    pred_mlp = run_neural_model(data, cfg, model_kind="mlp", seed_offset=10)
    pred_cnn = run_neural_model(data, cfg, model_kind="cnn_baseline", seed_offset=20)
    pred_cnn_se = run_neural_model(data, cfg, model_kind="cnn_se", seed_offset=30)

    pred_map = {
        "Cross-correlation": pred_cc,
        "Parametric fitting": pred_pf,
        "MLP": pred_mlp,
        "CNN": pred_cnn,
        "CNN+SE": pred_cnn_se,
    }
    rows = [metric_row(name, y_true, pred_map[name]) for name in pred_map]

    out_dir = PROJECT_ROOT / "results" / "phase3b"
    write_outputs(out_dir, rows, y_true, pred_map)

    print(f"Saved: {out_dir / 'metrics_table.csv'}")
    print(f"Saved: {out_dir / 'metrics.json'}")
    print(f"Saved: {out_dir / 'comparison_plot.png'}")
    print(f"Saved: {out_dir / 'parity_plot.png'}")
    print(f"Saved: {out_dir / 'residual_plot.png'}")
    print(f"Saved: {out_dir / 'predictions.npz'}")
    for row in rows:
        print(
            f"{row['Method']:<18} RMSE={row['RMSE_nm']:.6f} nm | "
            f"MAE={row['MAE_nm']:.6f} nm | R2={row['R2']:.6f}"
        )


if __name__ == "__main__":
    main()
