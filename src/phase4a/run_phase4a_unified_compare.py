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
from phase3.common import metrics_dict, set_seed
from phase3.models import build_model
from phase3.train_utils import make_loaders, predict, train_model
from phase4a.common import load_config, resolve_project_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase4-B unified comparison on Dataset_C_phase4a")
    parser.add_argument("--config", type=str, default="config/phase4a.yaml")
    return parser.parse_args()


def load_phase4a_dataset(cfg: dict) -> dict:
    data_path = resolve_project_path(cfg["phase4a"]["dataset_path"])
    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset missing: {data_path}\n"
            "Run `python src\\phase4a\\generate_dataset_phase4a.py --config config\\phase4a.yaml` first."
        )
    data = np.load(data_path)
    needed = ["X_local", "Y_dlambda_target", "wavelengths", "idx_train", "idx_val", "idx_test"]
    missing = [k for k in needed if k not in data]
    if missing:
        raise KeyError(f"Dataset missing keys: {missing}")
    return {
        "x": data["X_local"].astype(np.float32),
        "y": data["Y_dlambda_target"].astype(np.float32),
        "wavelengths": data["wavelengths"].astype(np.float32),
        "idx_train": data["idx_train"].astype(np.int64),
        "idx_val": data["idx_val"].astype(np.int64),
        "idx_test": data["idx_test"].astype(np.int64),
    }


def run_cross_correlation(ds: dict, cfg: dict) -> np.ndarray:
    wl = ds["wavelengths"]
    step_nm = float(wl[1] - wl[0])
    lambda0 = float(cfg["array"]["lambda0_nm"])
    sigma = float(cfg["array"]["linewidth_sigma_nm"])

    ref = gaussian_spectrum(wl, center_nm=lambda0, sigma_nm=sigma, amplitude=1.0, baseline=0.0)
    if str(cfg["local_window"]["normalize"]) == "minmax_per_sample":
        ref = normalize_minmax(ref)

    y_pred = np.zeros(len(ds["idx_test"]), dtype=np.float32)
    for i, idx in enumerate(ds["idx_test"]):
        y_pred[i] = estimate_shift_by_cross_correlation(ref, ds["x"][idx], step_nm)
    return y_pred


def run_parametric_fitting(ds: dict, cfg: dict) -> np.ndarray:
    lambda0 = float(cfg["array"]["lambda0_nm"])
    fit_window_points = int(cfg["compare"]["fit_window_points"])
    baseline_percentile = float(cfg["compare"]["baseline_percentile"])

    y_pred = np.zeros(len(ds["idx_test"]), dtype=np.float32)
    for i, idx in enumerate(ds["idx_test"]):
        center = estimate_center_by_parametric_fit(
            ds["wavelengths"],
            ds["x"][idx],
            fit_window_points=fit_window_points,
            baseline_percentile=baseline_percentile,
        )
        y_pred[i] = np.float32(center - lambda0)
    return y_pred


def run_neural(ds: dict, cfg: dict, model_key: str, seed_offset: int) -> np.ndarray:
    set_seed(int(cfg["phase4a"]["seed"]) + seed_offset)
    train_loader, val_loader = make_loaders(
        ds["x"],
        ds["y"],
        ds["idx_train"],
        ds["idx_val"],
        batch_size=int(cfg["train"]["batch_size"]),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if model_key == "mlp":
        model = MLPRegressor(input_dim=ds["x"].shape[1]).to(device)
    elif model_key in {"cnn_baseline", "cnn_se"}:
        model = build_model(model_key, input_dim=ds["x"].shape[1]).to(device)
    else:
        raise ValueError(f"Unsupported model_key: {model_key}")

    model = train_model(model, train_loader, val_loader, cfg["train"], device)
    return predict(model, ds["x"], ds["idx_test"], device)


def metric_row(name: str, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    m = metrics_dict(y_true, y_pred)
    return {"Method": name, "RMSE_nm": m["rmse"], "MAE_nm": m["mae"], "R2": m["r2"]}


def save_outputs(out_dir: Path, rows: list[dict], y_true: np.ndarray, pred_map: dict[str, np.ndarray]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "metrics_table.csv", "w", encoding="utf-8") as f:
        f.write("Method,RMSE_nm,MAE_nm,R2\n")
        for r in rows:
            f.write(f"{r['Method']},{r['RMSE_nm']:.8f},{r['MAE_nm']:.8f},{r['R2']:.8f}\n")

    metrics_json = {
        r["Method"]: {"RMSE_nm": r["RMSE_nm"], "MAE_nm": r["MAE_nm"], "R2": r["R2"]}
        for r in rows
    }
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_json, f, indent=2)

    np.savez(
        out_dir / "predictions.npz",
        y_true=y_true.astype(np.float32),
        pred_cross_correlation=pred_map["Cross-correlation"].astype(np.float32),
        pred_parametric_fitting=pred_map["Parametric fitting"].astype(np.float32),
        pred_mlp=pred_map["MLP"].astype(np.float32),
        pred_cnn=pred_map["CNN"].astype(np.float32),
        pred_cnn_se=pred_map["CNN+SE"].astype(np.float32),
    )

    labels = [r["Method"] for r in rows]
    rmse_vals = [r["RMSE_nm"] for r in rows]
    x = np.arange(len(labels))
    colors = ["#4e79a7", "#f28e2b", "#59a14f", "#e15759", "#76b7b2"]

    fig, ax = plt.subplots(figsize=(9.2, 4.2), constrained_layout=True)
    bars = ax.bar(x, rmse_vals, color=colors, width=0.62)
    ax.set_title("Phase4-B on Dataset_C (Local Distorted Spectrum)")
    ax.set_ylabel("RMSE (nm)")
    ax.set_xticks(x, labels, rotation=14)
    ax.grid(True, axis="y", alpha=0.25)
    for b, v in zip(bars, rmse_vals):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.6f}", ha="center", va="bottom", fontsize=8)
    fig.savefig(out_dir / "comparison_plot.png", dpi=300)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    ds = load_phase4a_dataset(cfg)
    y_true = ds["y"][ds["idx_test"]]

    set_seed(int(cfg["phase4a"]["seed"]))
    pred_cc = run_cross_correlation(ds, cfg)
    pred_pf = run_parametric_fitting(ds, cfg)
    pred_mlp = run_neural(ds, cfg, model_key="mlp", seed_offset=11)
    pred_cnn = run_neural(ds, cfg, model_key="cnn_baseline", seed_offset=21)
    pred_cnn_se = run_neural(ds, cfg, model_key="cnn_se", seed_offset=31)

    pred_map = {
        "Cross-correlation": pred_cc,
        "Parametric fitting": pred_pf,
        "MLP": pred_mlp,
        "CNN": pred_cnn,
        "CNN+SE": pred_cnn_se,
    }
    rows = [metric_row(name, y_true, pred_map[name]) for name in pred_map]

    out_dir = resolve_project_path(cfg["phase4a"]["results_dir"]) / "phase4b_compare"
    save_outputs(out_dir, rows, y_true, pred_map)

    print(f"Saved: {out_dir / 'metrics_table.csv'}")
    print(f"Saved: {out_dir / 'metrics.json'}")
    print(f"Saved: {out_dir / 'comparison_plot.png'}")
    for r in rows:
        print(
            f"{r['Method']:<18} RMSE={r['RMSE_nm']:.6f} nm | "
            f"MAE={r['MAE_nm']:.6f} nm | R2={r['R2']:.6f}"
        )


if __name__ == "__main__":
    main()

