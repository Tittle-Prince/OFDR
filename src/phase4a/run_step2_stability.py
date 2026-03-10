from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

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


SEEDS = [42, 123, 2026, 3407, 7777]
METHODS = ["Cross-correlation", "Parametric fitting", "MLP", "CNN", "CNN+SE"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Step2 stability experiment on dataset_c_final")
    p.add_argument("--output-dir", type=str, default="results/paper_results_step2")
    p.add_argument("--config", type=str, default="config/phase4_array.yaml")
    return p.parse_args()


def load_cfg(config_path: Path) -> dict:
    import yaml

    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def detect_dataset_path(cfg: dict) -> Path:
    cands = []
    cfg_ds = cfg.get("phase4a", {}).get("dataset_path")
    if cfg_ds:
        cands.append(PROJECT_ROOT / str(cfg_ds))
    cands.extend(
        [
            PROJECT_ROOT / "data" / "processed" / "dataset_c_final.npz",
            PROJECT_ROOT / "data" / "processed" / "dataset_b_array_se_hard.npz",
            PROJECT_ROOT / "data" / "processed" / "dataset_b_array.npz",
            PROJECT_ROOT / "data" / "processed" / "dataset_c_phase4a.npz",
            PROJECT_ROOT / "data" / "processed" / "dataset_b_phase3.npz",
        ]
    )
    for p in cands:
        if p.exists():
            return p
    raise FileNotFoundError("No usable dataset found in data/processed/")


def load_dataset(path: Path, cfg: dict) -> dict:
    d = np.load(path)
    if "X_local" in d:
        x = d["X_local"].astype(np.float32)
    elif "X" in d:
        x = d["X"].astype(np.float32)
    else:
        raise KeyError("Dataset missing X_local/X")

    if "Y_dlambda_target" in d:
        y = d["Y_dlambda_target"].astype(np.float32)
    elif "Y_dlambda" in d:
        y = d["Y_dlambda"].astype(np.float32)
    else:
        raise KeyError("Dataset missing Y_dlambda_target/Y_dlambda")

    wl = d["wavelengths"].astype(np.float32) if "wavelengths" in d else np.arange(x.shape[1], dtype=np.float32)

    if all(k in d for k in ["idx_train", "idx_val", "idx_test"]):
        idx_train = d["idx_train"].astype(np.int64)
        idx_val = d["idx_val"].astype(np.int64)
        idx_test = d["idx_test"].astype(np.int64)
    else:
        n = len(x)
        rng = np.random.default_rng(42)
        idx = rng.permutation(n)
        tr = float(cfg.get("dataset", {}).get("train_ratio", 0.7))
        vr = float(cfg.get("dataset", {}).get("val_ratio", 0.15))
        n_train = int(n * tr)
        n_val = int(n * vr)
        idx_train = idx[:n_train]
        idx_val = idx[n_train : n_train + n_val]
        idx_test = idx[n_train + n_val :]

    return {
        "x": x,
        "y": y,
        "wavelengths": wl,
        "idx_train": idx_train,
        "idx_val": idx_val,
        "idx_test": idx_test,
    }


def run_cross_correlation(ds: dict, cfg: dict) -> np.ndarray:
    wl = ds["wavelengths"]
    step_nm = float(wl[1] - wl[0])
    lambda0 = float(cfg.get("array", {}).get("lambda0_nm", 1550.0))
    sigma = float(cfg.get("array", {}).get("linewidth_sigma_nm", 0.08))
    ref = gaussian_spectrum(wl, center_nm=lambda0, sigma_nm=sigma, amplitude=1.0, baseline=0.0)
    if str(cfg.get("local_window", {}).get("normalize", "minmax_per_sample")) == "minmax_per_sample":
        ref = normalize_minmax(ref)
    pred = np.zeros(len(ds["idx_test"]), dtype=np.float32)
    for i, idx in enumerate(ds["idx_test"]):
        pred[i] = estimate_shift_by_cross_correlation(ref, ds["x"][idx], step_nm)
    return pred


def run_parametric(ds: dict, cfg: dict) -> np.ndarray:
    lambda0 = float(cfg.get("array", {}).get("lambda0_nm", 1550.0))
    fit_window_points = int(cfg.get("compare", {}).get("fit_window_points", 61))
    baseline_percentile = float(cfg.get("compare", {}).get("baseline_percentile", 10.0))
    pred = np.zeros(len(ds["idx_test"]), dtype=np.float32)
    for i, idx in enumerate(ds["idx_test"]):
        center = estimate_center_by_parametric_fit(
            ds["wavelengths"], ds["x"][idx], fit_window_points=fit_window_points, baseline_percentile=baseline_percentile
        )
        pred[i] = np.float32(center - lambda0)
    return pred


def make_loaders(ds: dict, batch_size: int) -> tuple[DataLoader, DataLoader]:
    x_train = torch.tensor(ds["x"][ds["idx_train"]], dtype=torch.float32)
    y_train = torch.tensor(ds["y"][ds["idx_train"]], dtype=torch.float32)
    x_val = torch.tensor(ds["x"][ds["idx_val"]], dtype=torch.float32)
    y_val = torch.tensor(ds["y"][ds["idx_val"]], dtype=torch.float32)
    train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(x_val, y_val), batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


def train_with_history(model: nn.Module, ds: dict, cfg_train: dict, seed: int) -> tuple[nn.Module, list[dict]]:
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    batch_size = int(cfg_train["batch_size"])
    lr = float(cfg_train["lr"])
    wd = float(cfg_train["weight_decay"])
    epochs = int(cfg_train["epochs"])
    patience = int(cfg_train["patience"])

    train_loader, val_loader = make_loaders(ds, batch_size=batch_size)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    criterion = nn.MSELoss()

    history = []
    best_state = None
    best_val = float("inf")
    stale = 0
    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            opt.step()
            train_losses.append(float(loss.item()))

        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                v = criterion(model(xb), yb)
                val_losses.append(float(v.item()))
        train_loss = float(np.mean(train_losses))
        val_loss = float(np.mean(val_losses))
        val_rmse = math.sqrt(max(val_loss, 0.0))
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "val_rmse": val_rmse})

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history


def predict_model(model: nn.Module, ds: dict) -> np.ndarray:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    x_test = torch.tensor(ds["x"][ds["idx_test"]], dtype=torch.float32, device=device)
    with torch.no_grad():
        pred = model(x_test).cpu().numpy()
    return pred.astype(np.float32)


def save_history_csv(path: Path, history: list[dict]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "train_loss", "val_loss", "val_rmse"])
        for r in history:
            w.writerow([r["epoch"], f"{r['train_loss']:.10f}", f"{r['val_loss']:.10f}", f"{r['val_rmse']:.10f}"])


def aggregate_metrics(rows: list[dict]) -> list[dict]:
    grouped = defaultdict(list)
    for r in rows:
        grouped[r["Method"]].append(r)
    out = []
    for m in METHODS:
        arr = grouped[m]
        rmse = np.array([x["RMSE_nm"] for x in arr], dtype=np.float64)
        mae = np.array([x["MAE_nm"] for x in arr], dtype=np.float64)
        r2 = np.array([x["R2"] for x in arr], dtype=np.float64)
        out.append(
            {
                "Method": m,
                "RMSE_mean": float(rmse.mean()),
                "RMSE_std": float(rmse.std()),
                "MAE_mean": float(mae.mean()),
                "MAE_std": float(mae.std()),
                "R2_mean": float(r2.mean()),
                "R2_std": float(r2.std()),
            }
        )
    return out


def plot_method_bar(summary: list[dict], out_path: Path) -> None:
    names = [r["Method"] for r in summary]
    rmse = [r["RMSE_mean"] for r in summary]
    mae = [r["MAE_mean"] for r in summary]
    r2 = [r["R2_mean"] for r in summary]
    x = np.arange(len(names))
    colors = ["#4e79a7", "#f28e2b", "#59a14f", "#e15759", "#76b7b2"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    axes[0].bar(x, rmse, color=colors)
    axes[0].set_title("RMSE mean")
    axes[0].set_xticks(x, names, rotation=14)
    axes[1].bar(x, mae, color=colors)
    axes[1].set_title("MAE mean")
    axes[1].set_xticks(x, names, rotation=14)
    axes[2].bar(x, r2, color=colors)
    axes[2].set_title("R2 mean")
    axes[2].set_xticks(x, names, rotation=14)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_errorbar(summary: list[dict], out_path: Path) -> None:
    names = [r["Method"] for r in summary]
    mean = np.array([r["RMSE_mean"] for r in summary], dtype=np.float64)
    std = np.array([r["RMSE_std"] for r in summary], dtype=np.float64)
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(9, 4.2), constrained_layout=True)
    ax.errorbar(x, mean, yerr=std, fmt="o", capsize=4, linewidth=1.8, color="#1f77b4")
    ax.set_xticks(x, names, rotation=14)
    ax.set_ylabel("RMSE (nm)")
    ax.set_title("RMSE mean ± std (5 seeds)")
    ax.grid(True, axis="y", alpha=0.25)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_parity(pred_best: dict[str, np.ndarray], y_true: np.ndarray, out_path: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), constrained_layout=True)
    flat = axes.flatten()
    lo = float(min(y_true.min(), min(v.min() for v in pred_best.values())))
    hi = float(max(y_true.max(), max(v.max() for v in pred_best.values())))
    pad = 0.03 * (hi - lo + 1e-12)
    lo -= pad
    hi += pad
    colors = ["#4e79a7", "#f28e2b", "#59a14f", "#e15759", "#76b7b2"]
    for i, m in enumerate(METHODS):
        ax = flat[i]
        pred = pred_best[m]
        mt = metrics_dict(y_true, pred)
        ax.scatter(y_true, pred, s=10, alpha=0.35, c=colors[i], edgecolors="none")
        ax.plot([lo, hi], [lo, hi], "k--", linewidth=1.0)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(m)
        ax.set_xlabel("True Δλ (nm)")
        ax.set_ylabel("Pred Δλ (nm)")
        ax.text(0.03, 0.96, f"RMSE={mt['rmse']:.4f}", transform=ax.transAxes, va="top", ha="left", fontsize=8)
    flat[-1].axis("off")
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_training_curves(histories: dict[str, dict[int, list[dict]]], out_path: Path) -> None:
    models = ["MLP", "CNN", "CNN+SE"]
    fig, axes = plt.subplots(3, 1, figsize=(11, 10), constrained_layout=True)
    for ax, m in zip(axes, models):
        seed_hist = histories[m]
        max_epoch = max(len(h) for h in seed_hist.values())
        train_mat = np.full((len(seed_hist), max_epoch), np.nan, dtype=np.float64)
        val_mat = np.full((len(seed_hist), max_epoch), np.nan, dtype=np.float64)
        rmse_mat = np.full((len(seed_hist), max_epoch), np.nan, dtype=np.float64)
        for i, s in enumerate(sorted(seed_hist.keys())):
            h = seed_hist[s]
            for j, row in enumerate(h):
                train_mat[i, j] = row["train_loss"]
                val_mat[i, j] = row["val_loss"]
                rmse_mat[i, j] = row["val_rmse"]

        ep = np.arange(1, max_epoch + 1)
        train_mean = np.nanmean(train_mat, axis=0)
        val_mean = np.nanmean(val_mat, axis=0)
        rmse_mean = np.nanmean(rmse_mat, axis=0)
        ax.plot(ep, train_mean, label="train_loss", color="#1f77b4")
        ax.plot(ep, val_mean, label="val_loss", color="#ff7f0e")
        ax2 = ax.twinx()
        ax2.plot(ep, rmse_mean, label="val_RMSE", color="#2ca02c", linestyle="--")
        ax.set_title(f"{m} training history (mean over seeds)")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax2.set_ylabel("Val RMSE")
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines + lines2, labels + labels2, loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.25)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_input_schematic(ds: dict, out_path: Path) -> None:
    idx_train = ds["idx_train"]
    y = ds["y"]
    order = np.argsort(y[idx_train])
    pick = np.linspace(0, len(order) - 1, 8, dtype=int)
    chosen = idx_train[order[pick]]
    fig, ax = plt.subplots(figsize=(10.5, 4.8), constrained_layout=True)
    cmap = plt.get_cmap("viridis")
    for i, idx in enumerate(chosen):
        ax.plot(ds["wavelengths"], ds["x"][idx], color=cmap(i / max(1, len(chosen) - 1)), linewidth=1.4, label=f"Δλ={y[idx]:.3f}")
    ax.set_title("Input Schematic: Local Distorted Spectra")
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Normalized intensity")
    ax.grid(True, alpha=0.25)
    ax.legend(ncol=2, fontsize=8)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_typical_samples(ds: dict, out_path: Path) -> None:
    idx_test = ds["idx_test"]
    y_test = ds["y"][idx_test]
    order = np.argsort(y_test)
    chosen = [idx_test[order[20]], idx_test[order[len(order) // 2]], idx_test[order[-20]]]
    titles = ["Low label sample", "Mid label sample", "High label sample"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    for ax, idx, ttl in zip(axes, chosen, titles):
        ax.plot(ds["wavelengths"], ds["x"][idx], color="#1f77b4", linewidth=1.5)
        ax.set_title(f"{ttl}\nΔλ={ds['y'][idx]:.3f} nm")
        ax.set_xlabel("Wavelength (nm)")
        ax.set_ylabel("Intensity")
        ax.grid(True, alpha=0.25)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    out_root = PROJECT_ROOT / args.output_dir
    fig_dir = out_root / "figures"
    table_dir = out_root / "tables"
    raw_dir = out_root / "raw"
    fig_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_cfg(PROJECT_ROOT / args.config)
    dataset_path = detect_dataset_path(cfg)
    ds = load_dataset(dataset_path, cfg)
    y_true = ds["y"][ds["idx_test"]]

    (raw_dir / "dataset_used.txt").write_text(str(dataset_path), encoding="utf-8")

    seed_rows = []
    pred_store = defaultdict(dict)  # method -> seed -> pred
    histories: dict[str, dict[int, list[dict]]] = {"MLP": {}, "CNN": {}, "CNN+SE": {}}

    for seed in SEEDS:
        # deterministic/traditional
        set_seed(seed)
        pred_cc = run_cross_correlation(ds, cfg)
        pred_pf = run_parametric(ds, cfg)
        for name, pred in [("Cross-correlation", pred_cc), ("Parametric fitting", pred_pf)]:
            m = metrics_dict(y_true, pred)
            seed_rows.append(
                {"Seed": seed, "Method": name, "RMSE_nm": m["rmse"], "MAE_nm": m["mae"], "R2": m["r2"]}
            )
            pred_store[name][seed] = pred

        # neural
        train_cfg = cfg.get("train", {"batch_size": 128, "lr": 1e-3, "weight_decay": 0.0, "epochs": 45, "patience": 10})

        model_mlp, h_mlp = train_with_history(MLPRegressor(input_dim=ds["x"].shape[1]), ds, train_cfg, seed=seed)
        pred_mlp = predict_model(model_mlp, ds)
        m_mlp = metrics_dict(y_true, pred_mlp)
        seed_rows.append({"Seed": seed, "Method": "MLP", "RMSE_nm": m_mlp["rmse"], "MAE_nm": m_mlp["mae"], "R2": m_mlp["r2"]})
        pred_store["MLP"][seed] = pred_mlp
        histories["MLP"][seed] = h_mlp
        save_history_csv(raw_dir / f"history_MLP_seed{seed}.csv", h_mlp)

        model_cnn, h_cnn = train_with_history(build_model("cnn_baseline", input_dim=ds["x"].shape[1]), ds, train_cfg, seed=seed + 1000)
        pred_cnn = predict_model(model_cnn, ds)
        m_cnn = metrics_dict(y_true, pred_cnn)
        seed_rows.append({"Seed": seed, "Method": "CNN", "RMSE_nm": m_cnn["rmse"], "MAE_nm": m_cnn["mae"], "R2": m_cnn["r2"]})
        pred_store["CNN"][seed] = pred_cnn
        histories["CNN"][seed] = h_cnn
        save_history_csv(raw_dir / f"history_CNN_seed{seed}.csv", h_cnn)

        model_se, h_se = train_with_history(build_model("cnn_se", input_dim=ds["x"].shape[1]), ds, train_cfg, seed=seed + 2000)
        pred_se = predict_model(model_se, ds)
        m_se = metrics_dict(y_true, pred_se)
        seed_rows.append({"Seed": seed, "Method": "CNN+SE", "RMSE_nm": m_se["rmse"], "MAE_nm": m_se["mae"], "R2": m_se["r2"]})
        pred_store["CNN+SE"][seed] = pred_se
        histories["CNN+SE"][seed] = h_se
        save_history_csv(raw_dir / f"history_CNNSE_seed{seed}.csv", h_se)

    # Raw per-seed table
    with open(raw_dir / "seed_metrics.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Seed", "Method", "RMSE_nm", "MAE_nm", "R2"])
        for r in seed_rows:
            w.writerow([r["Seed"], r["Method"], f"{r['RMSE_nm']:.10f}", f"{r['MAE_nm']:.10f}", f"{r['R2']:.10f}"])

    summary = aggregate_metrics(seed_rows)
    with open(table_dir / "paper_main_table.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Method", "RMSE_mean", "RMSE_std", "MAE_mean", "MAE_std", "R2_mean", "R2_std"])
        for r in summary:
            w.writerow(
                [
                    r["Method"],
                    f"{r['RMSE_mean']:.10f}",
                    f"{r['RMSE_std']:.10f}",
                    f"{r['MAE_mean']:.10f}",
                    f"{r['MAE_std']:.10f}",
                    f"{r['R2_mean']:.10f}",
                    f"{r['R2_std']:.10f}",
                ]
            )
    # duplicate at root for convenience
    with open(out_root / "paper_main_table.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Method", "RMSE_mean", "RMSE_std", "MAE_mean", "MAE_std", "R2_mean", "R2_std"])
        for r in summary:
            w.writerow([r["Method"], r["RMSE_mean"], r["RMSE_std"], r["MAE_mean"], r["MAE_std"], r["R2_mean"], r["R2_std"]])

    with open(raw_dir / "summary_metrics.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # choose best seed per method for parity plot
    best_pred = {}
    for m in METHODS:
        rows_m = [r for r in seed_rows if r["Method"] == m]
        best_seed = min(rows_m, key=lambda x: x["RMSE_nm"])["Seed"]
        best_pred[m] = pred_store[m][best_seed]
    np.savez(
        raw_dir / "best_seed_predictions.npz",
        y_true=y_true.astype(np.float32),
        pred_cross_correlation=best_pred["Cross-correlation"].astype(np.float32),
        pred_parametric_fitting=best_pred["Parametric fitting"].astype(np.float32),
        pred_mlp=best_pred["MLP"].astype(np.float32),
        pred_cnn=best_pred["CNN"].astype(np.float32),
        pred_cnn_se=best_pred["CNN+SE"].astype(np.float32),
    )

    # Figures
    plot_method_bar(summary, fig_dir / "fig_method_comparison_bar.png")
    plot_errorbar(summary, fig_dir / "fig_method_rmse_errorbar.png")
    plot_parity(best_pred, y_true, fig_dir / "fig_parity_scatter.png")
    plot_training_curves(histories, fig_dir / "fig_training_curves.png")
    plot_input_schematic(ds, fig_dir / "fig_input_schematic.png")
    plot_typical_samples(ds, fig_dir / "fig_typical_samples.png")

    print(f"Dataset used: {dataset_path}")
    print(f"Saved table: {table_dir / 'paper_main_table.csv'}")
    print(f"Saved figures under: {fig_dir}")
    print(f"Saved raw logs under: {raw_dir}")


if __name__ == "__main__":
    main()
