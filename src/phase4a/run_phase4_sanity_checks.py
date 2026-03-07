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
from phase4a.array_simulator import build_wavelength_axis, simulate_identical_array_spectra
from phase4a.common import load_config, resolve_project_path
from phase4a.local_window import apply_local_effects


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run sanity checks 1-6 for Phase4 array dataset")
    p.add_argument("--config", type=str, default="config/phase4_array.yaml")
    p.add_argument("--results-dir", type=str, default="results/phase4_checks")
    return p.parse_args()


def load_dataset(cfg: dict) -> dict:
    path = resolve_project_path(cfg["phase4a"]["dataset_path"])
    d = np.load(path)
    return {
        "x": d["X_local"].astype(np.float32),
        "y": d["Y_dlambda_target"].astype(np.float32),
        "wavelengths": d["wavelengths"].astype(np.float32),
        "idx_train": d["idx_train"].astype(np.int64),
        "idx_val": d["idx_val"].astype(np.int64),
        "idx_test": d["idx_test"].astype(np.int64),
    }


def run_cross_corr(ds: dict, cfg: dict) -> np.ndarray:
    wl = ds["wavelengths"]
    step_nm = float(wl[1] - wl[0])
    lambda0 = float(cfg["array"]["lambda0_nm"])
    sigma = float(cfg["array"]["linewidth_sigma_nm"])
    ref = gaussian_spectrum(wl, center_nm=lambda0, sigma_nm=sigma, amplitude=1.0, baseline=0.0)
    if str(cfg["local_window"]["normalize"]) == "minmax_per_sample":
        ref = normalize_minmax(ref)
    pred = np.zeros(len(ds["idx_test"]), dtype=np.float32)
    for i, idx in enumerate(ds["idx_test"]):
        pred[i] = estimate_shift_by_cross_correlation(ref, ds["x"][idx], step_nm)
    return pred


def run_parametric(ds: dict, cfg: dict) -> np.ndarray:
    lambda0 = float(cfg["array"]["lambda0_nm"])
    fit_window_points = int(cfg["compare"]["fit_window_points"])
    baseline_percentile = float(cfg["compare"]["baseline_percentile"])
    pred = np.zeros(len(ds["idx_test"]), dtype=np.float32)
    for i, idx in enumerate(ds["idx_test"]):
        center = estimate_center_by_parametric_fit(
            ds["wavelengths"],
            ds["x"][idx],
            fit_window_points=fit_window_points,
            baseline_percentile=baseline_percentile,
        )
        pred[i] = np.float32(center - lambda0)
    return pred


def run_neural(ds: dict, cfg: dict, model_key: str, seed: int) -> np.ndarray:
    set_seed(seed)
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
    else:
        model = build_model(model_key, input_dim=ds["x"].shape[1]).to(device)
    model = train_model(model, train_loader, val_loader, cfg["train"], device)
    return predict(model, ds["x"], ds["idx_test"], device)


def run_ridge(ds: dict, ridge_alpha: float = 1e-3) -> np.ndarray:
    x = ds["x"].astype(np.float64)
    y = ds["y"].astype(np.float64)
    tr, va, te = ds["idx_train"], ds["idx_val"], ds["idx_test"]

    mu = x[tr].mean(axis=0, keepdims=True)
    std = x[tr].std(axis=0, keepdims=True) + 1e-8
    xz = (x - mu) / std

    x_train = np.hstack([xz[tr], np.ones((len(tr), 1), dtype=np.float64)])
    x_val = np.hstack([xz[va], np.ones((len(va), 1), dtype=np.float64)])
    x_test = np.hstack([xz[te], np.ones((len(te), 1), dtype=np.float64)])
    y_train = y[tr]
    y_val = y[va]

    alphas = [0.0, 1e-6, 1e-4, ridge_alpha, 1e-2, 1e-1]
    best_w = None
    best_rmse = float("inf")
    for a in alphas:
        eye = np.eye(x_train.shape[1], dtype=np.float64)
        eye[-1, -1] = 0.0
        w = np.linalg.solve(x_train.T @ x_train + a * eye, x_train.T @ y_train)
        pred_val = x_val @ w
        m = metrics_dict(y[va].astype(np.float32), pred_val.astype(np.float32))
        if m["rmse"] < best_rmse:
            best_rmse = m["rmse"]
            best_w = w
    pred_test = x_test @ best_w
    return pred_test.astype(np.float32)


def check1_plot_inputs(ds: dict, out_dir: Path) -> None:
    idx_test = ds["idx_test"]
    y_test = ds["y"][idx_test]
    order = np.argsort(y_test)
    k = 8
    pick = np.linspace(0, len(order) - 1, k, dtype=int)
    chosen = idx_test[order[pick]]

    fig, ax = plt.subplots(figsize=(10.0, 4.8), constrained_layout=True)
    cmap = plt.get_cmap("viridis")
    for i, idx in enumerate(chosen):
        color = cmap(i / max(1, k - 1))
        ax.plot(ds["wavelengths"], ds["x"][idx], color=color, linewidth=1.4, label=f"Δλ={ds['y'][idx]:.3f} nm")
    ax.set_title("Check1: Input Spectra Sorted by Label (Test Samples)")
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Input intensity (normalized)")
    ax.grid(True, alpha=0.25)
    ax.legend(ncol=2, fontsize=8)
    fig.savefig(out_dir / "check1_input_vs_label.png", dpi=300)
    plt.close(fig)


def check2_permutation(ds: dict, cfg: dict, out_dir: Path, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    perm = rng.permutation(ds["x"].shape[1])
    ds_perm = dict(ds)
    ds_perm["x"] = ds["x"][:, perm]

    pred_mlp_orig = run_neural(ds, cfg, "mlp", seed=seed + 100)
    pred_cnn_orig = run_neural(ds, cfg, "cnn_baseline", seed=seed + 200)
    pred_mlp_perm = run_neural(ds_perm, cfg, "mlp", seed=seed + 100)
    pred_cnn_perm = run_neural(ds_perm, cfg, "cnn_baseline", seed=seed + 200)

    y_true = ds["y"][ds["idx_test"]]
    m = {
        "MLP_original": metrics_dict(y_true, pred_mlp_orig),
        "MLP_permuted": metrics_dict(y_true, pred_mlp_perm),
        "CNN_original": metrics_dict(y_true, pred_cnn_orig),
        "CNN_permuted": metrics_dict(y_true, pred_cnn_perm),
    }

    rows = [
        ("MLP", m["MLP_original"]["rmse"], m["MLP_permuted"]["rmse"]),
        ("CNN", m["CNN_original"]["rmse"], m["CNN_permuted"]["rmse"]),
    ]
    with open(out_dir / "check2_permutation.csv", "w", encoding="utf-8") as f:
        f.write("Model,RMSE_original,RMSE_permuted,Relative_drop_percent\n")
        for name, o, p in rows:
            drop = (p - o) / (o + 1e-12) * 100.0
            f.write(f"{name},{o:.8f},{p:.8f},{drop:.2f}\n")
    with open(out_dir / "check2_permutation.json", "w", encoding="utf-8") as f:
        json.dump(m, f, indent=2)
    return m


def generate_leakage_shift_dataset(cfg: dict, seed: int) -> dict:
    set_seed(seed)
    rng = np.random.default_rng(seed)

    n = int(cfg["dataset"]["num_samples"])
    tr = float(cfg["dataset"]["train_ratio"])
    vr = float(cfg["dataset"]["val_ratio"])
    wavelengths = build_wavelength_axis(cfg)
    dmin = float(cfg["label"]["delta_lambda_target_min_nm"])
    dmax = float(cfg["label"]["delta_lambda_target_max_nm"])
    target_index = int(cfg["array"]["target_index"])

    idx = rng.permutation(n)
    n_train = int(n * tr)
    n_val = int(n * vr)
    idx_train = idx[:n_train]
    idx_val = idx[n_train : n_train + n_val]
    idx_test = idx[n_train + n_val :]
    split = np.full(n, 2, dtype=np.int64)
    split[idx_train] = 0
    split[idx_val] = 1

    x = np.zeros((n, len(wavelengths)), dtype=np.float32)
    y = np.zeros(n, dtype=np.float32)
    alpha_values = np.zeros(n, dtype=np.float32)

    for i in range(n):
        dlam = float(rng.uniform(dmin, dmax))
        per_grating, _, _ = simulate_identical_array_spectra(wavelengths, cfg, dlam)
        if split[i] in (0, 1):
            alpha = float(rng.uniform(0.05, 0.15))
        else:
            alpha = float(rng.uniform(0.15, 0.25))
        alpha_values[i] = alpha

        # Keep center weight fixed at 1; use alpha as adjacent leakage; second neighbor 0.5*alpha.
        local_clean = (
            1.0 * per_grating[target_index]
            + alpha * (per_grating[target_index - 1] + per_grating[target_index + 1])
            + 0.5 * alpha * (per_grating[target_index - 2] + per_grating[target_index + 2])
        )
        x[i] = apply_local_effects(local_clean, cfg["local_window"], rng)
        y[i] = np.float32(dlam)

    return {
        "x": x,
        "y": y,
        "wavelengths": wavelengths.astype(np.float32),
        "idx_train": idx_train.astype(np.int64),
        "idx_val": idx_val.astype(np.int64),
        "idx_test": idx_test.astype(np.int64),
        "alpha": alpha_values,
    }


def check4_shift(ds_orig: dict, cfg: dict, out_dir: Path, seed: int, base_metrics: dict[str, dict]) -> dict:
    ds_shift = generate_leakage_shift_dataset(cfg, seed=seed)
    y_true = ds_shift["y"][ds_shift["idx_test"]]

    pred_cc = run_cross_corr(ds_shift, cfg)
    pred_pf = run_parametric(ds_shift, cfg)
    pred_mlp = run_neural(ds_shift, cfg, "mlp", seed=seed + 300)
    pred_cnn = run_neural(ds_shift, cfg, "cnn_baseline", seed=seed + 400)
    pred_cnn_se = run_neural(ds_shift, cfg, "cnn_se", seed=seed + 500)

    shift_metrics = {
        "Cross-correlation": metrics_dict(y_true, pred_cc),
        "Parametric fitting": metrics_dict(y_true, pred_pf),
        "MLP": metrics_dict(y_true, pred_mlp),
        "CNN": metrics_dict(y_true, pred_cnn),
        "CNN+SE": metrics_dict(y_true, pred_cnn_se),
    }

    with open(out_dir / "check4_shift_metrics.json", "w", encoding="utf-8") as f:
        json.dump(shift_metrics, f, indent=2)

    with open(out_dir / "check4_shift_drop.csv", "w", encoding="utf-8") as f:
        f.write("Method,RMSE_base,RMSE_shift,RMSE_change_percent\n")
        for k in shift_metrics:
            b = base_metrics[k]["rmse"]
            s = shift_metrics[k]["rmse"]
            d = (s - b) / (b + 1e-12) * 100.0
            f.write(f"{k},{b:.8f},{s:.8f},{d:.2f}\n")

    # Save alpha distribution quick plot
    alpha = ds_shift["alpha"]
    fig, ax = plt.subplots(figsize=(7.2, 3.8), constrained_layout=True)
    ax.hist(alpha[ds_shift["idx_train"]], bins=40, alpha=0.5, label="train+val alpha", color="#4e79a7")
    ax.hist(alpha[ds_shift["idx_test"]], bins=40, alpha=0.5, label="test alpha", color="#e15759")
    ax.set_title("Check4: Leakage Coefficient Distribution Shift")
    ax.set_xlabel("alpha")
    ax.set_ylabel("Count")
    ax.legend()
    ax.grid(True, alpha=0.25)
    fig.savefig(out_dir / "check4_alpha_shift.png", dpi=300)
    plt.close(fig)
    return shift_metrics


def check5_units(ds: dict, cfg: dict, out_dir: Path, base_metrics: dict[str, dict]) -> None:
    lambda0 = float(cfg["array"]["lambda0_nm"])
    k_t = 0.01  # project convention in earlier configs/scripts
    y = ds["y"]
    txt = []
    txt.append("Check5: Unit and Reference Alignment")
    txt.append("")
    txt.append(f"Label: Y_dlambda_target in nm, range [{float(y.min()):.4f}, {float(y.max()):.4f}]")
    txt.append("MLP/CNN/CNN+SE output: trained directly on Y_dlambda_target -> output unit is nm")
    txt.append("Cross-correlation output: estimated wavelength shift = lag * wavelength_step -> nm")
    txt.append("Parametric fitting output: fitted_center - lambda0 -> nm")
    txt.append(f"Reference wavelength used by baselines: lambda0 = {lambda0:.4f} nm")
    txt.append(f"Temperature conversion convention (if needed): dT = dLambda / K_T, with K_T={k_t:.4f} nm/degC")
    txt.append("")
    txt.append("Base metrics snapshot:")
    for k, m in base_metrics.items():
        txt.append(f"- {k}: RMSE={m['rmse']:.6f} nm, MAE={m['mae']:.6f} nm, R2={m['r2']:.6f}")
    (out_dir / "check5_units_alignment.txt").write_text("\n".join(txt), encoding="utf-8")


def check6_multiseed(ds: dict, cfg: dict, out_dir: Path) -> dict:
    seeds = [42, 123, 2026]
    models = [("MLP", "mlp"), ("CNN", "cnn_baseline"), ("CNN+SE", "cnn_se")]
    y_true = ds["y"][ds["idx_test"]]

    all_rows = []
    summary = {}
    for name, key in models:
        rmses = []
        maes = []
        r2s = []
        for s in seeds:
            pred = run_neural(ds, cfg, key, seed=s)
            m = metrics_dict(y_true, pred)
            rmses.append(m["rmse"])
            maes.append(m["mae"])
            r2s.append(m["r2"])
            all_rows.append((name, s, m["rmse"], m["mae"], m["r2"]))
        summary[name] = {
            "rmse_mean": float(np.mean(rmses)),
            "rmse_std": float(np.std(rmses)),
            "mae_mean": float(np.mean(maes)),
            "mae_std": float(np.std(maes)),
            "r2_mean": float(np.mean(r2s)),
            "r2_std": float(np.std(r2s)),
        }

    with open(out_dir / "check6_multiseed_runs.csv", "w", encoding="utf-8") as f:
        f.write("Model,Seed,RMSE_nm,MAE_nm,R2\n")
        for r in all_rows:
            f.write(f"{r[0]},{r[1]},{r[2]:.8f},{r[3]:.8f},{r[4]:.8f}\n")

    with open(out_dir / "check6_multiseed_summary.csv", "w", encoding="utf-8") as f:
        f.write("Model,RMSE_mean,RMSE_std,MAE_mean,MAE_std,R2_mean,R2_std\n")
        for k, v in summary.items():
            f.write(
                f"{k},{v['rmse_mean']:.8f},{v['rmse_std']:.8f},"
                f"{v['mae_mean']:.8f},{v['mae_std']:.8f},{v['r2_mean']:.8f},{v['r2_std']:.8f}\n"
            )
    with open(out_dir / "check6_multiseed_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    out_dir = resolve_project_path(args.results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ds = load_dataset(cfg)
    y_true = ds["y"][ds["idx_test"]]

    print("Running Check 1 ...")
    check1_plot_inputs(ds, out_dir)

    print("Running base metrics snapshot (for checks 4/5) ...")
    set_seed(int(cfg["phase4a"]["seed"]))
    base_metrics = {
        "Cross-correlation": metrics_dict(y_true, run_cross_corr(ds, cfg)),
        "Parametric fitting": metrics_dict(y_true, run_parametric(ds, cfg)),
        "MLP": metrics_dict(y_true, run_neural(ds, cfg, "mlp", seed=53)),
        "CNN": metrics_dict(y_true, run_neural(ds, cfg, "cnn_baseline", seed=63)),
        "CNN+SE": metrics_dict(y_true, run_neural(ds, cfg, "cnn_se", seed=73)),
    }
    with open(out_dir / "base_metrics_snapshot.json", "w", encoding="utf-8") as f:
        json.dump(base_metrics, f, indent=2)

    print("Running Check 2 ...")
    check2_permutation(ds, cfg, out_dir, seed=777)

    print("Running Check 3 ...")
    pred_ridge = run_ridge(ds)
    ridge_metrics = metrics_dict(y_true, pred_ridge)
    with open(out_dir / "check3_ridge.json", "w", encoding="utf-8") as f:
        json.dump(ridge_metrics, f, indent=2)

    print("Running Check 4 ...")
    check4_shift(ds, cfg, out_dir, seed=2027, base_metrics=base_metrics)

    print("Running Check 5 ...")
    check5_units(ds, cfg, out_dir, base_metrics=base_metrics)

    print("Running Check 6 ...")
    check6_multiseed(ds, cfg, out_dir)

    print(f"All checks completed. Outputs in: {out_dir}")


if __name__ == "__main__":
    main()

