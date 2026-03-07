from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

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
    p = argparse.ArgumentParser(description="Strict Check4: leakage distribution shift (train fixed, test shifted)")
    p.add_argument("--config", type=str, default="config/phase4_array.yaml")
    p.add_argument("--out-dir", type=str, default="results/phase4_checks")
    return p.parse_args()


def build_dataset(cfg: dict, seed: int, train_alpha: tuple[float, float], test_alpha: tuple[float, float]) -> dict:
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
    alpha = np.zeros(n, dtype=np.float32)

    for i in range(n):
        dlam = float(rng.uniform(dmin, dmax))
        per_grating, _, _ = simulate_identical_array_spectra(wavelengths, cfg, dlam)
        if split[i] == 2:
            a = float(rng.uniform(test_alpha[0], test_alpha[1]))
        else:
            a = float(rng.uniform(train_alpha[0], train_alpha[1]))
        alpha[i] = a
        local_clean = (
            per_grating[target_index]
            + a * (per_grating[target_index - 1] + per_grating[target_index + 1])
            + 0.5 * a * (per_grating[target_index - 2] + per_grating[target_index + 2])
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
        "alpha": alpha,
    }


def run_cross(ds: dict, cfg: dict) -> np.ndarray:
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


def run_param(ds: dict, cfg: dict) -> np.ndarray:
    lambda0 = float(cfg["array"]["lambda0_nm"])
    pred = np.zeros(len(ds["idx_test"]), dtype=np.float32)
    for i, idx in enumerate(ds["idx_test"]):
        c = estimate_center_by_parametric_fit(
            ds["wavelengths"],
            ds["x"][idx],
            fit_window_points=int(cfg["compare"]["fit_window_points"]),
            baseline_percentile=float(cfg["compare"]["baseline_percentile"]),
        )
        pred[i] = np.float32(c - lambda0)
    return pred


def run_neural(ds: dict, cfg: dict, key: str, seed: int) -> np.ndarray:
    set_seed(seed)
    train_loader, val_loader = make_loaders(
        ds["x"],
        ds["y"],
        ds["idx_train"],
        ds["idx_val"],
        batch_size=int(cfg["train"]["batch_size"]),
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if key == "mlp":
        model = MLPRegressor(input_dim=ds["x"].shape[1]).to(device)
    else:
        model = build_model(key, input_dim=ds["x"].shape[1]).to(device)
    model = train_model(model, train_loader, val_loader, cfg["train"], device)
    return predict(model, ds["x"], ds["idx_test"], device)


def eval_all(ds: dict, cfg: dict, seed_base: int) -> dict:
    y_true = ds["y"][ds["idx_test"]]
    metrics = {
        "Cross-correlation": metrics_dict(y_true, run_cross(ds, cfg)),
        "Parametric fitting": metrics_dict(y_true, run_param(ds, cfg)),
        "MLP": metrics_dict(y_true, run_neural(ds, cfg, "mlp", seed_base + 1)),
        "CNN": metrics_dict(y_true, run_neural(ds, cfg, "cnn_baseline", seed_base + 2)),
        "CNN+SE": metrics_dict(y_true, run_neural(ds, cfg, "cnn_se", seed_base + 3)),
    }
    return metrics


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    out_dir = resolve_project_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_alpha = (0.05, 0.15)
    in_dist_test_alpha = (0.05, 0.15)
    shifted_test_alpha = (0.15, 0.25)

    ds_in = build_dataset(cfg, seed=5001, train_alpha=train_alpha, test_alpha=in_dist_test_alpha)
    ds_shift = build_dataset(cfg, seed=5001, train_alpha=train_alpha, test_alpha=shifted_test_alpha)

    m_in = eval_all(ds_in, cfg, seed_base=8000)
    m_shift = eval_all(ds_shift, cfg, seed_base=8000)

    with open(out_dir / "check4_strict_in_metrics.json", "w", encoding="utf-8") as f:
        json.dump(m_in, f, indent=2)
    with open(out_dir / "check4_strict_shift_metrics.json", "w", encoding="utf-8") as f:
        json.dump(m_shift, f, indent=2)

    with open(out_dir / "check4_strict_drop.csv", "w", encoding="utf-8") as f:
        f.write("Method,RMSE_in_dist,RMSE_shift,RMSE_change_percent\n")
        for k in m_in:
            a = m_in[k]["rmse"]
            b = m_shift[k]["rmse"]
            d = (b - a) / (a + 1e-12) * 100.0
            f.write(f"{k},{a:.8f},{b:.8f},{d:.2f}\n")

    print(f"Saved: {out_dir / 'check4_strict_drop.csv'}")


if __name__ == "__main__":
    main()

