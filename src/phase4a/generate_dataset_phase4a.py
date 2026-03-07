from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from phase4a.array_simulator import build_wavelength_axis, simulate_identical_array_spectra
from phase4a.common import load_config, resolve_project_path, set_seed
from phase4a.local_window import apply_local_effects, extract_local_distorted_spectrum, sample_leakage_weights


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Phase4-A dataset: identical UWBFG array to local distorted spectra")
    parser.add_argument("--config", type=str, default="config/phase4a.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    seed = int(cfg["phase4a"]["seed"])
    set_seed(seed)
    rng = np.random.default_rng(seed)

    wavelengths = build_wavelength_axis(cfg)
    n_samples = int(cfg["dataset"]["num_samples"])
    n_points = len(wavelengths)
    n_gratings = int(cfg["array"]["n_gratings"])
    target_index = int(cfg["array"]["target_index"])

    dmin = float(cfg["label"]["delta_lambda_target_min_nm"])
    dmax = float(cfg["label"]["delta_lambda_target_max_nm"])

    x_local = np.zeros((n_samples, n_points), dtype=np.float32)
    y_dlambda = np.zeros(n_samples, dtype=np.float32)
    y_centers = np.zeros((n_samples, n_gratings), dtype=np.float32)
    x_total = np.zeros((n_samples, n_points), dtype=np.float32)
    leakage_weights_all = np.zeros((n_samples, n_gratings), dtype=np.float32)
    neighbor_mode_all = np.zeros(n_samples, dtype=np.int64)
    window_shift_all = np.zeros(n_samples, dtype=np.float32)
    amp_scales_all = np.zeros((n_samples, n_gratings), dtype=np.float32)
    sigma_scales_all = np.zeros((n_samples, n_gratings), dtype=np.float32)

    arr_rand = cfg.get("array_random", {})
    amp_min = float(arr_rand.get("amplitude_scale_min", 1.0))
    amp_max = float(arr_rand.get("amplitude_scale_max", 1.0))
    sig_min = float(arr_rand.get("linewidth_scale_min", 1.0))
    sig_max = float(arr_rand.get("linewidth_scale_max", 1.0))

    lw = cfg["local_window"]
    shift_min = float(lw.get("target_window_shift_min_nm", 0.0))
    shift_max = float(lw.get("target_window_shift_max_nm", 0.0))

    for i in range(n_samples):
        dlam = float(rng.uniform(dmin, dmax))
        amp_scales = rng.uniform(amp_min, amp_max, size=n_gratings).astype(np.float64)
        sigma_scales = rng.uniform(sig_min, sig_max, size=n_gratings).astype(np.float64)
        per_grating, total, centers = simulate_identical_array_spectra(
            wavelengths,
            cfg,
            dlam,
            amplitude_scales=amp_scales,
            linewidth_scales=sigma_scales,
        )

        weights, mode = sample_leakage_weights(n_gratings, target_index, lw, rng)
        local_clean, _ = extract_local_distorted_spectrum(per_grating, target_index, lw, weights=weights)

        window_shift = float(rng.uniform(shift_min, shift_max))
        local_noisy = apply_local_effects(
            local_clean,
            lw,
            rng,
            wavelengths=wavelengths,
            shift_nm=window_shift,
        )

        x_local[i] = local_noisy
        x_total[i] = total.astype(np.float32)
        y_dlambda[i] = np.float32(dlam)
        y_centers[i] = centers.astype(np.float32)
        leakage_weights_all[i] = weights.astype(np.float32)
        neighbor_mode_all[i] = np.int64(mode)
        window_shift_all[i] = np.float32(window_shift)
        amp_scales_all[i] = amp_scales.astype(np.float32)
        sigma_scales_all[i] = sigma_scales.astype(np.float32)

    tr = float(cfg["dataset"]["train_ratio"])
    vr = float(cfg["dataset"]["val_ratio"])
    ter = float(cfg["dataset"]["test_ratio"])
    if abs(tr + vr + ter - 1.0) > 1e-8:
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")

    idx = rng.permutation(n_samples)
    n_train = int(n_samples * tr)
    n_val = int(n_samples * vr)
    idx_train = idx[:n_train]
    idx_val = idx[n_train : n_train + n_val]
    idx_test = idx[n_train + n_val :]

    dataset_path = resolve_project_path(cfg["phase4a"]["dataset_path"])
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        dataset_path,
        X_local=x_local,
        X_total=x_total,
        Y_dlambda_target=y_dlambda,
        wavelengths=wavelengths.astype(np.float32),
        centers_nm=y_centers,
        leakage_weights=leakage_weights_all,
        neighbor_mode=neighbor_mode_all,
        window_shift_nm=window_shift_all,
        amplitude_scales=amp_scales_all,
        linewidth_scales=sigma_scales_all,
        idx_train=idx_train.astype(np.int64),
        idx_val=idx_val.astype(np.int64),
        idx_test=idx_test.astype(np.int64),
        target_index=np.array([target_index], dtype=np.int64),
    )
    print(f"Saved dataset: {dataset_path}")
    print(f"X_local shape: {x_local.shape}")
    print(f"Split sizes: train={len(idx_train)}, val={len(idx_val)}, test={len(idx_test)}")


if __name__ == "__main__":
    main()
