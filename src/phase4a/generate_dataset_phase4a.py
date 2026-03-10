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


def _build_regime_pool(cfg: dict) -> tuple[list[dict], np.ndarray]:
    regs = cfg.get("se_regimes", [])
    if not isinstance(regs, list) or len(regs) == 0:
        return [], np.zeros(0, dtype=np.float64)

    probs = []
    out = []
    for r in regs:
        if not isinstance(r, dict):
            continue
        out.append(r)
        probs.append(float(r.get("prob", 1.0)))
    if len(out) == 0:
        return [], np.zeros(0, dtype=np.float64)

    p = np.asarray(probs, dtype=np.float64)
    p = np.clip(p, 0.0, None)
    if float(np.sum(p)) <= 0.0:
        p = np.full(len(out), 1.0 / len(out), dtype=np.float64)
    else:
        p /= float(np.sum(p))
    return out, p


def _sample_regime(regimes: list[dict], probs: np.ndarray, rng: np.random.Generator) -> tuple[int, dict]:
    if len(regimes) == 0:
        return -1, {}
    ridx = int(rng.choice(len(regimes), p=probs))
    return ridx, regimes[ridx]


def _scale_key(cfg_local: dict, key: str, scale: float) -> None:
    if key in cfg_local:
        cfg_local[key] = float(cfg_local[key]) * scale


def _apply_regime_to_local_config(cfg_local: dict, regime: dict) -> dict:
    if len(regime) == 0:
        return dict(cfg_local)

    out = dict(cfg_local)
    noise_scale = float(regime.get("noise_scale", 1.0))
    baseline_scale = float(regime.get("baseline_scale", 1.0))
    shift_scale = float(regime.get("shift_scale", 1.0))
    ripple_scale = float(regime.get("ripple_scale", 1.0))
    colored_noise_scale = float(regime.get("colored_noise_scale", 1.0))
    impulse_scale = float(regime.get("impulse_scale", 1.0))
    dropout_scale = float(regime.get("dropout_scale", 1.0))
    blur_scale = float(regime.get("blur_scale", 1.0))

    for k in ["additive_noise_std", "additive_noise_std_min", "additive_noise_std_max"]:
        _scale_key(out, k, noise_scale)
    for k in [
        "baseline_offset_min",
        "baseline_offset_max",
        "baseline_component_amp_min",
        "baseline_component_amp_max",
    ]:
        _scale_key(out, k, baseline_scale)
    for k in ["target_window_shift_min_nm", "target_window_shift_max_nm"]:
        _scale_key(out, k, shift_scale)
    for k in ["hf_ripple_amp_min", "hf_ripple_amp_max"]:
        _scale_key(out, k, ripple_scale)
    for k in ["colored_noise_std_min", "colored_noise_std_max"]:
        _scale_key(out, k, colored_noise_scale)
    for k in ["impulse_amp_min", "impulse_amp_max"]:
        _scale_key(out, k, impulse_scale)
    for k in ["instrument_blur_sigma_points_min", "instrument_blur_sigma_points_max"]:
        _scale_key(out, k, blur_scale)

    if "dropout_prob" in out:
        out["dropout_prob"] = float(np.clip(float(out["dropout_prob"]) * dropout_scale, 0.0, 1.0))
    if "impulse_prob" in out:
        out["impulse_prob"] = float(np.clip(float(out["impulse_prob"]) * impulse_scale, 0.0, 1.0))
    return out


def _sample_neighbor_delta_lambdas(
    cfg_label: dict,
    n_gratings: int,
    target_index: int,
    delta_lambda_target_nm: float,
    rng: np.random.Generator,
    regime: dict,
) -> np.ndarray:
    base = float(cfg_label.get("delta_lambda_neighbors_nm", 0.0))
    out = np.full(n_gratings, base, dtype=np.float64)
    mode = str(cfg_label.get("neighbor_shift_mode", "fixed")).lower()
    scale = float(regime.get("neighbor_shift_scale", 1.0))

    if mode in {"random_shared", "shared_random"}:
        lo = float(cfg_label.get("neighbor_delta_lambda_min_nm", base))
        hi = float(cfg_label.get("neighbor_delta_lambda_max_nm", base))
        if hi < lo:
            hi = lo
        v = float(rng.uniform(lo, hi)) * scale
        out.fill(v)
    elif mode in {"random_per_neighbor", "per_neighbor"}:
        lo = float(cfg_label.get("neighbor_delta_lambda_min_nm", base))
        hi = float(cfg_label.get("neighbor_delta_lambda_max_nm", base))
        if hi < lo:
            hi = lo
        out = rng.uniform(lo, hi, size=n_gratings).astype(np.float64) * scale
    else:
        out = out * scale

    coupling = float(cfg_label.get("neighbor_target_coupling", 0.0))
    jitter_lo = float(cfg_label.get("neighbor_jitter_min_nm", 0.0))
    jitter_hi = float(cfg_label.get("neighbor_jitter_max_nm", 0.0))
    if jitter_hi < jitter_lo:
        jitter_hi = jitter_lo
    if abs(coupling) > 1e-15 or abs(jitter_lo) > 1e-15 or abs(jitter_hi) > 1e-15:
        for i in range(n_gratings):
            if i == target_index:
                continue
            jitter = float(rng.uniform(jitter_lo, jitter_hi))
            out[i] = out[i] + coupling * float(delta_lambda_target_nm) + jitter
    return out


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
    regime_id_all = np.full(n_samples, -1, dtype=np.int64)
    neighbor_delta_all = np.zeros((n_samples, n_gratings), dtype=np.float32)

    arr_rand = cfg.get("array_random", {})
    amp_min = float(arr_rand.get("amplitude_scale_min", 1.0))
    amp_max = float(arr_rand.get("amplitude_scale_max", 1.0))
    sig_min = float(arr_rand.get("linewidth_scale_min", 1.0))
    sig_max = float(arr_rand.get("linewidth_scale_max", 1.0))

    lw_base = cfg["local_window"]
    label_cfg = cfg.get("label", {})
    regimes, regime_probs = _build_regime_pool(cfg)
    regime_names = np.array([str(r.get("name", f"regime_{i}")) for i, r in enumerate(regimes)], dtype=np.str_)

    for i in range(n_samples):
        dlam = float(rng.uniform(dmin, dmax))
        regime_idx, regime = _sample_regime(regimes, regime_probs, rng)
        lw = _apply_regime_to_local_config(lw_base, regime)

        amp_scales = rng.uniform(amp_min, amp_max, size=n_gratings).astype(np.float64)
        sigma_scales = rng.uniform(sig_min, sig_max, size=n_gratings).astype(np.float64)
        neighbor_deltas = _sample_neighbor_delta_lambdas(
            label_cfg,
            n_gratings=n_gratings,
            target_index=target_index,
            delta_lambda_target_nm=dlam,
            rng=rng,
            regime=regime,
        )
        per_grating, total, centers = simulate_identical_array_spectra(
            wavelengths,
            cfg,
            dlam,
            amplitude_scales=amp_scales,
            linewidth_scales=sigma_scales,
            neighbor_delta_lambdas_nm=neighbor_deltas,
        )

        weights, mode = sample_leakage_weights(n_gratings, target_index, lw, rng)
        leak_scale = float(regime.get("leakage_scale", 1.0))
        if abs(leak_scale - 1.0) > 1e-15:
            for j in range(n_gratings):
                if j != target_index:
                    weights[j] *= leak_scale
        local_clean, _ = extract_local_distorted_spectrum(per_grating, target_index, lw, weights=weights)

        shift_min = float(lw.get("target_window_shift_min_nm", 0.0))
        shift_max = float(lw.get("target_window_shift_max_nm", 0.0))
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
        regime_id_all[i] = np.int64(regime_idx)
        neighbor_delta_all[i] = neighbor_deltas.astype(np.float32)

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
        regime_id=regime_id_all,
        neighbor_delta_lambdas_nm=neighbor_delta_all,
        regime_names=regime_names,
        idx_train=idx_train.astype(np.int64),
        idx_val=idx_val.astype(np.int64),
        idx_test=idx_test.astype(np.int64),
        target_index=np.array([target_index], dtype=np.int64),
    )
    print(f"Saved dataset: {dataset_path}")
    print(f"X_local shape: {x_local.shape}")
    print(f"Split sizes: train={len(idx_train)}, val={len(idx_val)}, test={len(idx_test)}")
    if len(regime_names) > 0:
        counts = np.bincount(np.maximum(regime_id_all, 0), minlength=len(regime_names))
        summary = ", ".join([f"{regime_names[i]}={int(counts[i])}" for i in range(len(regime_names))])
        print(f"Regime distribution: {summary}")


if __name__ == "__main__":
    main()
