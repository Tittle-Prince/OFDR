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
    p = argparse.ArgumentParser(description="Generate isolated time-series dataset for Phase4 hard mode")
    p.add_argument("--config", type=str, default="config/phase4_array_se_hard_ts.yaml")
    return p.parse_args()


def _build_regime_pool(cfg: dict) -> tuple[list[dict], np.ndarray]:
    regs = cfg.get("se_regimes", [])
    if not isinstance(regs, list) or len(regs) == 0:
        return [], np.zeros(0, dtype=np.float64)
    out = []
    probs = []
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


def _sample_neighbor_base(label_cfg: dict, n_gratings: int, target_index: int, rng: np.random.Generator) -> np.ndarray:
    base = float(label_cfg.get("delta_lambda_neighbors_nm", 0.0))
    mode = str(label_cfg.get("neighbor_shift_mode", "fixed")).lower()
    out = np.full(n_gratings, base, dtype=np.float64)
    if mode in {"random_shared", "shared_random"}:
        lo = float(label_cfg.get("neighbor_delta_lambda_min_nm", base))
        hi = float(label_cfg.get("neighbor_delta_lambda_max_nm", base))
        if hi < lo:
            hi = lo
        out.fill(float(rng.uniform(lo, hi)))
    elif mode in {"random_per_neighbor", "per_neighbor"}:
        lo = float(label_cfg.get("neighbor_delta_lambda_min_nm", base))
        hi = float(label_cfg.get("neighbor_delta_lambda_max_nm", base))
        if hi < lo:
            hi = lo
        out = rng.uniform(lo, hi, size=n_gratings).astype(np.float64)
    out[target_index] = base
    return out


def _traj_split(
    n_traj: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-8:
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")
    order = rng.permutation(n_traj)
    n_train = int(n_traj * train_ratio)
    n_val = int(n_traj * val_ratio)
    train_ids = np.sort(order[:n_train])
    val_ids = np.sort(order[n_train : n_train + n_val])
    test_ids = np.sort(order[n_train + n_val :])
    return train_ids.astype(np.int64), val_ids.astype(np.int64), test_ids.astype(np.int64)


def main() -> None:
    args = parse_args()
    ts_cfg = load_config(args.config)
    src_cfg = load_config(ts_cfg["phase4ts"]["source_config"])

    seed = int(ts_cfg["phase4ts"]["seed"])
    set_seed(seed)
    rng = np.random.default_rng(seed)

    dataset_cfg = ts_cfg["dataset_ts"]
    dyn_cfg = dataset_cfg["dynamics"]
    n_traj = int(dataset_cfg["num_trajectories"])
    t_len = int(dataset_cfg["trajectory_length"])

    wavelengths = build_wavelength_axis(src_cfg)
    n_points = len(wavelengths)
    n_gratings = int(src_cfg["array"]["n_gratings"])
    target_index = int(src_cfg["array"]["target_index"])
    n_total = n_traj * t_len

    x_local = np.zeros((n_total, n_points), dtype=np.float32)
    x_total = np.zeros((n_total, n_points), dtype=np.float32)
    y_dlambda = np.zeros(n_total, dtype=np.float32)
    centers_all = np.zeros((n_total, n_gratings), dtype=np.float32)
    traj_id = np.zeros(n_total, dtype=np.int64)
    t_index = np.zeros(n_total, dtype=np.int64)
    regime_id = np.full(n_total, -1, dtype=np.int64)
    leakage_weights = np.zeros((n_total, n_gratings), dtype=np.float32)
    neighbor_delta = np.zeros((n_total, n_gratings), dtype=np.float32)
    amp_scales_all = np.zeros((n_total, n_gratings), dtype=np.float32)
    lw_scales_all = np.zeros((n_total, n_gratings), dtype=np.float32)
    window_shift_all = np.zeros(n_total, dtype=np.float32)
    neighbor_mode_all = np.zeros(n_total, dtype=np.int64)

    arr_rand = src_cfg.get("array_random", {})
    amp_min = float(arr_rand.get("amplitude_scale_min", 1.0))
    amp_max = float(arr_rand.get("amplitude_scale_max", 1.0))
    lw_min = float(arr_rand.get("linewidth_scale_min", 1.0))
    lw_max = float(arr_rand.get("linewidth_scale_max", 1.0))

    label_cfg = src_cfg.get("label", {})
    local_base = src_cfg["local_window"]
    use_regimes = bool(dataset_cfg.get("use_se_regimes", True))
    regime_stay_prob = float(dataset_cfg.get("regime_stay_prob", 0.92))
    regimes, regime_probs = _build_regime_pool(src_cfg if use_regimes else {})
    regime_names = np.array([str(r.get("name", f"regime_{i}")) for i, r in enumerate(regimes)], dtype=np.str_)

    tr_ids, va_ids, te_ids = _traj_split(
        n_traj=n_traj,
        train_ratio=float(dataset_cfg["train_ratio"]),
        val_ratio=float(dataset_cfg["val_ratio"]),
        test_ratio=float(dataset_cfg["test_ratio"]),
        rng=rng,
    )

    d_min = float(dyn_cfg["dlambda_init_min_nm"])
    d_max = float(dyn_cfg["dlambda_init_max_nm"])
    vel_std = float(dyn_cfg["velocity_std_nm"])
    vel_rho = float(dyn_cfg["velocity_rho"])
    jump_prob = float(dyn_cfg["jump_prob"])
    jump_min = float(dyn_cfg["jump_min_nm"])
    jump_max = float(dyn_cfg["jump_max_nm"])
    clip_min = float(dyn_cfg["clip_min_nm"])
    clip_max = float(dyn_cfg["clip_max_nm"])

    shift_rw_std = float(dyn_cfg["window_shift_rw_std_nm"])
    neighbor_rw_std = float(dyn_cfg["neighbor_rw_std_nm"])
    amp_rw_std = float(dyn_cfg["amp_rw_std"])
    lw_rw_std = float(dyn_cfg["linewidth_rw_std"])
    leak_rw_std = float(dyn_cfg["leakage_rw_std"])

    coupling = float(label_cfg.get("neighbor_target_coupling", 0.0))
    n_jit_lo = float(label_cfg.get("neighbor_jitter_min_nm", 0.0))
    n_jit_hi = float(label_cfg.get("neighbor_jitter_max_nm", 0.0))
    if n_jit_hi < n_jit_lo:
        n_jit_hi = n_jit_lo

    print(
        f"Generate time-series dataset | trajectories={n_traj} | length={t_len} | "
        f"total_samples={n_total} | seed={seed}"
    )
    for tr in range(n_traj):
        if tr % max(1, n_traj // 10) == 0 or tr == n_traj - 1:
            print(f"  trajectory {tr + 1}/{n_traj} ({100.0 * (tr + 1) / n_traj:5.1f}%)")

        # Target drift trajectory: AR(1) velocity + random jump.
        d_seq = np.zeros(t_len, dtype=np.float64)
        d_seq[0] = float(rng.uniform(d_min, d_max))
        vel = float(rng.normal(0.0, vel_std))
        for t in range(1, t_len):
            vel = vel_rho * vel + float(rng.normal(0.0, vel_std))
            jump = float(rng.uniform(jump_min, jump_max)) if float(rng.uniform(0.0, 1.0)) < jump_prob else 0.0
            d_seq[t] = np.clip(d_seq[t - 1] + vel + jump, clip_min, clip_max)

        # Time-correlated nuisance parameters.
        amp_state = rng.uniform(amp_min, amp_max, size=n_gratings).astype(np.float64)
        lw_state = rng.uniform(lw_min, lw_max, size=n_gratings).astype(np.float64)
        n_base = _sample_neighbor_base(label_cfg, n_gratings, target_index, rng)
        n_rw = np.zeros(n_gratings, dtype=np.float64)
        win_shift = 0.0

        local_for_weights = dict(local_base)
        w_prev, mode_prev = sample_leakage_weights(n_gratings, target_index, local_for_weights, rng)

        rid = int(rng.choice(len(regimes), p=regime_probs)) if len(regimes) > 0 else -1
        for t in range(t_len):
            if len(regimes) > 0 and t > 0 and float(rng.uniform(0.0, 1.0)) > regime_stay_prob:
                rid = int(rng.choice(len(regimes), p=regime_probs))
            regime = regimes[rid] if (rid >= 0 and rid < len(regimes)) else {}
            lw_cfg = _apply_regime_to_local_config(local_base, regime)

            amp_state = np.clip(amp_state + rng.normal(0.0, amp_rw_std, size=n_gratings), amp_min * 0.6, amp_max * 1.4)
            lw_state = np.clip(lw_state + rng.normal(0.0, lw_rw_std, size=n_gratings), lw_min * 0.7, lw_max * 1.3)

            n_rw = 0.92 * n_rw + rng.normal(0.0, neighbor_rw_std, size=n_gratings)
            n_shift = n_base + n_rw + coupling * float(d_seq[t]) + rng.uniform(n_jit_lo, n_jit_hi, size=n_gratings)
            n_shift[target_index] = float(d_seq[t])

            win_shift = 0.9 * win_shift + float(rng.normal(0.0, shift_rw_std))
            shift_min = float(lw_cfg.get("target_window_shift_min_nm", -0.06))
            shift_max = float(lw_cfg.get("target_window_shift_max_nm", 0.06))
            win_shift = float(np.clip(win_shift, shift_min, shift_max))

            per_grating, total, centers = simulate_identical_array_spectra(
                wavelengths,
                src_cfg,
                delta_lambda_target_nm=float(d_seq[t]),
                amplitude_scales=amp_state,
                linewidth_scales=lw_state,
                neighbor_delta_lambdas_nm=n_shift,
            )

            w_samp, mode = sample_leakage_weights(n_gratings, target_index, lw_cfg, rng)
            w_new = 0.82 * w_prev + 0.18 * w_samp + rng.normal(0.0, leak_rw_std, size=n_gratings)
            w_new = np.clip(w_new, 0.0, None)
            w_new[target_index] = float(lw_cfg.get("center_weight", 1.0))
            w_prev = w_new
            mode_prev = mode

            local_clean, _ = extract_local_distorted_spectrum(per_grating, target_index, lw_cfg, weights=w_new)
            local_noisy = apply_local_effects(local_clean, lw_cfg, rng, wavelengths=wavelengths, shift_nm=win_shift)

            idx = tr * t_len + t
            x_local[idx] = local_noisy
            x_total[idx] = total.astype(np.float32)
            y_dlambda[idx] = np.float32(d_seq[t])
            centers_all[idx] = centers.astype(np.float32)
            traj_id[idx] = np.int64(tr)
            t_index[idx] = np.int64(t)
            regime_id[idx] = np.int64(rid)
            leakage_weights[idx] = w_new.astype(np.float32)
            neighbor_delta[idx] = n_shift.astype(np.float32)
            amp_scales_all[idx] = amp_state.astype(np.float32)
            lw_scales_all[idx] = lw_state.astype(np.float32)
            window_shift_all[idx] = np.float32(win_shift)
            neighbor_mode_all[idx] = np.int64(mode_prev)

    is_train = np.isin(traj_id, tr_ids)
    is_val = np.isin(traj_id, va_ids)
    is_test = np.isin(traj_id, te_ids)
    idx_train = np.flatnonzero(is_train).astype(np.int64)
    idx_val = np.flatnonzero(is_val).astype(np.int64)
    idx_test = np.flatnonzero(is_test).astype(np.int64)

    out_path = resolve_project_path(ts_cfg["phase4ts"]["dataset_path"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_path,
        X_local=x_local,
        X_total=x_total,
        Y_dlambda_target=y_dlambda,
        wavelengths=wavelengths.astype(np.float32),
        centers_nm=centers_all,
        traj_id=traj_id.astype(np.int64),
        t_index=t_index.astype(np.int64),
        regime_id=regime_id.astype(np.int64),
        regime_names=regime_names,
        leakage_weights=leakage_weights,
        neighbor_delta_lambdas_nm=neighbor_delta,
        amplitude_scales=amp_scales_all,
        linewidth_scales=lw_scales_all,
        window_shift_nm=window_shift_all,
        neighbor_mode=neighbor_mode_all,
        traj_ids_train=tr_ids,
        traj_ids_val=va_ids,
        traj_ids_test=te_ids,
        idx_train=idx_train,
        idx_val=idx_val,
        idx_test=idx_test,
        target_index=np.array([target_index], dtype=np.int64),
    )
    print(f"Saved dataset: {out_path}")
    print(f"Shapes: X_local={x_local.shape}, X_total={x_total.shape}, Y={y_dlambda.shape}")
    print(f"Frame split sizes: train={len(idx_train)}, val={len(idx_val)}, test={len(idx_test)}")
    print(f"Trajectory split sizes: train={len(tr_ids)}, val={len(va_ids)}, test={len(te_ids)}")


if __name__ == "__main__":
    main()

