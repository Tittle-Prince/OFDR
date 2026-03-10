from __future__ import annotations

import numpy as np


def normalize_minmax(x: np.ndarray) -> np.ndarray:
    x_min = float(np.min(x))
    x_max = float(np.max(x))
    return (x - x_min) / (x_max - x_min + 1e-12)


def distance_weight(distance: int, cfg_local: dict) -> float:
    if distance == 0:
        return float(cfg_local["center_weight"])
    if distance == 1:
        return float(cfg_local["first_neighbor_weight"])
    if distance == 2:
        return float(cfg_local["second_neighbor_weight"])
    return 0.0


def _sample_range(cfg_local: dict, key_min: str, key_max: str, rng: np.random.Generator, default_val: float) -> float:
    if key_min in cfg_local and key_max in cfg_local:
        return float(rng.uniform(float(cfg_local[key_min]), float(cfg_local[key_max])))
    return float(default_val)


def _sample_int_range(cfg_local: dict, key_min: str, key_max: str, rng: np.random.Generator, default_val: int) -> int:
    if key_min in cfg_local and key_max in cfg_local:
        lo = int(cfg_local[key_min])
        hi = int(cfg_local[key_max])
        if hi < lo:
            hi = lo
        return int(rng.integers(lo, hi + 1))
    return int(default_val)


def _gaussian_kernel1d(sigma_points: float) -> np.ndarray:
    sigma = float(max(1e-6, sigma_points))
    radius = int(max(1, round(4.0 * sigma)))
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    k = np.exp(-0.5 * (x / sigma) ** 2)
    k /= np.sum(k)
    return k


def _smooth_noise(n_points: int, sigma_points: float, rng: np.random.Generator) -> np.ndarray:
    raw = rng.normal(0.0, 1.0, size=n_points)
    k = _gaussian_kernel1d(sigma_points)
    return np.convolve(raw, k, mode="same")


def sample_leakage_weights(
    n_gratings: int,
    target_index: int,
    cfg_local: dict,
    rng: np.random.Generator,
) -> tuple[np.ndarray, int]:
    weights = np.zeros(n_gratings, dtype=np.float64)
    weights[target_index] = float(cfg_local.get("center_weight", 1.0))

    modes = cfg_local.get("neighbor_mode_options", [4])
    mode = int(rng.choice(modes)) if len(modes) > 0 else 4
    mode = 2 if mode == 2 else 4

    # First-order neighbors (always enabled in mode 2/4)
    if target_index - 1 >= 0:
        weights[target_index - 1] = _sample_range(
            cfg_local,
            "first_neighbor_left_weight_min",
            "first_neighbor_left_weight_max",
            rng,
            cfg_local.get("first_neighbor_weight", 0.35),
        )
    if target_index + 1 < n_gratings:
        weights[target_index + 1] = _sample_range(
            cfg_local,
            "first_neighbor_right_weight_min",
            "first_neighbor_right_weight_max",
            rng,
            cfg_local.get("first_neighbor_weight", 0.35),
        )

    # Second-order neighbors (enabled only in mode 4)
    if mode == 4:
        if target_index - 2 >= 0:
            weights[target_index - 2] = _sample_range(
                cfg_local,
                "second_neighbor_left_weight_min",
                "second_neighbor_left_weight_max",
                rng,
                cfg_local.get("second_neighbor_weight", 0.15),
            )
        if target_index + 2 < n_gratings:
            weights[target_index + 2] = _sample_range(
                cfg_local,
                "second_neighbor_right_weight_min",
                "second_neighbor_right_weight_max",
                rng,
                cfg_local.get("second_neighbor_weight", 0.15),
            )

    leak_scale = _sample_range(
        cfg_local,
        "leakage_global_scale_min",
        "leakage_global_scale_max",
        rng,
        1.0,
    )
    if abs(leak_scale - 1.0) > 1e-15:
        for i in range(n_gratings):
            if i != target_index:
                weights[i] *= leak_scale

    tilt = _sample_range(
        cfg_local,
        "leakage_left_right_tilt_min",
        "leakage_left_right_tilt_max",
        rng,
        0.0,
    )
    if abs(tilt) > 1e-15:
        for i in range(n_gratings):
            if i < target_index:
                weights[i] *= max(0.0, 1.0 + tilt)
            elif i > target_index:
                weights[i] *= max(0.0, 1.0 - tilt)
    return weights, mode


def extract_local_distorted_spectrum(
    per_grating: np.ndarray,
    target_index: int,
    cfg_local: dict,
    weights: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    n_gratings = per_grating.shape[0]
    if weights is None:
        weights = np.zeros(n_gratings, dtype=np.float64)
        for i in range(n_gratings):
            dist = abs(i - target_index)
            weights[i] = distance_weight(dist, cfg_local)
    local_clean = np.sum(weights[:, None] * per_grating, axis=0)
    return local_clean, weights


def apply_window_shift(local_clean: np.ndarray, wavelengths: np.ndarray, shift_nm: float) -> np.ndarray:
    if abs(shift_nm) < 1e-15:
        return local_clean
    source_wl = wavelengths - float(shift_nm)
    shifted = np.interp(
        wavelengths,
        source_wl,
        local_clean,
        left=float(local_clean[0]),
        right=float(local_clean[-1]),
    )
    return shifted


def generate_smooth_lowfreq_baseline(n_points: int, cfg_local: dict, rng: np.random.Generator) -> np.ndarray:
    x = np.linspace(0.0, 1.0, n_points, dtype=np.float64)
    baseline = np.full(n_points, float(rng.uniform(float(cfg_local.get("baseline_offset_min", 0.0)), float(cfg_local.get("baseline_offset_max", 0.0)))), dtype=np.float64)

    n_min = int(cfg_local.get("baseline_num_components_min", 0))
    n_max = int(cfg_local.get("baseline_num_components_max", 0))
    if n_max < n_min:
        n_max = n_min
    n_comp = int(rng.integers(n_min, n_max + 1)) if n_max > 0 else 0

    amp_min = float(cfg_local.get("baseline_component_amp_min", 0.0))
    amp_max = float(cfg_local.get("baseline_component_amp_max", 0.0))
    freq_min = float(cfg_local.get("baseline_freq_min", 0.3))
    freq_max = float(cfg_local.get("baseline_freq_max", 2.0))

    for _ in range(n_comp):
        amp = float(rng.uniform(amp_min, amp_max))
        freq = float(rng.uniform(freq_min, freq_max))
        phase = float(rng.uniform(0.0, 2.0 * np.pi))
        baseline += amp * np.sin(2.0 * np.pi * freq * x + phase)

    return baseline


def _apply_dropout_notches(signal: np.ndarray, cfg_local: dict, rng: np.random.Generator) -> np.ndarray:
    prob = float(cfg_local.get("dropout_prob", 0.0))
    if prob <= 0.0 or float(rng.uniform(0.0, 1.0)) >= prob:
        return signal

    out = signal.copy()
    n_points = len(out)
    count = _sample_int_range(cfg_local, "dropout_count_min", "dropout_count_max", rng, 1)
    width_default = max(3.0, 0.015 * n_points)

    idx = np.arange(n_points, dtype=np.float64)
    for _ in range(count):
        center = int(rng.integers(0, n_points))
        width = _sample_range(cfg_local, "dropout_width_points_min", "dropout_width_points_max", rng, width_default)
        depth = _sample_range(cfg_local, "dropout_depth_min", "dropout_depth_max", rng, 0.15)
        notch = 1.0 - max(0.0, depth) * np.exp(-0.5 * ((idx - center) / max(1.0, width)) ** 2)
        out *= notch
    return out


def _apply_highfreq_ripple(signal: np.ndarray, cfg_local: dict, rng: np.random.Generator) -> np.ndarray:
    amp = _sample_range(cfg_local, "hf_ripple_amp_min", "hf_ripple_amp_max", rng, 0.0)
    if amp <= 0.0:
        return signal

    x = np.linspace(0.0, 1.0, len(signal), dtype=np.float64)
    freq = _sample_range(cfg_local, "hf_ripple_freq_min", "hf_ripple_freq_max", rng, 8.0)
    phase = _sample_range(cfg_local, "hf_ripple_phase_min", "hf_ripple_phase_max", rng, float(rng.uniform(0.0, 2.0 * np.pi)))
    return signal + amp * np.sin(2.0 * np.pi * freq * x + phase)


def _apply_blur(signal: np.ndarray, cfg_local: dict, rng: np.random.Generator) -> np.ndarray:
    sigma = _sample_range(cfg_local, "instrument_blur_sigma_points_min", "instrument_blur_sigma_points_max", rng, 0.0)
    if sigma <= 1e-12:
        return signal
    k = _gaussian_kernel1d(sigma)
    return np.convolve(signal, k, mode="same")


def _apply_smooth_colored_noise(signal: np.ndarray, cfg_local: dict, rng: np.random.Generator) -> np.ndarray:
    std = _sample_range(cfg_local, "colored_noise_std_min", "colored_noise_std_max", rng, 0.0)
    if std <= 0.0:
        return signal

    n_points = len(signal)
    smooth = _sample_range(cfg_local, "colored_noise_smooth_points_min", "colored_noise_smooth_points_max", rng, 10.0)
    colored = _smooth_noise(n_points, sigma_points=max(1.0, smooth), rng=rng)
    colored = colored / (float(np.std(colored)) + 1e-12) * std
    return signal + colored


def _apply_impulsive_noise(signal: np.ndarray, cfg_local: dict, rng: np.random.Generator) -> np.ndarray:
    prob = float(cfg_local.get("impulse_prob", 0.0))
    if prob <= 0.0 or float(rng.uniform(0.0, 1.0)) >= prob:
        return signal

    out = signal.copy()
    n_points = len(out)
    count = _sample_int_range(cfg_local, "impulse_count_min", "impulse_count_max", rng, 1)
    amp_min = float(cfg_local.get("impulse_amp_min", 0.0))
    amp_max = float(cfg_local.get("impulse_amp_max", amp_min))
    if amp_max < amp_min:
        amp_max = amp_min
    for _ in range(count):
        idx = int(rng.integers(0, n_points))
        amp = float(rng.uniform(amp_min, amp_max))
        sign = -1.0 if float(rng.uniform(0.0, 1.0)) < 0.5 else 1.0
        out[idx] += sign * amp
    return out


def apply_local_effects(
    local_clean: np.ndarray,
    cfg_local: dict,
    rng: np.random.Generator,
    wavelengths: np.ndarray | None = None,
    shift_nm: float = 0.0,
) -> np.ndarray:
    signal = local_clean
    if wavelengths is not None:
        signal = apply_window_shift(signal, wavelengths, shift_nm)

    baseline = generate_smooth_lowfreq_baseline(signal.shape[0], cfg_local, rng)
    noisy = signal + baseline
    noisy = _apply_highfreq_ripple(noisy, cfg_local, rng)
    noisy = _apply_dropout_notches(noisy, cfg_local, rng)
    noisy = _apply_blur(noisy, cfg_local, rng)
    noisy = _apply_smooth_colored_noise(noisy, cfg_local, rng)

    noise_std = _sample_range(
        cfg_local,
        "additive_noise_std_min",
        "additive_noise_std_max",
        rng,
        float(cfg_local.get("additive_noise_std", 0.0)),
    )
    noisy = noisy + rng.normal(0.0, noise_std, size=signal.shape[0])
    noisy = _apply_impulsive_noise(noisy, cfg_local, rng)

    gain = _sample_range(cfg_local, "gain_min", "gain_max", rng, 1.0)
    bias = _sample_range(cfg_local, "post_bias_min", "post_bias_max", rng, 0.0)
    noisy = gain * noisy + bias

    sat = _sample_range(cfg_local, "saturation_level_min", "saturation_level_max", rng, 0.0)
    if sat > 0.0:
        noisy = sat * np.tanh(noisy / sat)

    gamma = _sample_range(cfg_local, "gamma_min", "gamma_max", rng, 1.0)
    if abs(gamma - 1.0) > 1e-12:
        noisy = np.sign(noisy) * (np.abs(noisy) + 1e-12) ** gamma

    if bool(cfg_local["clip_to_nonnegative"]):
        noisy = np.clip(noisy, 0.0, None)
    if str(cfg_local["normalize"]) == "minmax_per_sample":
        noisy = normalize_minmax(noisy)
    return noisy.astype(np.float32)
