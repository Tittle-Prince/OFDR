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


def _apply_spike_noise(signal: np.ndarray, cfg_local: dict, rng: np.random.Generator) -> np.ndarray:
    # Backward-compatible: disabled by default unless explicitly enabled.
    if not bool(cfg_local.get("spike_noise_enable", False)):
        return signal

    out = signal.copy()
    n_points = len(out)

    n_min = int(cfg_local.get("spike_noise_num_min", 0))
    n_max = int(cfg_local.get("spike_noise_num_max", n_min))
    if n_max < n_min:
        n_max = n_min
    n_spikes = int(rng.integers(n_min, n_max + 1))
    if n_spikes <= 0:
        return out

    amp_min = float(cfg_local.get("spike_noise_amp_min", 0.0))
    amp_max = float(cfg_local.get("spike_noise_amp_max", amp_min))
    if amp_max < amp_min:
        amp_max = amp_min

    w_min = int(cfg_local.get("spike_noise_width_points_min", 1))
    w_max = int(cfg_local.get("spike_noise_width_points_max", w_min))
    if w_max < w_min:
        w_max = w_min
    w_min = max(1, w_min)
    w_max = max(1, w_max)

    for _ in range(n_spikes):
        pos = int(rng.integers(0, n_points))
        amp = float(rng.uniform(amp_min, amp_max))
        sign = -1.0 if float(rng.uniform(0.0, 1.0)) < 0.5 else 1.0
        width = int(rng.integers(w_min, w_max + 1))

        left = int(max(0, pos - width // 2))
        right = int(min(n_points, left + width))
        if right <= left:
            continue

        span = right - left
        if span == 1:
            out[left] += sign * amp
        else:
            shape = np.hanning(span + 2)[1:-1]
            shape = shape / (float(np.max(shape)) + 1e-12)
            out[left:right] += sign * amp * shape
    return out


def _sample_sign(cfg_local: dict, key: str, rng: np.random.Generator) -> float:
    mode = str(cfg_local.get(key, "both")).lower()
    if mode == "positive":
        return 1.0
    if mode == "negative":
        return -1.0
    return -1.0 if float(rng.uniform(0.0, 1.0)) < 0.5 else 1.0


def _artifact_region_bounds(signal: np.ndarray, cfg_local: dict) -> tuple[int, int]:
    n_points = len(signal)
    mode = str(cfg_local.get("targeted_artifact_region_mode", "center_band")).lower()

    if mode == "target_peak_nearby":
        center = int(np.argmax(signal))
        half = int(cfg_local.get("targeted_artifact_peak_half_window_points", max(10, n_points // 32)))
        left = max(0, center - half)
        right = min(n_points, center + half + 1)
        if right - left >= 3:
            return left, right

    left_ratio = float(cfg_local.get("targeted_artifact_region_left_ratio", 0.35))
    right_ratio = float(cfg_local.get("targeted_artifact_region_right_ratio", 0.65))
    left = int(np.floor(np.clip(left_ratio, 0.0, 1.0) * n_points))
    right = int(np.ceil(np.clip(right_ratio, 0.0, 1.0) * n_points))
    left = max(0, min(left, n_points - 1))
    right = max(left + 2, min(right, n_points))
    return left, right


def _apply_targeted_artifacts(signal: np.ndarray, cfg_local: dict, rng: np.random.Generator) -> np.ndarray:
    # Backward-compatible: disabled by default unless explicitly enabled.
    if not bool(cfg_local.get("targeted_artifact_enable", False)):
        return signal

    out = signal.copy()
    n_min = int(cfg_local.get("targeted_artifact_num_min", 0))
    n_max = int(cfg_local.get("targeted_artifact_num_max", n_min))
    if n_max < n_min:
        n_max = n_min
    n_art = int(rng.integers(n_min, n_max + 1))
    if n_art <= 0:
        return out

    amp_min = float(cfg_local.get("targeted_artifact_amp_min", 0.01))
    amp_max = float(cfg_local.get("targeted_artifact_amp_max", amp_min))
    if amp_max < amp_min:
        amp_max = amp_min

    w_min = int(cfg_local.get("targeted_artifact_width_points_min", 1))
    w_max = int(cfg_local.get("targeted_artifact_width_points_max", w_min))
    if w_max < w_min:
        w_max = w_min
    w_min = max(1, w_min)
    w_max = max(1, w_max)

    region_l, region_r = _artifact_region_bounds(signal, cfg_local)
    if region_r - region_l < 2:
        return out

    idx = np.arange(len(out), dtype=np.float64)
    for _ in range(n_art):
        c = int(rng.integers(region_l, region_r))
        amp = float(rng.uniform(amp_min, amp_max))
        sign = _sample_sign(cfg_local, "targeted_artifact_sign_mode", rng)
        width = int(rng.integers(w_min, w_max + 1))

        # Narrow Gaussian bump to mimic local burr/spur around key region.
        sigma = max(0.5, 0.5 * float(width))
        g = np.exp(-0.5 * ((idx - float(c)) / sigma) ** 2)
        g /= float(np.max(g)) + 1e-12
        out += sign * amp * g
    return out


def _apply_hf_ripple(signal: np.ndarray, cfg_local: dict, rng: np.random.Generator) -> np.ndarray:
    # Backward-compatible: disabled by default unless explicitly enabled.
    if not bool(cfg_local.get("hf_ripple_enable", False)):
        return signal

    amp_min = float(cfg_local.get("hf_ripple_amp_min", 0.0))
    amp_max = float(cfg_local.get("hf_ripple_amp_max", amp_min))
    if amp_max < amp_min:
        amp_max = amp_min
    amp = float(rng.uniform(amp_min, amp_max))
    if amp <= 0.0:
        return signal

    f_min = float(cfg_local.get("hf_ripple_freq_min", 8.0))
    f_max = float(cfg_local.get("hf_ripple_freq_max", f_min))
    if f_max < f_min:
        f_max = f_min
    freq = float(rng.uniform(f_min, f_max))

    phase_random = bool(cfg_local.get("hf_ripple_phase_random", True))
    if phase_random:
        phase = float(rng.uniform(0.0, 2.0 * np.pi))
    else:
        phase = float(cfg_local.get("hf_ripple_phase", 0.0))

    x = np.linspace(0.0, 1.0, len(signal), dtype=np.float64)
    ripple = amp * np.sin(2.0 * np.pi * freq * x + phase)
    return signal + ripple


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
    noise_std = float(cfg_local["additive_noise_std"])
    noisy = signal + baseline + rng.normal(0.0, noise_std, size=signal.shape[0])
    noisy = _apply_hf_ripple(noisy, cfg_local, rng)
    noisy = _apply_spike_noise(noisy, cfg_local, rng)
    noisy = _apply_targeted_artifacts(noisy, cfg_local, rng)

    if bool(cfg_local["clip_to_nonnegative"]):
        noisy = np.clip(noisy, 0.0, None)
    if str(cfg_local["normalize"]) == "minmax_per_sample":
        noisy = normalize_minmax(noisy)
    return noisy.astype(np.float32)
