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

    if bool(cfg_local["clip_to_nonnegative"]):
        noisy = np.clip(noisy, 0.0, None)
    if str(cfg_local["normalize"]) == "minmax_per_sample":
        noisy = normalize_minmax(noisy)
    return noisy.astype(np.float32)
