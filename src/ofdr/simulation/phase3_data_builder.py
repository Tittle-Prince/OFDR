from __future__ import annotations

import numpy as np

from phase3.common import Phase3Data


def gaussian_spectrum(lambda_axis: np.ndarray, center_nm: float, sigma_nm: float) -> np.ndarray:
    return np.exp(-0.5 * ((lambda_axis - center_nm) / sigma_nm) ** 2)


def normalize_minmax(x: np.ndarray) -> np.ndarray:
    x_min = float(np.min(x))
    x_max = float(np.max(x))
    return (x - x_min) / (x_max - x_min + 1e-12)


def _u(rng: np.random.Generator, lo: float, hi: float) -> float:
    return float(rng.uniform(lo, hi))


def build_dataset_b(source_a: Phase3Data, cfg: dict, rng: np.random.Generator) -> Phase3Data:
    dcfg = cfg["dataset_b"]

    lambda_b0 = float(dcfg["lambda_b0_nm"])
    base_sigma = float(dcfg["base_linewidth_nm"])
    clip_nonneg = bool(dcfg["clip_to_nonnegative"])
    normalize_mode = str(dcfg["normalize"])

    x_norm = np.linspace(-1.0, 1.0, source_a.x.shape[1], dtype=np.float64)
    wavelengths = source_a.wavelengths.astype(np.float64)

    out_x = np.zeros_like(source_a.x, dtype=np.float32)
    out_y_dlambda = source_a.y_dlambda.copy().astype(np.float32)
    out_y_dt = source_a.y_dt.copy().astype(np.float32)

    for i in range(source_a.x.shape[0]):
        dlam = float(out_y_dlambda[i])

        amp = _u(rng, float(dcfg["amplitude_scale_min"]), float(dcfg["amplitude_scale_max"]))
        sigma_scale = _u(rng, float(dcfg["width_scale_min"]), float(dcfg["width_scale_max"]))
        sigma = base_sigma * sigma_scale
        skew = _u(rng, float(dcfg["skew_min"]), float(dcfg["skew_max"]))

        center = lambda_b0 + dlam
        base = amp * gaussian_spectrum(wavelengths, center_nm=center, sigma_nm=sigma)
        wl_span = float(np.ptp(wavelengths))
        skew_term = 1.0 + skew * ((wavelengths - center) / (wl_span + 1e-12))
        shaped = base * skew_term

        offset = _u(rng, float(dcfg["baseline_offset_min"]), float(dcfg["baseline_offset_max"]))
        slope = _u(rng, float(dcfg["baseline_slope_min"]), float(dcfg["baseline_slope_max"]))
        curve = _u(rng, float(dcfg["baseline_curve_min"]), float(dcfg["baseline_curve_max"]))
        baseline = offset + slope * x_norm + curve * (x_norm**2 - 1.0 / 3.0)

        ripple_amp = _u(rng, float(dcfg["ripple_amp_min"]), float(dcfg["ripple_amp_max"]))
        ripple_freq = _u(rng, float(dcfg["ripple_freq_min"]), float(dcfg["ripple_freq_max"]))
        ripple_phase = _u(rng, float(dcfg["ripple_phase_min"]), float(dcfg["ripple_phase_max"]))
        ripple = ripple_amp * np.sin(2.0 * np.pi * ripple_freq * (x_norm + 1.0) * 0.5 + ripple_phase)

        noise_std = _u(rng, float(dcfg["white_noise_std_min"]), float(dcfg["white_noise_std_max"]))
        noise = rng.normal(0.0, noise_std, size=source_a.x.shape[1])

        y = shaped + baseline + ripple + noise
        if clip_nonneg:
            y = np.clip(y, 0.0, None)
        if normalize_mode == "minmax_per_sample":
            y = normalize_minmax(y)

        out_x[i] = y.astype(np.float32)

    return Phase3Data(
        x=out_x,
        y_dlambda=out_y_dlambda,
        y_dt=out_y_dt,
        wavelengths=source_a.wavelengths.copy(),
        idx_train=source_a.idx_train.copy(),
        idx_val=source_a.idx_val.copy(),
        idx_test=source_a.idx_test.copy(),
    )
