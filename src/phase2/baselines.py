from __future__ import annotations

import numpy as np


def normalize_minmax(x: np.ndarray) -> np.ndarray:
    x_min = x.min()
    x_max = x.max()
    return (x - x_min) / (x_max - x_min + 1e-12)


def gaussian_spectrum(
    lambda_axis: np.ndarray,
    center_nm: float,
    sigma_nm: float,
    amplitude: float = 1.0,
    baseline: float = 0.0,
) -> np.ndarray:
    return baseline + amplitude * np.exp(-0.5 * ((lambda_axis - center_nm) / sigma_nm) ** 2)


def estimate_shift_by_cross_correlation(reference: np.ndarray, sample: np.ndarray, step_nm: float) -> float:
    ref = reference - reference.mean()
    sig = sample - sample.mean()
    corr = np.correlate(sig, ref, mode="full")
    peak = int(np.argmax(corr))
    lag = peak - (len(reference) - 1)

    delta = 0.0
    if 0 < peak < len(corr) - 1:
        y1, y2, y3 = corr[peak - 1], corr[peak], corr[peak + 1]
        denom = y1 - 2.0 * y2 + y3
        if abs(denom) > 1e-12:
            delta = 0.5 * (y1 - y3) / denom
    return float((lag + delta) * step_nm)


def estimate_center_by_parametric_fit(
    lambda_axis: np.ndarray,
    spectrum: np.ndarray,
    fit_window_points: int = 61,
    baseline_percentile: float = 10.0,
) -> float:
    peak_idx = int(np.argmax(spectrum))
    half = fit_window_points // 2
    left = max(0, peak_idx - half)
    right = min(len(spectrum), peak_idx + half + 1)

    x = lambda_axis[left:right].astype(np.float64)
    y = spectrum[left:right].astype(np.float64)
    if len(x) < 5:
        return float(lambda_axis[peak_idx])

    baseline = np.percentile(y, baseline_percentile)
    z = y - baseline
    valid = z > 1e-10
    if valid.sum() < 5:
        return float(lambda_axis[peak_idx])

    xv = x[valid]
    lv = np.log(z[valid])
    try:
        a2, a1, _ = np.polyfit(xv, lv, deg=2)
    except np.linalg.LinAlgError:
        return float(lambda_axis[peak_idx])

    if a2 >= 0:
        weights = np.clip(z, 0.0, None)
        wsum = float(np.sum(weights))
        if wsum <= 1e-12:
            return float(lambda_axis[peak_idx])
        return float(np.sum(x * weights) / wsum)

    center = -a1 / (2.0 * a2)
    center = np.clip(center, float(x.min()), float(x.max()))
    return float(center)

