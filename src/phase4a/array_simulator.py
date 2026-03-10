from __future__ import annotations

import numpy as np


def gaussian_reflectivity(wavelengths: np.ndarray, center_nm: float, sigma_nm: float, amplitude: float = 1.0) -> np.ndarray:
    return amplitude * np.exp(-0.5 * ((wavelengths - center_nm) / sigma_nm) ** 2)


def build_wavelength_axis(cfg: dict) -> np.ndarray:
    arr = cfg["array"]
    return np.linspace(
        float(arr["wavelength_start_nm"]),
        float(arr["wavelength_end_nm"]),
        int(arr["num_points"]),
        dtype=np.float64,
    )


def simulate_identical_array_spectra(
    wavelengths: np.ndarray,
    cfg: dict,
    delta_lambda_target_nm: float,
    amplitude_scales: np.ndarray | None = None,
    linewidth_scales: np.ndarray | None = None,
    neighbor_delta_lambdas_nm: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    arr = cfg["array"]
    label = cfg["label"]

    n_gratings = int(arr["n_gratings"])
    target_index = int(arr["target_index"])
    if not (0 <= target_index < n_gratings):
        raise ValueError("target_index must be within [0, n_gratings)")

    lambda0 = float(arr["lambda0_nm"])
    sigma = float(arr["linewidth_sigma_nm"])
    amplitude = float(arr["amplitude"])

    neighbor_shift = float(label["delta_lambda_neighbors_nm"])
    centers = np.full(n_gratings, lambda0 + neighbor_shift, dtype=np.float64)
    if neighbor_delta_lambdas_nm is not None:
        shifts = np.asarray(neighbor_delta_lambdas_nm, dtype=np.float64).reshape(-1)
        if len(shifts) != n_gratings:
            raise ValueError("neighbor_delta_lambdas_nm must have shape (n_gratings,)")
        centers = lambda0 + shifts
    centers[target_index] = lambda0 + float(delta_lambda_target_nm)

    if amplitude_scales is None:
        amplitude_scales = np.ones(n_gratings, dtype=np.float64)
    if linewidth_scales is None:
        linewidth_scales = np.ones(n_gratings, dtype=np.float64)
    if len(amplitude_scales) != n_gratings or len(linewidth_scales) != n_gratings:
        raise ValueError("amplitude_scales and linewidth_scales must match n_gratings")

    per_grating = np.zeros((n_gratings, len(wavelengths)), dtype=np.float64)
    for i in range(n_gratings):
        per_grating[i] = gaussian_reflectivity(
            wavelengths,
            centers[i],
            sigma_nm=sigma * float(linewidth_scales[i]),
            amplitude=amplitude * float(amplitude_scales[i]),
        )

    total = np.sum(per_grating, axis=0)
    return per_grating, total, centers
