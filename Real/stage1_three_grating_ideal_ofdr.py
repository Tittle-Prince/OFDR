from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import find_peaks
from scipy.signal.windows import hann, tukey


@dataclass
class SimulationConfig:
    # Optical axis configuration.
    lambda_center_m: float = 1550e-9
    scan_span_m: float = 2.0e-9
    num_samples: int = 32768

    # Effective index used in the OFDR phase term.
    n_eff: float = 1.4682

    # Three-grating spectral parameters.
    lambda_left_m: float = 1549.70e-9
    lambda_target_m: float = 1550.00e-9
    lambda_right_m: float = 1550.30e-9
    sigma_left_m: float = 0.08e-9
    sigma_target_m: float = 0.08e-9
    sigma_right_m: float = 0.08e-9
    amplitude_left: float = 0.90
    amplitude_target: float = 1.00
    amplitude_right: float = 0.92

    # Spatial positions of the three gratings.
    z_left_m: float = 2.0
    z_target_m: float = 4.0
    z_right_m: float = 6.0

    # Ideal reference interferometer for later k-linearization interface.
    z_ref_m: float = 0.3

    # Signal processing.
    target_gate_width_m: float = 0.8
    output_dir: str = "Real/stage1_three_grating_ideal_ofdr_outputs"


def build_lambda_axis(cfg: SimulationConfig) -> np.ndarray:
    lambda_start = cfg.lambda_center_m - 0.5 * cfg.scan_span_m
    lambda_end = cfg.lambda_center_m + 0.5 * cfg.scan_span_m
    return np.linspace(lambda_start, lambda_end, cfg.num_samples, dtype=np.float64)


def build_k_axis(lambda_axis_m: np.ndarray) -> np.ndarray:
    return 2.0 * np.pi / lambda_axis_m


def gaussian_reflectivity(lambda_axis_m: np.ndarray, center_m: float, sigma_m: float, amplitude: float) -> np.ndarray:
    return amplitude * np.exp(-0.5 * ((lambda_axis_m - center_m) / sigma_m) ** 2)


def simulate_three_gratings_reflectivity(
    lambda_axis_m: np.ndarray,
    cfg: SimulationConfig,
    delta_lambda_target_m: float,
) -> dict[str, np.ndarray]:
    left = gaussian_reflectivity(lambda_axis_m, cfg.lambda_left_m, cfg.sigma_left_m, cfg.amplitude_left)
    target = gaussian_reflectivity(
        lambda_axis_m,
        cfg.lambda_target_m + delta_lambda_target_m,
        cfg.sigma_target_m,
        cfg.amplitude_target,
    )
    right = gaussian_reflectivity(lambda_axis_m, cfg.lambda_right_m, cfg.sigma_right_m, cfg.amplitude_right)
    total = left + target + right
    centers_m = np.array(
        [
            cfg.lambda_left_m,
            cfg.lambda_target_m + delta_lambda_target_m,
            cfg.lambda_right_m,
        ],
        dtype=np.float64,
    )
    return {
        "left": left,
        "target": target,
        "right": right,
        "total": total,
        "centers_m": centers_m,
    }


def simulate_measurement_signal(
    k_axis_lambda_m: np.ndarray,
    reflectivity: dict[str, np.ndarray],
    cfg: SimulationConfig,
) -> dict[str, np.ndarray]:
    left_field = reflectivity["left"] * np.exp(1j * 2.0 * cfg.n_eff * k_axis_lambda_m * cfg.z_left_m)
    target_field = reflectivity["target"] * np.exp(1j * 2.0 * cfg.n_eff * k_axis_lambda_m * cfg.z_target_m)
    right_field = reflectivity["right"] * np.exp(1j * 2.0 * cfg.n_eff * k_axis_lambda_m * cfg.z_right_m)

    measurement_complex = left_field + target_field + right_field
    measurement_real = np.real(measurement_complex)

    reference_complex = np.exp(1j * 2.0 * cfg.n_eff * k_axis_lambda_m * cfg.z_ref_m)
    reference_real = np.real(reference_complex)

    return {
        "measurement_complex_lambda": measurement_complex,
        "measurement_real_lambda": measurement_real,
        "reference_complex_lambda": reference_complex,
        "reference_real_lambda": reference_real,
    }


def interpolate_to_uniform_k(k_axis_lambda_m: np.ndarray, complex_signal_lambda: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    sort_idx = np.argsort(k_axis_lambda_m)
    k_sorted = k_axis_lambda_m[sort_idx]
    signal_sorted = complex_signal_lambda[sort_idx]
    k_uniform = np.linspace(k_sorted[0], k_sorted[-1], k_sorted.size, dtype=np.float64)

    interp_real = interp1d(k_sorted, np.real(signal_sorted), kind="linear", bounds_error=False, fill_value="extrapolate")
    interp_imag = interp1d(k_sorted, np.imag(signal_sorted), kind="linear", bounds_error=False, fill_value="extrapolate")
    signal_uniform = interp_real(k_uniform) + 1j * interp_imag(k_uniform)
    return k_uniform, signal_uniform


def reconstruct_spatial_response(
    k_axis_lambda_m: np.ndarray,
    measurement_complex_lambda: np.ndarray,
    cfg: SimulationConfig,
) -> dict[str, np.ndarray]:
    k_uniform, measurement_k = interpolate_to_uniform_k(k_axis_lambda_m, measurement_complex_lambda)
    dk = float(k_uniform[1] - k_uniform[0])
    analysis_window = hann(len(k_uniform), sym=False)
    measurement_k_windowed = measurement_k * analysis_window

    spatial_response = np.fft.fftshift(np.fft.fft(measurement_k_windowed))
    freq_axis = np.fft.fftshift(np.fft.fftfreq(len(k_uniform), d=dk))
    z_axis_m = np.pi * freq_axis / cfg.n_eff

    return {
        "k_uniform": k_uniform,
        "measurement_k": measurement_k,
        "measurement_k_windowed": measurement_k_windowed,
        "analysis_window": analysis_window,
        "z_axis_m": z_axis_m,
        "spatial_response": spatial_response,
    }


def detect_spatial_peaks(z_axis_m: np.ndarray, spatial_response: np.ndarray, cfg: SimulationConfig) -> dict[str, np.ndarray]:
    positive_mask = z_axis_m > 0.0
    z_pos = z_axis_m[positive_mask]
    amp_pos = np.abs(spatial_response[positive_mask])

    min_distance_samples = max(20, int(round(0.8 / max(np.diff(z_pos).mean(), 1e-12))))
    peaks, properties = find_peaks(amp_pos, prominence=0.05 * amp_pos.max(), distance=min_distance_samples)
    if len(peaks) < 3:
        raise RuntimeError("Failed to detect three spatial peaks in the ideal chain.")

    order = np.argsort(properties["prominences"])[::-1]
    peak_indices = peaks[order[:3]]
    peak_indices = peak_indices[np.argsort(z_pos[peak_indices])]
    peak_positions_m = z_pos[peak_indices]

    expected = np.array([cfg.z_left_m, cfg.z_target_m, cfg.z_right_m], dtype=np.float64)
    if np.max(np.abs(peak_positions_m - expected)) > 0.6:
        raise RuntimeError("Detected spatial peaks do not align with the expected grating positions.")

    return {
        "positive_z_axis_m": z_pos,
        "positive_amplitude": amp_pos,
        "peak_positions_m": peak_positions_m,
    }


def spatial_gate_target(
    z_axis_m: np.ndarray,
    spatial_response: np.ndarray,
    target_position_m: float,
    gate_width_m: float,
) -> dict[str, np.ndarray]:
    gate = np.zeros_like(z_axis_m, dtype=np.float64)
    mask = np.abs(z_axis_m - target_position_m) <= 0.5 * gate_width_m
    idx = np.where(mask)[0]
    if idx.size < 8:
        raise RuntimeError("Spatial gate is too narrow for the target peak.")

    gate_segment = tukey(idx.size, alpha=0.5, sym=True)
    gate[idx] = gate_segment
    gated_spatial_response = spatial_response * gate
    return {
        "gate": gate,
        "gated_spatial_response": gated_spatial_response,
    }


def recover_target_spectrum(
    gated_spatial_response: np.ndarray,
    k_uniform: np.ndarray,
) -> dict[str, np.ndarray]:
    recovered_complex_k = np.fft.ifft(np.fft.ifftshift(gated_spatial_response))
    recovered_amplitude = np.abs(recovered_complex_k)
    lambda_from_k_m = 2.0 * np.pi / k_uniform

    sort_idx = np.argsort(lambda_from_k_m)
    return {
        "lambda_axis_m": lambda_from_k_m[sort_idx],
        "recovered_complex_k": recovered_complex_k[sort_idx],
        "recovered_amplitude": recovered_amplitude[sort_idx],
    }


def estimate_spectrum_center(lambda_axis_m: np.ndarray, spectrum: np.ndarray) -> float:
    spec = np.asarray(spectrum, dtype=np.float64)
    lam = np.asarray(lambda_axis_m, dtype=np.float64)
    spec = np.clip(spec, 0.0, None)
    threshold = 0.20 * float(spec.max())
    mask = spec >= threshold
    if not np.any(mask):
        raise RuntimeError("Spectrum center estimation failed because the recovered spectrum is empty.")
    weights = spec[mask]
    return float(np.sum(lam[mask] * weights) / np.sum(weights))


def plot_results(
    cfg: SimulationConfig,
    delta_lambda_target_m: float,
    lambda_axis_m: np.ndarray,
    k_axis_lambda_m: np.ndarray,
    reflectivity: dict[str, np.ndarray],
    measurement: dict[str, np.ndarray],
    spatial: dict[str, np.ndarray],
    peak_info: dict[str, np.ndarray],
    gate_info: dict[str, np.ndarray],
    recovered: dict[str, np.ndarray],
    ideal_target_uniform_sorted: np.ndarray,
    center_true_m: float,
    center_est_m: float,
    output_dir: Path,
) -> None:
    lambda_nm = lambda_axis_m * 1e9
    delta_pm = delta_lambda_target_m * 1e12
    z_axis_m = spatial["z_axis_m"]
    spatial_amp = np.abs(spatial["spatial_response"])
    gated_amp = np.abs(gate_info["gated_spatial_response"])

    lambda_recovered_nm = recovered["lambda_axis_m"] * 1e9
    recovered_norm = recovered["recovered_amplitude"] / (np.max(recovered["recovered_amplitude"]) + 1e-12)
    ideal_norm = ideal_target_uniform_sorted / (np.max(ideal_target_uniform_sorted) + 1e-12)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5), constrained_layout=True)
    fig.suptitle(f"Ideal three-grating OFDR chain | target shift = {delta_pm:.1f} pm", fontsize=14)

    ax = axes[0, 0]
    ax.plot(lambda_nm, reflectivity["left"], label="Left grating", linewidth=1.5)
    ax.plot(lambda_nm, reflectivity["target"], label="Target grating", linewidth=1.8)
    ax.plot(lambda_nm, reflectivity["right"], label="Right grating", linewidth=1.5)
    ax.plot(lambda_nm, reflectivity["total"], "k--", linewidth=1.6, label="Total reflectivity")
    ax.set_title("Reflectivity spectra")
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Reflectivity (a.u.)")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False, fontsize=9)

    ax = axes[0, 1]
    ax.plot(lambda_nm, measurement["measurement_real_lambda"], color="#1f77b4", linewidth=1.2, label="Main interferogram")
    ax.plot(lambda_nm, measurement["reference_real_lambda"], color="#d62728", linewidth=1.0, alpha=0.8, label="Reference signal")
    ax.set_title("Ideal measurement signals")
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Signal (a.u.)")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False, fontsize=9)

    ax = axes[0, 2]
    ax.plot(z_axis_m, spatial_amp, color="black", linewidth=1.3)
    for z_peak in peak_info["peak_positions_m"]:
        ax.axvline(z_peak, color="#1f77b4", linestyle="--", linewidth=1.0)
    ax.set_xlim(0.0, cfg.z_right_m + 1.5)
    ax.set_title("Spatial-domain response")
    ax.set_xlabel("Distance z (m)")
    ax.set_ylabel("|FFT| (a.u.)")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.plot(z_axis_m, spatial_amp / (spatial_amp.max() + 1e-12), color="#7f7f7f", linewidth=1.2, label="Original spatial response")
    ax.plot(z_axis_m, gate_info["gate"], color="#d62728", linewidth=1.8, label="Target gate")
    ax.plot(z_axis_m, gated_amp / (gated_amp.max() + 1e-12), color="#1f77b4", linewidth=1.4, label="Gated response")
    ax.set_xlim(cfg.z_target_m - 2.0, cfg.z_target_m + 2.0)
    ax.set_title("Target spatial gating")
    ax.set_xlabel("Distance z (m)")
    ax.set_ylabel("Normalized amplitude")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False, fontsize=9)

    ax = axes[1, 1]
    ax.plot(lambda_recovered_nm, ideal_norm, color="#7f7f7f", linestyle="--", linewidth=1.8, label="Ideal target spectrum")
    ax.plot(lambda_recovered_nm, recovered_norm, color="#1f77b4", linewidth=1.8, label="Recovered target spectrum")
    ax.axvline(center_true_m * 1e9, color="black", linestyle="--", linewidth=1.0, label="True center")
    ax.axvline(center_est_m * 1e9, color="#d62728", linestyle="-", linewidth=1.2, label="Estimated center")
    ax.set_title("Recovered target local spectrum")
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Normalized amplitude")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1, 2]
    error_pm = (center_est_m - center_true_m) * 1e12
    ax.axis("off")
    ax.set_title("Center estimation")
    text = (
        f"True target center: {center_true_m * 1e9:.6f} nm\n"
        f"Recovered center:   {center_est_m * 1e9:.6f} nm\n"
        f"Center error:       {error_pm:+.3f} pm\n\n"
        f"Expected spatial peaks (m):\n"
        f"  left   = {cfg.z_left_m:.2f}\n"
        f"  target = {cfg.z_target_m:.2f}\n"
        f"  right  = {cfg.z_right_m:.2f}\n\n"
        f"Detected peaks (m):\n"
        f"  {peak_info['peak_positions_m'][0]:.3f}, "
        f"{peak_info['peak_positions_m'][1]:.3f}, "
        f"{peak_info['peak_positions_m'][2]:.3f}"
    )
    ax.text(
        0.02,
        0.98,
        text,
        va="top",
        ha="left",
        fontsize=10,
        family="monospace",
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": "#b0b0b0"},
    )

    out_path = output_dir / f"case_delta_{int(round(delta_pm)):+d}pm.png"
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def run_single_case(cfg: SimulationConfig, delta_lambda_target_m: float) -> dict[str, float]:
    lambda_axis_m = build_lambda_axis(cfg)
    k_axis_lambda_m = build_k_axis(lambda_axis_m)

    reflectivity = simulate_three_gratings_reflectivity(lambda_axis_m, cfg, delta_lambda_target_m)
    measurement = simulate_measurement_signal(k_axis_lambda_m, reflectivity, cfg)
    spatial = reconstruct_spatial_response(k_axis_lambda_m, measurement["measurement_complex_lambda"], cfg)
    peak_info = detect_spatial_peaks(spatial["z_axis_m"], spatial["spatial_response"], cfg)
    gate_info = spatial_gate_target(spatial["z_axis_m"], spatial["spatial_response"], cfg.z_target_m, cfg.target_gate_width_m)
    recovered = recover_target_spectrum(gate_info["gated_spatial_response"], spatial["k_uniform"])

    ideal_target_uniform = np.interp(
        recovered["lambda_axis_m"],
        lambda_axis_m,
        reflectivity["target"],
        left=float(reflectivity["target"][0]),
        right=float(reflectivity["target"][-1]),
    )
    ideal_target_uniform *= np.interp(
        recovered["lambda_axis_m"],
        2.0 * np.pi / spatial["k_uniform"][::-1],
        spatial["analysis_window"][::-1],
        left=float(spatial["analysis_window"][0]),
        right=float(spatial["analysis_window"][-1]),
    )

    center_true_m = float(cfg.lambda_target_m + delta_lambda_target_m)
    center_est_m = estimate_spectrum_center(recovered["lambda_axis_m"], recovered["recovered_amplitude"])

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_results(
        cfg=cfg,
        delta_lambda_target_m=delta_lambda_target_m,
        lambda_axis_m=lambda_axis_m,
        k_axis_lambda_m=k_axis_lambda_m,
        reflectivity=reflectivity,
        measurement=measurement,
        spatial=spatial,
        peak_info=peak_info,
        gate_info=gate_info,
        recovered=recovered,
        ideal_target_uniform_sorted=ideal_target_uniform,
        center_true_m=center_true_m,
        center_est_m=center_est_m,
        output_dir=output_dir,
    )

    return {
        "delta_lambda_target_pm": delta_lambda_target_m * 1e12,
        "center_true_nm": center_true_m * 1e9,
        "center_est_nm": center_est_m * 1e9,
        "center_error_pm": (center_est_m - center_true_m) * 1e12,
    }


def plot_summary_shift(results: list[dict[str, float]], cfg: SimulationConfig) -> None:
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    deltas_pm = np.array([r["delta_lambda_target_pm"] for r in results], dtype=np.float64)
    true_centers_nm = np.array([r["center_true_nm"] for r in results], dtype=np.float64)
    est_centers_nm = np.array([r["center_est_nm"] for r in results], dtype=np.float64)

    fig, ax = plt.subplots(figsize=(7.5, 5.0), constrained_layout=True)
    ax.plot(deltas_pm, true_centers_nm, "o--", linewidth=1.8, label="True target center")
    ax.plot(deltas_pm, est_centers_nm, "s-", linewidth=1.8, label="Recovered target center")
    for d_pm, est_nm, true_nm in zip(deltas_pm, est_centers_nm, true_centers_nm):
        ax.text(d_pm + 0.5, est_nm + 0.00002, f"{(est_nm - true_nm) * 1e3:+.2f} pm", fontsize=9)
    ax.set_title("Recovered target center follows imposed wavelength shift")
    ax.set_xlabel("Imposed target shift (pm)")
    ax.set_ylabel("Target center (nm)")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False)
    fig.savefig(output_dir / "summary_center_shift.png", dpi=300)
    plt.close(fig)


def main() -> None:
    cfg = SimulationConfig()
    test_shifts_pm = [0.0, 20.0]
    test_shifts_m = [v * 1e-12 for v in test_shifts_pm]

    print("Running ideal stage-1 three-grating OFDR simulation...")
    print(f"Output directory: {Path(cfg.output_dir).resolve()}")
    print("")

    results = []
    for delta_m in test_shifts_m:
        result = run_single_case(cfg, delta_m)
        results.append(result)
        print(
            f"delta_lambda_target = {result['delta_lambda_target_pm']:+.1f} pm | "
            f"true center = {result['center_true_nm']:.6f} nm | "
            f"recovered center = {result['center_est_nm']:.6f} nm | "
            f"error = {result['center_error_pm']:+.3f} pm"
        )

    plot_summary_shift(results, cfg)

    recovered_shift_pm = (results[1]["center_est_nm"] - results[0]["center_est_nm"]) * 1e3
    true_shift_pm = results[1]["delta_lambda_target_pm"] - results[0]["delta_lambda_target_pm"]
    print("")
    print(f"Imposed target center shift:   {true_shift_pm:+.3f} pm")
    print(f"Recovered target center shift: {recovered_shift_pm:+.3f} pm")

    if abs(recovered_shift_pm - true_shift_pm) > 5.0:
        print("Warning: recovered center shift deviates noticeably from the imposed shift.")

    print("")
    print("Stage-1 simplifications:")
    print("1. Ideal linear sweep is assumed; no sweep nonlinearity or k-clock correction error is included.")
    print("2. Only first-order single reflections from three gratings are modeled.")
    print("3. No additive noise, phase noise, baseline drift, spike, or artifact is included.")
    print("4. Each grating is modeled by a Gaussian spectral envelope with fixed linewidth.")
    print("5. Spatial gating uses a simple smooth Tukey window around the target peak.")
    print("6. Center estimation uses a thresholded spectral centroid, not a full parametric fit.")
    print("7. Phase-2 can add sweep nonlinearity, noise, linewidth variation, and more realistic FBG spectra.")


if __name__ == "__main__":
    main()
