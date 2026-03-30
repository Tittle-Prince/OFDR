from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.interpolate import interp1d
from scipy.signal import find_peaks, hilbert
from scipy.signal.windows import hann

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Real.stage1_three_grating_ideal_ofdr import (
    SimulationConfig,
    detect_spatial_peaks,
    estimate_spectrum_center,
    recover_target_spectrum,
    spatial_gate_target,
)
from Real.stage1_three_grating_ideal_ofdr_bridge import (
    BridgeConfig,
    compose_xlocal_style_spectrum,
    load_local_grid,
)
from src.ofdr.models.phase3_cnn import build_model


@dataclass
class PreexperimentConfig:
    output_dir: str = "Real/stage2_ofdr_preexperiment_outputs"

    # OFDR sweep and optical path.
    lambda_center_m: float = 1550.0e-9
    scan_span_m: float = 2.0e-9
    num_raw_samples: int = 8192
    n_eff: float = 1.4682
    z_left_m: float = 0.40
    z_target_m: float = 0.80
    z_right_m: float = 1.20
    z_ref_m: float = 0.20
    spatial_gate_width_m: float = 0.18

    # Nominal grating locations on the wavelength axis.
    lambda_left_nominal_m: float = 1550.00e-9
    lambda_target_nominal_m: float = 1550.00e-9
    lambda_right_nominal_m: float = 1550.00e-9
    sigma_nominal_m: float = 0.080e-9

    # Reference sweep nonlinearity.
    sweep_quad: float = 0.020
    sweep_cubic: float = -0.010

    # Small Monte Carlo scale for the first preexperiment.
    num_cases_per_scenario: int = 120
    random_seed: int = 20260323

    # Model evaluation.
    baseline_ckpt: str = "results/phase4a_shift004_linewidth_l3/method_enhance_tailaware_baseline_seed45/model_cnn.pt"
    tail_ckpt: str = "results/phase4a_shift004_linewidth_l3/method_enhance_tailaware_hard_seed45/model_cnn.pt"

    # Bridge adapter.
    bridge_cfg: BridgeConfig = BridgeConfig()


@dataclass
class OFDRScene:
    scenario_name: str
    delta_target_m: float
    lambda_left_m: float
    lambda_target_m: float
    lambda_right_m: float
    sigma_left_m: float
    sigma_target_left_m: float
    sigma_target_right_m: float
    sigma_right_m: float
    amp_left: float
    amp_target: float
    amp_right: float
    meas_noise_std: float
    ref_noise_std: float
    baseline_amp: float
    ripple_amp: float
    ripple_cycles: float
    spike_amp: float
    spike_pos: float
    spike_width: float
    bridge_background_scale: float


def minmax_normalize(x: np.ndarray) -> np.ndarray:
    x_min = float(np.min(x))
    x_max = float(np.max(x))
    return (x - x_min) / (x_max - x_min + 1e-12)


def build_linear_lambda_axis(cfg: PreexperimentConfig) -> np.ndarray:
    lam0 = cfg.lambda_center_m - 0.5 * cfg.scan_span_m
    lam1 = cfg.lambda_center_m + 0.5 * cfg.scan_span_m
    return np.linspace(lam0, lam1, cfg.num_raw_samples, dtype=np.float64)


def build_nonlinear_k_axis(cfg: PreexperimentConfig) -> tuple[np.ndarray, np.ndarray]:
    lambda_linear = build_linear_lambda_axis(cfg)
    k_linear = np.sort(2.0 * np.pi / lambda_linear)
    xi = np.linspace(-1.0, 1.0, cfg.num_raw_samples, dtype=np.float64)
    k_span = float(k_linear[-1] - k_linear[0])

    distortion = k_span * (cfg.sweep_quad * xi**2 + cfg.sweep_cubic * xi**3)
    distortion -= np.linspace(distortion[0], distortion[-1], distortion.size)
    k_true = k_linear + distortion
    if not np.all(np.diff(k_true) > 0.0):
        raise RuntimeError("Nonlinear k-axis must remain strictly monotonic after distortion.")
    return k_linear.astype(np.float64), k_true.astype(np.float64)


def gaussian_reflectivity(lambda_axis_m: np.ndarray, center_m: float, sigma_m: float, amplitude: float) -> np.ndarray:
    return amplitude * np.exp(-0.5 * ((lambda_axis_m - center_m) / sigma_m) ** 2)


def asymmetric_gaussian_reflectivity(
    lambda_axis_m: np.ndarray,
    center_m: float,
    sigma_left_m: float,
    sigma_right_m: float,
    amplitude: float,
) -> np.ndarray:
    left = lambda_axis_m <= center_m
    right = ~left
    out = np.zeros_like(lambda_axis_m, dtype=np.float64)
    out[left] = amplitude * np.exp(-0.5 * ((lambda_axis_m[left] - center_m) / sigma_left_m) ** 2)
    out[right] = amplitude * np.exp(-0.5 * ((lambda_axis_m[right] - center_m) / sigma_right_m) ** 2)
    return out


def sample_neighbor_shift_case(cfg: PreexperimentConfig, rng: np.random.Generator) -> OFDRScene:
    delta_target_m = rng.uniform(-30.0, 30.0) * 1e-12
    left_extra_pm = rng.uniform(-35.0, -6.0)
    right_extra_pm = rng.uniform(6.0, 35.0)
    return OFDRScene(
        scenario_name="neighbor_shift",
        delta_target_m=delta_target_m,
        lambda_left_m=cfg.lambda_left_nominal_m + left_extra_pm * 1e-12,
        lambda_target_m=cfg.lambda_target_nominal_m + delta_target_m,
        lambda_right_m=cfg.lambda_right_nominal_m + right_extra_pm * 1e-12,
        sigma_left_m=cfg.sigma_nominal_m,
        sigma_target_left_m=cfg.sigma_nominal_m,
        sigma_target_right_m=cfg.sigma_nominal_m,
        sigma_right_m=cfg.sigma_nominal_m,
        amp_left=float(rng.uniform(0.95, 1.05)),
        amp_target=float(rng.uniform(0.98, 1.02)),
        amp_right=float(rng.uniform(0.95, 1.05)),
        meas_noise_std=0.0015,
        ref_noise_std=0.0008,
        baseline_amp=0.002,
        ripple_amp=0.0015,
        ripple_cycles=float(rng.uniform(4.0, 9.0)),
        spike_amp=0.0,
        spike_pos=0.5,
        spike_width=0.01,
        bridge_background_scale=0.85,
    )


def sample_reference_clean_case(cfg: PreexperimentConfig, rng: np.random.Generator) -> OFDRScene:
    delta_target_m = rng.uniform(-30.0, 30.0) * 1e-12
    return OFDRScene(
        scenario_name="reference_clean",
        delta_target_m=delta_target_m,
        lambda_left_m=cfg.lambda_left_nominal_m - 10.0e-12,
        lambda_target_m=cfg.lambda_target_nominal_m + delta_target_m,
        lambda_right_m=cfg.lambda_right_nominal_m + 10.0e-12,
        sigma_left_m=cfg.sigma_nominal_m,
        sigma_target_left_m=cfg.sigma_nominal_m,
        sigma_target_right_m=cfg.sigma_nominal_m,
        sigma_right_m=cfg.sigma_nominal_m,
        amp_left=1.00,
        amp_target=1.00,
        amp_right=1.00,
        meas_noise_std=0.0008,
        ref_noise_std=0.0005,
        baseline_amp=0.0008,
        ripple_amp=0.0006,
        ripple_cycles=4.0,
        spike_amp=0.0,
        spike_pos=0.5,
        spike_width=0.01,
        bridge_background_scale=0.70,
    )


def sample_linewidth_asymmetry_case(cfg: PreexperimentConfig, rng: np.random.Generator) -> OFDRScene:
    delta_target_m = rng.uniform(-30.0, 30.0) * 1e-12
    width_scale = float(rng.uniform(0.95, 1.10))
    asym_ratio = float(rng.uniform(0.82, 1.18))
    target_sigma_left = cfg.sigma_nominal_m * width_scale * asym_ratio
    target_sigma_right = cfg.sigma_nominal_m * width_scale / asym_ratio
    return OFDRScene(
        scenario_name="linewidth_asymmetry",
        delta_target_m=delta_target_m,
        lambda_left_m=cfg.lambda_left_nominal_m + rng.uniform(-22.0, -5.0) * 1e-12,
        lambda_target_m=cfg.lambda_target_nominal_m + delta_target_m,
        lambda_right_m=cfg.lambda_right_nominal_m + rng.uniform(5.0, 22.0) * 1e-12,
        sigma_left_m=cfg.sigma_nominal_m * float(rng.uniform(0.98, 1.02)),
        sigma_target_left_m=target_sigma_left,
        sigma_target_right_m=target_sigma_right,
        sigma_right_m=cfg.sigma_nominal_m * float(rng.uniform(0.98, 1.02)),
        amp_left=float(rng.uniform(0.96, 1.04)),
        amp_target=float(rng.uniform(0.98, 1.03)),
        amp_right=float(rng.uniform(0.96, 1.04)),
        meas_noise_std=0.0015,
        ref_noise_std=0.0008,
        baseline_amp=0.0025,
        ripple_amp=0.0015,
        ripple_cycles=float(rng.uniform(4.0, 9.0)),
        spike_amp=0.0,
        spike_pos=0.5,
        spike_width=0.01,
        bridge_background_scale=0.95,
    )


def sample_system_artifact_case(cfg: PreexperimentConfig, rng: np.random.Generator) -> OFDRScene:
    delta_target_m = rng.uniform(-30.0, 30.0) * 1e-12
    return OFDRScene(
        scenario_name="system_artifact",
        delta_target_m=delta_target_m,
        lambda_left_m=cfg.lambda_left_nominal_m + rng.uniform(-28.0, -4.0) * 1e-12,
        lambda_target_m=cfg.lambda_target_nominal_m + delta_target_m,
        lambda_right_m=cfg.lambda_right_nominal_m + rng.uniform(4.0, 28.0) * 1e-12,
        sigma_left_m=cfg.sigma_nominal_m,
        sigma_target_left_m=cfg.sigma_nominal_m,
        sigma_target_right_m=cfg.sigma_nominal_m,
        sigma_right_m=cfg.sigma_nominal_m,
        amp_left=float(rng.uniform(0.95, 1.05)),
        amp_target=float(rng.uniform(0.98, 1.02)),
        amp_right=float(rng.uniform(0.95, 1.05)),
        meas_noise_std=0.0025,
        ref_noise_std=0.0012,
        baseline_amp=float(rng.uniform(0.004, 0.012)),
        ripple_amp=float(rng.uniform(0.004, 0.010)),
        ripple_cycles=float(rng.uniform(8.0, 18.0)),
        spike_amp=float(rng.uniform(0.008, 0.030)),
        spike_pos=float(rng.uniform(0.20, 0.85)),
        spike_width=float(rng.uniform(0.0015, 0.0045)),
        bridge_background_scale=1.10,
    )


def sample_combined_hard_case(cfg: PreexperimentConfig, rng: np.random.Generator) -> OFDRScene:
    delta_target_m = rng.uniform(-35.0, 35.0) * 1e-12
    width_scale = float(rng.uniform(1.00, 1.14))
    asym_ratio = float(rng.uniform(0.78, 1.22))
    target_sigma_left = cfg.sigma_nominal_m * width_scale * asym_ratio
    target_sigma_right = cfg.sigma_nominal_m * width_scale / asym_ratio
    return OFDRScene(
        scenario_name="combined_hard",
        delta_target_m=delta_target_m,
        lambda_left_m=cfg.lambda_left_nominal_m + rng.uniform(-38.0, -8.0) * 1e-12,
        lambda_target_m=cfg.lambda_target_nominal_m + delta_target_m,
        lambda_right_m=cfg.lambda_right_nominal_m + rng.uniform(8.0, 38.0) * 1e-12,
        sigma_left_m=cfg.sigma_nominal_m * float(rng.uniform(0.97, 1.04)),
        sigma_target_left_m=target_sigma_left,
        sigma_target_right_m=target_sigma_right,
        sigma_right_m=cfg.sigma_nominal_m * float(rng.uniform(0.97, 1.04)),
        amp_left=float(rng.uniform(0.92, 1.06)),
        amp_target=float(rng.uniform(0.97, 1.04)),
        amp_right=float(rng.uniform(0.92, 1.06)),
        meas_noise_std=0.0025,
        ref_noise_std=0.0012,
        baseline_amp=float(rng.uniform(0.004, 0.012)),
        ripple_amp=float(rng.uniform(0.003, 0.010)),
        ripple_cycles=float(rng.uniform(8.0, 18.0)),
        spike_amp=float(rng.uniform(0.005, 0.020)),
        spike_pos=float(rng.uniform(0.18, 0.85)),
        spike_width=float(rng.uniform(0.0015, 0.0040)),
        bridge_background_scale=1.10,
    )


def sample_scene(cfg: PreexperimentConfig, scenario_name: str, rng: np.random.Generator) -> OFDRScene:
    if scenario_name == "reference_clean":
        return sample_reference_clean_case(cfg, rng)
    if scenario_name == "neighbor_shift":
        return sample_neighbor_shift_case(cfg, rng)
    if scenario_name == "linewidth_asymmetry":
        return sample_linewidth_asymmetry_case(cfg, rng)
    if scenario_name == "system_artifact":
        return sample_system_artifact_case(cfg, rng)
    if scenario_name == "combined_hard":
        return sample_combined_hard_case(cfg, rng)
    raise ValueError(f"Unknown scenario: {scenario_name}")


def simulate_raw_ofdr_signals(scene: OFDRScene, cfg: PreexperimentConfig, rng: np.random.Generator) -> dict[str, np.ndarray]:
    _, k_true = build_nonlinear_k_axis(cfg)
    lambda_true = 2.0 * np.pi / k_true
    x = np.linspace(0.0, 1.0, cfg.num_raw_samples, dtype=np.float64)

    left = gaussian_reflectivity(lambda_true, scene.lambda_left_m, scene.sigma_left_m, scene.amp_left)
    target = asymmetric_gaussian_reflectivity(
        lambda_true,
        scene.lambda_target_m,
        scene.sigma_target_left_m,
        scene.sigma_target_right_m,
        scene.amp_target,
    )
    right = gaussian_reflectivity(lambda_true, scene.lambda_right_m, scene.sigma_right_m, scene.amp_right)

    measurement_complex = (
        left * np.exp(1j * 2.0 * cfg.n_eff * k_true * cfg.z_left_m)
        + target * np.exp(1j * 2.0 * cfg.n_eff * k_true * cfg.z_target_m)
        + right * np.exp(1j * 2.0 * cfg.n_eff * k_true * cfg.z_right_m)
    )
    reference_complex = np.exp(1j * 2.0 * cfg.n_eff * k_true * cfg.z_ref_m)

    measurement_real = np.real(measurement_complex)
    reference_real = np.real(reference_complex)

    baseline = scene.baseline_amp * (x - 0.5)
    ripple = scene.ripple_amp * np.sin(2.0 * np.pi * scene.ripple_cycles * x + 0.7)
    spike = scene.spike_amp * np.exp(-0.5 * ((x - scene.spike_pos) / scene.spike_width) ** 2)

    measurement_real = measurement_real + baseline + ripple + spike + rng.normal(0.0, scene.meas_noise_std, size=measurement_real.shape)
    reference_real = reference_real + 0.2 * baseline + rng.normal(0.0, scene.ref_noise_std, size=reference_real.shape)

    return {
        "lambda_true_m": lambda_true.astype(np.float64),
        "k_true": k_true.astype(np.float64),
        "measurement_real": measurement_real.astype(np.float64),
        "reference_real": reference_real.astype(np.float64),
        "reflect_left": left.astype(np.float64),
        "reflect_target": target.astype(np.float64),
        "reflect_right": right.astype(np.float64),
    }


def estimate_uniform_k_from_reference(reference_real: np.ndarray, cfg: PreexperimentConfig) -> tuple[np.ndarray, np.ndarray]:
    analytic = hilbert(reference_real)
    phase = np.unwrap(np.angle(analytic))
    delta_k = (phase - phase[0]) / (2.0 * cfg.n_eff * cfg.z_ref_m)
    nominal_lambda = build_linear_lambda_axis(cfg)
    nominal_k = np.sort(2.0 * np.pi / nominal_lambda)
    delta_k_span = float(delta_k[-1] - delta_k[0])
    if abs(delta_k_span) < 1e-12:
        raise RuntimeError("Reference phase variation is too small to recover k-axis.")
    delta_k_norm = (delta_k - delta_k[0]) / delta_k_span
    k_est = nominal_k[0] + delta_k_norm * (nominal_k[-1] - nominal_k[0])
    order = np.argsort(k_est)
    k_sorted = k_est[order]
    k_uniform = np.linspace(k_sorted[0], k_sorted[-1], k_sorted.size, dtype=np.float64)
    return k_uniform, order


def interp_complex_to_uniform_k(k_raw: np.ndarray, complex_signal: np.ndarray, k_uniform: np.ndarray) -> np.ndarray:
    interp_real = interp1d(k_raw, np.real(complex_signal), kind="linear", bounds_error=False, fill_value="extrapolate")
    interp_imag = interp1d(k_raw, np.imag(complex_signal), kind="linear", bounds_error=False, fill_value="extrapolate")
    return interp_real(k_uniform) + 1j * interp_imag(k_uniform)


def reconstruct_from_raw_ofdr(raw: dict[str, np.ndarray], cfg: PreexperimentConfig) -> dict[str, np.ndarray]:
    measurement_analytic = hilbert(raw["measurement_real"])
    k_uniform, order = estimate_uniform_k_from_reference(raw["reference_real"], cfg)
    analytic_ref = hilbert(raw["reference_real"])
    phase = np.unwrap(np.angle(analytic_ref))
    delta_k = (phase - phase[0]) / (2.0 * cfg.n_eff * cfg.z_ref_m)
    nominal_lambda = build_linear_lambda_axis(cfg)
    nominal_k = np.sort(2.0 * np.pi / nominal_lambda)
    delta_k_norm = (delta_k - delta_k[0]) / (delta_k[-1] - delta_k[0] + 1e-12)
    k_est_sorted = (nominal_k[0] + delta_k_norm * (nominal_k[-1] - nominal_k[0]))[order]
    meas_sorted = measurement_analytic[order]
    measurement_uniform = interp_complex_to_uniform_k(k_est_sorted, meas_sorted, k_uniform)

    window = hann(k_uniform.size, sym=False)
    measurement_windowed = measurement_uniform * window
    spatial_response = np.fft.fftshift(np.fft.fft(measurement_windowed))
    dk = float(k_uniform[1] - k_uniform[0])
    freq_axis = np.fft.fftshift(np.fft.fftfreq(k_uniform.size, d=dk))
    z_axis_m = np.pi * freq_axis / cfg.n_eff

    return {
        "k_uniform": k_uniform.astype(np.float64),
        "measurement_uniform": measurement_uniform.astype(np.complex128),
        "measurement_windowed": measurement_windowed.astype(np.complex128),
        "window": window.astype(np.float64),
        "z_axis_m": z_axis_m.astype(np.float64),
        "spatial_response": spatial_response.astype(np.complex128),
    }


def detect_three_spatial_peaks_robust(reconstruction: dict[str, np.ndarray], cfg: PreexperimentConfig) -> np.ndarray:
    z_axis = reconstruction["z_axis_m"]
    amp = np.abs(reconstruction["spatial_response"])
    positive = z_axis > 0.0
    z_pos = z_axis[positive]
    amp_pos = amp[positive]

    expected_positions = [cfg.z_left_m, cfg.z_target_m, cfg.z_right_m]
    peak_positions: list[float] = []
    for expected in expected_positions:
        local_mask = np.abs(z_pos - expected) <= 0.20
        if not np.any(local_mask):
            raise RuntimeError(f"No spatial samples available near expected z={expected:.2f} m")
        z_local = z_pos[local_mask]
        amp_local = amp_pos[local_mask]
        local_peaks, _ = find_peaks(amp_local, prominence=0.02 * np.max(amp_pos))
        if local_peaks.size == 0:
            best_idx = int(np.argmax(amp_local))
        else:
            best_idx = int(local_peaks[np.argmax(amp_local[local_peaks])])
        peak_positions.append(float(z_local[best_idx]))
    return np.array(peak_positions, dtype=np.float64)


def recover_three_spectral_kernels(reconstruction: dict[str, np.ndarray], cfg: PreexperimentConfig) -> dict[str, dict[str, np.ndarray | float]]:
    sim_cfg = SimulationConfig(
        n_eff=cfg.n_eff,
        z_left_m=cfg.z_left_m,
        z_target_m=cfg.z_target_m,
        z_right_m=cfg.z_right_m,
    )
    try:
        peak_info = detect_spatial_peaks(reconstruction["z_axis_m"], reconstruction["spatial_response"], sim_cfg)
        peak_positions = peak_info["peak_positions_m"]
    except RuntimeError:
        peak_positions = detect_three_spatial_peaks_robust(reconstruction, cfg)
    kernels: dict[str, dict[str, np.ndarray | float]] = {}
    for name, z_peak in zip(["left", "target", "right"], peak_positions):
        gate = spatial_gate_target(
            reconstruction["z_axis_m"],
            reconstruction["spatial_response"],
            target_position_m=float(z_peak),
            gate_width_m=cfg.spatial_gate_width_m,
        )
        recovered = recover_target_spectrum(gate["gated_spatial_response"], reconstruction["k_uniform"])
        kernel = recovered["recovered_amplitude"].astype(np.float64)
        kernel = kernel / (np.max(kernel) + 1e-12)
        center = estimate_spectrum_center(recovered["lambda_axis_m"], kernel)
        kernels[name] = {
            "lambda_axis_m": recovered["lambda_axis_m"].astype(np.float64),
            "kernel": kernel.astype(np.float64),
            "center_m": float(center),
            "spatial_peak_m": float(z_peak),
        }
    return kernels


def compose_model_input_from_recovered_kernels(
    kernels: dict[str, dict[str, np.ndarray | float]],
    local_grid_m: np.ndarray,
    bridge_cfg: BridgeConfig,
    scene: OFDRScene,
) -> dict[str, np.ndarray | float]:
    left_shift_m = scene.lambda_left_m - bridge_cfg.nominal_center_m
    right_shift_m = scene.lambda_right_m - bridge_cfg.nominal_center_m
    target_delta_m = scene.lambda_target_m - bridge_cfg.nominal_center_m

    sigma_nominal = 0.080e-9
    sigma_target_avg = 0.5 * (scene.sigma_target_left_m + scene.sigma_target_right_m)
    left_width_scale = float(np.clip(scene.sigma_left_m / sigma_nominal, 0.96, 1.05))
    target_width_scale = float(np.clip(sigma_target_avg / sigma_nominal, 0.96, 1.08))
    right_width_scale = float(np.clip(scene.sigma_right_m / sigma_nominal, 0.96, 1.05))

    # Recovered kernels are individually normalized. Re-inject relative amplitudes so the bridge
    # remains close to phase4a X_local semantics, where the target dominates and neighbors leak in.
    left_amp_scale = float(np.clip(0.34 * scene.amp_left / max(scene.amp_target, 1e-12), 0.26, 0.42))
    target_amp_scale = 1.0
    right_amp_scale = float(np.clip(0.34 * scene.amp_right / max(scene.amp_target, 1e-12), 0.26, 0.42))

    return compose_xlocal_style_spectrum(
        local_grid_m=local_grid_m,
        kernels=kernels,
        cfg_bridge=bridge_cfg,
        target_delta_m=target_delta_m,
        left_neighbor_shift_m=left_shift_m,
        right_neighbor_shift_m=right_shift_m,
        left_width_scale=left_width_scale,
        target_width_scale=target_width_scale,
        right_width_scale=right_width_scale,
        left_amp_scale=left_amp_scale,
        target_amp_scale=target_amp_scale,
        right_amp_scale=right_amp_scale,
        background_scale=scene.bridge_background_scale,
    )


def load_model(checkpoint_path: Path, input_dim: int, device: torch.device) -> torch.nn.Module:
    model = build_model("cnn_baseline", input_dim=input_dim).to(device)
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


def run_model_batch(model: torch.nn.Module, x: np.ndarray, device: torch.device) -> np.ndarray:
    x_tensor = torch.tensor(x, dtype=torch.float32, device=device)
    with torch.no_grad():
        return model(x_tensor).detach().cpu().numpy().reshape(-1).astype(np.float64)


def compute_metrics(y_true_nm: np.ndarray, y_pred_nm: np.ndarray) -> dict[str, float]:
    err = y_pred_nm - y_true_nm
    abs_err = np.abs(err)
    return {
        "RMSE_nm": float(np.sqrt(np.mean(err**2))),
        "MAE_nm": float(np.mean(abs_err)),
        "P95_nm": float(np.quantile(abs_err, 0.95)),
        "P99_nm": float(np.quantile(abs_err, 0.99)),
    }


def plot_representative_case(
    scenario_output_dir: Path,
    scenario_name: str,
    raw: dict[str, np.ndarray],
    reconstruction: dict[str, np.ndarray],
    bridge_payload: dict[str, np.ndarray | float],
    scene: OFDRScene,
    pred_baseline_nm: float,
    pred_tail_nm: float,
) -> None:
    lambda_nm = raw["lambda_true_m"] * 1e9
    x_idx = np.arange(raw["measurement_real"].size)
    z_axis = reconstruction["z_axis_m"]
    spatial_amp = np.abs(reconstruction["spatial_response"])
    local_grid_nm = load_local_grid(BridgeConfig()) * 1e9

    fig, axes = plt.subplots(2, 3, figsize=(15.0, 8.0), constrained_layout=True)
    ax = axes[0, 0]
    ax.plot(x_idx, raw["measurement_real"], color="#1f77b4", linewidth=1.0)
    ax.set_title(f"{scenario_name}: simulated raw measurement")
    ax.set_xlabel("Raw sample index")
    ax.set_ylabel("Signal (a.u.)")
    ax.grid(True, alpha=0.25)

    ax = axes[0, 1]
    ax.plot(x_idx, raw["reference_real"], color="#7f7f7f", linewidth=1.0)
    ax.set_title("Reference signal")
    ax.set_xlabel("Raw sample index")
    ax.set_ylabel("Signal (a.u.)")
    ax.grid(True, alpha=0.25)

    ax = axes[0, 2]
    ax.plot(z_axis, spatial_amp, color="black", linewidth=1.2)
    z_positive = z_axis[z_axis > 0.0]
    z_limit = float(z_positive.max()) if z_positive.size else float(z_axis.max())
    ax.set_xlim(0.0, max(1.35, 0.95 * z_limit))
    ax.set_title("Spatial response")
    ax.set_xlabel("Distance z (m)")
    ax.set_ylabel("|FFT| (a.u.)")
    ax.grid(True, alpha=0.25)

    ax = axes[1, 0]
    ax.plot(lambda_nm, raw["reflect_left"], linewidth=1.2, label="left")
    ax.plot(lambda_nm, raw["reflect_target"], linewidth=1.4, label="target")
    ax.plot(lambda_nm, raw["reflect_right"], linewidth=1.2, label="right")
    ax.set_title("True grating reflectivities")
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Amplitude")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, fontsize=9)

    ax = axes[1, 1]
    ax.plot(local_grid_nm, np.asarray(bridge_payload["composed_norm"], dtype=np.float64), color="#1f77b4", linewidth=1.8)
    ax.axvline(scene.lambda_target_m * 1e9, color="#d62728", linestyle="--", linewidth=1.0, label="True target center")
    ax.set_title("Bridge X_local spectrum")
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Normalized amplitude")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, fontsize=9)

    ax = axes[1, 2]
    labels = ["True", "Baseline", "Tail"]
    vals = [scene.delta_target_m * 1e12, pred_baseline_nm * 1e3, pred_tail_nm * 1e3]
    colors = ["#333333", "#7f7f7f", "#1f77b4"]
    ax.bar(labels, vals, color=colors)
    ax.set_title("Prediction summary")
    ax.set_ylabel("delta_lambda_target (pm)")
    ax.grid(True, axis="y", alpha=0.25)
    for i, v in enumerate(vals):
        ax.text(i, v, f"{v:+.1f}", ha="center", va="bottom", fontsize=9)

    fig.savefig(scenario_output_dir / "representative_case.png", dpi=300)
    plt.close(fig)


def run_single_scenario(
    cfg: PreexperimentConfig,
    scenario_name: str,
    baseline_model: torch.nn.Module,
    tail_model: torch.nn.Module,
    device: torch.device,
    rng: np.random.Generator,
) -> dict[str, object]:
    scenario_output_dir = PROJECT_ROOT / cfg.output_dir / scenario_name
    scenario_output_dir.mkdir(parents=True, exist_ok=True)

    local_grid_m = load_local_grid(cfg.bridge_cfg)

    x_batch: list[np.ndarray] = []
    y_true_nm: list[float] = []
    rows: list[dict[str, float | int | str]] = []

    representative_scene = None
    representative_raw = None
    representative_recon = None
    representative_bridge = None

    for i in range(cfg.num_cases_per_scenario):
        scene = sample_scene(cfg, scenario_name, rng)
        raw = simulate_raw_ofdr_signals(scene, cfg, rng)
        reconstruction = reconstruct_from_raw_ofdr(raw, cfg)
        kernels = recover_three_spectral_kernels(reconstruction, cfg)
        bridge_payload = compose_model_input_from_recovered_kernels(kernels, local_grid_m, cfg.bridge_cfg, scene)

        x_batch.append(np.asarray(bridge_payload["composed_norm"], dtype=np.float32))
        y_true_nm.append(float(scene.delta_target_m * 1e9))
        rows.append(
            {
                "sample_index": i,
                "delta_target_pm": float(scene.delta_target_m * 1e12),
                "target_center_true_nm": float(scene.lambda_target_m * 1e9),
                "target_center_bridge_est_nm": float(bridge_payload["target_center_est_m"]) * 1e9,
            }
        )

        if representative_scene is None:
            representative_scene = scene
            representative_raw = raw
            representative_recon = reconstruction
            representative_bridge = bridge_payload

    x_batch_arr = np.stack(x_batch, axis=0)[:, None, :]
    y_true_nm_arr = np.array(y_true_nm, dtype=np.float64)
    pred_baseline_nm = run_model_batch(baseline_model, x_batch_arr, device=device)
    pred_tail_nm = run_model_batch(tail_model, x_batch_arr, device=device)

    baseline_metrics = compute_metrics(y_true_nm_arr, pred_baseline_nm)
    tail_metrics = compute_metrics(y_true_nm_arr, pred_tail_nm)

    # Save representative visualization once predictions are available.
    plot_representative_case(
        scenario_output_dir=scenario_output_dir,
        scenario_name=scenario_name,
        raw=representative_raw,
        reconstruction=representative_recon,
        bridge_payload=representative_bridge,
        scene=representative_scene,
        pred_baseline_nm=float(pred_baseline_nm[0]),
        pred_tail_nm=float(pred_tail_nm[0]),
    )

    # Save arrays and metrics.
    np.savez(
        scenario_output_dir / "predictions.npz",
        X_bridge=x_batch_arr.astype(np.float32),
        y_true_nm=y_true_nm_arr.astype(np.float64),
        pred_baseline_nm=pred_baseline_nm.astype(np.float64),
        pred_tail_nm=pred_tail_nm.astype(np.float64),
        abs_err_baseline_nm=np.abs(pred_baseline_nm - y_true_nm_arr).astype(np.float64),
        abs_err_tail_nm=np.abs(pred_tail_nm - y_true_nm_arr).astype(np.float64),
    )

    lines = [
        "method,RMSE_nm,MAE_nm,P95_nm,P99_nm",
        "BaselineCNN,{RMSE_nm:.8f},{MAE_nm:.8f},{P95_nm:.8f},{P99_nm:.8f}".format(**baseline_metrics),
        "TailAwareBest,{RMSE_nm:.8f},{MAE_nm:.8f},{P95_nm:.8f},{P99_nm:.8f}".format(**tail_metrics),
    ]
    (scenario_output_dir / "metrics_table.csv").write_text("\n".join(lines), encoding="utf-8")

    return {
        "scenario_name": scenario_name,
        "baseline_metrics": baseline_metrics,
        "tail_metrics": tail_metrics,
        "num_cases": cfg.num_cases_per_scenario,
    }


def plot_summary_metrics(results: list[dict[str, object]], output_dir: Path) -> None:
    scenarios = [str(r["scenario_name"]) for r in results]
    baseline_p99 = [float(r["baseline_metrics"]["P99_nm"]) for r in results]
    tail_p99 = [float(r["tail_metrics"]["P99_nm"]) for r in results]
    baseline_rmse = [float(r["baseline_metrics"]["RMSE_nm"]) for r in results]
    tail_rmse = [float(r["tail_metrics"]["RMSE_nm"]) for r in results]

    x = np.arange(len(scenarios))
    width = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.8), constrained_layout=True)

    ax = axes[0]
    ax.bar(x - width / 2, baseline_p99, width=width, color="#7f7f7f", label="Baseline CNN")
    ax.bar(x + width / 2, tail_p99, width=width, color="#1f77b4", label="Tail-aware best")
    ax.set_title("P99 across OFDR preexperiment scenarios")
    ax.set_ylabel("P99 (nm)")
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios, rotation=20)
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(frameon=False)

    ax = axes[1]
    ax.bar(x - width / 2, baseline_rmse, width=width, color="#7f7f7f", label="Baseline CNN")
    ax.bar(x + width / 2, tail_rmse, width=width, color="#1f77b4", label="Tail-aware best")
    ax.set_title("RMSE across OFDR preexperiment scenarios")
    ax.set_ylabel("RMSE (nm)")
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios, rotation=20)
    ax.grid(True, axis="y", alpha=0.25)

    fig.savefig(output_dir / "scenario_metric_summary.png", dpi=300)
    plt.close(fig)


def write_summary_table(results: list[dict[str, object]], output_dir: Path) -> None:
    lines = ["scenario,method,num_cases,RMSE_nm,MAE_nm,P95_nm,P99_nm"]
    for r in results:
        for method_key, label in [("baseline_metrics", "BaselineCNN"), ("tail_metrics", "TailAwareBest")]:
            m = r[method_key]
            lines.append(
                f"{r['scenario_name']},{label},{r['num_cases']},"
                f"{m['RMSE_nm']:.8f},{m['MAE_nm']:.8f},{m['P95_nm']:.8f},{m['P99_nm']:.8f}"
            )
    (output_dir / "metrics_summary.csv").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    cfg = PreexperimentConfig()
    output_dir = PROJECT_ROOT / cfg.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    baseline_model = load_model(PROJECT_ROOT / cfg.baseline_ckpt, input_dim=512, device=device)
    tail_model = load_model(PROJECT_ROOT / cfg.tail_ckpt, input_dim=512, device=device)

    rng = np.random.default_rng(cfg.random_seed)
    scenario_names = [
        "reference_clean",
        "neighbor_shift",
        "linewidth_asymmetry",
        "system_artifact",
        "combined_hard",
    ]
    results: list[dict[str, object]] = []

    print("Running stage-2 OFDR preexperiment pipeline...")
    print(f"Output directory: {output_dir.resolve()}")
    print("")

    for scenario_name in scenario_names:
        print(f"[Scenario] {scenario_name} | num_cases = {cfg.num_cases_per_scenario}")
        scenario_result = run_single_scenario(cfg, scenario_name, baseline_model, tail_model, device, rng)
        results.append(scenario_result)

        baseline_metrics = scenario_result["baseline_metrics"]
        tail_metrics = scenario_result["tail_metrics"]
        print(
            "  Baseline CNN  | "
            f"RMSE={baseline_metrics['RMSE_nm']:.6f} nm, "
            f"MAE={baseline_metrics['MAE_nm']:.6f} nm, "
            f"P95={baseline_metrics['P95_nm']:.6f} nm, "
            f"P99={baseline_metrics['P99_nm']:.6f} nm"
        )
        print(
            "  Tail-aware    | "
            f"RMSE={tail_metrics['RMSE_nm']:.6f} nm, "
            f"MAE={tail_metrics['MAE_nm']:.6f} nm, "
            f"P95={tail_metrics['P95_nm']:.6f} nm, "
            f"P99={tail_metrics['P99_nm']:.6f} nm"
        )
        print("")

    plot_summary_metrics(results, output_dir)
    write_summary_table(results, output_dir)

    print("Summary:")
    print("1. This preexperiment simulates quasi-realistic OFDR raw signals under three physically motivated difficulty scenarios.")
    print("2. The raw signals are reconstructed, bridged into phase4a-like X_local spectra, and then fed into the trained models.")
    print("3. The resulting metrics quantify whether the current models remain usable after a realistic signal-chain front end.")
    print("4. The next step is to inspect which scenario dominates the tail and then refine the bridge or retraining strategy accordingly.")


if __name__ == "__main__":
    main()
