from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Real.stage1_three_grating_ideal_ofdr import (
    SimulationConfig,
    build_k_axis,
    build_lambda_axis,
    detect_spatial_peaks,
    estimate_spectrum_center,
    main as stage1_main,
    reconstruct_spatial_response,
    recover_target_spectrum,
    simulate_measurement_signal,
    simulate_three_gratings_reflectivity,
    spatial_gate_target,
)


@dataclass
class BridgeConfig:
    dataset_path: str = "data/processed/dataset_c_phase4a_shift004_linewidth_l3.npz"
    output_dir: str = "Real/stage1_three_grating_ideal_ofdr_bridge_outputs"
    local_num_points: int = 512
    local_window_nm: tuple[float, float] = (1549.0, 1551.0)
    nominal_center_m: float = 1550.0e-9
    gate_width_m: float = 0.8
    background_offset: float = 0.010
    background_slope: float = 0.008
    background_bump: float = 0.012
    background_bump_center: float = 0.58
    background_bump_width: float = 0.16


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 1.5 bridge demo: stage1 kernels -> phase4a-like local spectra")
    parser.add_argument("--mode", type=str, default="stage1_5", choices=["stage1", "stage1_5"])
    return parser.parse_args()


def minmax_normalize(x: np.ndarray) -> np.ndarray:
    x_min = float(np.min(x))
    x_max = float(np.max(x))
    return (x - x_min) / (x_max - x_min + 1e-12)


def load_local_grid(cfg_bridge: BridgeConfig) -> np.ndarray:
    dataset_path = PROJECT_ROOT / cfg_bridge.dataset_path
    if dataset_path.exists():
        data = np.load(dataset_path)
        wavelengths = data["wavelengths"].astype(np.float64)
        if wavelengths.ndim == 1 and wavelengths.size == cfg_bridge.local_num_points:
            # phase4a stores wavelengths in nm, while this bridge code uses SI meters internally.
            if np.max(np.abs(wavelengths)) > 1e-6:
                wavelengths = wavelengths * 1e-9
            return wavelengths
    return np.linspace(
        cfg_bridge.local_window_nm[0] * 1e-9,
        cfg_bridge.local_window_nm[1] * 1e-9,
        cfg_bridge.local_num_points,
        dtype=np.float64,
    )


def extract_spectral_kernels(
    sim_cfg: SimulationConfig,
    cfg_bridge: BridgeConfig,
    delta_lambda_target_m: float = 0.0,
) -> dict[str, dict[str, np.ndarray | float]]:
    """
    Recover left / target / right single-peak spectral kernels from the ideal stage-1 OFDR chain.

    These kernels are not yet phase4a-like inputs. They are the bridge basis functions that will
    later be shifted, broadened and recomposed into a local overlapped spectrum.
    """
    lambda_axis_m = build_lambda_axis(sim_cfg)
    k_axis_lambda_m = build_k_axis(lambda_axis_m)
    reflectivity = simulate_three_gratings_reflectivity(lambda_axis_m, sim_cfg, delta_lambda_target_m)
    measurement = simulate_measurement_signal(k_axis_lambda_m, reflectivity, sim_cfg)
    spatial = reconstruct_spatial_response(k_axis_lambda_m, measurement["measurement_complex_lambda"], sim_cfg)
    peak_info = detect_spatial_peaks(spatial["z_axis_m"], spatial["spatial_response"], sim_cfg)

    kernels: dict[str, dict[str, np.ndarray | float]] = {}
    names = ["left", "target", "right"]
    for name, z_peak in zip(names, peak_info["peak_positions_m"]):
        gate_info = spatial_gate_target(
            spatial["z_axis_m"],
            spatial["spatial_response"],
            target_position_m=float(z_peak),
            gate_width_m=cfg_bridge.gate_width_m,
        )
        recovered = recover_target_spectrum(gate_info["gated_spatial_response"], spatial["k_uniform"])
        kernel_lambda_m = recovered["lambda_axis_m"]
        kernel_amp = recovered["recovered_amplitude"].astype(np.float64)
        kernel_amp = kernel_amp / (np.max(kernel_amp) + 1e-12)
        center_est_m = estimate_spectrum_center(kernel_lambda_m, kernel_amp)

        kernels[name] = {
            "lambda_axis_m": kernel_lambda_m.astype(np.float64),
            "kernel": kernel_amp.astype(np.float64),
            "center_m": float(center_est_m),
            "spatial_peak_m": float(z_peak),
        }
    return kernels


def resample_kernel_to_local_grid(
    kernel_lambda_m: np.ndarray,
    kernel: np.ndarray,
    local_grid_m: np.ndarray,
) -> np.ndarray:
    return np.interp(
        local_grid_m,
        kernel_lambda_m,
        kernel,
        left=0.0,
        right=0.0,
    ).astype(np.float64)


def shift_kernel(
    local_grid_m: np.ndarray,
    kernel: np.ndarray,
    shift_m: float,
) -> np.ndarray:
    """
    Sub-pixel spectral shift via interpolation.
    Positive shift moves the kernel to longer wavelength.
    """
    return np.interp(
        local_grid_m - float(shift_m),
        local_grid_m,
        kernel,
        left=0.0,
        right=0.0,
    ).astype(np.float64)


def scale_kernel_width(
    local_grid_m: np.ndarray,
    kernel: np.ndarray,
    center_m: float,
    width_scale: float,
) -> np.ndarray:
    """
    Light width scaling by horizontal coordinate stretching around the chosen center.
    width_scale > 1 broadens the kernel; width_scale < 1 narrows it.
    """
    width_scale = float(width_scale)
    if abs(width_scale - 1.0) < 1e-12:
        return kernel.astype(np.float64)
    mapped = float(center_m) + (local_grid_m - float(center_m)) / width_scale
    return np.interp(mapped, local_grid_m, kernel, left=0.0, right=0.0).astype(np.float64)


def scale_kernel_amplitude(kernel: np.ndarray, amplitude_scale: float) -> np.ndarray:
    return np.asarray(kernel, dtype=np.float64) * float(amplitude_scale)


def add_smooth_background(
    local_grid_m: np.ndarray,
    spectrum: np.ndarray,
    offset: float,
    slope: float,
    bump_amp: float,
    bump_center_ratio: float,
    bump_width_ratio: float,
) -> np.ndarray:
    x = np.linspace(0.0, 1.0, len(local_grid_m), dtype=np.float64)
    baseline = float(offset) + float(slope) * (x - 0.5)
    bump = float(bump_amp) * np.exp(-0.5 * ((x - float(bump_center_ratio)) / float(bump_width_ratio)) ** 2)
    return np.asarray(spectrum, dtype=np.float64) + baseline + bump


def estimate_target_center_from_composite(
    local_grid_m: np.ndarray,
    spectrum: np.ndarray,
    center_guess_m: float,
    half_window_m: float = 0.06e-9,
) -> float:
    mask = np.abs(local_grid_m - center_guess_m) <= half_window_m
    if not np.any(mask):
        return float(center_guess_m)
    return estimate_spectrum_center(local_grid_m[mask], spectrum[mask])


def compose_xlocal_style_spectrum(
    local_grid_m: np.ndarray,
    kernels: dict[str, dict[str, np.ndarray | float]],
    cfg_bridge: BridgeConfig,
    target_delta_m: float,
    left_neighbor_shift_m: float,
    right_neighbor_shift_m: float,
    left_width_scale: float = 1.0,
    target_width_scale: float = 1.0,
    right_width_scale: float = 1.0,
    left_amp_scale: float = 0.35,
    target_amp_scale: float = 1.0,
    right_amp_scale: float = 0.35,
    background_scale: float = 1.0,
) -> dict[str, np.ndarray | float]:
    desired_centers = {
        "left": cfg_bridge.nominal_center_m + float(left_neighbor_shift_m),
        "target": cfg_bridge.nominal_center_m + float(target_delta_m),
        "right": cfg_bridge.nominal_center_m + float(right_neighbor_shift_m),
    }
    width_scales = {
        "left": float(left_width_scale),
        "target": float(target_width_scale),
        "right": float(right_width_scale),
    }
    amp_scales = {
        "left": float(left_amp_scale),
        "target": float(target_amp_scale),
        "right": float(right_amp_scale),
    }

    components: dict[str, np.ndarray] = {}
    for name in ["left", "target", "right"]:
        kernel = resample_kernel_to_local_grid(
            np.asarray(kernels[name]["lambda_axis_m"], dtype=np.float64),
            np.asarray(kernels[name]["kernel"], dtype=np.float64),
            local_grid_m,
        )
        original_center = float(kernels[name]["center_m"])
        desired_center = float(desired_centers[name])
        shifted = shift_kernel(local_grid_m, kernel, desired_center - original_center)
        width_scaled = scale_kernel_width(local_grid_m, shifted, desired_center, width_scales[name])
        amp_scaled = scale_kernel_amplitude(width_scaled, amp_scales[name])
        components[name] = amp_scaled

    composed_raw = components["left"] + components["target"] + components["right"]
    composed_with_bg = add_smooth_background(
        local_grid_m,
        composed_raw,
        offset=cfg_bridge.background_offset * background_scale,
        slope=cfg_bridge.background_slope * background_scale,
        bump_amp=cfg_bridge.background_bump * background_scale,
        bump_center_ratio=cfg_bridge.background_bump_center,
        bump_width_ratio=cfg_bridge.background_bump_width,
    )
    composed_with_bg = np.clip(composed_with_bg, 0.0, None)
    composed_norm = minmax_normalize(composed_with_bg)

    target_center_est_m = estimate_target_center_from_composite(
        local_grid_m,
        composed_norm,
        center_guess_m=float(desired_centers["target"]),
    )

    return {
        "components_left": components["left"].astype(np.float64),
        "components_target": components["target"].astype(np.float64),
        "components_right": components["right"].astype(np.float64),
        "composed_raw": composed_raw.astype(np.float64),
        "composed_with_background": composed_with_bg.astype(np.float64),
        "composed_norm": composed_norm.astype(np.float64),
        "target_center_true_m": float(desired_centers["target"]),
        "target_center_est_m": float(target_center_est_m),
    }


def generate_bridge_examples(
    local_grid_m: np.ndarray,
    kernels: dict[str, dict[str, np.ndarray | float]],
    cfg_bridge: BridgeConfig,
) -> list[dict[str, np.ndarray | float | str]]:
    cases = [
        {
            "name": "CaseA_clean_like_delta0",
            "delta_pm": 0.0,
            "left_neighbor_pm": -8.0,
            "right_neighbor_pm": +8.0,
            "left_width": 1.00,
            "target_width": 1.00,
            "right_width": 1.00,
            "left_amp": 0.34,
            "target_amp": 1.00,
            "right_amp": 0.34,
            "background_scale": 0.8,
        },
        {
            "name": "CaseB_clean_like_delta20",
            "delta_pm": +20.0,
            "left_neighbor_pm": -8.0,
            "right_neighbor_pm": +8.0,
            "left_width": 1.00,
            "target_width": 1.00,
            "right_width": 1.00,
            "left_amp": 0.34,
            "target_amp": 1.00,
            "right_amp": 0.34,
            "background_scale": 0.8,
        },
        {
            "name": "CaseC_small_neighbor_shift",
            "delta_pm": 0.0,
            "left_neighbor_pm": -18.0,
            "right_neighbor_pm": +12.0,
            "left_width": 1.00,
            "target_width": 1.00,
            "right_width": 1.00,
            "left_amp": 0.38,
            "target_amp": 1.00,
            "right_amp": 0.32,
            "background_scale": 1.0,
        },
        {
            "name": "CaseD_mild_width_variation",
            "delta_pm": +20.0,
            "left_neighbor_pm": -12.0,
            "right_neighbor_pm": +10.0,
            "left_width": 0.98,
            "target_width": 1.03,
            "right_width": 1.01,
            "left_amp": 0.35,
            "target_amp": 1.00,
            "right_amp": 0.33,
            "background_scale": 1.0,
        },
        {
            "name": "CaseE_mild_amp_variation",
            "delta_pm": +20.0,
            "left_neighbor_pm": -10.0,
            "right_neighbor_pm": +14.0,
            "left_width": 1.00,
            "target_width": 1.01,
            "right_width": 1.00,
            "left_amp": 0.30,
            "target_amp": 1.02,
            "right_amp": 0.39,
            "background_scale": 1.1,
        },
    ]

    outputs: list[dict[str, np.ndarray | float | str]] = []
    for case in cases:
        composed = compose_xlocal_style_spectrum(
            local_grid_m=local_grid_m,
            kernels=kernels,
            cfg_bridge=cfg_bridge,
            target_delta_m=float(case["delta_pm"]) * 1e-12,
            left_neighbor_shift_m=float(case["left_neighbor_pm"]) * 1e-12,
            right_neighbor_shift_m=float(case["right_neighbor_pm"]) * 1e-12,
            left_width_scale=float(case["left_width"]),
            target_width_scale=float(case["target_width"]),
            right_width_scale=float(case["right_width"]),
            left_amp_scale=float(case["left_amp"]),
            target_amp_scale=float(case["target_amp"]),
            right_amp_scale=float(case["right_amp"]),
            background_scale=float(case["background_scale"]),
        )
        outputs.append({**case, **composed})
    return outputs


def compare_stage1_vs_bridge(
    local_grid_m: np.ndarray,
    kernels: dict[str, dict[str, np.ndarray | float]],
    examples: list[dict[str, np.ndarray | float | str]],
    cfg_bridge: BridgeConfig,
) -> None:
    out_dir = PROJECT_ROOT / cfg_bridge.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    lambda_local_nm = local_grid_m * 1e9
    target_kernel_local = resample_kernel_to_local_grid(
        np.asarray(kernels["target"]["lambda_axis_m"], dtype=np.float64),
        np.asarray(kernels["target"]["kernel"], dtype=np.float64),
        local_grid_m,
    )
    target_kernel_local = minmax_normalize(target_kernel_local)

    # 1) Kernel comparison.
    fig, ax = plt.subplots(figsize=(10, 5.2), constrained_layout=True)
    for name, color in zip(["left", "target", "right"], ["#1f77b4", "#d62728", "#2ca02c"]):
        kernel_local = resample_kernel_to_local_grid(
            np.asarray(kernels[name]["lambda_axis_m"], dtype=np.float64),
            np.asarray(kernels[name]["kernel"], dtype=np.float64),
            local_grid_m,
        )
        kernel_local = minmax_normalize(kernel_local)
        ax.plot(lambda_local_nm, kernel_local, linewidth=1.8, label=f"{name} kernel", color=color)
        ax.axvline(float(kernels[name]["center_m"]) * 1e9, color=color, linestyle="--", linewidth=1.0, alpha=0.7)
    ax.set_title("Recovered spectral kernels from stage-1 OFDR chain")
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Normalized amplitude")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False)
    fig.savefig(out_dir / "bridge_kernels_overview.png", dpi=300)
    plt.close(fig)

    # 2) Stage1 recovered target vs bridge composite.
    case_a = next(c for c in examples if c["name"] == "CaseA_clean_like_delta0")
    fig, ax = plt.subplots(figsize=(10, 5.2), constrained_layout=True)
    ax.plot(lambda_local_nm, target_kernel_local, color="#7f7f7f", linestyle="--", linewidth=2.0, label="Stage1 recovered target kernel")
    ax.plot(lambda_local_nm, np.asarray(case_a["composed_norm"], dtype=np.float64), color="#1f77b4", linewidth=2.0, label="Bridge composite spectrum")
    ax.axvline(float(case_a["target_center_true_m"]) * 1e9, color="#d62728", linewidth=1.2, label="Target center")
    ax.set_title("Stage-1 single target spectrum vs bridge local-overlap spectrum")
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Normalized amplitude")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False)
    fig.savefig(out_dir / "bridge_stage1_vs_composite.png", dpi=300)
    plt.close(fig)

    # 3) Delta comparison between 0 pm and +20 pm.
    case_b = next(c for c in examples if c["name"] == "CaseB_clean_like_delta20")
    fig, ax = plt.subplots(figsize=(10, 5.2), constrained_layout=True)
    ax.plot(lambda_local_nm, np.asarray(case_a["composed_norm"], dtype=np.float64), color="#1f77b4", linewidth=2.0, label="Bridge delta = 0 pm")
    ax.plot(lambda_local_nm, np.asarray(case_b["composed_norm"], dtype=np.float64), color="#d62728", linewidth=2.0, label="Bridge delta = +20 pm")
    ax.axvline(float(case_a["target_center_true_m"]) * 1e9, color="#1f77b4", linestyle="--", linewidth=1.0)
    ax.axvline(float(case_b["target_center_true_m"]) * 1e9, color="#d62728", linestyle="--", linewidth=1.0)
    ax.set_title("Bridge spectra under different target delta settings")
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Normalized amplitude")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False)
    fig.savefig(out_dir / "bridge_delta_comparison.png", dpi=300)
    plt.close(fig)

    # 4) Component decomposition for two representative cases.
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2), constrained_layout=True, sharey=True)
    for ax, case in zip(axes, [case_a, case_b]):
        ax.plot(lambda_local_nm, np.asarray(case["components_left"], dtype=np.float64), linewidth=1.4, label="left component")
        ax.plot(lambda_local_nm, np.asarray(case["components_target"], dtype=np.float64), linewidth=1.6, label="target component")
        ax.plot(lambda_local_nm, np.asarray(case["components_right"], dtype=np.float64), linewidth=1.4, label="right component")
        ax.plot(lambda_local_nm, np.asarray(case["composed_norm"], dtype=np.float64), "k--", linewidth=1.8, label="composed spectrum")
        ax.axvline(float(case["target_center_true_m"]) * 1e9, color="#d62728", linestyle="--", linewidth=1.0, label="target center")
        ax.set_title(f"{case['name']} | delta = {float(case['delta_pm']):+.1f} pm")
        ax.set_xlabel("Wavelength (nm)")
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("Amplitude (a.u.)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncol=5, loc="upper center", bbox_to_anchor=(0.5, 1.02))
    fig.savefig(out_dir / "bridge_component_decomposition.png", dpi=300)
    plt.close(fig)


def export_model_ready_array(
    examples: list[dict[str, np.ndarray | float | str]],
    cfg_bridge: BridgeConfig,
) -> dict[str, np.ndarray]:
    x = np.stack([np.asarray(c["composed_norm"], dtype=np.float32) for c in examples], axis=0)
    x = x[:, None, :]
    y_delta_pm = np.array([float(c["delta_pm"]) for c in examples], dtype=np.float32)
    y_center_est_pm = np.array([float(c["target_center_est_m"]) * 1e12 for c in examples], dtype=np.float32)
    y_center_true_pm = np.array([float(c["target_center_true_m"]) * 1e12 for c in examples], dtype=np.float32)

    out_dir = PROJECT_ROOT / cfg_bridge.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_dir / "bridge_model_ready_examples.npz",
        X_bridge=x,
        Y_delta_target_pm=y_delta_pm,
        Y_center_true_pm=y_center_true_pm,
        Y_center_est_pm=y_center_est_pm,
    )
    return {
        "X_bridge": x,
        "Y_delta_target_pm": y_delta_pm,
        "Y_center_true_pm": y_center_true_pm,
        "Y_center_est_pm": y_center_est_pm,
    }


def run_stage1_5_bridge_demo() -> None:
    sim_cfg = SimulationConfig()
    cfg_bridge = BridgeConfig()
    out_dir = PROJECT_ROOT / cfg_bridge.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    local_grid_m = load_local_grid(cfg_bridge)
    kernels = extract_spectral_kernels(sim_cfg, cfg_bridge, delta_lambda_target_m=0.0)
    examples = generate_bridge_examples(local_grid_m, kernels, cfg_bridge)
    compare_stage1_vs_bridge(local_grid_m, kernels, examples, cfg_bridge)
    exported = export_model_ready_array(examples, cfg_bridge)

    print("Stage 1.5 bridge demo completed.")
    print(f"Output directory: {out_dir.resolve()}")
    print("")
    print("Bridge case summary:")
    for case in examples:
        true_center_nm = float(case["target_center_true_m"]) * 1e9
        est_center_nm = float(case["target_center_est_m"]) * 1e9
        center_err_pm = (float(case["target_center_est_m"]) - float(case["target_center_true_m"])) * 1e12
        print(
            f"{case['name']:<28} "
            f"delta={float(case['delta_pm']):+6.1f} pm | "
            f"target_center_est={est_center_nm:10.6f} nm | "
            f"center_error={center_err_pm:+8.3f} pm"
        )

    case_a = next(c for c in examples if c["name"] == "CaseA_clean_like_delta0")
    case_b = next(c for c in examples if c["name"] == "CaseB_clean_like_delta20")
    center_diff_pm = (float(case_b["target_center_est_m"]) - float(case_a["target_center_est_m"])) * 1e12
    print("")
    print(f"Estimated center difference between Case B and Case A: {center_diff_pm:+.3f} pm")
    print(f"Exported model-ready array shape: {exported['X_bridge'].shape}")
    print("")
    print("Summary:")
    print("1. Stage 1.5 extracts real spectral kernels from the ideal OFDR chain and reuses them as basis functions.")
    print("2. The bridge spectra are not a more realistic OFDR system simulation; they are a distribution-alignment layer.")
    print("3. The goal is to move stage1 outputs closer to the phase4a local-overlap input semantics.")
    print("4. The next practical step is to feed these bridge spectra to the trained models and inspect whether the domain gap shrinks.")
    print("5. After that, artifact/spike or stronger local perturbations can be added incrementally if still needed.")


def main() -> None:
    args = parse_args()
    if args.mode == "stage1":
        stage1_main()
        return
    run_stage1_5_bridge_demo()


if __name__ == "__main__":
    main()
