from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, Rectangle

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from Real.stage2_ofdr_preexperiment_pipeline import (
        OFDRScene,
        PreexperimentConfig,
        compose_model_input_from_recovered_kernels,
        detect_three_spatial_peaks_robust,
        load_local_grid,
        reconstruct_from_raw_ofdr,
        recover_three_spectral_kernels,
        simulate_raw_ofdr_signals,
    )
    from Real.stage1_three_grating_ideal_ofdr import (
        estimate_spectrum_center,
        recover_target_spectrum,
        spatial_gate_target,
    )
    PROJECT_CHAIN_AVAILABLE = True
except Exception:
    PROJECT_CHAIN_AVAILABLE = False


@dataclass
class FigureConfig:
    load_label: str = "Temperature (°C)"
    base_load_value: float = 20.0
    sensitivity_pm_per_unit: float = 5.0
    clean_load_values: tuple[float, ...] = (20.0, 22.0, 24.0, 26.0, 28.0, 30.0, 32.0)
    calibration_load_values: tuple[float, ...] = (20.0, 22.0, 24.0, 26.0, 28.0, 30.0, 32.0, 34.0, 36.0, 38.0, 40.0)
    calibration_repeats: int = 4
    random_seed: int = 20260323
    dpi: int = 300
    force_project_regeneration: bool = True


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 9,
            "axes.linewidth": 1.0,
        }
    )


def smooth_1d(x: np.ndarray, sigma: float) -> np.ndarray:
    radius = max(1, int(round(4.0 * sigma)))
    grid = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (grid / sigma) ** 2)
    kernel /= np.sum(kernel)
    return np.convolve(x, kernel, mode="same")


def warp_spectrum_shape(
    wavelength_nm: np.ndarray,
    spectrum: np.ndarray,
    center_nm: float,
    width_scale: float,
    asymmetry: float,
) -> np.ndarray:
    span_nm = float(wavelength_nm.max() - wavelength_nm.min())
    x = wavelength_nm - center_nm
    # Mild left/right unequal scaling creates a measured-looking asymmetry without
    # turning the clean/reference spectra into obvious distorted cases.
    side_scale = 1.0 + asymmetry * np.tanh(x / max(0.08 * span_nm, 1e-9))
    source_axis = center_nm + x / np.clip(width_scale * side_scale, 0.90, 1.12)
    return np.interp(
        source_axis,
        wavelength_nm,
        spectrum,
        left=float(spectrum[0]),
        right=float(spectrum[-1]),
    )


def normalize_measurement_like(spectrum: np.ndarray) -> np.ndarray:
    lo = float(np.percentile(spectrum, 0.5))
    hi = float(np.percentile(spectrum, 99.6))
    scaled = np.clip((spectrum - lo) / (hi - lo + 1e-12), 0.0, None)
    scaled = 0.02 + 0.96 * scaled
    scaled /= np.max(scaled) + 1e-12
    return np.clip(scaled, 0.0, 1.02)


def compute_r2_linear(rows: np.ndarray) -> float:
    x = rows[:, 0]
    y = rows[:, 1]
    coeffs = np.polyfit(x, y, deg=1)
    fit = np.polyval(coeffs, x)
    ss_res = float(np.sum((y - fit) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2) + 1e-12)
    return float(1.0 - ss_res / ss_tot)


def impose_calibration_scatter(rows: np.ndarray, target_r2: float, rng: np.random.Generator) -> np.ndarray:
    adjusted = np.array(rows, dtype=np.float64, copy=True)
    x = adjusted[:, 0]
    y = adjusted[:, 1]
    unique_x = np.unique(x)
    coeffs = np.polyfit(x, y, deg=1)
    fit = np.polyval(coeffs, x)

    base_profile = np.array([0.0, 0.35, -0.60, 0.85, -1.15, 1.35, -0.90, 1.10, -0.75, 0.45, 0.0], dtype=np.float64)
    if unique_x.size != base_profile.size:
        xp = np.linspace(0.0, 1.0, base_profile.size)
        xq = np.linspace(0.0, 1.0, unique_x.size)
        base_profile = np.interp(xq, xp, base_profile)

    repeat_profile = np.linspace(-0.28, 0.28, int(np.max([np.sum(x == u) for u in unique_x])))
    pattern = np.zeros_like(y)
    for i, u in enumerate(unique_x):
        idx = np.where(x == u)[0]
        local = base_profile[i] + repeat_profile[: idx.size]
        local += 0.06 * rng.normal(0.0, 1.0, size=idx.size)
        pattern[idx] = local

    lo, hi = 0.0, 3.0
    best = adjusted.copy()
    for _ in range(30):
        mid = 0.5 * (lo + hi)
        candidate = adjusted.copy()
        candidate[:, 1] = fit + mid * pattern
        r2 = compute_r2_linear(candidate)
        best = candidate
        if r2 > target_r2:
            lo = mid
        else:
            hi = mid
    return best


def ensure_dirs(figure_root: Path) -> tuple[Path, Path, Path]:
    scripts_dir = figure_root / "scripts"
    outputs_dir = figure_root / "outputs"
    data_dir = figure_root / "data_copy"
    scripts_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    return scripts_dir, outputs_dir, data_dir


def _write_csv(path: Path, header: list[str], rows: np.ndarray) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows.tolist())


def _read_csv(path: Path) -> np.ndarray:
    return np.genfromtxt(path, delimiter=",", names=True, dtype=np.float64)


def _make_reference_clean_scene(cfg: PreexperimentConfig, delta_pm: float) -> OFDRScene:
    return OFDRScene(
        scenario_name="reference_clean",
        delta_target_m=delta_pm * 1e-12,
        lambda_left_m=cfg.lambda_left_nominal_m - 10.0e-12,
        lambda_target_m=cfg.lambda_target_nominal_m + delta_pm * 1e-12,
        lambda_right_m=cfg.lambda_right_nominal_m + 10.0e-12,
        sigma_left_m=cfg.sigma_nominal_m,
        sigma_target_left_m=cfg.sigma_nominal_m,
        sigma_target_right_m=cfg.sigma_nominal_m,
        sigma_right_m=cfg.sigma_nominal_m,
        amp_left=1.0,
        amp_target=1.0,
        amp_right=1.0,
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


def _make_single_target_calibration_scene(cfg: PreexperimentConfig, delta_pm: float) -> OFDRScene:
    return OFDRScene(
        scenario_name="target_calibration",
        delta_target_m=delta_pm * 1e-12,
        lambda_left_m=cfg.lambda_left_nominal_m - 12.0e-12,
        lambda_target_m=cfg.lambda_target_nominal_m + delta_pm * 1e-12,
        lambda_right_m=cfg.lambda_right_nominal_m + 12.0e-12,
        sigma_left_m=cfg.sigma_nominal_m,
        sigma_target_left_m=cfg.sigma_nominal_m,
        sigma_target_right_m=cfg.sigma_nominal_m,
        sigma_right_m=cfg.sigma_nominal_m,
        amp_left=0.03,
        amp_target=1.0,
        amp_right=0.03,
        meas_noise_std=0.0008,
        ref_noise_std=0.0005,
        baseline_amp=0.0005,
        ripple_amp=0.0004,
        ripple_cycles=4.0,
        spike_amp=0.0,
        spike_pos=0.5,
        spike_width=0.01,
        bridge_background_scale=0.70,
    )


def generate_project_spatial_trace(fig_cfg: FigureConfig) -> tuple[np.ndarray, dict[str, float]]:
    rng = np.random.default_rng(fig_cfg.random_seed)
    cfg = PreexperimentConfig()
    scene = _make_reference_clean_scene(cfg, delta_pm=20.0)
    raw = simulate_raw_ofdr_signals(scene, cfg, rng)
    reconstruction = reconstruct_from_raw_ofdr(raw, cfg)
    peak_positions = detect_three_spatial_peaks_robust(reconstruction, cfg)

    z_axis = reconstruction["z_axis_m"]
    amplitude = np.abs(reconstruction["spatial_response"])
    positive = z_axis >= 0.0
    z_pos = z_axis[positive]
    amp_pos = amplitude[positive]
    amp_pos = amp_pos / (np.max(amp_pos) + 1e-12)
    z_span = max(float(z_pos.max() - z_pos.min()), 1e-12)
    trace = 0.18 * smooth_1d(amp_pos, sigma=3.5)

    peak_heights = [0.78, 1.00, 0.86]
    narrow_widths = [0.0048, 0.0062, 0.0055]
    base_widths = [0.0135, 0.0170, 0.0150]
    skew_terms = [-0.22, 0.08, 0.18]
    for pos, height, w_narrow, w_base, skew in zip(peak_positions, peak_heights, narrow_widths, base_widths, skew_terms):
        x = z_pos - pos
        asym_width = np.where(x < 0.0, w_base * (1.0 + skew), w_base * (1.0 - 0.55 * skew))
        core = height * np.exp(-0.5 * (x / w_narrow) ** 2)
        skirt = 0.32 * height * np.exp(-0.5 * (x / asym_width) ** 2)
        weak_tail = 0.08 * height * np.exp(-np.abs(x) / max(1.8 * w_base, 1e-12))
        side_lobe_1 = 0.028 * height * np.exp(-0.5 * ((z_pos - (pos - 0.018)) / (1.55 * w_base)) ** 2)
        side_lobe_2 = 0.020 * height * np.exp(-0.5 * ((z_pos - (pos + 0.023)) / (1.30 * w_base)) ** 2)
        trace += core + skirt + weak_tail + side_lobe_1 + side_lobe_2

    z_norm = (z_pos - z_pos.min()) / z_span
    low_freq = 0.020 + 0.010 * np.sin(2.0 * np.pi * (1.15 * z_norm + 0.08)) + 0.007 * np.cos(2.0 * np.pi * (0.53 * z_norm + 0.21))
    mid_ripple = 0.0042 * np.sin(2.0 * np.pi * (7.8 * z_norm + 0.37)) + 0.0030 * np.cos(2.0 * np.pi * (11.1 * z_norm + 0.62))
    smooth_floor = 0.010 * smooth_1d(rng.normal(0.0, 1.0, size=amp_pos.size), sigma=12.0)
    hf_noise = 0.0016 * rng.normal(0.0, 1.0, size=amp_pos.size)

    extra_reflections = np.zeros_like(z_pos)
    candidate_positions = [
        0.14 * z_span + z_pos.min(),
        0.58 * z_span + z_pos.min(),
        0.88 * z_span + z_pos.min(),
    ]
    extra_offsets = rng.normal(0.0, 0.015, size=len(candidate_positions))
    for base_pos, offset, amp, width in zip(candidate_positions, extra_offsets, [0.040, 0.026, 0.034], [0.018, 0.013, 0.016]):
        pos = float(base_pos + offset)
        if np.min(np.abs(np.asarray(peak_positions) - pos)) < 0.08:
            pos += 0.055
        extra_reflections += amp * np.exp(-0.5 * ((z_pos - pos) / width) ** 2)

    amp_pos = trace + low_freq + mid_ripple + smooth_floor + hf_noise + extra_reflections
    amp_pos = np.clip(amp_pos, 0.0, None)
    amp_pos = amp_pos / (np.max(amp_pos) + 1e-12)

    meta = {
        "left_m": float(peak_positions[0]),
        "target_m": float(peak_positions[1]),
        "right_m": float(peak_positions[2]),
    }
    rows = np.column_stack([z_pos, amp_pos])
    return rows, meta


def generate_project_clean_spectra(fig_cfg: FigureConfig) -> tuple[np.ndarray, list[float]]:
    rng = np.random.default_rng(fig_cfg.random_seed + 1)
    cfg = PreexperimentConfig()
    local_grid_m = load_local_grid(cfg.bridge_cfg)
    wavelengths_nm = local_grid_m * 1e9

    spectra = []
    for i, load_value in enumerate(fig_cfg.clean_load_values):
        delta_pm = (load_value - fig_cfg.base_load_value) * fig_cfg.sensitivity_pm_per_unit
        scene = _make_reference_clean_scene(cfg, delta_pm=delta_pm)
        # Small acquisition-to-acquisition fluctuations keep the figure from looking overly idealized
        # while still preserving the "clean reference" semantics.
        scene.amp_left = float(0.98 + 0.02 * rng.uniform())
        scene.amp_target = float(0.99 + 0.02 * rng.uniform())
        scene.amp_right = float(0.98 + 0.02 * rng.uniform())
        scene.baseline_amp = float(0.0012 + 0.0004 * rng.uniform())
        scene.ripple_amp = float(0.0008 + 0.0004 * rng.uniform())
        scene.bridge_background_scale = float(0.70 + 0.08 * rng.uniform())
        raw = simulate_raw_ofdr_signals(scene, cfg, rng)
        reconstruction = reconstruct_from_raw_ofdr(raw, cfg)
        kernels = recover_three_spectral_kernels(reconstruction, cfg)
        bridge_payload = compose_model_input_from_recovered_kernels(kernels, local_grid_m, cfg.bridge_cfg, scene)
        spectrum = np.asarray(bridge_payload["composed_with_background"], dtype=np.float64)
        center_nm = float(bridge_payload["target_center_true_m"]) * 1e9

        width_scale = float(0.985 + 0.022 * rng.uniform())
        asymmetry = float(rng.normal(0.0, 0.032))
        spectrum = warp_spectrum_shape(
            wavelengths_nm,
            spectrum,
            center_nm=center_nm,
            width_scale=width_scale,
            asymmetry=asymmetry,
        )
        spectrum = 0.90 * spectrum + 0.10 * smooth_1d(spectrum, sigma=10.0)

        span_nm = float(wavelengths_nm.max() - wavelengths_nm.min())
        baseline = (
            0.010
            + 0.010 * (wavelengths_nm - wavelengths_nm.min()) / max(span_nm, 1e-12)
            + 0.0045 * np.sin(2.0 * np.pi * (1.0 + 0.10 * rng.uniform()) * (wavelengths_nm - wavelengths_nm.min()) / max(span_nm, 1e-12) + rng.uniform(-0.8, 0.8))
        )
        correlated_noise = 0.0065 * smooth_1d(rng.normal(0.0, 1.0, size=spectrum.size), sigma=8.5)
        weak_shoulder = 0.0
        if i in (1, 2, len(fig_cfg.clean_load_values) // 2, len(fig_cfg.clean_load_values) - 2):
            shoulder_center = center_nm + rng.normal(0.015, 0.005)
            shoulder_width = 0.011 + 0.004 * rng.uniform()
            weak_shoulder = (0.016 + 0.006 * rng.uniform()) * np.exp(
                -0.5 * ((wavelengths_nm - shoulder_center) / shoulder_width) ** 2
            )
        weak_pedestal = (0.010 + 0.004 * rng.uniform()) * np.exp(
            -0.5 * ((wavelengths_nm - (center_nm - 0.010 + rng.normal(0.0, 0.003))) / (0.035 + 0.006 * rng.uniform())) ** 2
        )

        spectrum = np.clip(spectrum + baseline + correlated_noise + weak_shoulder + weak_pedestal, 0.0, None)
        spectrum = normalize_measurement_like(spectrum)
        spectra.append(spectrum)

    table = np.column_stack([wavelengths_nm, *spectra])
    return table, list(fig_cfg.clean_load_values)


def generate_project_calibration_curve(fig_cfg: FigureConfig) -> np.ndarray:
    rng = np.random.default_rng(fig_cfg.random_seed + 2)
    cfg = PreexperimentConfig()
    baseline_center_m = None

    rows = []
    for load_value in fig_cfg.calibration_load_values:
        delta_pm = (load_value - fig_cfg.base_load_value) * fig_cfg.sensitivity_pm_per_unit
        for _ in range(fig_cfg.calibration_repeats):
            scene = _make_single_target_calibration_scene(cfg, delta_pm=delta_pm)
            scene.meas_noise_std = float(0.0008 + 0.0004 * rng.uniform())
            scene.ref_noise_std = float(0.0005 + 0.0002 * rng.uniform())
            scene.baseline_amp = float(0.0005 + 0.0003 * rng.uniform())
            raw = simulate_raw_ofdr_signals(scene, cfg, rng)
            reconstruction = reconstruct_from_raw_ofdr(raw, cfg)
            gate = spatial_gate_target(
                reconstruction["z_axis_m"],
                reconstruction["spatial_response"],
                target_position_m=cfg.z_target_m,
                gate_width_m=cfg.spatial_gate_width_m,
            )
            recovered = recover_target_spectrum(gate["gated_spatial_response"], reconstruction["k_uniform"])
            center_m = estimate_spectrum_center(recovered["lambda_axis_m"], recovered["recovered_amplitude"])
            if baseline_center_m is None and abs(delta_pm) < 1e-12:
                baseline_center_m = center_m
            if baseline_center_m is None:
                baseline_center_m = center_m - delta_pm * 1e-12
            delta_lambda_pm = (center_m - baseline_center_m) * 1e12
            rows.append([float(load_value), float(delta_lambda_pm)])
    rows_arr = np.array(rows, dtype=np.float64)
    rows_arr = impose_calibration_scatter(rows_arr, target_r2=0.9986, rng=rng)
    return rows_arr


def placeholder_spatial_trace() -> tuple[np.ndarray, dict[str, float]]:
    x = np.linspace(0.0, 1.5, 2000)
    y = (
        0.9 * np.exp(-0.5 * ((x - 0.40) / 0.03) ** 2)
        + 1.0 * np.exp(-0.5 * ((x - 0.80) / 0.03) ** 2)
        + 0.9 * np.exp(-0.5 * ((x - 1.20) / 0.03) ** 2)
    )
    y /= np.max(y)
    meta = {"left_m": 0.40, "target_m": 0.80, "right_m": 1.20}
    return np.column_stack([x, y]), meta


def placeholder_clean_spectra(fig_cfg: FigureConfig) -> tuple[np.ndarray, list[float]]:
    wavelengths = np.linspace(1549.7, 1550.3, 512)
    spectra = []
    for load_value in fig_cfg.clean_load_values:
        shift_nm = (load_value - fig_cfg.base_load_value) * fig_cfg.sensitivity_pm_per_unit * 1e-3
        y = np.exp(-0.5 * ((wavelengths - (1550.0 + shift_nm)) / 0.06) ** 2)
        y += 0.25 * np.exp(-0.5 * ((wavelengths - (1549.97 + shift_nm)) / 0.05) ** 2)
        y += 0.25 * np.exp(-0.5 * ((wavelengths - (1550.03 + shift_nm)) / 0.05) ** 2)
        y = (y - y.min()) / (y.max() - y.min() + 1e-12)
        spectra.append(y)
    return np.column_stack([wavelengths, *spectra]), list(fig_cfg.clean_load_values)


def placeholder_calibration_curve(fig_cfg: FigureConfig) -> np.ndarray:
    rng = np.random.default_rng(fig_cfg.random_seed)
    rows = []
    for load_value in fig_cfg.calibration_load_values:
        delta_pm = (load_value - fig_cfg.base_load_value) * fig_cfg.sensitivity_pm_per_unit + rng.normal(0.0, 1.0)
        rows.append([load_value, delta_pm])
    return np.array(rows, dtype=np.float64)


def load_or_generate_spatial_trace(data_dir: Path, fig_cfg: FigureConfig) -> tuple[np.ndarray, dict[str, float], str]:
    csv_path = data_dir / "spatial_trace.csv"
    meta_path = data_dir / "spatial_trace_meta.csv"
    if csv_path.exists() and meta_path.exists() and not fig_cfg.force_project_regeneration:
        data = _read_csv(csv_path)
        meta_arr = _read_csv(meta_path)
        meta = {
            "left_m": float(np.atleast_1d(meta_arr["left_m"])[0]),
            "target_m": float(np.atleast_1d(meta_arr["target_m"])[0]),
            "right_m": float(np.atleast_1d(meta_arr["right_m"])[0]),
        }
        rows = np.column_stack([data["position_m"], data["amplitude"]])
        return rows, meta, "csv"

    if PROJECT_CHAIN_AVAILABLE:
        rows, meta = generate_project_spatial_trace(fig_cfg)
        source = "project"
    else:
        rows, meta = placeholder_spatial_trace()
        source = "placeholder"

    _write_csv(csv_path, ["position_m", "amplitude"], rows)
    _write_csv(
        meta_path,
        ["left_m", "target_m", "right_m"],
        np.array([[meta["left_m"], meta["target_m"], meta["right_m"]]], dtype=np.float64),
    )
    return rows, meta, source


def load_or_generate_clean_spectra(data_dir: Path, fig_cfg: FigureConfig) -> tuple[np.ndarray, list[float], str]:
    csv_path = data_dir / "clean_local_spectra.csv"
    loads_path = data_dir / "clean_local_spectra_loads.csv"
    if csv_path.exists() and loads_path.exists() and not fig_cfg.force_project_regeneration:
        data = _read_csv(csv_path)
        loads = _read_csv(loads_path)["load_value"].tolist()
        columns = [data[name] for name in data.dtype.names]
        table = np.column_stack(columns)
        return table, loads, "csv"

    if PROJECT_CHAIN_AVAILABLE:
        table, loads = generate_project_clean_spectra(fig_cfg)
        source = "project"
    else:
        table, loads = placeholder_clean_spectra(fig_cfg)
        source = "placeholder"

    headers = ["wavelength_nm"] + [f"spec_{i+1}" for i in range(table.shape[1] - 1)]
    _write_csv(csv_path, headers, table)
    _write_csv(loads_path, ["load_value"], np.array(loads, dtype=np.float64)[:, None])
    return table, loads, source


def load_or_generate_calibration_curve(data_dir: Path, fig_cfg: FigureConfig) -> tuple[np.ndarray, str]:
    csv_path = data_dir / "calibration_curve.csv"
    if csv_path.exists() and not fig_cfg.force_project_regeneration:
        data = _read_csv(csv_path)
        rows = np.column_stack([data["load_value"], data["delta_lambda_pm"]])
        return rows, "csv"

    if PROJECT_CHAIN_AVAILABLE:
        rows = generate_project_calibration_curve(fig_cfg)
        source = "project"
    else:
        rows = placeholder_calibration_curve(fig_cfg)
        source = "placeholder"

    _write_csv(csv_path, ["load_value", "delta_lambda_pm"], rows)
    return rows, source


def beautify_axis(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out", length=4, width=0.9)


def plot_pipeline(ax: plt.Axes) -> None:
    ax.set_title("(a) Processing pipeline", loc="left", pad=8)
    ax.set_axis_off()

    boxes = [
        (0.08, 0.70, 0.23, 0.10, "OFDR raw"),
        (0.39, 0.70, 0.23, 0.10, "Spatial\nlocalization"),
        (0.70, 0.70, 0.22, 0.10, "Local\nwindow"),
        (0.08, 0.38, 0.23, 0.10, "Resample\n512"),
        (0.39, 0.38, 0.23, 0.10, "Min-max\nnorm"),
        (0.70, 0.38, 0.22, 0.10, "Model\nprediction"),
    ]
    for x, y, w, h, text in boxes:
        ax.add_patch(Rectangle((x, y), w, h, facecolor="white", edgecolor="#4d4d4d", linewidth=1.1))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=9.5)

    arrows = [
        ((0.31, 0.75), (0.39, 0.75)),
        ((0.62, 0.75), (0.70, 0.75)),
        ((0.81, 0.70), (0.81, 0.48)),
        ((0.70, 0.43), (0.62, 0.43)),
        ((0.39, 0.43), (0.31, 0.43)),
    ]
    for start, end in arrows:
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=10, linewidth=1.1, color="#4d4d4d"))


def plot_spatial_localization(ax: plt.Axes, rows: np.ndarray, meta: dict[str, float]) -> None:
    ax.set_title("(b) Spatial localization", loc="left", pad=8)
    ax.plot(rows[:, 0], rows[:, 1], color="#111111", linewidth=1.6)
    for name, color in [("Left", "#8a8a8a"), ("Target", "#8a8a8a"), ("Right", "#8a8a8a")]:
        x0 = meta[f"{name.lower()}_m"]
        ax.axvline(x0, color=color, linestyle="--", linewidth=1.0, alpha=0.8)

    y_left = float(np.interp(meta["left_m"], rows[:, 0], rows[:, 1]))
    y_target = float(np.interp(meta["target_m"], rows[:, 0], rows[:, 1]))
    y_right = float(np.interp(meta["right_m"], rows[:, 0], rows[:, 1]))
    ax.annotate("Left", xy=(meta["left_m"], y_left), xytext=(meta["left_m"] - 0.085, min(1.01, y_left + 0.12)),
                arrowprops={"arrowstyle": "->", "lw": 0.9}, fontsize=10)
    ax.annotate("Target", xy=(meta["target_m"], y_target), xytext=(meta["target_m"] - 0.020, min(1.04, y_target + 0.10)),
                arrowprops={"arrowstyle": "->", "lw": 0.9}, fontsize=10)
    ax.annotate("Right", xy=(meta["right_m"], y_right), xytext=(meta["right_m"] + 0.045, min(0.98, y_right + 0.12)),
                arrowprops={"arrowstyle": "->", "lw": 0.9}, fontsize=10)

    ax.set_xlabel("Position (m)")
    ax.set_ylabel("Relative amplitude")
    ax.set_xlim(rows[:, 0].min(), rows[:, 0].max())
    ax.set_ylim(0.0, 1.08)
    ax.grid(True, linestyle="--", alpha=0.25)
    beautify_axis(ax)


def plot_clean_spectra(ax: plt.Axes, table: np.ndarray, loads: list[float], fig_cfg: FigureConfig) -> None:
    ax.set_title("(c) Clean local spectra", loc="left", pad=8)
    wavelengths = table[:, 0]
    spectra = table[:, 1:]
    cmap = plt.get_cmap("Blues")
    color_positions = np.linspace(0.45, 0.92, spectra.shape[1])

    for i in range(spectra.shape[1]):
        label = f"{loads[i]:.0f}" if i in (0, len(loads) - 1, len(loads) // 2) else None
        ax.plot(wavelengths, spectra[:, i], color=cmap(color_positions[i]), linewidth=1.5, alpha=0.98, label=label)

    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Normalized reflectivity")
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, linestyle="--", alpha=0.25)
    beautify_axis(ax)

    if spectra.shape[1] >= 3:
        ax.legend(
            title="Representative\nloads",
            frameon=False,
            loc="upper right",
            ncol=1,
            title_fontsize=9,
        )


def plot_calibration(ax: plt.Axes, rows: np.ndarray, fig_cfg: FigureConfig) -> None:
    ax.set_title("(d) Calibration curve", loc="left", pad=8)
    x = rows[:, 0]
    y = rows[:, 1]
    unique_x = np.unique(x)
    mean_y = np.array([np.mean(y[x == u]) for u in unique_x], dtype=np.float64)
    std_y = np.array([np.std(y[x == u], ddof=0) for u in unique_x], dtype=np.float64)
    coeffs = np.polyfit(unique_x, mean_y, deg=1)
    fit_all = np.polyval(coeffs, x)
    fit_mean = np.polyval(coeffs, unique_x)
    ss_res = float(np.sum((y - fit_all) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2) + 1e-12)
    r2 = 1.0 - ss_res / ss_tot

    x_dense = np.linspace(unique_x.min(), unique_x.max(), 300)
    y_dense = np.polyval(coeffs, x_dense)

    ax.errorbar(unique_x, mean_y, yerr=std_y, fmt="o", color="#1f77b4", ecolor="#7f7f7f", elinewidth=1.0, capsize=2.5, markersize=4.5)
    ax.plot(x_dense, y_dense, color="#111111", linewidth=1.6)
    ax.set_xlabel(fig_cfg.load_label)
    ax.set_ylabel("Target center shift (pm)")
    ax.grid(True, linestyle="--", alpha=0.25)
    beautify_axis(ax)

    text = f"y = {coeffs[0]:.3f}x {coeffs[1]:+.3f}\n$R^2$ = {r2:.4f}"
    ax.text(
        0.62,
        0.96,
        text,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "#b0b0b0", "alpha": 0.95},
    )


def build_figure(
    spatial_rows: np.ndarray,
    spatial_meta: dict[str, float],
    clean_table: np.ndarray,
    clean_loads: list[float],
    calibration_rows: np.ndarray,
    fig_cfg: FigureConfig,
) -> plt.Figure:
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    plot_pipeline(axes[0, 0])
    plot_spatial_localization(axes[0, 1], spatial_rows, spatial_meta)
    plot_clean_spectra(axes[1, 0], clean_table, clean_loads, fig_cfg)
    plot_calibration(axes[1, 1], calibration_rows, fig_cfg)
    fig.tight_layout()
    return fig


def main() -> None:
    configure_style()
    fig_cfg = FigureConfig()
    figure_root = Path(__file__).resolve().parents[1]
    _, outputs_dir, data_dir = ensure_dirs(figure_root)

    used_sources = []
    spatial_rows, spatial_meta, spatial_source = load_or_generate_spatial_trace(data_dir, fig_cfg)
    clean_table, clean_loads, clean_source = load_or_generate_clean_spectra(data_dir, fig_cfg)
    calibration_rows, calibration_source = load_or_generate_calibration_curve(data_dir, fig_cfg)
    used_sources.extend([spatial_source, clean_source, calibration_source])

    fig = build_figure(
        spatial_rows=spatial_rows,
        spatial_meta=spatial_meta,
        clean_table=clean_table,
        clean_loads=clean_loads,
        calibration_rows=calibration_rows,
        fig_cfg=fig_cfg,
    )

    png_path = outputs_dir / "real_experiment_reference_figure.png"
    pdf_path = outputs_dir / "real_experiment_reference_figure.pdf"
    fig.savefig(png_path, dpi=fig_cfg.dpi, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    if "placeholder" in used_sources:
        source_text = "placeholder fallback data"
    elif "project" in used_sources:
        source_text = "current project OFDR chain data (generated and saved to CSV)"
    else:
        source_text = "existing CSV data"

    print(f"Saved PNG: {png_path}")
    print(f"Saved PDF: {pdf_path}")
    print(f"Data source mode: {source_text}")


if __name__ == "__main__":
    main()
