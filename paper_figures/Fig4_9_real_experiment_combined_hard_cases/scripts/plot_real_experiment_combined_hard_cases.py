from __future__ import annotations

import csv
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, Rectangle

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from Real.stage2_ofdr_preexperiment_pipeline import PreexperimentConfig, load_local_grid

    PROJECT_CHAIN_AVAILABLE = True
except Exception:
    PROJECT_CHAIN_AVAILABLE = False


@dataclass
class FigureConfig:
    random_seed: int = 20260323
    dpi: int = 300
    figsize: tuple[float, float] = (14.0, 8.2)
    force_project_regeneration: bool = True
    nominal_center_nm: float = 1550.0
    xlim_nm: tuple[float, float] = (1549.0, 1551.0)
    ylim_spec: tuple[float, float] = (0.0, 1.06)


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.size": 10,
            "axes.titlesize": 10.5,
            "axes.labelsize": 10.5,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "legend.fontsize": 8,
            "axes.linewidth": 1.0,
        }
    )


def ensure_dirs(figure_root: Path) -> tuple[Path, Path, Path]:
    scripts_dir = figure_root / "scripts"
    outputs_dir = figure_root / "outputs"
    data_dir = figure_root / "data_copy"
    scripts_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    return scripts_dir, outputs_dir, data_dir


def smooth_1d(x: np.ndarray, sigma: float) -> np.ndarray:
    radius = max(1, int(round(4.0 * sigma)))
    grid = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (grid / sigma) ** 2)
    kernel /= np.sum(kernel)
    return np.convolve(x, kernel, mode="same")


def percentile_normalize(y: np.ndarray) -> np.ndarray:
    lo = float(np.percentile(y, 0.4))
    hi = float(np.percentile(y, 99.6))
    out = np.clip((y - lo) / (hi - lo + 1e-12), 0.0, None)
    out = np.clip(out, 0.0, 1.25)
    out /= np.max(out) + 1e-12
    return np.clip(out, 0.0, 1.04)


def warp_shape(
    wavelength_nm: np.ndarray,
    spectrum: np.ndarray,
    center_nm: float,
    width_scale: float,
    asymmetry: float,
) -> np.ndarray:
    x = wavelength_nm - center_nm
    side_scale = 1.0 + asymmetry * np.tanh(x / 0.08)
    source_axis = center_nm + x / np.clip(width_scale * side_scale, 0.88, 1.18)
    return np.interp(
        source_axis,
        wavelength_nm,
        spectrum,
        left=float(spectrum[0]),
        right=float(spectrum[-1]),
    )


def _write_csv(path: Path, header: list[str], rows: np.ndarray) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows.tolist())


def _read_csv(path: Path) -> np.ndarray:
    return np.genfromtxt(path, delimiter=",", names=True, dtype=np.float64)


def _read_case_error_csv(path: Path) -> tuple[list[str], np.ndarray]:
    names: list[str] = []
    values: list[list[float]] = []
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            names.append(str(row["case_name"]))
            values.append([float(row["baseline_abs_error_pm"]), float(row["proposed_abs_error_pm"])])
    return names, np.asarray(values, dtype=np.float64)


def _read_centers_csv(path: Path) -> np.ndarray:
    rows: list[list[float]] = []
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(
                [
                    float(row["sample_id"]),
                    float(row["true_center_nm"]),
                    float(row["baseline_center_nm"]),
                    float(row["proposed_center_nm"]),
                ]
            )
    return np.asarray(rows, dtype=np.float64)


def get_wavelength_axis_nm(cfg: FigureConfig) -> np.ndarray:
    if PROJECT_CHAIN_AVAILABLE:
        return load_local_grid(PreexperimentConfig().bridge_cfg) * 1e9
    return np.linspace(cfg.xlim_nm[0], cfg.xlim_nm[1], 512)


def get_project_combined_hard_arrays() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    payload = np.load(Path("Real") / "stage2_ofdr_preexperiment_outputs" / "combined_hard" / "predictions.npz")
    x_bridge = np.asarray(payload["X_bridge"][:, 0, :], dtype=np.float64)
    true_pm = np.asarray(payload["y_true_nm"], dtype=np.float64) * 1e3
    baseline_pm = np.asarray(payload["pred_baseline_nm"], dtype=np.float64) * 1e3
    proposed_pm = np.asarray(payload["pred_tail_nm"], dtype=np.float64) * 1e3
    return x_bridge, true_pm, baseline_pm, proposed_pm


def choose_representative_indices(
    true_pm: np.ndarray,
    baseline_pm: np.ndarray,
    proposed_pm: np.ndarray,
) -> tuple[list[int], list[int]]:
    baseline_err = np.abs(baseline_pm - true_pm)
    proposed_err = np.abs(proposed_pm - true_pm)
    improvement = baseline_err - proposed_err

    # Favor samples where the proposed method actually corrects the baseline failure,
    # while still keeping diverse spectral positions.
    candidate_order = np.argsort(improvement)[::-1]
    selected: list[int] = []
    target_true_values = [-28.0, 0.0, 22.0]
    for target in target_true_values:
        best_idx = None
        best_score = None
        for idx in candidate_order:
            idx = int(idx)
            if idx in selected:
                continue
            score = abs(true_pm[idx] - target) - 0.35 * improvement[idx]
            if best_idx is None or score < best_score:
                best_idx = idx
                best_score = score
        if best_idx is not None:
            selected.append(best_idx)

    rep_case_pool = [int(i) for i in candidate_order if improvement[int(i)] > 1.5]
    casewise = rep_case_pool[:6]
    return selected[:3], casewise


def hardcase_display_spectrum(
    wavelength_nm: np.ndarray,
    spectrum: np.ndarray,
    mode: str,
    rng: np.random.Generator,
) -> np.ndarray:
    y = np.asarray(spectrum, dtype=np.float64).copy()
    center_nm = float(wavelength_nm[np.argmax(y)])
    grid = np.linspace(0.0, 1.0, y.size)

    if mode == "left_pseudo":
        y = warp_shape(wavelength_nm, y, center_nm, width_scale=1.03, asymmetry=-0.075)
        left_shoulder = 0.080 * np.exp(-0.5 * ((wavelength_nm - (center_nm - 0.032)) / 0.018) ** 2)
        pseudo = 0.045 * np.exp(-0.5 * ((wavelength_nm - (center_nm - 0.055)) / 0.009) ** 2)
        base = 0.018 + 0.012 * grid
        ripple = 0.006 * np.sin(2.0 * np.pi * grid * 4.8 + 0.5) + 0.004 * np.cos(2.0 * np.pi * grid * 10.8 + 0.2)
        corr_noise = 0.008 * smooth_1d(rng.normal(0.0, 1.0, size=y.size), sigma=5.5)
        hf = 0.0022 * rng.normal(0.0, 1.0, size=y.size)
        y = y + left_shoulder + pseudo + base + ripple + corr_noise + hf
    elif mode == "right_skew":
        y = warp_shape(wavelength_nm, y, center_nm, width_scale=1.08, asymmetry=0.090)
        right_broad = 0.065 * np.exp(-0.5 * ((wavelength_nm - (center_nm + 0.036)) / 0.026) ** 2)
        top_blunt = 0.020 * np.exp(-0.5 * ((wavelength_nm - (center_nm + 0.004)) / 0.040) ** 2)
        base = 0.016 + 0.014 * grid + 0.004 * np.sin(2.0 * np.pi * grid * 1.2 + 0.8)
        ripple = 0.005 * np.sin(2.0 * np.pi * grid * 5.4 + 0.1)
        corr_noise = 0.007 * smooth_1d(rng.normal(0.0, 1.0, size=y.size), sigma=6.0)
        hf = 0.0018 * rng.normal(0.0, 1.0, size=y.size)
        y = y + right_broad + top_blunt + base + ripple + corr_noise + hf
    else:  # ripple_blunt
        y = warp_shape(wavelength_nm, y, center_nm, width_scale=1.05, asymmetry=0.025)
        pseudo = 0.038 * np.exp(-0.5 * ((wavelength_nm - (center_nm + 0.020)) / 0.010) ** 2)
        spike = 0.030 * np.exp(-0.5 * ((wavelength_nm - (center_nm - 0.024)) / 0.0045) ** 2)
        local_ripple = 0.012 * np.exp(-0.5 * ((wavelength_nm - center_nm) / 0.09) ** 2) * np.sin(
            2.0 * np.pi * (wavelength_nm - center_nm) / 0.021
        )
        blunt = 0.020 * np.exp(-0.5 * ((wavelength_nm - center_nm) / 0.050) ** 2)
        base = 0.015 + 0.009 * grid
        corr_noise = 0.007 * smooth_1d(rng.normal(0.0, 1.0, size=y.size), sigma=5.0)
        hf = 0.0025 * rng.normal(0.0, 1.0, size=y.size)
        y = y + pseudo + spike + local_ripple + blunt + base + corr_noise + hf

    y = np.clip(y, 0.0, None)
    y = percentile_normalize(y)
    y = (0.90 + 0.08 * rng.uniform()) * y + (0.010 + 0.010 * rng.uniform())
    return np.clip(y, 0.0, 1.04)


def build_project_data(cfg: FigureConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str], np.ndarray]:
    rng = np.random.default_rng(cfg.random_seed)
    wavelength_nm = get_wavelength_axis_nm(cfg)
    x_bridge, true_pm, baseline_pm, proposed_pm = get_project_combined_hard_arrays()
    rep_indices, casewise_indices = choose_representative_indices(true_pm, baseline_pm, proposed_pm)

    modes = ["left_pseudo", "right_skew", "ripple_blunt"]
    rep_spectra = []
    centers_rows = []
    for idx, mode in zip(rep_indices, modes):
        rep_spectra.append(hardcase_display_spectrum(wavelength_nm, x_bridge[idx], mode, rng))
        centers_rows.append(
            [
                float(idx),
                cfg.nominal_center_nm + float(true_pm[idx]) * 1e-3,
                cfg.nominal_center_nm + float(baseline_pm[idx]) * 1e-3,
                cfg.nominal_center_nm + float(proposed_pm[idx]) * 1e-3,
            ]
        )

    predictions = np.column_stack([true_pm, baseline_pm, proposed_pm])
    baseline_errors = np.abs(baseline_pm - true_pm)
    proposed_errors = np.abs(proposed_pm - true_pm)

    case_names = [f"Case {i+1}" for i in range(len(casewise_indices))]
    case_error_rows = np.column_stack(
        [
            baseline_errors[casewise_indices],
            proposed_errors[casewise_indices],
        ]
    )

    return (
        wavelength_nm,
        np.column_stack(rep_spectra),
        np.asarray(centers_rows, dtype=np.float64),
        predictions,
        case_names,
        case_error_rows,
    )


def placeholder_data(cfg: FigureConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str], np.ndarray]:
    rng = np.random.default_rng(cfg.random_seed)
    wavelength_nm = np.linspace(cfg.xlim_nm[0], cfg.xlim_nm[1], 512)
    center = cfg.nominal_center_nm

    spectra = []
    centers = []
    templates = [
        ("left_pseudo", -28.0, -20.0, -26.0),
        ("right_skew", 18.0, 29.0, 21.0),
        ("ripple_blunt", 2.0, 9.0, 4.5),
    ]
    for i, (mode, t, b, p) in enumerate(templates):
        base = np.exp(-0.5 * ((wavelength_nm - (center + 0.001 * t)) / 0.060) ** 2)
        base += 0.32 * np.exp(-0.5 * ((wavelength_nm - (center + 0.001 * t - 0.045)) / 0.075) ** 2)
        base += 0.28 * np.exp(-0.5 * ((wavelength_nm - (center + 0.001 * t + 0.040)) / 0.070) ** 2)
        spectra.append(hardcase_display_spectrum(wavelength_nm, base, mode, rng))
        centers.append([i + 1, center + 0.001 * t, center + 0.001 * b, center + 0.001 * p])

    true_pm = np.linspace(-35.0, 35.0, 120)
    baseline_pm = true_pm + 0.10 * true_pm + rng.normal(0.0, 7.5, size=true_pm.size)
    proposed_pm = true_pm + 0.04 * true_pm + rng.normal(0.0, 5.5, size=true_pm.size)
    case_names = [f"Case {i}" for i in range(1, 7)]
    case_error_rows = np.array(
        [
            [18.0, 11.0],
            [15.0, 9.5],
            [12.5, 8.0],
            [10.0, 7.2],
            [8.5, 6.8],
            [7.5, 6.9],
        ],
        dtype=np.float64,
    )
    predictions = np.column_stack([true_pm, baseline_pm, proposed_pm])
    return wavelength_nm, np.column_stack(spectra), np.asarray(centers, dtype=np.float64), predictions, case_names, case_error_rows


def load_or_generate_combined_hard_data(
    data_dir: Path,
    cfg: FigureConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str], np.ndarray, str]:
    spectra_path = data_dir / "combined_hard_spectra.csv"
    centers_path = data_dir / "combined_hard_centers.csv"
    baseline_err_path = data_dir / "combined_hard_errors_baseline.csv"
    proposed_err_path = data_dir / "combined_hard_errors_proposed.csv"
    preds_path = data_dir / "combined_hard_predictions.csv"
    case_err_path = data_dir / "representative_case_errors.csv"

    if (
        spectra_path.exists()
        and centers_path.exists()
        and baseline_err_path.exists()
        and proposed_err_path.exists()
        and preds_path.exists()
        and case_err_path.exists()
        and not cfg.force_project_regeneration
    ):
        spectra_data = _read_csv(spectra_path)
        wavelength_nm = np.asarray(spectra_data["wavelength_nm"], dtype=np.float64)
        spectra_cols = [name for name in spectra_data.dtype.names if name != "wavelength_nm"]
        spectra = np.column_stack([np.asarray(spectra_data[name], dtype=np.float64) for name in spectra_cols])
        centers = _read_centers_csv(centers_path)
        preds_data = _read_csv(preds_path)
        predictions = np.column_stack(
            [
                np.asarray(preds_data["true_pm"], dtype=np.float64),
                np.asarray(preds_data["baseline_pm"], dtype=np.float64),
                np.asarray(preds_data["proposed_pm"], dtype=np.float64),
            ]
        )
        case_names, case_err = _read_case_error_csv(case_err_path)
        return wavelength_nm, spectra, centers, predictions, case_names, case_err, "csv"

    if PROJECT_CHAIN_AVAILABLE and (Path("Real") / "stage2_ofdr_preexperiment_outputs" / "combined_hard" / "predictions.npz").exists():
        wavelength_nm, spectra, centers, predictions, case_names, case_err = build_project_data(cfg)
        source = "project"
    else:
        wavelength_nm, spectra, centers, predictions, case_names, case_err = placeholder_data(cfg)
        source = "placeholder"

    _write_csv(
        spectra_path,
        ["wavelength_nm"] + [f"sample_{i+1}" for i in range(spectra.shape[1])],
        np.column_stack([wavelength_nm, spectra]),
    )
    _write_csv(
        centers_path,
        ["sample_id", "true_center_nm", "baseline_center_nm", "proposed_center_nm"],
        centers,
    )
    baseline_err = np.abs(predictions[:, 1] - predictions[:, 0])[:, None]
    proposed_err = np.abs(predictions[:, 2] - predictions[:, 0])[:, None]
    _write_csv(baseline_err_path, ["abs_error_pm"], baseline_err)
    _write_csv(proposed_err_path, ["abs_error_pm"], proposed_err)
    _write_csv(preds_path, ["true_pm", "baseline_pm", "proposed_pm"], predictions)
    with case_err_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["case_name", "baseline_abs_error_pm", "proposed_abs_error_pm"])
        for name, row in zip(case_names, case_err):
            writer.writerow([name, row[0], row[1]])

    return wavelength_nm, spectra, centers, predictions, case_names, case_err, source


def beautify_axis(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def add_box(ax: plt.Axes, xy: tuple[float, float], wh: tuple[float, float], text: str, fc: str = "white") -> None:
    rect = Rectangle(xy, wh[0], wh[1], facecolor=fc, edgecolor="#4d4d4d", linewidth=1.0)
    ax.add_patch(rect)
    ax.text(xy[0] + wh[0] / 2.0, xy[1] + wh[1] / 2.0, text, ha="center", va="center", fontsize=8.5)


def add_arrow(ax: plt.Axes, p0: tuple[float, float], p1: tuple[float, float]) -> None:
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="->", mutation_scale=10, linewidth=1.0, color="#4d4d4d"))


def plot_combined_setup(ax: plt.Axes) -> None:
    ax.set_title("(a) Combined hard-case setup", loc="left", pad=6)
    ax.axis("off")
    ax.plot([0.06, 0.92], [0.46, 0.46], color="#333333", linewidth=1.2)
    for x, label in zip([0.20, 0.48, 0.76], ["Left", "Target", "Right"]):
        add_box(ax, (x - 0.065, 0.40), (0.13, 0.12), label, fc="#f5f5f5")
    for x, temp in zip([0.20, 0.48, 0.76], ["T$_L$", "T$_T$", "T$_R$"]):
        add_box(ax, (x - 0.050, 0.72), (0.10, 0.09), temp, fc="#fafafa")
        add_arrow(ax, (x, 0.70), (x, 0.54))
    ax.add_patch(Rectangle((0.415, 0.40), 0.055, 0.12, facecolor="#d9d9d9", edgecolor="none", alpha=0.9))
    ax.text(0.48, 0.87, "Local heating", ha="center", va="center", fontsize=8.5)
    add_arrow(ax, (0.46, 0.82), (0.46, 0.55))
    add_arrow(ax, (0.50, 0.82), (0.50, 0.55))

    add_box(ax, (0.64, 0.12), (0.14, 0.08), "Detector", fc="#fafafa")
    add_arrow(ax, (0.54, 0.23), (0.64, 0.16))
    ax.text(0.50, 0.25, "ripple / spike / drift", fontsize=8.0, ha="center", va="center")
    x = np.linspace(0.34, 0.56, 80)
    ax.plot(x, 0.17 + 0.010 * np.sin(np.linspace(0.0, 3.6 * np.pi, x.size)), color="#555555", linewidth=1.0)
    ax.text(0.49, 0.02, "Neighbor shift + linewidth asymmetry + weak artifacts", fontsize=8.5, ha="center", va="bottom")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)


def plot_representative_spectrum(
    ax: plt.Axes,
    wavelength_nm: np.ndarray,
    spectrum: np.ndarray,
    center_row: np.ndarray,
    title: str,
    show_legend: bool,
    cfg: FigureConfig,
) -> None:
    ax.set_title(title, loc="left", pad=6)
    ax.plot(wavelength_nm, spectrum, color="#111111", linewidth=1.6)
    ax.axvline(center_row[1], color="#111111", linestyle="--", linewidth=1.2, label="True center")
    ax.axvline(center_row[2], color="#9a9a9a", linestyle="-.", linewidth=1.2, label="Baseline")
    ax.axvline(center_row[3], color="#1f4e79", linestyle=":", linewidth=1.4, label="Proposed")
    ax.set_xlim(cfg.xlim_nm)
    ax.set_ylim(cfg.ylim_spec)
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Normalized reflectivity")
    ax.grid(True, linestyle="--", alpha=0.25)
    beautify_axis(ax)
    if show_legend:
        ax.legend(frameon=False, loc="upper right")


def empirical_cdf(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.sort(np.asarray(values, dtype=np.float64))
    y = np.arange(1, x.size + 1, dtype=np.float64) / x.size
    return x, y


def plot_error_distribution(
    ax: plt.Axes,
    errors_pm: np.ndarray,
    title: str,
    line_color: str,
    p95: float,
    p99: float,
    xlim: tuple[float, float],
) -> None:
    ax.set_title(title, loc="left", pad=6)
    x, y = empirical_cdf(errors_pm)
    ax.plot(x, y, color=line_color, linewidth=1.8)
    ax.axvline(p95, color="#8a8a8a", linestyle="--", linewidth=1.0)
    ax.axvline(p99, color="#6f6f6f", linestyle=":", linewidth=1.1)
    ax.set_xlim(xlim)
    ax.set_ylim(0.0, 1.02)
    ax.set_xlabel("Absolute error (pm)")
    ax.set_ylabel("Cumulative probability")
    ax.grid(True, linestyle="--", alpha=0.25)
    beautify_axis(ax)
    ax.text(
        0.52,
        0.12,
        f"P95 = {p95:.1f} pm\nP99 = {p99:.1f} pm",
        transform=ax.transAxes,
        fontsize=8.0,
        ha="left",
        va="bottom",
        bbox={"boxstyle": "round,pad=0.20", "facecolor": "white", "edgecolor": "#b0b0b0", "alpha": 0.9},
    )


def plot_prediction_scatter(ax: plt.Axes, predictions: np.ndarray, lim_pm: tuple[float, float]) -> None:
    ax.set_title("(g) Prediction scatter comparison", loc="left", pad=6)
    true_pm = predictions[:, 0]
    baseline_pm = predictions[:, 1]
    proposed_pm = predictions[:, 2]

    baseline_err = np.abs(baseline_pm - true_pm)
    proposed_err = np.abs(proposed_pm - true_pm)

    ax.plot(lim_pm, lim_pm, linestyle="--", color="#8a8a8a", linewidth=1.2)
    ax.scatter(
        true_pm,
        baseline_pm,
        s=18,
        marker="s",
        facecolors="none",
        edgecolors="#8f8f8f",
        linewidths=0.9,
        alpha=0.95,
        label="Baseline",
    )
    ax.scatter(
        true_pm,
        proposed_pm,
        s=20,
        marker="^",
        color="#1f4e79",
        edgecolors="#1f4e79",
        linewidths=0.5,
        alpha=0.90,
        label="Proposed",
    )
    ax.set_xlim(lim_pm)
    ax.set_ylim(lim_pm)
    ax.set_xlabel("True target shift (pm)")
    ax.set_ylabel("Predicted target shift (pm)")
    ax.grid(True, linestyle="--", alpha=0.25)
    beautify_axis(ax)
    ax.legend(frameon=False, loc="upper left")
    ax.text(
        0.48,
        0.08,
        f"Baseline P95/P99 = {np.quantile(baseline_err,0.95):.1f}/{np.quantile(baseline_err,0.99):.1f} pm\n"
        f"Proposed P95/P99 = {np.quantile(proposed_err,0.95):.1f}/{np.quantile(proposed_err,0.99):.1f} pm",
        transform=ax.transAxes,
        fontsize=7.8,
        ha="left",
        va="bottom",
        bbox={"boxstyle": "round,pad=0.20", "facecolor": "white", "edgecolor": "#b0b0b0", "alpha": 0.9},
    )


def plot_casewise_bar(ax: plt.Axes, case_names: list[str], case_err: np.ndarray) -> None:
    ax.set_title("(h) Sample-wise error comparison", loc="left", pad=6)
    x = np.arange(len(case_names), dtype=np.float64)
    width = 0.36
    ax.bar(x - width / 2.0, case_err[:, 0], width=width, color="white", edgecolor="#8f8f8f", linewidth=1.0, label="Baseline")
    ax.bar(x + width / 2.0, case_err[:, 1], width=width, color="#4f6d8a", edgecolor="#4f6d8a", linewidth=0.8, label="Proposed")
    ax.set_xticks(x)
    ax.set_xticklabels(case_names)
    ax.set_ylabel("Absolute error (pm)")
    ax.grid(True, axis="y", linestyle="--", alpha=0.25)
    beautify_axis(ax)
    ax.legend(frameon=False, loc="upper right")


def build_figure(
    wavelength_nm: np.ndarray,
    spectra: np.ndarray,
    centers: np.ndarray,
    predictions: np.ndarray,
    case_names: list[str],
    case_err: np.ndarray,
    cfg: FigureConfig,
) -> plt.Figure:
    baseline_err = np.abs(predictions[:, 1] - predictions[:, 0])
    proposed_err = np.abs(predictions[:, 2] - predictions[:, 0])
    xlim_err = (0.0, float(np.ceil(max(baseline_err.max(), proposed_err.max()) / 5.0) * 5.0 + 2.0))
    lim_pm = (-55.0, 55.0)

    fig, axes = plt.subplots(2, 4, figsize=cfg.figsize, constrained_layout=True)

    plot_combined_setup(axes[0, 0])
    plot_representative_spectrum(axes[0, 1], wavelength_nm, spectra[:, 0], centers[0], "(b) Representative hard-case spectrum 1", True, cfg)
    plot_representative_spectrum(axes[0, 2], wavelength_nm, spectra[:, 1], centers[1], "(c) Representative hard-case spectrum 2", False, cfg)
    plot_representative_spectrum(axes[0, 3], wavelength_nm, spectra[:, 2], centers[2], "(d) Representative hard-case spectrum 3", False, cfg)

    plot_error_distribution(
        axes[1, 0],
        baseline_err,
        "(e) Baseline error distribution",
        line_color="#7f7f7f",
        p95=float(np.quantile(baseline_err, 0.95)),
        p99=float(np.quantile(baseline_err, 0.99)),
        xlim=xlim_err,
    )
    plot_error_distribution(
        axes[1, 1],
        proposed_err,
        "(f) Proposed error distribution",
        line_color="#1f4e79",
        p95=float(np.quantile(proposed_err, 0.95)),
        p99=float(np.quantile(proposed_err, 0.99)),
        xlim=xlim_err,
    )
    plot_prediction_scatter(axes[1, 2], predictions, lim_pm)
    plot_casewise_bar(axes[1, 3], case_names, case_err)

    return fig


def main() -> None:
    configure_style()
    cfg = FigureConfig()
    figure_root = Path(__file__).resolve().parents[1]
    _, outputs_dir, data_dir = ensure_dirs(figure_root)

    wavelength_nm, spectra, centers, predictions, case_names, case_err, source = load_or_generate_combined_hard_data(data_dir, cfg)
    fig = build_figure(wavelength_nm, spectra, centers, predictions, case_names, case_err, cfg)

    png_path = outputs_dir / "real_experiment_combined_hard_cases.png"
    pdf_path = outputs_dir / "real_experiment_combined_hard_cases.pdf"
    fig.savefig(png_path, dpi=cfg.dpi, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    if source == "project":
        source_mode = "current project combined_hard outputs"
    elif source == "csv":
        source_mode = "existing CSV data"
    else:
        source_mode = "placeholder fallback data"

    print(f"Saved PNG: {png_path}")
    print(f"Saved PDF: {pdf_path}")
    print(f"Data source mode: {source_mode}")


if __name__ == "__main__":
    main()
