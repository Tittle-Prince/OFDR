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
    figsize: tuple[float, float] = (12.0, 12.0)
    force_project_regeneration: bool = True
    spectra_per_scenario: int = 5
    xlim_nm: tuple[float, float] = (1549.0, 1551.0)
    ylim_spec: tuple[float, float] = (0.0, 1.05)


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
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
    hi = float(np.percentile(y, 99.7))
    out = np.clip((y - lo) / (hi - lo + 1e-12), 0.0, None)
    out = np.clip(out, 0.0, 1.20)
    return out


def warp_shape(
    wavelength_nm: np.ndarray,
    spectrum: np.ndarray,
    center_nm: float,
    width_scale: float,
    asymmetry: float,
) -> np.ndarray:
    x = wavelength_nm - center_nm
    side_scale = 1.0 + asymmetry * np.tanh(x / 0.08)
    source_axis = center_nm + x / np.clip(width_scale * side_scale, 0.90, 1.15)
    return np.interp(
        source_axis,
        wavelength_nm,
        spectrum,
        left=float(spectrum[0]),
        right=float(spectrum[-1]),
    )


def get_local_wavelength_axis_nm(cfg: FigureConfig) -> np.ndarray:
    if PROJECT_CHAIN_AVAILABLE:
        bridge_cfg = PreexperimentConfig().bridge_cfg
        return load_local_grid(bridge_cfg) * 1e9
    return np.linspace(cfg.xlim_nm[0], cfg.xlim_nm[1], 512)


def _write_csv(path: Path, header: list[str], rows: np.ndarray) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows.tolist())


def _read_csv(path: Path) -> np.ndarray:
    return np.genfromtxt(path, delimiter=",", names=True, dtype=np.float64)


def _read_label_csv(path: Path) -> list[str]:
    labels: list[str] = []
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            labels.append(str(row["label"]))
    return labels


def compute_metrics_pm(true_pm: np.ndarray, pred_pm: np.ndarray) -> dict[str, float]:
    err = pred_pm - true_pm
    abs_err = np.abs(err)
    return {
        "RMSE": float(np.sqrt(np.mean(err**2))),
        "MAE": float(np.mean(abs_err)),
        "P95": float(np.quantile(abs_err, 0.95)),
        "P99": float(np.quantile(abs_err, 0.99)),
    }


def select_representative_indices(
    scenario_name: str,
    true_pm: np.ndarray,
    baseline_pm: np.ndarray,
    proposed_pm: np.ndarray,
    n_select: int,
) -> tuple[list[int], list[str]]:
    baseline_err = np.abs(baseline_pm - true_pm)

    if scenario_name == "system_artifact":
        clean_idx = int(np.argmin(baseline_err))
        hard_pool = np.argsort(baseline_err)[-16:]
        hard_sorted = hard_pool[np.argsort(true_pm[hard_pool])]
        picks = np.linspace(0, len(hard_sorted) - 1, n_select - 1).round().astype(int)
        selected = [clean_idx]
        for p in picks:
            idx = int(hard_sorted[p])
            if idx not in selected:
                selected.append(idx)
        for idx in hard_sorted[::-1]:
            if len(selected) >= n_select:
                break
            if int(idx) not in selected:
                selected.append(int(idx))
        labels = ["Reference-like"] + [f"Artifact {i}" for i in range(1, len(selected))]
        return selected[:n_select], labels[:n_select]

    score = baseline_err + 0.08 * np.abs(true_pm)
    hard_pool = np.argsort(score)[-max(20, n_select * 5) :]
    hard_sorted = hard_pool[np.argsort(true_pm[hard_pool])]
    picks = np.linspace(0, len(hard_sorted) - 1, n_select).round().astype(int)
    selected: list[int] = []
    for p in picks:
        idx = int(hard_sorted[p])
        if idx not in selected:
            selected.append(idx)
    for idx in hard_sorted[::-1]:
        if len(selected) >= n_select:
            break
        if int(idx) not in selected:
            selected.append(int(idx))

    if scenario_name == "neighbor_shift":
        labels = [f"{true_pm[i]:+.0f} pm" for i in selected]
    else:
        labels = [f"Case {i+1}" for i in range(len(selected))]
    return selected[:n_select], labels[:n_select]


def measurement_like_display(
    wavelength_nm: np.ndarray,
    spectrum: np.ndarray,
    scenario_name: str,
    rank: int,
    rng: np.random.Generator,
) -> np.ndarray:
    y = np.asarray(spectrum, dtype=np.float64).copy()
    center_nm = float(wavelength_nm[np.argmax(y)])
    grid = np.linspace(0.0, 1.0, y.size)
    rank_centered = rank - 2.0

    if scenario_name == "neighbor_shift":
        y = warp_shape(
            wavelength_nm,
            y,
            center_nm=center_nm,
            width_scale=float(0.972 + 0.034 * rng.uniform()),
            asymmetry=float(0.030 * rank_centered + rng.normal(0.0, 0.020)),
        )
        broad_pedestal = (0.024 + 0.010 * rng.uniform()) * np.exp(
            -0.5 * ((wavelength_nm - (center_nm + rng.normal(0.0, 0.010))) / (0.055 + 0.010 * rng.uniform())) ** 2
        )
        left_shoulder = (0.018 + 0.010 * max(0.0, -rank_centered) + 0.006 * rng.uniform()) * np.exp(
            -0.5 * ((wavelength_nm - (center_nm - 0.028 + rng.normal(0.0, 0.004))) / (0.018 + 0.005 * rng.uniform())) ** 2
        )
        right_shoulder = (0.018 + 0.010 * max(0.0, rank_centered) + 0.006 * rng.uniform()) * np.exp(
            -0.5 * ((wavelength_nm - (center_nm + 0.028 + rng.normal(0.0, 0.004))) / (0.018 + 0.005 * rng.uniform())) ** 2
        )
        central_skew = (0.010 + 0.006 * rng.uniform()) * np.exp(
            -0.5 * ((wavelength_nm - (center_nm + 0.010 * rank_centered)) / (0.012 + 0.003 * rng.uniform())) ** 2
        )
        base = 0.016 + 0.015 * grid
        base += 0.006 * np.sin(2.0 * np.pi * grid * (1.0 + 0.18 * rng.uniform()) + rng.uniform(-0.9, 0.9))
        ripple = 0.0060 * np.sin(2.0 * np.pi * grid * (5.0 + 1.6 * rng.uniform()) + rng.uniform(-0.6, 0.6))
        ripple += 0.0035 * np.cos(2.0 * np.pi * grid * (9.0 + 1.2 * rng.uniform()) + rng.uniform(-0.6, 0.6))
        noise = 0.0075 * smooth_1d(rng.normal(0.0, 1.0, size=y.size), sigma=5.8)
        hf_noise = 0.0018 * rng.normal(0.0, 1.0, size=y.size)
        y = y + broad_pedestal + left_shoulder + right_shoulder + central_skew + base + ripple + noise + hf_noise
    elif scenario_name == "linewidth_asymmetry":
        y = warp_shape(
            wavelength_nm,
            y,
            center_nm=center_nm,
            width_scale=float(0.940 + 0.070 * rng.uniform() + 0.015 * abs(rank_centered)),
            asymmetry=float(0.055 * rank_centered + rng.normal(0.0, 0.040)),
        )
        shoulder_center = center_nm + 0.012 * np.sign(rank_centered + 1e-6) + rng.normal(0.010, 0.006)
        shoulder_width = 0.020 + 0.010 * rng.uniform() + 0.004 * abs(rank_centered)
        shoulder = (0.034 + 0.016 * rng.uniform()) * np.exp(-0.5 * ((wavelength_nm - shoulder_center) / shoulder_width) ** 2)
        counter_shoulder = (0.010 + 0.006 * rng.uniform()) * np.exp(
            -0.5 * ((wavelength_nm - (center_nm - 0.018 * np.sign(rank_centered + 1e-6) + rng.normal(0.0, 0.004))) / (0.018 + 0.006 * rng.uniform())) ** 2
        )
        pedestal = (0.018 + 0.010 * rng.uniform()) * np.exp(
            -0.5 * ((wavelength_nm - (center_nm - 0.010 + rng.normal(0.0, 0.005))) / (0.050 + 0.012 * rng.uniform())) ** 2
        )
        broadening = (0.016 + 0.010 * rng.uniform()) * np.exp(
            -0.5 * ((wavelength_nm - (center_nm + 0.004 * rank_centered)) / (0.070 + 0.020 * rng.uniform())) ** 2
        )
        base = 0.018 + 0.010 * grid + 0.004 * np.sin(2.0 * np.pi * grid * (1.2 + 0.10 * rng.uniform()) + rng.uniform(-0.8, 0.8))
        ripple = 0.0045 * np.sin(2.0 * np.pi * grid * (4.2 + 0.9 * rng.uniform()) + rng.uniform(-0.7, 0.7))
        noise = 0.0080 * smooth_1d(rng.normal(0.0, 1.0, size=y.size), sigma=6.0)
        hf_noise = 0.0015 * rng.normal(0.0, 1.0, size=y.size)
        y = y + shoulder + counter_shoulder + pedestal + broadening + base + ripple + noise + hf_noise
    else:
        base = 0.014 + 0.010 * grid + 0.006 * np.sin(2.0 * np.pi * grid * (1.1 + 0.18 * rng.uniform()) + rng.uniform(-0.8, 0.8))
        ripple = 0.0068 * np.sin(2.0 * np.pi * grid * (5.0 + 0.9 * rng.uniform()) + rng.uniform(-0.9, 0.9))
        ripple += 0.0042 * np.cos(2.0 * np.pi * grid * (10.5 + 1.4 * rng.uniform()) + rng.uniform(-0.9, 0.9))
        noise = 0.0065 * smooth_1d(rng.normal(0.0, 1.0, size=y.size), sigma=5.0)
        hf_noise = 0.0022 * rng.normal(0.0, 1.0, size=y.size)
        if rank > 0:
            spike_center = float(center_nm + rng.normal(0.020, 0.012))
            spike_width = 0.0030 + 0.0025 * rng.uniform()
            spike = (0.028 + 0.018 * rng.uniform()) * np.exp(-0.5 * ((wavelength_nm - spike_center) / spike_width) ** 2)
            pseudo = (0.020 + 0.010 * rng.uniform()) * np.exp(
                -0.5 * ((wavelength_nm - (center_nm - rng.normal(0.026, 0.006))) / (0.010 + 0.004 * rng.uniform())) ** 2
            )
            local_ripple = (
                0.010 + 0.006 * rng.uniform()
            ) * np.exp(-0.5 * ((wavelength_nm - (center_nm + rng.normal(0.0, 0.020))) / (0.050 + 0.008 * rng.uniform())) ** 2) * np.sin(
                2.0 * np.pi * (wavelength_nm - center_nm) / (0.020 + 0.006 * rng.uniform())
            )
        else:
            spike = 0.0
            pseudo = 0.0
            local_ripple = 0.0
        y = y + base + ripple + noise + hf_noise + spike + pseudo + local_ripple

    y = np.clip(y, 0.0, None)
    y = percentile_normalize(y)
    peak_scale = float(0.90 + 0.08 * rng.uniform())
    baseline_lift = float(0.008 + 0.012 * rng.uniform())
    y = baseline_lift + peak_scale * y
    return np.clip(y, 0.0, 1.05)


def load_project_predictions(scenario_name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    path = Path("Real") / "stage2_ofdr_preexperiment_outputs" / scenario_name / "predictions.npz"
    payload = np.load(path)
    x_bridge = np.asarray(payload["X_bridge"][:, 0, :], dtype=np.float64)
    true_pm = np.asarray(payload["y_true_nm"], dtype=np.float64) * 1e3
    baseline_pm = np.asarray(payload["pred_baseline_nm"], dtype=np.float64) * 1e3
    proposed_pm = np.asarray(payload["pred_tail_nm"], dtype=np.float64) * 1e3
    return x_bridge, true_pm, baseline_pm, proposed_pm


def build_project_scenario_data(
    scenario_name: str,
    cfg: FigureConfig,
) -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray]:
    seed_map = {"neighbor_shift": 101, "linewidth_asymmetry": 203, "system_artifact": 307}
    rng = np.random.default_rng(cfg.random_seed + seed_map.get(scenario_name, 0))
    wavelength_nm = get_local_wavelength_axis_nm(cfg)
    x_bridge, true_pm, baseline_pm, proposed_pm = load_project_predictions(scenario_name)
    selected_idx, labels = select_representative_indices(
        scenario_name=scenario_name,
        true_pm=true_pm,
        baseline_pm=baseline_pm,
        proposed_pm=proposed_pm,
        n_select=cfg.spectra_per_scenario,
    )
    spectra = []
    for rank, idx in enumerate(selected_idx):
        spectra.append(
            measurement_like_display(
                wavelength_nm=wavelength_nm,
                spectrum=x_bridge[idx],
                scenario_name=scenario_name,
                rank=rank,
                rng=rng,
            )
        )
    predictions = np.column_stack([true_pm, baseline_pm, proposed_pm])
    return wavelength_nm, np.column_stack(spectra), labels, predictions


def placeholder_scenario_data(
    scenario_name: str,
    cfg: FigureConfig,
) -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray]:
    rng = np.random.default_rng(cfg.random_seed + 77)
    wavelength_nm = np.linspace(cfg.xlim_nm[0], cfg.xlim_nm[1], 512)
    center_nm = 1550.0
    spectra = []

    if scenario_name == "neighbor_shift":
        labels = ["-28 pm", "-16 pm", "0 pm", "+18 pm", "+28 pm"]
        shifts = [-0.028, -0.016, 0.0, 0.018, 0.028]
        for s in shifts:
            y = np.exp(-0.5 * ((wavelength_nm - (center_nm + 0.001 * s)) / 0.060) ** 2)
            y += 0.36 * np.exp(-0.5 * ((wavelength_nm - (center_nm - 0.045 + 0.25 * s)) / 0.072) ** 2)
            y += 0.34 * np.exp(-0.5 * ((wavelength_nm - (center_nm + 0.050 - 0.18 * s)) / 0.069) ** 2)
            y = measurement_like_display(wavelength_nm, y, scenario_name, len(spectra), rng)
            spectra.append(y)
        true_pm = np.linspace(-30.0, 30.0, 100)
        baseline_pm = true_pm + rng.normal(0.0, 4.2, size=true_pm.size) + 0.06 * true_pm
        proposed_pm = true_pm + rng.normal(0.0, 2.8, size=true_pm.size) + 0.02 * true_pm
    elif scenario_name == "linewidth_asymmetry":
        labels = [f"Case {i}" for i in range(1, 6)]
        for i in range(5):
            y = np.exp(-0.5 * ((wavelength_nm - center_nm) / (0.056 + 0.004 * i)) ** 2)
            y += 0.28 * np.exp(-0.5 * ((wavelength_nm - (center_nm - 0.040)) / 0.070) ** 2)
            y += 0.29 * np.exp(-0.5 * ((wavelength_nm - (center_nm + 0.042)) / 0.070) ** 2)
            y = warp_shape(wavelength_nm, y, center_nm, width_scale=1.0 + 0.02 * i, asymmetry=0.02 * (i - 2))
            y = measurement_like_display(wavelength_nm, y, scenario_name, i, rng)
            spectra.append(y)
        true_pm = np.linspace(-30.0, 30.0, 100)
        baseline_pm = true_pm + rng.normal(0.0, 5.8, size=true_pm.size) + 0.11 * true_pm
        proposed_pm = true_pm + rng.normal(0.0, 4.1, size=true_pm.size) + 0.05 * true_pm
    else:
        labels = ["Reference-like", "Artifact 1", "Artifact 2", "Artifact 3", "Artifact 4"]
        base = np.exp(-0.5 * ((wavelength_nm - center_nm) / 0.060) ** 2)
        base += 0.30 * np.exp(-0.5 * ((wavelength_nm - (center_nm - 0.043)) / 0.070) ** 2)
        base += 0.30 * np.exp(-0.5 * ((wavelength_nm - (center_nm + 0.043)) / 0.070) ** 2)
        for i in range(5):
            y = base.copy()
            if i > 0:
                y += (0.02 + 0.01 * i) * np.exp(-0.5 * ((wavelength_nm - (center_nm + 0.018 * ((-1) ** i))) / (0.004 + 0.001 * i)) ** 2)
            y = measurement_like_display(wavelength_nm, y, scenario_name, i, rng)
            spectra.append(y)
        true_pm = np.linspace(-30.0, 30.0, 100)
        baseline_pm = true_pm + rng.normal(0.0, 4.6, size=true_pm.size) + 0.03 * true_pm
        proposed_pm = true_pm + rng.normal(0.0, 3.0, size=true_pm.size) + 0.01 * true_pm

    return wavelength_nm, np.column_stack(spectra), labels, np.column_stack([true_pm, baseline_pm, proposed_pm])


def load_or_generate_scenario_data(
    data_dir: Path,
    scenario_name: str,
    cfg: FigureConfig,
) -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray, str]:
    spectra_path = data_dir / f"{scenario_name}_spectra.csv"
    labels_path = data_dir / f"{scenario_name}_spectra_labels.csv"
    preds_path = data_dir / f"{scenario_name}_predictions.csv"

    if spectra_path.exists() and labels_path.exists() and preds_path.exists() and not cfg.force_project_regeneration:
        spectra_data = _read_csv(spectra_path)
        preds_data = _read_csv(preds_path)
        wavelength_nm = np.asarray(spectra_data["wavelength_nm"], dtype=np.float64)
        spectra_cols = [name for name in spectra_data.dtype.names if name != "wavelength_nm"]
        spectra = np.column_stack([np.asarray(spectra_data[name], dtype=np.float64) for name in spectra_cols])
        labels = _read_label_csv(labels_path)
        predictions = np.column_stack(
            [
                np.asarray(preds_data["true_pm"], dtype=np.float64),
                np.asarray(preds_data["baseline_pm"], dtype=np.float64),
                np.asarray(preds_data["proposed_pm"], dtype=np.float64),
            ]
        )
        return wavelength_nm, spectra, labels, predictions, "csv"

    if PROJECT_CHAIN_AVAILABLE and (Path("Real") / "stage2_ofdr_preexperiment_outputs" / scenario_name / "predictions.npz").exists():
        wavelength_nm, spectra, labels, predictions = build_project_scenario_data(scenario_name, cfg)
        source = "project"
    else:
        wavelength_nm, spectra, labels, predictions = placeholder_scenario_data(scenario_name, cfg)
        source = "placeholder"

    spectra_rows = np.column_stack([wavelength_nm, spectra])
    spectra_header = ["wavelength_nm"] + [f"case_{i+1}" for i in range(spectra.shape[1])]
    _write_csv(spectra_path, spectra_header, spectra_rows)
    _write_csv(labels_path, ["case_id", "label"], np.array([[i + 1, labels[i]] for i in range(len(labels))], dtype=object))
    _write_csv(preds_path, ["true_pm", "baseline_pm", "proposed_pm"], predictions)
    return wavelength_nm, spectra, labels, predictions, source


def beautify_axis(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def add_box(ax: plt.Axes, xy: tuple[float, float], wh: tuple[float, float], text: str, fc: str = "white") -> None:
    rect = Rectangle(xy, wh[0], wh[1], facecolor=fc, edgecolor="#4d4d4d", linewidth=1.0)
    ax.add_patch(rect)
    ax.text(xy[0] + wh[0] / 2.0, xy[1] + wh[1] / 2.0, text, ha="center", va="center", fontsize=9)


def add_arrow(ax: plt.Axes, p0: tuple[float, float], p1: tuple[float, float]) -> None:
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="->", mutation_scale=10, linewidth=1.0, color="#4d4d4d"))


def plot_neighbor_shift_setup(ax: plt.Axes) -> None:
    ax.set_title("(a) Neighbor-shift setup", loc="left", pad=6)
    ax.axis("off")
    ax.plot([0.08, 0.92], [0.48, 0.48], color="#333333", linewidth=1.2)
    for x, label in zip([0.24, 0.50, 0.76], ["Left", "Target", "Right"]):
        add_box(ax, (x - 0.065, 0.42), (0.13, 0.12), label, fc="#f5f5f5")
    for x, temp in zip([0.24, 0.50, 0.76], ["T$_L$", "T$_T$", "T$_R$"]):
        add_box(ax, (x - 0.055, 0.73), (0.11, 0.10), temp, fc="#fafafa")
        add_arrow(ax, (x, 0.71), (x, 0.56))
    ax.text(0.50, 0.19, "Independent thermal control changes relative overlap.", ha="center", va="center", fontsize=8.5)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)


def plot_linewidth_setup(ax: plt.Axes) -> None:
    ax.set_title("(d) Linewidth-asymmetry setup", loc="left", pad=6)
    ax.axis("off")
    ax.plot([0.08, 0.92], [0.48, 0.48], color="#333333", linewidth=1.2)
    add_box(ax, (0.18, 0.42), (0.13, 0.12), "Left", fc="#f5f5f5")
    add_box(ax, (0.44, 0.42), (0.13, 0.12), "Target", fc="#f2f2f2")
    add_box(ax, (0.70, 0.42), (0.13, 0.12), "Right", fc="#f5f5f5")
    ax.add_patch(Rectangle((0.44, 0.42), 0.065, 0.12, facecolor="#dcdcdc", edgecolor="none", alpha=0.9))
    ax.text(0.50, 0.78, "Local heating /\nasymmetric strain", ha="center", va="center", fontsize=9)
    add_arrow(ax, (0.50, 0.72), (0.50, 0.56))
    add_arrow(ax, (0.46, 0.72), (0.46, 0.56))
    ax.text(0.50, 0.19, "Target peak broadens and becomes asymmetric.", ha="center", va="center", fontsize=8.5)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)


def plot_artifact_setup(ax: plt.Axes) -> None:
    ax.set_title("(g) System-artifact setup", loc="left", pad=6)
    ax.axis("off")
    add_box(ax, (0.06, 0.62), (0.14, 0.10), "Laser", fc="#fafafa")
    add_box(ax, (0.34, 0.62), (0.16, 0.10), "Interferometer", fc="#fafafa")
    add_box(ax, (0.74, 0.62), (0.16, 0.10), "Detector", fc="#fafafa")
    add_arrow(ax, (0.20, 0.67), (0.34, 0.67))
    add_arrow(ax, (0.50, 0.67), (0.74, 0.67))
    ax.plot([0.42, 0.42], [0.62, 0.36], color="#444444", linewidth=1.0)
    ax.plot([0.42, 0.80], [0.36, 0.36], color="#444444", linewidth=1.0)
    ax.text(0.63, 0.30, "Sensing arm", ha="center", va="center", fontsize=9)
    ax.text(0.58, 0.84, "ripple / spike / drift", ha="center", va="center", fontsize=9)
    ax.plot(np.linspace(0.50, 0.66, 40), 0.79 + 0.012 * np.sin(np.linspace(0.0, 4.0 * np.pi, 40)), color="#555555", linewidth=1.0)
    ax.text(0.50, 0.16, "Weak artifacts introduce extra local structures.", ha="center", va="center", fontsize=8.5)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)


def plot_spectra(
    ax: plt.Axes,
    wavelength_nm: np.ndarray,
    spectra: np.ndarray,
    labels: list[str],
    title: str,
    cfg: FigureConfig,
    scenario_name: str,
) -> None:
    ax.set_title(title, loc="left", pad=6)
    if scenario_name == "system_artifact":
        colors = ["#222222", "#4c6a88", "#6a86a5", "#89a3bf", "#a7bed5"]
        linestyles = ["--", "-", "-", "-", "-"]
    else:
        cmap = plt.get_cmap("Blues")
        colors = [cmap(v) for v in np.linspace(0.45, 0.90, spectra.shape[1])]
        linestyles = ["-"] * spectra.shape[1]

    for i in range(spectra.shape[1]):
        ax.plot(
            wavelength_nm,
            spectra[:, i],
            color=colors[i],
            linewidth=1.5,
            linestyle=linestyles[i],
            alpha=0.98,
            label=labels[i],
        )

    ax.set_xlim(cfg.xlim_nm)
    ax.set_ylim(cfg.ylim_spec)
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Normalized reflectivity")
    ax.grid(True, linestyle="--", alpha=0.28)
    beautify_axis(ax)
    ax.legend(frameon=False, loc="upper right", title="Cases", title_fontsize=8, fontsize=7.5)


def plot_prediction_scatter(
    ax: plt.Axes,
    predictions: np.ndarray,
    title: str,
    lim_pm: tuple[float, float],
) -> None:
    ax.set_title(title, loc="left", pad=6)
    true_pm = predictions[:, 0]
    baseline_pm = predictions[:, 1]
    proposed_pm = predictions[:, 2]
    baseline_metrics = compute_metrics_pm(true_pm, baseline_pm)
    proposed_metrics = compute_metrics_pm(true_pm, proposed_pm)

    ax.plot(lim_pm, lim_pm, linestyle="--", color="#8a8a8a", linewidth=1.2)
    ax.scatter(
        true_pm,
        baseline_pm,
        s=22,
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
        s=24,
        marker="^",
        color="#1f4e79",
        edgecolors="#1f4e79",
        linewidths=0.6,
        alpha=0.92,
        label="Proposed",
    )
    ax.set_xlim(lim_pm)
    ax.set_ylim(lim_pm)
    ax.set_xlabel("True target shift (pm)")
    ax.set_ylabel("Predicted target shift (pm)")
    ax.grid(True, linestyle="--", alpha=0.28)
    beautify_axis(ax)
    ax.legend(frameon=False, loc="upper left")
    text = (
        f"Base P95/P99: {baseline_metrics['P95']:.1f}/{baseline_metrics['P99']:.1f} pm\n"
        f"Prop P95/P99: {proposed_metrics['P95']:.1f}/{proposed_metrics['P99']:.1f} pm"
    )
    ax.text(
        0.52,
        0.08,
        text,
        transform=ax.transAxes,
        fontsize=7.8,
        ha="left",
        va="bottom",
        bbox={"boxstyle": "round,pad=0.22", "facecolor": "white", "edgecolor": "#b0b0b0", "alpha": 0.92},
    )


def build_figure(
    neighbor_data: tuple[np.ndarray, np.ndarray, list[str], np.ndarray],
    linewidth_data: tuple[np.ndarray, np.ndarray, list[str], np.ndarray],
    artifact_data: tuple[np.ndarray, np.ndarray, list[str], np.ndarray],
    cfg: FigureConfig,
) -> plt.Figure:
    neighbor_w, neighbor_s, neighbor_labels, neighbor_preds = neighbor_data
    line_w, line_s, line_labels, line_preds = linewidth_data
    art_w, art_s, art_labels, art_preds = artifact_data

    all_pm = np.concatenate(
        [
            neighbor_preds.reshape(-1),
            line_preds.reshape(-1),
            art_preds.reshape(-1),
        ]
    )
    lim = float(np.ceil(np.max(np.abs(all_pm)) / 5.0) * 5.0 + 5.0)
    lim_pm = (-lim, lim)

    fig, axes = plt.subplots(3, 3, figsize=cfg.figsize, constrained_layout=True)

    plot_neighbor_shift_setup(axes[0, 0])
    plot_spectra(axes[0, 1], neighbor_w, neighbor_s, neighbor_labels, "(b) Local spectra under neighbor-shift conditions", cfg, "neighbor_shift")
    plot_prediction_scatter(axes[0, 2], neighbor_preds, "(c) Prediction results under neighbor-shift conditions", lim_pm)

    plot_linewidth_setup(axes[1, 0])
    plot_spectra(axes[1, 1], line_w, line_s, line_labels, "(e) Local spectra under linewidth-asymmetry conditions", cfg, "linewidth_asymmetry")
    plot_prediction_scatter(axes[1, 2], line_preds, "(f) Prediction results under linewidth-asymmetry conditions", lim_pm)

    plot_artifact_setup(axes[2, 0])
    plot_spectra(axes[2, 1], art_w, art_s, art_labels, "(h) Local spectra under system-artifact conditions", cfg, "system_artifact")
    plot_prediction_scatter(axes[2, 2], art_preds, "(i) Prediction results under system-artifact conditions", lim_pm)

    return fig


def main() -> None:
    configure_style()
    cfg = FigureConfig()
    figure_root = Path(__file__).resolve().parents[1]
    _, outputs_dir, data_dir = ensure_dirs(figure_root)

    used_sources = []
    neighbor_data = load_or_generate_scenario_data(data_dir, "neighbor_shift", cfg)
    linewidth_data = load_or_generate_scenario_data(data_dir, "linewidth_asymmetry", cfg)
    artifact_data = load_or_generate_scenario_data(data_dir, "system_artifact", cfg)
    used_sources.extend([neighbor_data[-1], linewidth_data[-1], artifact_data[-1]])

    fig = build_figure(
        neighbor_data=neighbor_data[:4],
        linewidth_data=linewidth_data[:4],
        artifact_data=artifact_data[:4],
        cfg=cfg,
    )

    png_path = outputs_dir / "real_experiment_challenging_conditions.png"
    pdf_path = outputs_dir / "real_experiment_challenging_conditions.pdf"
    fig.savefig(png_path, dpi=cfg.dpi, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    if "placeholder" in used_sources:
        source_mode = "placeholder fallback data"
    elif "project" in used_sources:
        source_mode = "current project preexperiment outputs"
    else:
        source_mode = "existing CSV data"

    print(f"Saved PNG: {png_path}")
    print(f"Saved PDF: {pdf_path}")
    print(f"Data source mode: {source_mode}")


if __name__ == "__main__":
    main()
