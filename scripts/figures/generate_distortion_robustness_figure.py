from __future__ import annotations

import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from phase2.baselines import (  # noqa: E402
    estimate_center_by_parametric_fit,
    estimate_shift_by_cross_correlation,
    gaussian_spectrum,
    normalize_minmax,
)
from phase3.models import build_model  # noqa: E402
from phase4a.array_simulator import build_wavelength_axis, simulate_identical_array_spectra  # noqa: E402
from phase4a.common import load_config  # noqa: E402
from phase4a.local_window import apply_local_effects  # noqa: E402


METHODS = ["Cross-correlation", "Parametric fitting", "CNN", "CNN+SE"]
COLORS = {
    "Cross-correlation": "#3b3b3b",  # dark gray
    "Parametric fitting": "#5f7692",  # gray-blue
    "CNN": "#4f7f6d",  # muted green
    "CNN+SE": "#7a2a2a",  # dark red
}
MARKERS = {"Cross-correlation": "o", "Parametric fitting": "s", "CNN": "^", "CNN+SE": "D"}


@dataclass
class SweepResult:
    condition_type: str
    level: float
    method: str
    rmse_mean: float
    rmse_std: float


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_pred - y_true) ** 2)))


def load_state(model: torch.nn.Module, ckpt_path: Path) -> None:
    ck = torch.load(ckpt_path, map_location="cpu")
    if isinstance(ck, dict) and "model_state_dict" in ck:
        sd = ck["model_state_dict"]
    elif isinstance(ck, dict) and all(isinstance(k, str) for k in ck.keys()):
        sd = ck
    else:
        sd = ck
    model.load_state_dict(sd, strict=False)


def load_models(input_dim: int) -> tuple[torch.nn.Module, torch.nn.Module, dict[str, str]]:
    ckpt_cnn = PROJECT_ROOT / "results" / "phase3" / "cnn_baseline" / "model.pt"
    ckpt_se = PROJECT_ROOT / "results" / "phase3" / "cnn_se" / "model.pt"
    if not ckpt_cnn.exists() or not ckpt_se.exists():
        raise FileNotFoundError("Required model checkpoints not found: phase3/cnn_baseline/model.pt and phase3/cnn_se/model.pt")

    model_cnn = build_model("cnn_baseline", input_dim=input_dim)
    model_se = build_model("cnn_se", input_dim=input_dim)
    load_state(model_cnn, ckpt_cnn)
    load_state(model_se, ckpt_se)
    model_cnn.eval()
    model_se.eval()
    return model_cnn, model_se, {"CNN": str(ckpt_cnn), "CNN+SE": str(ckpt_se)}


def predict_model(model: torch.nn.Module, x: np.ndarray, batch: int = 512) -> np.ndarray:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    out = np.zeros(len(x), dtype=np.float32)
    with torch.no_grad():
        for i in range(0, len(x), batch):
            xb = torch.tensor(x[i : i + batch], dtype=torch.float32, device=device)
            out[i : i + batch] = model(xb).detach().cpu().numpy().astype(np.float32)
    return out


def predict_cross_and_param(x: np.ndarray, wavelengths: np.ndarray, cfg: dict) -> tuple[np.ndarray, np.ndarray]:
    lambda0 = float(cfg["array"]["lambda0_nm"])
    sigma = float(cfg["array"]["linewidth_sigma_nm"])
    step_nm = float(wavelengths[1] - wavelengths[0])
    fit_window_points = int(cfg["compare"]["fit_window_points"])
    baseline_percentile = float(cfg["compare"]["baseline_percentile"])

    ref = gaussian_spectrum(wavelengths, center_nm=lambda0, sigma_nm=sigma, amplitude=1.0, baseline=0.0)
    if str(cfg["local_window"]["normalize"]) == "minmax_per_sample":
        ref = normalize_minmax(ref)

    pred_cc = np.zeros(len(x), dtype=np.float32)
    pred_pf = np.zeros(len(x), dtype=np.float32)
    for i in range(len(x)):
        pred_cc[i] = estimate_shift_by_cross_correlation(ref, x[i], step_nm)
        center = estimate_center_by_parametric_fit(
            wavelengths,
            x[i],
            fit_window_points=fit_window_points,
            baseline_percentile=baseline_percentile,
        )
        pred_pf[i] = np.float32(center - lambda0)
    return pred_cc, pred_pf


def generate_testset(
    cfg: dict,
    n_samples: int,
    noise_level: float,
    leakage_level: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    wavelengths = build_wavelength_axis(cfg).astype(np.float32)
    n_points = len(wavelengths)
    n_gratings = int(cfg["array"]["n_gratings"])
    target = int(cfg["array"]["target_index"])

    dmin = float(cfg["label"]["delta_lambda_target_min_nm"])
    dmax = float(cfg["label"]["delta_lambda_target_max_nm"])

    arr_rand = cfg.get("array_random", {})
    amp_min = float(arr_rand.get("amplitude_scale_min", 1.0))
    amp_max = float(arr_rand.get("amplitude_scale_max", 1.0))
    sig_min = float(arr_rand.get("linewidth_scale_min", 1.0))
    sig_max = float(arr_rand.get("linewidth_scale_max", 1.0))

    lw_cfg = dict(cfg["local_window"])
    lw_cfg["additive_noise_std"] = float(noise_level)

    x = np.zeros((n_samples, n_points), dtype=np.float32)
    y = np.zeros(n_samples, dtype=np.float32)
    for i in range(n_samples):
        dlam = float(rng.uniform(dmin, dmax))
        amp_scales = rng.uniform(amp_min, amp_max, size=n_gratings).astype(np.float64)
        sigma_scales = rng.uniform(sig_min, sig_max, size=n_gratings).astype(np.float64)
        per_grating, _, _ = simulate_identical_array_spectra(
            wavelengths.astype(np.float64),
            cfg,
            dlam,
            amplitude_scales=amp_scales,
            linewidth_scales=sigma_scales,
        )

        # Fixed leakage strength sweep with mild asymmetry
        weights = np.zeros(n_gratings, dtype=np.float64)
        weights[target] = 1.0
        if target - 1 >= 0:
            weights[target - 1] = leakage_level * float(rng.uniform(0.85, 1.15))
        if target + 1 < n_gratings:
            weights[target + 1] = leakage_level * float(rng.uniform(0.85, 1.15))
        if target - 2 >= 0:
            weights[target - 2] = 0.5 * leakage_level * float(rng.uniform(0.85, 1.15))
        if target + 2 < n_gratings:
            weights[target + 2] = 0.5 * leakage_level * float(rng.uniform(0.85, 1.15))

        local_clean = np.sum(weights[:, None] * per_grating, axis=0)
        shift_nm = float(rng.uniform(float(lw_cfg.get("target_window_shift_min_nm", 0.0)), float(lw_cfg.get("target_window_shift_max_nm", 0.0))))
        local_noisy = apply_local_effects(local_clean, lw_cfg, rng, wavelengths=wavelengths.astype(np.float64), shift_nm=shift_nm)

        x[i] = local_noisy.astype(np.float32)
        y[i] = np.float32(dlam)
    return x, y, wavelengths


def run_sweep(
    condition_type: str,
    levels: list[float],
    cfg: dict,
    model_cnn: torch.nn.Module,
    model_se: torch.nn.Module,
    repeats: list[int],
    n_samples: int,
    fixed_noise: float,
    fixed_leakage: float,
) -> list[SweepResult]:
    rows: list[SweepResult] = []
    for lv in levels:
        rmse_store = {m: [] for m in METHODS}
        for rseed in repeats:
            if condition_type == "noise":
                noise_level = float(lv)
                leakage_level = float(fixed_leakage)
            else:
                noise_level = float(fixed_noise)
                leakage_level = float(lv)

            x, y_true, wl = generate_testset(
                cfg=cfg,
                n_samples=n_samples,
                noise_level=noise_level,
                leakage_level=leakage_level,
                seed=int(rseed + int(round(lv * 10000))),
            )

            pred_cc, pred_pf = predict_cross_and_param(x, wl, cfg)
            pred_cnn = predict_model(model_cnn, x)
            pred_se = predict_model(model_se, x)

            rmse_store["Cross-correlation"].append(rmse(y_true, pred_cc))
            rmse_store["Parametric fitting"].append(rmse(y_true, pred_pf))
            rmse_store["CNN"].append(rmse(y_true, pred_cnn))
            rmse_store["CNN+SE"].append(rmse(y_true, pred_se))

        for m in METHODS:
            arr = np.array(rmse_store[m], dtype=float)
            rows.append(
                SweepResult(
                    condition_type=condition_type,
                    level=float(lv),
                    method=m,
                    rmse_mean=float(np.mean(arr)),
                    rmse_std=float(np.std(arr)),
                )
            )
    return rows


def write_csv(rows: list[SweepResult], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["condition_type", "level", "method", "rmse_mean", "rmse_std"])
        for r in rows:
            w.writerow([r.condition_type, f"{r.level:.6f}", r.method, f"{r.rmse_mean:.10f}", f"{r.rmse_std:.10f}"])


def plot_figure(rows: list[SweepResult], out_png: Path) -> None:
    def pick(cond: str, method: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        rr = [r for r in rows if r.condition_type == cond and r.method == method]
        rr.sort(key=lambda x: x.level)
        x = np.array([r.level for r in rr], dtype=float)
        y = np.array([r.rmse_mean for r in rr], dtype=float)
        e = np.array([r.rmse_std for r in rr], dtype=float)
        return x, y, e

    plt.rcParams["font.family"] = "Arial"
    dpi = 300
    fig_w, fig_h = 1800 / dpi, 800 / dpi
    fig, axes = plt.subplots(1, 2, figsize=(fig_w, fig_h), dpi=dpi)
    fig.patch.set_facecolor("white")

    titles = ["(a) Noise robustness", "(b) Leakage robustness"]
    conds = ["noise", "leakage"]
    xlabels = ["Noise level", "Leakage strength"]

    for ax, ttl, cond, xl in zip(axes, titles, conds, xlabels):
        ax.set_facecolor("white")
        for m in METHODS:
            x, y, e = pick(cond, m)
            ax.errorbar(
                x,
                y,
                yerr=e,
                color=COLORS[m],
                marker=MARKERS[m],
                markersize=5,
                linewidth=1.6,
                capsize=3,
                label=m,
            )
        ax.set_title(ttl, fontsize=11, pad=4)
        ax.set_xlabel(xl, fontsize=10)
        ax.set_ylabel("RMSE (nm)", fontsize=10)
        ax.tick_params(axis="both", labelsize=8, width=1.0, length=4)
        for sp in ax.spines.values():
            sp.set_linewidth(1.1)
        ax.grid(False)

    axes[0].legend(frameon=False, fontsize=8.5, loc="upper left")
    fig.subplots_adjust(left=0.08, right=0.99, bottom=0.16, top=0.90, wspace=0.22)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=dpi)
    plt.close(fig)


def summarize(rows: list[SweepResult], noise_levels: list[float], leakage_levels: list[float]) -> dict:
    def get(cond: str, level: float, method: str) -> float:
        for r in rows:
            if r.condition_type == cond and abs(r.level - level) < 1e-12 and r.method == method:
                return r.rmse_mean
        raise KeyError(cond, level, method)

    low_noise = noise_levels[0]
    high_noise = noise_levels[-1]
    low_leak = leakage_levels[0]
    high_leak = leakage_levels[-1]

    # Low-distortion check using noise=0, leakage fixed low (configured below)
    low_cond = {m: get("noise", low_noise, m) for m in METHODS}
    high_cond = {m: get("noise", high_noise, m) for m in METHODS}
    high_leak_cond = {m: get("leakage", high_leak, m) for m in METHODS}

    traditional_ok = (low_cond["Cross-correlation"] < 0.02) and (low_cond["Parametric fitting"] < 0.02)

    # Degradation speed from low->high leakage (ratio).
    ratios = {m: high_leak_cond[m] / max(get("leakage", low_leak, m), 1e-12) for m in METHODS}
    slowest = min(ratios.keys(), key=lambda k: ratios[k])

    # CNN+SE vs CNN at high distortion (average high two levels in both sweeps).
    hn1, hn2 = noise_levels[-2], noise_levels[-1]
    hl1, hl2 = leakage_levels[-2], leakage_levels[-1]
    cnn_hi = np.mean([get("noise", hn1, "CNN"), get("noise", hn2, "CNN"), get("leakage", hl1, "CNN"), get("leakage", hl2, "CNN")])
    se_hi = np.mean(
        [get("noise", hn1, "CNN+SE"), get("noise", hn2, "CNN+SE"), get("leakage", hl1, "CNN+SE"), get("leakage", hl2, "CNN+SE")]
    )
    se_better = se_hi < cnn_hi

    support_conclusion = (
        (high_noise > low_noise)
        and (high_cond["Cross-correlation"] > low_cond["Cross-correlation"])
        and (high_cond["Parametric fitting"] > low_cond["Parametric fitting"])
        and (high_cond["CNN"] < high_cond["Cross-correlation"])
        and (high_cond["CNN+SE"] < high_cond["Parametric fitting"])
    )

    return {
        "traditional_low_distortion_ok": traditional_ok,
        "degrade_ratio_high_leak": ratios,
        "slowest_degrade_method_high_leak": slowest,
        "cnn_high_distortion_avg_rmse": float(cnn_hi),
        "cnnse_high_distortion_avg_rmse": float(se_hi),
        "cnnse_stable_advantage_over_cnn": bool(se_better),
        "supports_main_claim": bool(support_conclusion),
    }


def main() -> None:
    cfg_path = PROJECT_ROOT / "config" / "phase4_array.yaml"
    cfg = load_config(str(cfg_path))

    # Sweep settings
    noise_levels = [0.0, 0.005, 0.01, 0.02, 0.03]
    leakage_levels = [0.0, 0.1, 0.2, 0.3, 0.4]
    repeats = [2026, 3407, 7777]  # fixed repeats for mean/std
    n_samples = 800

    fixed_leakage_for_noise = 0.0
    fixed_noise_for_leakage = 0.005

    input_dim = int(cfg["array"]["num_points"])
    model_cnn, model_se, model_paths = load_models(input_dim=input_dim)

    noise_rows = run_sweep(
        condition_type="noise",
        levels=noise_levels,
        cfg=cfg,
        model_cnn=model_cnn,
        model_se=model_se,
        repeats=repeats,
        n_samples=n_samples,
        fixed_noise=fixed_noise_for_leakage,
        fixed_leakage=fixed_leakage_for_noise,
    )
    leak_rows = run_sweep(
        condition_type="leakage",
        levels=leakage_levels,
        cfg=cfg,
        model_cnn=model_cnn,
        model_se=model_se,
        repeats=repeats,
        n_samples=n_samples,
        fixed_noise=fixed_noise_for_leakage,
        fixed_leakage=fixed_leakage_for_noise,
    )

    rows = noise_rows + leak_rows
    out_png = PROJECT_ROOT / "results" / "paper_figures" / "FigX_distortion_robustness.png"
    out_csv = PROJECT_ROOT / "results" / "paper_figures" / "FigX_distortion_robustness_data.csv"
    write_csv(rows, out_csv)
    plot_figure(rows, out_png)

    summary = summarize(rows, noise_levels=noise_levels, leakage_levels=leakage_levels)
    summary_path = PROJECT_ROOT / "results" / "paper_figures" / "FigX_distortion_robustness_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # Required terminal outputs
    print("Image path:", out_png.as_posix())
    print(
        "Data generation parameters:",
        json.dumps(
            {
                "config": str(cfg_path),
                "n_samples_per_level": n_samples,
                "repeats": repeats,
                "fixed_leakage_for_noise_sweep": fixed_leakage_for_noise,
                "fixed_noise_for_leakage_sweep": fixed_noise_for_leakage,
                "noise_levels": noise_levels,
                "leakage_levels": leakage_levels,
            },
            ensure_ascii=False,
        ),
    )
    print("Model files:", json.dumps(model_paths, ensure_ascii=False))
    print("Left plot noise levels:", noise_levels)
    print("Right plot leakage levels:", leakage_levels)
    print("Low distortion traditional methods reasonable?:", "Yes" if summary["traditional_low_distortion_ok"] else "No")
    print("Slowest degradation at high leakage:", summary["slowest_degrade_method_high_leak"])
    print("CNN+SE stable advantage over CNN?:", "Yes" if summary["cnnse_stable_advantage_over_cnn"] else "No")
    print(
        "Supports main claim ('traditional degrade under severe distortion while CNN-class more robust')?:",
        "Yes" if summary["supports_main_claim"] else "Partially/No",
    )
    print("Intermediate csv:", out_csv.as_posix())


if __name__ == "__main__":
    main()

