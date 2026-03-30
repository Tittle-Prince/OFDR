from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Real.stage1_three_grating_ideal_ofdr import (
    SimulationConfig,
    build_lambda_axis,
    simulate_three_gratings_reflectivity,
)
from src.ofdr.models.phase3_cnn import build_model


@dataclass
class InferenceConfig:
    dataset_path: str = "data/processed/dataset_c_phase4a_shift004_linewidth_l3.npz"
    baseline_ckpt: str = "results/phase4a_shift004_linewidth_l3/method_enhance_tailaware_baseline_seed45/model_cnn.pt"
    tail_ckpt: str = "results/phase4a_shift004_linewidth_l3/method_enhance_tailaware_hard_seed45/model_cnn.pt"
    output_dir: str = "Real/stage1_three_grating_model_inference_outputs"

    # Match the local-spectrum semantics used in phase4a dataset generation.
    left_weight: float = 0.35
    target_weight: float = 1.0
    right_weight: float = 0.35


def minmax_normalize(x: np.ndarray) -> np.ndarray:
    x_min = float(np.min(x))
    x_max = float(np.max(x))
    return (x - x_min) / (x_max - x_min + 1e-12)


def load_model_wavelength_axis(repo_root: Path, dataset_path: str) -> np.ndarray:
    dataset = np.load(repo_root / dataset_path)
    wavelengths = dataset["wavelengths"].astype(np.float64)
    if wavelengths.ndim != 1 or wavelengths.size != 512:
        raise ValueError("Expected model wavelength axis to be a 1D array with 512 points.")
    return wavelengths


def build_model_input_spectrum(
    sim_cfg: SimulationConfig,
    model_wavelengths_m: np.ndarray,
    delta_lambda_target_m: float,
    inf_cfg: InferenceConfig,
) -> dict[str, np.ndarray]:
    sim_lambda_m = build_lambda_axis(sim_cfg)
    reflectivity = simulate_three_gratings_reflectivity(sim_lambda_m, sim_cfg, delta_lambda_target_m)

    local_weighted = (
        inf_cfg.left_weight * reflectivity["left"]
        + inf_cfg.target_weight * reflectivity["target"]
        + inf_cfg.right_weight * reflectivity["right"]
    )
    local_weighted_norm = minmax_normalize(local_weighted)

    local_model = np.interp(
        model_wavelengths_m,
        sim_lambda_m,
        local_weighted_norm,
        left=float(local_weighted_norm[0]),
        right=float(local_weighted_norm[-1]),
    )

    return {
        "sim_lambda_m": sim_lambda_m,
        "reflectivity_total": reflectivity["total"],
        "local_weighted_dense": local_weighted_norm,
        "local_model_input": local_model.astype(np.float32),
    }


def load_trained_model(checkpoint_path: Path, input_dim: int, device: torch.device) -> torch.nn.Module:
    model = build_model("cnn_baseline", input_dim=input_dim).to(device)
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


def run_model(model: torch.nn.Module, spectrum_1d: np.ndarray, device: torch.device) -> float:
    x = torch.tensor(spectrum_1d[None, :], dtype=torch.float32, device=device)
    with torch.no_grad():
        pred = model(x).detach().cpu().numpy().reshape(-1)
    return float(pred[0])


def plot_inference_results(
    repo_root: Path,
    model_wavelengths_m: np.ndarray,
    cases: list[dict[str, float | np.ndarray]],
    output_dir: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.0), constrained_layout=True)
    lambda_model_nm = model_wavelengths_m * 1e9

    for col, case in enumerate(cases):
        ax = axes[0, col]
        ax.plot(lambda_model_nm, case["local_model_input"], color="#1f77b4", linewidth=1.8)
        ax.set_title(f"Model input spectrum | delta = {case['delta_pm']:+.1f} pm")
        ax.set_xlabel("Wavelength (nm)")
        ax.set_ylabel("Normalized intensity")
        ax.grid(True, alpha=0.3)

        ax = axes[1, col]
        labels = ["True", "Baseline CNN", "Tail+Hard CNN"]
        values = [
            case["true_delta_pm"],
            case["pred_baseline_pm"],
            case["pred_tail_pm"],
        ]
        colors = ["#333333", "#7f7f7f", "#1f77b4"]
        ax.bar(labels, values, color=colors, alpha=0.9)
        ax.set_title(f"Prediction comparison | delta = {case['delta_pm']:+.1f} pm")
        ax.set_ylabel("Predicted delta_lambda_target (pm)")
        ax.grid(True, axis="y", alpha=0.3)
        for i, v in enumerate(values):
            ax.text(i, v + 0.8, f"{v:+.2f}", ha="center", va="bottom", fontsize=9)

    out_path = output_dir / "stage1_model_inference_comparison.png"
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    sim_cfg = SimulationConfig()
    inf_cfg = InferenceConfig()
    output_dir = repo_root / inf_cfg.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    model_wavelengths_m = load_model_wavelength_axis(repo_root, inf_cfg.dataset_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    baseline_model = load_trained_model(repo_root / inf_cfg.baseline_ckpt, input_dim=model_wavelengths_m.size, device=device)
    tail_model = load_trained_model(repo_root / inf_cfg.tail_ckpt, input_dim=model_wavelengths_m.size, device=device)

    deltas_pm = [0.0, 20.0]
    case_rows: list[dict[str, float | np.ndarray]] = []

    print("Feeding stage-1 ideal local spectra into trained models...")
    print("")
    for delta_pm in deltas_pm:
        delta_m = delta_pm * 1e-12
        payload = build_model_input_spectrum(sim_cfg, model_wavelengths_m, delta_m, inf_cfg)
        pred_baseline_nm = run_model(baseline_model, payload["local_model_input"], device=device)
        pred_tail_nm = run_model(tail_model, payload["local_model_input"], device=device)

        row = {
            "delta_pm": delta_pm,
            "true_delta_pm": delta_pm,
            "pred_baseline_pm": pred_baseline_nm * 1e3,
            "pred_tail_pm": pred_tail_nm * 1e3,
            "err_baseline_pm": pred_baseline_nm * 1e3 - delta_pm,
            "err_tail_pm": pred_tail_nm * 1e3 - delta_pm,
            "local_model_input": payload["local_model_input"],
        }
        case_rows.append(row)

        print(
            f"delta = {delta_pm:+.1f} pm | "
            f"baseline CNN = {row['pred_baseline_pm']:+.3f} pm (err {row['err_baseline_pm']:+.3f} pm) | "
            f"tail+hard CNN = {row['pred_tail_pm']:+.3f} pm (err {row['err_tail_pm']:+.3f} pm)"
        )

    plot_inference_results(repo_root, model_wavelengths_m, case_rows, output_dir)

    np.savez(
        output_dir / "stage1_model_inference_results.npz",
        wavelengths_m=model_wavelengths_m.astype(np.float64),
        delta_pm=np.array([r["delta_pm"] for r in case_rows], dtype=np.float64),
        true_delta_pm=np.array([r["true_delta_pm"] for r in case_rows], dtype=np.float64),
        pred_baseline_pm=np.array([r["pred_baseline_pm"] for r in case_rows], dtype=np.float64),
        pred_tail_pm=np.array([r["pred_tail_pm"] for r in case_rows], dtype=np.float64),
        err_baseline_pm=np.array([r["err_baseline_pm"] for r in case_rows], dtype=np.float64),
        err_tail_pm=np.array([r["err_tail_pm"] for r in case_rows], dtype=np.float64),
        input_case0=np.asarray(case_rows[0]["local_model_input"], dtype=np.float32),
        input_case1=np.asarray(case_rows[1]["local_model_input"], dtype=np.float32),
    )

    print("")
    print(f"Saved plot: {output_dir / 'stage1_model_inference_comparison.png'}")
    print(f"Saved data: {output_dir / 'stage1_model_inference_results.npz'}")
    print("")
    print("Note:")
    print("1. The trained models were learned on phase4a local spectra, not on raw OFDR interferograms.")
    print("2. Here we feed a model-compatible local overlapped spectrum derived from the stage-1 ideal scene.")
    print("3. This is an out-of-distribution smoke test for the ideal chain, not a formal benchmark result.")


if __name__ == "__main__":
    main()
