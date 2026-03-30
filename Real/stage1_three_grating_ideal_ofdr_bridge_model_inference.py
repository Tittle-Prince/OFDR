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

from src.ofdr.models.phase3_cnn import build_model


@dataclass
class BridgeInferenceConfig:
    bridge_npz: str = "Real/stage1_three_grating_ideal_ofdr_bridge_outputs/bridge_model_ready_examples.npz"
    baseline_ckpt: str = "results/phase4a_shift004_linewidth_l3/method_enhance_tailaware_baseline_seed45/model_cnn.pt"
    tail_ckpt: str = "results/phase4a_shift004_linewidth_l3/method_enhance_tailaware_hard_seed45/model_cnn.pt"
    output_dir: str = "Real/stage1_three_grating_bridge_model_inference_outputs"


def load_bridge_examples(npz_path: Path) -> dict[str, np.ndarray]:
    payload = np.load(npz_path)
    expected = {"X_bridge", "Y_delta_target_pm", "Y_center_true_pm", "Y_center_est_pm"}
    missing = expected.difference(payload.files)
    if missing:
        raise KeyError(f"Missing keys in bridge file: {sorted(missing)}")

    x_bridge = payload["X_bridge"].astype(np.float32)
    if x_bridge.ndim != 3 or x_bridge.shape[1] != 1 or x_bridge.shape[2] != 512:
        raise ValueError(f"Expected X_bridge shape [N, 1, 512], got {x_bridge.shape}")

    return {
        "X_bridge": x_bridge,
        "Y_delta_target_pm": payload["Y_delta_target_pm"].astype(np.float64),
        "Y_center_true_pm": payload["Y_center_true_pm"].astype(np.float64),
        "Y_center_est_pm": payload["Y_center_est_pm"].astype(np.float64),
    }


def load_trained_model(checkpoint_path: Path, input_dim: int, device: torch.device) -> torch.nn.Module:
    model = build_model("cnn_baseline", input_dim=input_dim).to(device)
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


def run_model_batch(model: torch.nn.Module, x_bridge: np.ndarray, device: torch.device) -> np.ndarray:
    x_tensor = torch.tensor(x_bridge, dtype=torch.float32, device=device)
    with torch.no_grad():
        pred = model(x_tensor).detach().cpu().numpy().reshape(-1)
    return pred.astype(np.float64)


def plot_bridge_predictions(
    x_bridge: np.ndarray,
    true_delta_pm: np.ndarray,
    pred_baseline_pm: np.ndarray,
    pred_tail_pm: np.ndarray,
    output_path: Path,
) -> None:
    num_cases = x_bridge.shape[0]
    fig, axes = plt.subplots(3, 2, figsize=(12.0, 9.0), constrained_layout=True)
    flat_axes = axes.ravel()

    for idx in range(num_cases):
        ax = flat_axes[idx]
        spectrum = x_bridge[idx, 0]
        ax.plot(np.arange(spectrum.size), spectrum, color="#1f77b4", linewidth=1.6)
        ax.set_title(
            f"Case {idx + 1} | true={true_delta_pm[idx]:+.1f} pm | "
            f"base={pred_baseline_pm[idx]:+.2f} pm | tail={pred_tail_pm[idx]:+.2f} pm",
            fontsize=10,
        )
        ax.set_xlabel("Local wavelength index")
        ax.set_ylabel("Normalized intensity")
        ax.grid(True, alpha=0.25)
        ax.text(
            0.03,
            0.97,
            f"err(base)={pred_baseline_pm[idx] - true_delta_pm[idx]:+.2f} pm\n"
            f"err(tail)={pred_tail_pm[idx] - true_delta_pm[idx]:+.2f} pm",
            transform=ax.transAxes,
            va="top",
            fontsize=9,
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.9, "edgecolor": "#bbbbbb"},
        )

    summary_ax = flat_axes[-1]
    summary_ax.axhline(0.0, color="#333333", linewidth=1.0, linestyle="--")
    x_positions = np.arange(num_cases)
    summary_ax.plot(x_positions, true_delta_pm, color="#333333", linewidth=1.8, marker="o", label="True")
    summary_ax.plot(x_positions, pred_baseline_pm, color="#7f7f7f", linewidth=1.8, marker="s", label="Baseline CNN")
    summary_ax.plot(x_positions, pred_tail_pm, color="#1f77b4", linewidth=1.8, marker="^", label="Tail+Hard CNN")
    summary_ax.set_title("Prediction summary across bridge examples", fontsize=10)
    summary_ax.set_xlabel("Bridge example index")
    summary_ax.set_ylabel("delta_lambda_target (pm)")
    summary_ax.set_xticks(x_positions)
    summary_ax.grid(True, alpha=0.25)
    summary_ax.legend(frameon=False, fontsize=9)

    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def main() -> None:
    cfg = BridgeInferenceConfig()
    repo_root = Path(__file__).resolve().parents[1]
    output_dir = repo_root / cfg.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    payload = load_bridge_examples(repo_root / cfg.bridge_npz)
    x_bridge = payload["X_bridge"]
    true_delta_pm = payload["Y_delta_target_pm"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    baseline_model = load_trained_model(repo_root / cfg.baseline_ckpt, input_dim=x_bridge.shape[-1], device=device)
    tail_model = load_trained_model(repo_root / cfg.tail_ckpt, input_dim=x_bridge.shape[-1], device=device)

    pred_baseline_pm = run_model_batch(baseline_model, x_bridge, device=device) * 1e3
    pred_tail_pm = run_model_batch(tail_model, x_bridge, device=device) * 1e3

    print("Feeding stage-1.5 bridge spectra into trained models...")
    print("")
    for idx in range(x_bridge.shape[0]):
        print(
            f"Case {idx + 1}: true={true_delta_pm[idx]:+.1f} pm | "
            f"baseline={pred_baseline_pm[idx]:+.3f} pm (err {pred_baseline_pm[idx] - true_delta_pm[idx]:+.3f} pm) | "
            f"tail+hard={pred_tail_pm[idx]:+.3f} pm (err {pred_tail_pm[idx] - true_delta_pm[idx]:+.3f} pm)"
        )

    plot_bridge_predictions(
        x_bridge=x_bridge,
        true_delta_pm=true_delta_pm,
        pred_baseline_pm=pred_baseline_pm,
        pred_tail_pm=pred_tail_pm,
        output_path=output_dir / "stage1_bridge_model_inference_comparison.png",
    )

    np.savez(
        output_dir / "stage1_bridge_model_inference_results.npz",
        X_bridge=x_bridge,
        true_delta_pm=true_delta_pm,
        pred_baseline_pm=pred_baseline_pm,
        pred_tail_pm=pred_tail_pm,
        err_baseline_pm=pred_baseline_pm - true_delta_pm,
        err_tail_pm=pred_tail_pm - true_delta_pm,
        y_center_true_pm=payload["Y_center_true_pm"],
        y_center_est_pm=payload["Y_center_est_pm"],
    )

    print("")
    print(f"Saved plot: {output_dir / 'stage1_bridge_model_inference_comparison.png'}")
    print(f"Saved data: {output_dir / 'stage1_bridge_model_inference_results.npz'}")
    print("")
    print("Interpretation:")
    print("1. These inputs are bridge spectra reconstructed from stage-1 kernels and remixed toward phase4a X_local semantics.")
    print("2. If predictions still collapse or stay insensitive to delta, the bridge is not yet sufficient for deployment on the trained models.")
    print("3. If predictions begin to separate between 0 pm and +20 pm cases, the bridge is partially reducing the domain gap.")


if __name__ == "__main__":
    main()
