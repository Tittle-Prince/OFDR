from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    in_file = project_root / "results" / "phase3b" / "predictions.npz"
    out_file = project_root / "results" / "phase3b" / "error_zoom_plot.png"

    if not in_file.exists():
        raise FileNotFoundError(
            f"Missing predictions file: {in_file}\n"
            "Run `python src\\phase3\\run_phase3b_unified_compare.py --config config\\phase3.yaml` first."
        )

    d = np.load(in_file)
    y_true = d["y_true"]
    preds = {
        "Cross-correlation": d["pred_cross_correlation"],
        "Parametric fitting": d["pred_parametric_fitting"],
        "MLP": d["pred_mlp"],
        "CNN": d["pred_cnn"],
        "CNN+SE": d["pred_cnn_se"],
    }

    colors = {
        "Cross-correlation": "#4e79a7",
        "Parametric fitting": "#f28e2b",
        "MLP": "#59a14f",
        "CNN": "#e15759",
        "CNN+SE": "#76b7b2",
    }

    residuals = {k: (v - y_true) for k, v in preds.items()}
    abs_max = max(float(np.max(np.abs(r))) for r in residuals.values())
    y_lim = min(0.012, max(0.004, abs_max * 1.15))

    fig, axes = plt.subplots(2, 3, figsize=(13.2, 7.2), constrained_layout=True)
    flat_axes = axes.flatten()

    for i, (name, r) in enumerate(residuals.items()):
        ax = flat_axes[i]
        ax.scatter(y_true, r, s=10, alpha=0.35, c=colors[name], edgecolors="none")
        ax.axhline(0.0, color="k", linestyle="--", linewidth=1.0)
        ax.set_title(name)
        ax.set_xlabel("True Δλ (nm)")
        ax.set_ylabel("Error (Pred-True) (nm)")
        ax.set_ylim(-y_lim, y_lim)
        rmse = float(np.sqrt(np.mean(r**2)))
        mae = float(np.mean(np.abs(r)))
        ax.text(
            0.03,
            0.97,
            f"RMSE={rmse:.6f}\nMAE={mae:.6f}",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=8,
            bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "alpha": 0.9},
        )
        ax.grid(True, alpha=0.25)

    flat_axes[-1].axis("off")
    fig.suptitle("Dataset_B Error Zoom View (Residual vs True)")
    fig.savefig(out_file, dpi=300)
    plt.close(fig)
    print(f"Saved: {out_file}")


if __name__ == "__main__":
    main()

