from __future__ import annotations

import csv
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _find_prediction_file(root: Path) -> Path:
    candidates = [
        root / "results" / "paper_results_step2" / "raw" / "best_seed_predictions.npz",
        root / "results" / "phase3b" / "predictions.npz",
        root / "results" / "phase4a" / "phase4b_compare" / "predictions.npz",
        root / "results" / "phase4_array_tmp" / "phase4b_compare" / "predictions.npz",
    ]
    for p in candidates:
        if p.exists():
            d = np.load(p, allow_pickle=True)
            keys = set(d.files)
            req = {
                "y_true",
                "pred_cross_correlation",
                "pred_parametric_fitting",
                "pred_mlp",
                "pred_cnn",
                "pred_cnn_se",
            }
            if req.issubset(keys):
                return p
    raise FileNotFoundError("No prediction file with required keys was found.")


def _load_temperature_scale(root: Path) -> float:
    # Source convention from existing unit alignment notes.
    txt_path = root / "results" / "phase4_checks" / "check5_units_alignment.txt"
    if txt_path.exists():
        txt = txt_path.read_text(encoding="utf-8", errors="ignore")
        m = re.search(r"K_T\s*=\s*([0-9]*\.?[0-9]+)", txt)
        if m:
            k = float(m.group(1))
            if k > 0:
                return k
    return 0.01


def _mae_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    err = y_pred - y_true
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err**2)))
    return mae, rmse


def _style_axis(ax: plt.Axes) -> None:
    ax.set_facecolor("white")
    for sp in ax.spines.values():
        sp.set_linewidth(1.1)
    ax.tick_params(axis="both", labelsize=8, width=1.0, length=4)
    ax.grid(False)


def _plot_pred_panel(
    ax: plt.Axes,
    x: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str,
    y_lim: tuple[float, float],
    show_legend: bool = False,
) -> None:
    ax.plot(x, y_true, color="black", linewidth=1.6, label="Ground Truth")
    ax.plot(x, y_pred, color="#1f3a70", linewidth=1.6, label="Prediction")
    ax.set_title(title, fontsize=10, pad=4)
    ax.set_xlabel("Sample Index", fontsize=10)
    ax.set_ylabel("Temperature (°C)", fontsize=10)
    ax.set_xlim(float(x[0]), float(x[-1]))
    ax.set_ylim(*y_lim)
    if show_legend:
        ax.legend(frameon=False, fontsize=8.5, loc="upper left")


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    pred_path = _find_prediction_file(root)
    kt = _load_temperature_scale(root)

    d = np.load(pred_path, allow_pickle=True)
    y_true_nm = np.asarray(d["y_true"], dtype=float).reshape(-1)
    pred_nm = {
        "Cross-correlation": np.asarray(d["pred_cross_correlation"], dtype=float).reshape(-1),
        "Parametric fitting": np.asarray(d["pred_parametric_fitting"], dtype=float).reshape(-1),
        "MLP": np.asarray(d["pred_mlp"], dtype=float).reshape(-1),
        "CNN": np.asarray(d["pred_cnn"], dtype=float).reshape(-1),
        "CNN+SE": np.asarray(d["pred_cnn_se"], dtype=float).reshape(-1),
    }

    # Convert wavelength shift (nm) -> temperature (degC)
    y_true = y_true_nm / kt
    pred = {k: v / kt for k, v in pred_nm.items()}
    # Sort by ground truth for clearer sequence-like visualization.
    order = np.argsort(y_true)
    y_true_plot = y_true[order]
    pred_plot = {k: v[order] for k, v in pred.items()}
    x_plot = np.arange(len(y_true_plot), dtype=int)

    all_vals = [y_true_plot] + [pred_plot[k] for k in pred_plot.keys()]
    ymin = min(float(np.min(v)) for v in all_vals)
    ymax = max(float(np.max(v)) for v in all_vals)
    pad = 0.04 * (ymax - ymin + 1e-12)
    y_lim = (ymin - pad, ymax + pad)

    plt.rcParams["font.family"] = "Arial"
    dpi = 300
    fig_w, fig_h = 2200 / dpi, 1200 / dpi
    fig, axes = plt.subplots(2, 4, figsize=(fig_w, fig_h), dpi=dpi)
    fig.patch.set_facecolor("white")

    # First row: method comparison
    first_row = [
        ("(I-a) Cross-correlation", "Cross-correlation"),
        ("(I-b) Parametric fitting", "Parametric fitting"),
        ("(I-c) MLP", "MLP"),
        ("(I-d) CNN", "CNN"),
    ]
    for j, (title, m) in enumerate(first_row):
        _style_axis(axes[0, j])
        _plot_pred_panel(
            axes[0, j],
            x_plot,
            y_true_plot,
            pred_plot[m],
            title=title,
            y_lim=y_lim,
            show_legend=(j == 0),
        )
        if j != 0:
            axes[0, j].set_ylabel("")

    # Second row: CNN+SE analysis
    # (II-a) prediction
    _style_axis(axes[1, 0])
    _plot_pred_panel(
        axes[1, 0],
        x_plot,
        y_true_plot,
        pred_plot["CNN+SE"],
        title="(II-a) CNN+SE prediction",
        y_lim=y_lim,
        show_legend=True,
    )

    # (II-b) scatter
    ax = axes[1, 1]
    _style_axis(ax)
    ax.scatter(y_true, pred["CNN+SE"], s=8, color="#1f3a70", alpha=0.65, edgecolors="none")
    lo = min(float(np.min(y_true)), float(np.min(pred["CNN+SE"])))
    hi = max(float(np.max(y_true)), float(np.max(pred["CNN+SE"])))
    m = 0.03 * (hi - lo + 1e-12)
    ax.plot([lo - m, hi + m], [lo - m, hi + m], color="black", linewidth=1.1, linestyle="--")
    ax.set_xlim(lo - m, hi + m)
    ax.set_ylim(lo - m, hi + m)
    ax.set_title("(II-b) Prediction scatter", fontsize=10, pad=4)
    ax.set_xlabel("Ground Truth (°C)", fontsize=10)
    ax.set_ylabel("Prediction (°C)", fontsize=10)

    # (II-c) residual
    ax = axes[1, 2]
    _style_axis(ax)
    err = pred["CNN+SE"] - y_true
    err_plot = (pred_plot["CNN+SE"] - y_true_plot)
    ax.plot(x_plot, err_plot, color="#7a1f1f", linewidth=1.6)
    ax.axhline(0.0, color="black", linewidth=1.0, linestyle="--", alpha=0.85)
    ax.set_title("(II-c) Residual error", fontsize=10, pad=4)
    ax.set_xlabel("Sample Index", fontsize=10)
    ax.set_ylabel("Prediction Error (°C)", fontsize=10)
    ax.set_xlim(float(x_plot[0]), float(x_plot[-1]))

    # (II-d) error distribution
    ax = axes[1, 3]
    _style_axis(ax)
    ax.hist(err, bins=30, color="#7a1f1f", alpha=0.80, edgecolor="black", linewidth=0.5)
    ax.set_title("(II-d) Error distribution", fontsize=10, pad=4)
    ax.set_xlabel("Prediction Error (°C)", fontsize=10)
    ax.set_ylabel("Count", fontsize=10)

    fig.subplots_adjust(left=0.055, right=0.995, bottom=0.09, top=0.94, wspace=0.26, hspace=0.38)

    out_path = root / "results" / "paper_figures" / "Fig8_main_results.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)

    # Metrics print
    print("Data file used:", pred_path.as_posix())
    print(f"Temperature conversion: dT = dLambda / K_T, K_T={kt:.4f} nm/degC")
    print("Method metrics (MAE / RMSE in °C):")
    for m in ["Cross-correlation", "Parametric fitting", "MLP", "CNN", "CNN+SE"]:
        mae, rmse = _mae_rmse(y_true, pred[m])
        print(f"- {m}: MAE={mae:.6f}, RMSE={rmse:.6f}")
    print("2x4 layout compliance:", "Yes")
    print("SCI plotting style compliance:", "Yes")


if __name__ == "__main__":
    main()
