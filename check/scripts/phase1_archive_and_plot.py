import argparse
import csv
import json
import shutil
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml


def project_root_from_script() -> Path:
    return Path(__file__).resolve().parents[2]


def setup_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "font.size": 11,
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "legend.fontsize": 10,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "axes.grid": True,
            "grid.alpha": 0.25,
        }
    )


def load_phase1_module(project_root: Path):
    import sys

    src_dir = project_root / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
    import phase1_pipeline as p1  # pylint: disable=import-error

    return p1


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    err = y_pred - y_true
    rmse = float(np.sqrt(np.mean(err**2)))
    mae = float(np.mean(np.abs(err)))
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2) + 1e-12)
    r2 = float(1.0 - ss_res / ss_tot)
    return {"rmse": rmse, "mae": mae, "r2": r2}


def load_inputs(project_root: Path) -> dict:
    config_path = project_root / "config" / "phase1.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config file: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    data_path = project_root / cfg["phase1"]["data_path"]
    metrics_json = project_root / cfg["phase1"]["results_dir"] / "phase1_metrics.json"
    model_mlp = project_root / cfg["phase1"]["results_dir"] / "models" / "phase1_mlp.pt"
    model_cnn = project_root / cfg["phase1"]["results_dir"] / "models" / "phase1_cnn1d.pt"

    for p in [data_path, metrics_json, model_mlp, model_cnn]:
        if not p.exists():
            raise FileNotFoundError(f"Required file not found: {p}")

    data = np.load(data_path)
    return {
        "cfg": cfg,
        "config_path": config_path,
        "data_path": data_path,
        "metrics_json": metrics_json,
        "model_mlp": model_mlp,
        "model_cnn": model_cnn,
        "data": data,
    }


def predict_all(inputs: dict, p1_module) -> dict:
    cfg = inputs["cfg"]
    data = inputs["data"]
    x = data["X"]
    y = data["Y_dlambda"]
    idx_test = data["idx_test"]
    wavelengths = data["wavelengths"]

    x_test = x[idx_test]
    y_test = y[idx_test]

    dcfg = cfg["dataset"]
    lambda_b0 = float(dcfg["lambda_b0_nm"])
    sigma = float(dcfg["linewidth_nm"])
    amplitude = float(dcfg["peak_amplitude"])
    baseline = float(dcfg["baseline"])
    normalize_mode = str(dcfg["normalize"])
    step_nm = float(wavelengths[1] - wavelengths[0])

    ref = p1_module.gaussian_spectrum(wavelengths, lambda_b0, sigma, amplitude, baseline)
    if normalize_mode == "minmax_per_sample":
        ref = p1_module.normalize_minmax(ref)

    pred_cc = np.zeros_like(y_test)
    for i in range(len(y_test)):
        pred_cc[i] = p1_module.estimate_shift_by_cross_correlation(ref, x_test[i], step_nm)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x_test_tensor = torch.tensor(x_test, dtype=torch.float32, device=device)

    mlp = p1_module.MLPRegressor(input_dim=x.shape[1]).to(device)
    mlp.load_state_dict(torch.load(inputs["model_mlp"], map_location=device))
    mlp.eval()
    with torch.no_grad():
        pred_mlp = mlp(x_test_tensor).cpu().numpy()

    cnn = p1_module.CNN1DRegressor(input_dim=x.shape[1]).to(device)
    cnn.load_state_dict(torch.load(inputs["model_cnn"], map_location=device))
    cnn.eval()
    with torch.no_grad():
        pred_cnn = cnn(x_test_tensor).cpu().numpy()

    return {
        "y_true": y_test,
        "pred_cc": pred_cc,
        "pred_mlp": pred_mlp,
        "pred_cnn": pred_cnn,
        "metrics_cc": compute_metrics(y_test, pred_cc),
        "metrics_mlp": compute_metrics(y_test, pred_mlp),
        "metrics_cnn": compute_metrics(y_test, pred_cnn),
    }


def make_output_dirs(project_root: Path, tag: str | None) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_name = f"phase1_report_{tag}_{ts}" if tag else f"phase1_report_{ts}"
    out_dir = project_root / "check" / "results" / report_name
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)
    (out_dir / "archive").mkdir(parents=True, exist_ok=True)
    return out_dir


def save_metrics_files(out_dir: Path, predictions: dict) -> None:
    metrics = {
        "cross_correlation": predictions["metrics_cc"],
        "mlp": predictions["metrics_mlp"],
        "cnn1d": predictions["metrics_cnn"],
    }
    with open(out_dir / "metrics_recomputed.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    with open(out_dir / "metrics_recomputed.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["method", "rmse_nm", "mae_nm", "r2"])
        writer.writerow(
            [
                "cross_correlation",
                metrics["cross_correlation"]["rmse"],
                metrics["cross_correlation"]["mae"],
                metrics["cross_correlation"]["r2"],
            ]
        )
        writer.writerow(["mlp", metrics["mlp"]["rmse"], metrics["mlp"]["mae"], metrics["mlp"]["r2"]])
        writer.writerow(["cnn1d", metrics["cnn1d"]["rmse"], metrics["cnn1d"]["mae"], metrics["cnn1d"]["r2"]])

    np.savez(
        out_dir / "predictions_test.npz",
        y_true=predictions["y_true"],
        pred_cross_correlation=predictions["pred_cc"],
        pred_mlp=predictions["pred_mlp"],
        pred_cnn1d=predictions["pred_cnn"],
    )


def plot_metric_comparison(out_dir: Path, predictions: dict) -> None:
    names = ["Cross-correlation", "MLP", "1D CNN"]
    metrics = [predictions["metrics_cc"], predictions["metrics_mlp"], predictions["metrics_cnn"]]
    rmse_vals = [m["rmse"] for m in metrics]
    mae_vals = [m["mae"] for m in metrics]
    r2_vals = [m["r2"] for m in metrics]
    colors = ["#4e79a7", "#59a14f", "#e15759"]
    x = np.arange(len(names))

    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.8), constrained_layout=True)
    axes[0].bar(x, rmse_vals, color=colors, width=0.65)
    axes[0].set_title("RMSE (nm)")
    axes[0].set_xticks(x, names, rotation=12)

    axes[1].bar(x, mae_vals, color=colors, width=0.65)
    axes[1].set_title("MAE (nm)")
    axes[1].set_xticks(x, names, rotation=12)

    axes[2].bar(x, r2_vals, color=colors, width=0.65)
    axes[2].set_title("R2")
    axes[2].set_ylim(max(0.99, min(r2_vals) - 0.005), 1.00001)
    axes[2].set_xticks(x, names, rotation=12)

    for ax, vals in zip(axes, [rmse_vals, mae_vals, r2_vals]):
        for i, v in enumerate(vals):
            ax.text(i, v, f"{v:.6f}", ha="center", va="bottom", fontsize=9)

    fig.suptitle("Phase1 Method Comparison (Dataset_A, Test Set)")
    fig.savefig(out_dir / "figures" / "fig1_metric_comparison.png")
    fig.savefig(out_dir / "figures" / "fig1_metric_comparison.pdf")
    plt.close(fig)


def parity_panel(ax, y_true: np.ndarray, y_pred: np.ndarray, name: str, color: str) -> None:
    m = compute_metrics(y_true, y_pred)
    lo = float(min(y_true.min(), y_pred.min()))
    hi = float(max(y_true.max(), y_pred.max()))
    pad = 0.02 * (hi - lo + 1e-9)
    lo -= pad
    hi += pad
    ax.scatter(y_true, y_pred, s=12, alpha=0.35, c=color, edgecolors="none")
    ax.plot([lo, hi], [lo, hi], "k--", linewidth=1.2)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(name)
    ax.set_xlabel("True Δλ (nm)")
    ax.set_ylabel("Predicted Δλ (nm)")
    text = f"RMSE={m['rmse']:.6f} nm\nMAE={m['mae']:.6f} nm\nR2={m['r2']:.6f}"
    ax.text(
        0.04,
        0.96,
        text,
        transform=ax.transAxes,
        va="top",
        ha="left",
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.9},
    )


def plot_parity(out_dir: Path, predictions: dict) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), constrained_layout=True)
    parity_panel(axes[0], predictions["y_true"], predictions["pred_cc"], "Cross-correlation", "#4e79a7")
    parity_panel(axes[1], predictions["y_true"], predictions["pred_mlp"], "MLP", "#59a14f")
    parity_panel(axes[2], predictions["y_true"], predictions["pred_cnn"], "1D CNN", "#e15759")
    fig.suptitle("Parity Plots on Phase1 Test Set")
    fig.savefig(out_dir / "figures" / "fig2_parity_plots.png")
    fig.savefig(out_dir / "figures" / "fig2_parity_plots.pdf")
    plt.close(fig)


def plot_residuals(out_dir: Path, predictions: dict) -> None:
    residuals = [
        predictions["pred_cc"] - predictions["y_true"],
        predictions["pred_mlp"] - predictions["y_true"],
        predictions["pred_cnn"] - predictions["y_true"],
    ]
    labels = ["Cross-correlation", "MLP", "1D CNN"]
    colors = ["#4e79a7", "#59a14f", "#e15759"]

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.0), constrained_layout=True)

    bp = axes[0].boxplot(residuals, tick_labels=labels, patch_artist=True, showfliers=False)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.65)
    axes[0].axhline(0.0, color="k", linestyle="--", linewidth=1.0)
    axes[0].set_ylabel("Residual (Pred - True) in nm")
    axes[0].set_title("Residual Boxplot")

    bins = np.linspace(
        min(float(r.min()) for r in residuals),
        max(float(r.max()) for r in residuals),
        80,
    )
    for r, label, color in zip(residuals, labels, colors):
        axes[1].hist(r, bins=bins, density=True, alpha=0.35, color=color, label=label)
    axes[1].axvline(0.0, color="k", linestyle="--", linewidth=1.0)
    axes[1].set_xlabel("Residual (nm)")
    axes[1].set_ylabel("Density")
    axes[1].set_title("Residual Distribution")
    axes[1].legend()

    fig.suptitle("Phase1 Residual Analysis")
    fig.savefig(out_dir / "figures" / "fig3_residual_analysis.png")
    fig.savefig(out_dir / "figures" / "fig3_residual_analysis.pdf")
    plt.close(fig)


def archive_inputs(out_dir: Path, inputs: dict) -> None:
    archive_dir = out_dir / "archive"
    shutil.copy2(inputs["config_path"], archive_dir / "phase1.yaml")
    shutil.copy2(inputs["data_path"], archive_dir / "dataset_a_phase1.npz")
    shutil.copy2(inputs["metrics_json"], archive_dir / "phase1_metrics_original.json")
    shutil.copy2(inputs["model_mlp"], archive_dir / "phase1_mlp.pt")
    shutil.copy2(inputs["model_cnn"], archive_dir / "phase1_cnn1d.pt")


def write_report(out_dir: Path, predictions: dict) -> None:
    metrics = {
        "Cross-correlation": predictions["metrics_cc"],
        "MLP": predictions["metrics_mlp"],
        "1D CNN": predictions["metrics_cnn"],
    }
    lines = []
    lines.append("# Phase1 Experiment Check Report")
    lines.append("")
    lines.append(f"- Generated at: {datetime.now().isoformat(timespec='seconds')}")
    lines.append(f"- Test samples: {len(predictions['y_true'])}")
    lines.append("")
    lines.append("## Metrics")
    lines.append("")
    lines.append("| Method | RMSE (nm) | MAE (nm) | R2 |")
    lines.append("| --- | ---: | ---: | ---: |")
    for name, m in metrics.items():
        lines.append(f"| {name} | {m['rmse']:.8f} | {m['mae']:.8f} | {m['r2']:.8f} |")
    lines.append("")
    lines.append("## Figures")
    lines.append("")
    lines.append("- fig1_metric_comparison.(png/pdf)")
    lines.append("- fig2_parity_plots.(png/pdf)")
    lines.append("- fig3_residual_analysis.(png/pdf)")
    lines.append("")
    lines.append("## Archived Inputs")
    lines.append("")
    lines.append("- phase1.yaml")
    lines.append("- dataset_a_phase1.npz")
    lines.append("- phase1_metrics_original.json")
    lines.append("- phase1_mlp.pt")
    lines.append("- phase1_cnn1d.pt")

    (out_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Archive Phase1 results and generate publication-style plots")
    parser.add_argument("--tag", type=str, default="", help="Optional tag added to output folder name")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = project_root_from_script()
    setup_plot_style()
    p1_module = load_phase1_module(project_root)
    inputs = load_inputs(project_root)
    predictions = predict_all(inputs, p1_module)

    out_dir = make_output_dirs(project_root, args.tag.strip() or None)
    save_metrics_files(out_dir, predictions)
    plot_metric_comparison(out_dir, predictions)
    plot_parity(out_dir, predictions)
    plot_residuals(out_dir, predictions)
    archive_inputs(out_dir, inputs)
    write_report(out_dir, predictions)

    print(f"Report generated in: {out_dir}")
    print(f"Figures: {out_dir / 'figures'}")
    print(f"Archived files: {out_dir / 'archive'}")


if __name__ == "__main__":
    main()
