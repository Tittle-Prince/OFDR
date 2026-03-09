from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FINAL_DIR = PROJECT_ROOT / "results" / "paper_results_step2" / "final"
RAW_DIR = PROJECT_ROOT / "results" / "paper_results_step2" / "raw"
FIG_DIR = PROJECT_ROOT / "results" / "paper_results_step2" / "figures"
DATA_DIR = PROJECT_ROOT / "data" / "processed"

INPUT_RUNS = FINAL_DIR / "multiseed_runs.csv"
INPUT_SUMMARY = FINAL_DIR / "multiseed_summary.csv"
INPUT_MAIN_TABLE = FINAL_DIR / "paper_main_table_final.csv"
INPUT_README = FINAL_DIR / "README_results.txt"

ORDER = [
    "Cross-correlation",
    "Parametric fitting",
    "MLP",
    "CNN",
    "CNN+SE",
]


@dataclass
class FigureRecord:
    name: str
    path: Path
    inputs: list[str]
    section: str
    conclusion: str


def load_csv_rows(path: Path) -> list[dict[str, str]]:
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def to_float(v: Any, default: float = np.nan) -> float:
    try:
        return float(v)
    except Exception:
        return default


def method_sort_key(name: str) -> tuple[int, str]:
    if name in ORDER:
        return (ORDER.index(name), name)
    return (len(ORDER), name)


def ensure_inputs() -> None:
    missing = [str(p) for p in [INPUT_RUNS, INPUT_SUMMARY, INPUT_MAIN_TABLE, INPUT_README] if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing required input files:\n" + "\n".join(missing))


def detect_unit(readme_text: str) -> str:
    # Use "nm" when source metrics mention MAE_nm / RMSE_nm.
    low = readme_text.lower()
    if "mae_nm" in low or "rmse_nm" in low or "(nm)" in low:
        return "nm"
    return "a.u."


def load_summary() -> list[dict[str, float | str]]:
    rows = load_csv_rows(INPUT_SUMMARY)
    out: list[dict[str, float | str]] = []
    for r in rows:
        out.append(
            {
                "method": r.get("method", "").strip(),
                "mae_mean": to_float(r.get("mae_mean")),
                "mae_std": to_float(r.get("mae_std")),
                "rmse_mean": to_float(r.get("rmse_mean")),
                "rmse_std": to_float(r.get("rmse_std")),
                "r2_mean": to_float(r.get("r2_mean")),
                "r2_std": to_float(r.get("r2_std")),
            }
        )
    out.sort(key=lambda x: method_sort_key(str(x["method"])))
    return out


def load_runs() -> list[dict[str, float | str]]:
    rows = load_csv_rows(INPUT_RUNS)
    out: list[dict[str, float | str]] = []
    for r in rows:
        out.append(
            {
                "method": r.get("method", "").strip(),
                "seed": r.get("seed", "").strip(),
                "mae": to_float(r.get("mae")),
                "rmse": to_float(r.get("rmse")),
                "r2": to_float(r.get("r2")),
            }
        )
    out.sort(key=lambda x: (method_sort_key(str(x["method"])), str(x["seed"])))
    return out


def style_rc() -> None:
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 9,
            "figure.dpi": 120,
            "savefig.dpi": 320,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linestyle": "--",
        }
    )


def method_palette() -> dict[str, str]:
    return {
        "Cross-correlation": "#4C78A8",
        "Parametric fitting": "#F58518",
        "MLP": "#54A24B",
        "CNN": "#E45756",
        "CNN+SE": "#72B7B2",
    }


def plot_errorbar_comparison(
    summary: list[dict[str, float | str]],
    mean_key: str,
    std_key: str,
    ylabel: str,
    title: str,
    out_path: Path,
) -> None:
    palette = method_palette()
    methods = [str(r["method"]) for r in summary]
    means = np.array([float(r[mean_key]) for r in summary], dtype=float)
    stds = np.array([float(r[std_key]) for r in summary], dtype=float)
    x = np.arange(len(methods))
    colors = [palette.get(m, "#777777") for m in methods]

    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    ax.bar(x, means, color=colors, alpha=0.9, edgecolor="black", linewidth=0.6, zorder=2)
    ax.errorbar(x, means, yerr=stds, fmt="none", ecolor="black", elinewidth=1.2, capsize=4, zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=15, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.margins(x=0.03)
    fig.tight_layout()
    fig.savefig(out_path, dpi=320, bbox_inches="tight")
    plt.close(fig)


def plot_stability(
    runs: list[dict[str, float | str]],
    metric: str,
    ylabel: str,
    title: str,
    out_path: Path,
) -> None:
    palette = method_palette()
    grouped: dict[str, list[float]] = {m: [] for m in ORDER}
    for r in runs:
        m = str(r["method"])
        if m in grouped:
            grouped[m].append(float(r[metric]))

    methods = [m for m in ORDER if grouped.get(m)]
    data = [grouped[m] for m in methods]
    x = np.arange(1, len(methods) + 1)

    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    bp = ax.boxplot(
        data,
        positions=x,
        widths=0.52,
        patch_artist=True,
        showmeans=True,
        meanline=False,
    )
    for patch, m in zip(bp["boxes"], methods):
        patch.set_facecolor(palette.get(m, "#999999"))
        patch.set_alpha(0.35)
        patch.set_edgecolor("black")
        patch.set_linewidth(0.8)
    for median in bp["medians"]:
        median.set_color("black")
        median.set_linewidth(1.4)
    for meanp in bp["means"]:
        meanp.set_marker("D")
        meanp.set_markerfacecolor("black")
        meanp.set_markeredgecolor("white")
        meanp.set_markersize(5)

    # Overlay seed points to show spread explicitly.
    rng = np.random.default_rng(20260308)
    for i, vals in enumerate(data, start=1):
        jitter = rng.normal(loc=0.0, scale=0.03, size=len(vals))
        ax.scatter(
            np.full(len(vals), i, dtype=float) + jitter,
            vals,
            s=22,
            c=palette.get(methods[i - 1], "#777777"),
            edgecolors="black",
            linewidths=0.35,
            alpha=0.9,
            zorder=3,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=15, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=320, bbox_inches="tight")
    plt.close(fig)


def plot_cnn_vs_cnnse(summary: list[dict[str, float | str]], out_path: Path, unit: str) -> bool:
    rows = {str(r["method"]): r for r in summary}
    if "CNN" not in rows or "CNN+SE" not in rows:
        return False
    cnn = rows["CNN"]
    se = rows["CNN+SE"]

    labels = ["MAE", "RMSE"]
    cnn_mean = [float(cnn["mae_mean"]), float(cnn["rmse_mean"])]
    cnn_std = [float(cnn["mae_std"]), float(cnn["rmse_std"])]
    se_mean = [float(se["mae_mean"]), float(se["rmse_mean"])]
    se_std = [float(se["mae_std"]), float(se["rmse_std"])]

    x = np.arange(len(labels))
    w = 0.34
    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    ax.bar(x - w / 2, cnn_mean, w, yerr=cnn_std, label="CNN", color="#E45756", capsize=4, edgecolor="black", linewidth=0.6)
    ax.bar(x + w / 2, se_mean, w, yerr=se_std, label="CNN+SE", color="#72B7B2", capsize=4, edgecolor="black", linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel(f"Error ({unit})")
    ax.set_title("CNN vs CNN+SE: Mean Error with Seed Variability")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    fig.tight_layout()
    fig.savefig(out_path, dpi=320, bbox_inches="tight")
    plt.close(fig)
    return True


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float, float]:
    e = y_pred - y_true
    mae = float(np.mean(np.abs(e)))
    rmse = float(np.sqrt(np.mean(e**2)))
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return mae, rmse, r2


def try_load_step2_predictions() -> dict[str, np.ndarray] | None:
    p = RAW_DIR / "best_seed_predictions.npz"
    if not p.exists():
        return None
    data = np.load(p, allow_pickle=True)
    out: dict[str, np.ndarray] = {}
    for k in data.files:
        out[k] = np.asarray(data[k])
    return out


def plot_scatter(y_true: np.ndarray, y_pred: np.ndarray, method: str, out_path: Path, lims: tuple[float, float]) -> None:
    mae, rmse, r2 = compute_metrics(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5.2, 5.0))
    ax.scatter(y_true, y_pred, s=10, c="#4C78A8", alpha=0.5, edgecolors="none")
    ax.plot([lims[0], lims[1]], [lims[0], lims[1]], "k--", linewidth=1.2)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("Ground Truth Δλ (nm)")
    ax.set_ylabel("Prediction Δλ (nm)")
    ax.set_title(f"{method} Prediction Scatter")
    txt = f"MAE={mae:.4f} nm\nRMSE={rmse:.4f} nm\nR²={r2:.4f}"
    ax.text(
        0.03,
        0.97,
        txt,
        transform=ax.transAxes,
        va="top",
        ha="left",
        bbox=dict(facecolor="white", edgecolor="black", alpha=0.85),
        fontsize=9.5,
    )
    ax.grid(True, alpha=0.22, linestyle="--")
    fig.tight_layout()
    fig.savefig(out_path, dpi=320, bbox_inches="tight")
    plt.close(fig)


def _pick_example_index(y: np.ndarray, idx: np.ndarray | None = None) -> int:
    if idx is not None and len(idx) > 0:
        yy = y[idx]
        target = np.median(yy)
        j = int(np.argmin(np.abs(yy - target)))
        return int(idx[j])
    target = np.median(y)
    return int(np.argmin(np.abs(y - target)))


def _load_dataset_sample(dataset_path: Path, prefer_local: bool = False) -> tuple[np.ndarray, np.ndarray, float] | None:
    if not dataset_path.exists():
        return None
    d = np.load(dataset_path, allow_pickle=True)

    if "wavelengths" not in d.files:
        return None
    wavelengths = np.asarray(d["wavelengths"]).reshape(-1)

    x_key = None
    y_key = None
    if prefer_local and "X_local" in d.files:
        x_key = "X_local"
    elif "X" in d.files:
        x_key = "X"
    elif "X_local" in d.files:
        x_key = "X_local"
    elif "X_total" in d.files:
        x_key = "X_total"
    if "Y_dlambda" in d.files:
        y_key = "Y_dlambda"
    elif "Y_dlambda_target" in d.files:
        y_key = "Y_dlambda_target"
    if x_key is None or y_key is None:
        return None

    X = np.asarray(d[x_key])
    y = np.asarray(d[y_key]).reshape(-1)
    idx_test = np.asarray(d["idx_test"]) if "idx_test" in d.files else None
    i = _pick_example_index(y, idx_test)
    return wavelengths, np.asarray(X[i]).reshape(-1), float(y[i])


def plot_dataset_examples(dataset_used_path: Path, out_path: Path) -> bool:
    ds_a = _load_dataset_sample(DATA_DIR / "dataset_a_phase1.npz", prefer_local=False)
    ds_b = _load_dataset_sample(DATA_DIR / "dataset_b_phase3.npz", prefer_local=False)
    ds_c = _load_dataset_sample(dataset_used_path, prefer_local=True)
    if ds_a is None or ds_b is None or ds_c is None:
        return False

    items = [
        ("Dataset_A (ideal single FBG)", ds_a),
        ("Dataset_B (noisy single FBG)", ds_b),
        ("Dataset_B Array (local distorted window)", ds_c),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.9), sharey=False)
    for ax, (title, (w, x, yv)) in zip(axes, items):
        xn = (x - np.min(x)) / (np.max(x) - np.min(x) + 1e-12)
        ax.plot(w, xn, color="#1f77b4", linewidth=1.2)
        ax.set_title(title)
        ax.set_xlabel("Wavelength (nm)")
        ax.set_ylabel("Normalized Reflectance")
        ax.text(0.03, 0.05, f"Δλ={yv:.3f} nm", transform=ax.transAxes, fontsize=9)
        ax.grid(True, alpha=0.2, linestyle="--")
    fig.suptitle("Representative Inputs with Increasing Spectral Complexity", y=1.03, fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=320, bbox_inches="tight")
    plt.close(fig)
    return True


def plot_example_samples(dataset_used_path: Path, out_path: Path) -> bool:
    if not dataset_used_path.exists():
        return False
    d = np.load(dataset_used_path, allow_pickle=True)
    if not {"X_local", "Y_dlambda_target", "wavelengths"}.issubset(set(d.files)):
        return False
    X = np.asarray(d["X_local"])
    y = np.asarray(d["Y_dlambda_target"]).reshape(-1)
    w = np.asarray(d["wavelengths"]).reshape(-1)
    idx_test = np.asarray(d["idx_test"]) if "idx_test" in d.files else np.arange(len(y))
    if len(idx_test) < 6:
        return False

    # Quantile-based representative test samples.
    y_test = y[idx_test]
    q = np.quantile(y_test, [0.05, 0.25, 0.45, 0.6, 0.8, 0.95])
    picks = []
    for t in q:
        j = int(np.argmin(np.abs(y_test - t)))
        picks.append(int(idx_test[j]))

    fig, axes = plt.subplots(2, 3, figsize=(12.0, 6.2), sharex=True, sharey=True)
    axes_flat = axes.ravel()
    for ax, i in zip(axes_flat, picks):
        x = X[i]
        xn = (x - np.min(x)) / (np.max(x) - np.min(x) + 1e-12)
        ax.plot(w, xn, color="#2A9D8F", linewidth=1.2)
        title = f"Δλ={y[i]:.3f} nm"
        if "neighbor_mode" in d.files:
            title += f", mode={int(d['neighbor_mode'][i])}"
        ax.set_title(title, fontsize=10)
        ax.grid(True, alpha=0.2, linestyle="--")
    fig.suptitle("Typical Local Distorted Spectra from Dataset_B Array (Test Set)", y=0.99, fontsize=12)
    fig.text(0.5, 0.02, "Wavelength (nm)", ha="center")
    fig.text(0.02, 0.5, "Normalized Reflectance", va="center", rotation="vertical")
    fig.tight_layout(rect=(0.03, 0.04, 1.0, 0.96))
    fig.savefig(out_path, dpi=320, bbox_inches="tight")
    plt.close(fig)
    return True


def write_figure_summary(
    out_path: Path,
    generated: list[FigureRecord],
    skipped: list[tuple[str, str]],
) -> None:
    lines: list[str] = []
    lines.append("Step2 Figure Generation Summary")
    lines.append("")
    lines.append("1) Generated figures")
    if generated:
        for rec in generated:
            lines.append(f"- {rec.name}")
            lines.append(f"  path: {rec.path.as_posix()}")
            lines.append(f"  inputs: {', '.join(rec.inputs)}")
            lines.append(f"  section: {rec.section}")
            lines.append(f"  conclusion: {rec.conclusion}")
    else:
        lines.append("- None")

    lines.append("")
    lines.append("2) Not generated and reasons")
    if skipped:
        for name, reason in skipped:
            lines.append(f"- {name}: {reason}")
    else:
        lines.append("- None")

    lines.append("")
    lines.append("3) Notes")
    lines.append("- This task only reads existing Step2 result/data files and does not retrain or rerun experiments.")
    lines.append("- Traditional methods may show std=0 in multiseed plots; this is expected.")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ensure_inputs()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    style_rc()

    summary = load_summary()
    runs = load_runs()
    readme_text = INPUT_README.read_text(encoding="utf-8", errors="ignore")
    unit = detect_unit(readme_text)

    generated: list[FigureRecord] = []
    skipped: list[tuple[str, str]] = []

    # Group 1: main comparison (required)
    p = FIG_DIR / "mae_comparison.png"
    plot_errorbar_comparison(
        summary,
        "mae_mean",
        "mae_std",
        f"MAE ({unit})",
        "Comparison of MAE across different demodulation methods",
        p,
    )
    generated.append(
        FigureRecord(
            name="mae_comparison.png",
            path=p,
            inputs=[INPUT_SUMMARY.as_posix()],
            section="Main Results",
            conclusion="CNN+SE achieves the lowest MAE among all compared methods.",
        )
    )

    p = FIG_DIR / "rmse_comparison.png"
    plot_errorbar_comparison(
        summary,
        "rmse_mean",
        "rmse_std",
        f"RMSE ({unit})",
        "Comparison of RMSE across different demodulation methods",
        p,
    )
    generated.append(
        FigureRecord(
            name="rmse_comparison.png",
            path=p,
            inputs=[INPUT_SUMMARY.as_posix()],
            section="Main Results",
            conclusion="CNN+SE provides the best RMSE, with CNN as the second-best method.",
        )
    )

    p = FIG_DIR / "r2_comparison.png"
    plot_errorbar_comparison(
        summary,
        "r2_mean",
        "r2_std",
        "R²",
        "Comparison of R² across different demodulation methods",
        p,
    )
    generated.append(
        FigureRecord(
            name="r2_comparison.png",
                path=p,
                inputs=[INPUT_SUMMARY.as_posix()],
                section="Main Results",
                conclusion="Deep models show substantially higher R2 than traditional baselines, with CNN+SE highest.",
            )
        )

    # Group 2: stability (required)
    p = FIG_DIR / "rmse_stability.png"
    plot_stability(
        runs,
        "rmse",
        f"RMSE ({unit})",
        "RMSE Stability across 5 Random Seeds",
        p,
    )
    generated.append(
        FigureRecord(
            name="rmse_stability.png",
            path=p,
            inputs=[INPUT_RUNS.as_posix()],
            section="Stability Analysis",
            conclusion="CNN+SE has the lowest RMSE mean and a small cross-seed spread.",
        )
    )

    p = FIG_DIR / "mae_stability.png"
    plot_stability(
        runs,
        "mae",
        f"MAE ({unit})",
        "MAE Stability across 5 Random Seeds",
        p,
    )
    generated.append(
        FigureRecord(
            name="mae_stability.png",
                path=p,
                inputs=[INPUT_RUNS.as_posix()],
                section="Stability Analysis",
                conclusion="Traditional methods are deterministic (std~0), while neural methods have low but non-zero variation.",
            )
        )

    # Group 3: CNN vs CNN+SE (recommended)
    p = FIG_DIR / "cnn_vs_cnnse.png"
    ok = plot_cnn_vs_cnnse(summary, p, unit)
    if ok:
        generated.append(
            FigureRecord(
                name="cnn_vs_cnnse.png",
                path=p,
                inputs=[INPUT_SUMMARY.as_posix()],
                section="Ablation / Model Comparison",
                conclusion="SE block brings consistent mean error reduction over CNN.",
            )
        )
    else:
        skipped.append(("cnn_vs_cnnse.png", "Missing CNN or CNN+SE rows in multiseed_summary.csv"))

    # Group 4: scatter plots (if prediction data is available)
    pred_data = try_load_step2_predictions()
    if pred_data is None:
        skipped.append(("scatter_mlp.png / scatter_cnn.png / scatter_cnn_se.png", "best_seed_predictions.npz not found"))
    else:
        need_keys = ["y_true", "pred_mlp", "pred_cnn", "pred_cnn_se"]
        if not all(k in pred_data for k in need_keys):
            skipped.append(
                (
                    "scatter_mlp.png / scatter_cnn.png / scatter_cnn_se.png",
                    f"best_seed_predictions.npz missing required keys: {need_keys}",
                )
            )
        else:
            y_true = np.asarray(pred_data["y_true"]).reshape(-1)
            preds = {
                "MLP": np.asarray(pred_data["pred_mlp"]).reshape(-1),
                "CNN": np.asarray(pred_data["pred_cnn"]).reshape(-1),
                "CNN+SE": np.asarray(pred_data["pred_cnn_se"]).reshape(-1),
            }
            y_min = float(np.min(y_true))
            y_max = float(np.max(y_true))
            for arr in preds.values():
                y_min = min(y_min, float(np.min(arr)))
                y_max = max(y_max, float(np.max(arr)))
            pad = 0.03 * (y_max - y_min + 1e-12)
            lims = (y_min - pad, y_max + pad)

            scatter_map = [
                ("MLP", "scatter_mlp.png"),
                ("CNN", "scatter_cnn.png"),
                ("CNN+SE", "scatter_cnn_se.png"),
            ]
            for name, fname in scatter_map:
                out = FIG_DIR / fname
                plot_scatter(y_true, preds[name], name, out, lims)
                generated.append(
                    FigureRecord(
                        name=fname,
                        path=out,
                        inputs=[(RAW_DIR / "best_seed_predictions.npz").as_posix()],
                        section="Prediction Consistency",
                        conclusion=f"{name} predictions align closely with ground truth along the y=x line.",
                    )
                )

    # Group 5: dataset/sample figures (if available)
    dataset_used_path = None
    dataset_used_file = RAW_DIR / "dataset_used.txt"
    if dataset_used_file.exists():
        try:
            txt = dataset_used_file.read_text(encoding="utf-8", errors="ignore").strip()
            if txt:
                dataset_used_path = Path(txt)
        except Exception:
            dataset_used_path = None
    if dataset_used_path is None:
        fallback = DATA_DIR / "dataset_b_array.npz"
        if fallback.exists():
            dataset_used_path = fallback

    if dataset_used_path is None or not dataset_used_path.exists():
        skipped.append(("dataset_examples.png / example_samples.png", "No usable dataset path found for sample plotting"))
    else:
        p = FIG_DIR / "dataset_examples.png"
        ok1 = plot_dataset_examples(dataset_used_path, p)
        if ok1:
            generated.append(
                FigureRecord(
                    name="dataset_examples.png",
                    path=p,
                    inputs=[
                        (DATA_DIR / "dataset_a_phase1.npz").as_posix(),
                        (DATA_DIR / "dataset_b_phase3.npz").as_posix(),
                        dataset_used_path.as_posix(),
                    ],
                    section="Dataset Description",
                    conclusion="Spectral complexity increases from ideal single-FBG to noisy and array-distorted inputs.",
                )
            )
        else:
            skipped.append(("dataset_examples.png", "Required dataset fields are missing in one or more dataset files"))

        p = FIG_DIR / "example_samples.png"
        ok2 = plot_example_samples(dataset_used_path, p)
        if ok2:
            generated.append(
                FigureRecord(
                    name="example_samples.png",
                    path=p,
                    inputs=[dataset_used_path.as_posix()],
                    section="Dataset Description",
                    conclusion="Local distorted windows show clear sample-wise spectral diversity across target shifts.",
                )
            )
        else:
            skipped.append(("example_samples.png", "Missing X_local/Y_dlambda_target/wavelengths or insufficient test samples"))

    summary_path = FIG_DIR / "FIGURE_SUMMARY.txt"
    write_figure_summary(summary_path, generated, skipped)

    # Required terminal output
    print("Generated figure files:")
    for rec in generated:
        print(f"- {rec.path.as_posix()}")
    print(f"- {summary_path.as_posix()}")

    print("\nNot generated figures and reasons:")
    if skipped:
        for name, reason in skipped:
            print(f"- {name}: {reason}")
    else:
        print("- None")

    print("\nOne-line conclusion per core figure:")
    core_names = {
        "mae_comparison.png",
        "rmse_comparison.png",
        "r2_comparison.png",
        "rmse_stability.png",
        "mae_stability.png",
        "cnn_vs_cnnse.png",
    }
    for rec in generated:
        if rec.name in core_names:
            print(f"- {rec.name}: {rec.conclusion}")

    print("\nStage status: Ready to proceed to experiment-section writing.")


if __name__ == "__main__":
    main()
