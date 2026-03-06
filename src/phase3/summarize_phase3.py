from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from phase3.common import load_config, resolve_project_path


METHODS = [
    ("cnn_baseline", "Baseline"),
    ("cnn_dilated", "Dilated"),
    ("cnn_se", "SE"),
    ("cnn_dilated_se", "Dilated+SE"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize Phase3 CNN robustness ablation")
    parser.add_argument("--config", type=str, default="config/phase3.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    root_out = resolve_project_path(cfg["phase3"]["results_dir"])
    summary_dir = root_out / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for key, label in METHODS:
        p = root_out / key / "metrics.json"
        if not p.exists():
            raise FileNotFoundError(f"Missing metrics file for {key}: {p}")
        with open(p, "r", encoding="utf-8") as f:
            m = json.load(f)
        rows.append({"method_key": key, "method": label, "rmse": m["rmse"], "mae": m["mae"], "r2": m["r2"]})

    with open(summary_dir / "phase3_metrics_table.csv", "w", encoding="utf-8") as f:
        f.write("method_key,method,rmse_nm,mae_nm,r2\n")
        for r in rows:
            f.write(f"{r['method_key']},{r['method']},{r['rmse']:.8f},{r['mae']:.8f},{r['r2']:.8f}\n")

    with open(summary_dir / "phase3_metrics_table.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)

    md_lines = [
        "# Phase3 CNN Structure Robustness (Dataset_B)",
        "",
        "| Model | RMSE (nm) | MAE (nm) | R2 |",
        "| --- | ---: | ---: | ---: |",
    ]
    for r in rows:
        md_lines.append(f"| {r['method']} | {r['rmse']:.8f} | {r['mae']:.8f} | {r['r2']:.8f} |")
    (summary_dir / "phase3_metrics_table.md").write_text("\n".join(md_lines), encoding="utf-8")

    txt_lines = []
    txt_lines.append("Phase3 CNN Structure Robustness (Dataset_B)")
    txt_lines.append("")
    txt_lines.append(f"{'Model':<16}{'RMSE (nm)':>14}{'MAE (nm)':>14}{'R2':>14}")
    txt_lines.append("-" * 58)
    for r in rows:
        txt_lines.append(f"{r['method']:<16}{r['rmse']:>14.8f}{r['mae']:>14.8f}{r['r2']:>14.8f}")
    (summary_dir / "phase3_metrics_table.txt").write_text("\n".join(txt_lines), encoding="utf-8")

    names = [r["method"] for r in rows]
    rmse = [r["rmse"] for r in rows]
    mae = [r["mae"] for r in rows]
    r2 = [r["r2"] for r in rows]
    colors = ["#4e79a7", "#f28e2b", "#59a14f", "#e15759"]
    x = range(len(names))

    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.8), constrained_layout=True)
    axes[0].bar(x, rmse, color=colors)
    axes[0].set_title("RMSE (nm)")
    axes[0].set_xticks(list(x), names, rotation=12)
    axes[1].bar(x, mae, color=colors)
    axes[1].set_title("MAE (nm)")
    axes[1].set_xticks(list(x), names, rotation=12)
    axes[2].bar(x, r2, color=colors)
    axes[2].set_title("R2")
    axes[2].set_ylim(max(0.97, min(r2) - 0.01), 1.00001)
    axes[2].set_xticks(list(x), names, rotation=12)

    for ax, vals in zip(axes, [rmse, mae, r2]):
        for i, v in enumerate(vals):
            ax.text(i, v, f"{v:.6f}", ha="center", va="bottom", fontsize=8)

    fig.suptitle("Phase3 Ablation on Dataset_B")
    fig.savefig(summary_dir / "phase3_ablation.png", dpi=300)
    fig.savefig(summary_dir / "phase3_ablation.pdf")
    plt.close(fig)

    baseline_rmse = rows[0]["rmse"]
    improv = []
    for r in rows:
        gain = (baseline_rmse - r["rmse"]) / baseline_rmse * 100.0
        improv.append({"method": r["method"], "rmse_improvement_vs_baseline_percent": gain})
    with open(summary_dir / "phase3_ablation_gain.json", "w", encoding="utf-8") as f:
        json.dump(improv, f, indent=2)

    print(f"Summary table saved to: {summary_dir / 'phase3_metrics_table.txt'}")
    print(f"Ablation figure saved to: {summary_dir / 'phase3_ablation.png'}")


if __name__ == "__main__":
    main()

