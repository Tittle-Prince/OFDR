from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Quick sweep for phase4ts GRU residual route")
    p.add_argument("--config", type=str, default="config/phase4_array_se_hard_ts.yaml")
    p.add_argument("--epochs", type=int, default=24)
    p.add_argument("--patience", type=int, default=6)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    experiments = [
        {"tag": "sweep_q0p005_w24_h48", "q": 0.005, "w": 24, "h": 48},
        {"tag": "sweep_q0p010_w24_h48", "q": 0.010, "w": 24, "h": 48},
        {"tag": "sweep_q0p020_w24_h48", "q": 0.020, "w": 24, "h": 48},
        {"tag": "sweep_q0p010_w32_h64", "q": 0.010, "w": 32, "h": 64},
        {"tag": "sweep_q0p020_w32_h64", "q": 0.020, "w": 32, "h": 64},
        {"tag": "sweep_q0p010_w40_h64", "q": 0.010, "w": 40, "h": 64},
    ]

    root = Path(__file__).resolve().parents[2]
    runner = root / "src" / "phase4ts" / "run_gru_residual_timeseries.py"

    for i, exp in enumerate(experiments, start=1):
        print(f"\n=== [{i}/{len(experiments)}] {exp['tag']} ===")
        cmd = [
            sys.executable,
            str(runner),
            "--config",
            str(args.config),
            "--skip-single-train",
            "--q-scale",
            str(exp["q"]),
            "--window",
            str(exp["w"]),
            "--hidden",
            str(exp["h"]),
            "--epochs",
            str(int(args.epochs)),
            "--patience",
            str(int(args.patience)),
            "--run-tag",
            exp["tag"],
        ]
        subprocess.run(cmd, check=True)

    res_dir = root / "results" / "phase4_array_se_hard_ts" / "gru_residual"
    rows: list[dict] = []
    for exp in experiments:
        m_path = res_dir / exp["tag"] / "metrics.json"
        if not m_path.exists():
            continue
        with open(m_path, "r", encoding="utf-8") as f:
            m = json.load(f)
        row = {
            "tag": exp["tag"],
            "q_scale": float(m.get("kalman_q_scale", exp["q"])),
            "window": int(m.get("window", exp["w"])),
            "hidden": int(m.get("gru_hidden", exp["h"])),
            "single_rmse": float(m["single_frame"]["rmse"]),
            "kalman_rmse": float(m["kalman"]["rmse"]),
            "gru_rmse": float(m["gru_residual"]["rmse"]),
            "gru_mae": float(m["gru_residual"]["mae"]),
            "gru_p95": float(m["gru_residual"]["p95"]),
            "gru_p99": float(m["gru_residual"]["p99"]),
        }
        rows.append(row)

    rows.sort(key=lambda x: x["gru_rmse"])
    out_csv = res_dir / "sweep_summary.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "tag",
                "q_scale",
                "window",
                "hidden",
                "single_rmse",
                "kalman_rmse",
                "gru_rmse",
                "gru_mae",
                "gru_p95",
                "gru_p99",
            ],
        )
        w.writeheader()
        w.writerows(rows)

    print(f"\nSaved sweep summary: {out_csv}")
    if len(rows) > 0:
        b = rows[0]
        print(
            f"Best GRU by RMSE: {b['tag']} | q={b['q_scale']} window={b['window']} hidden={b['hidden']} | "
            f"single={b['single_rmse']:.6f} kalman={b['kalman_rmse']:.6f} gru={b['gru_rmse']:.6f}"
        )


if __name__ == "__main__":
    main()

