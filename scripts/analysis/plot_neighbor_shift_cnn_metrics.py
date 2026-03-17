from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _load_row(path: Path) -> dict[str, float]:
    with open(path, "r", encoding="utf-8") as f:
        row = list(csv.DictReader(f))[0]
    return {
        "RMSE_nm": float(row["RMSE_nm"]),
        "MAE_nm": float(row["MAE_nm"]),
        "P95_nm": float(row["P95_nm"]),
        "P99_nm": float(row["P99_nm"]),
        "R2": float(row["R2"]),
    }


def main() -> None:
    specs = [
        (0.02, PROJECT_ROOT / "results" / "phase4a_neighbor_shift_002" / "cnn_only" / "metrics_table.csv"),
        (0.04, PROJECT_ROOT / "results" / "phase4a_neighbor_shift_004" / "cnn_only" / "metrics_table.csv"),
        (0.06, PROJECT_ROOT / "results" / "phase4a_neighbor_shift_006" / "cnn_only" / "metrics_table.csv"),
    ]

    shifts = []
    rmse = []
    mae = []
    p95 = []
    p99 = []
    for shift, path in specs:
        if not path.exists():
            raise FileNotFoundError(f"Missing metrics file: {path}")
        row = _load_row(path)
        shifts.append(shift)
        rmse.append(row["RMSE_nm"])
        mae.append(row["MAE_nm"])
        p95.append(row["P95_nm"])
        p99.append(row["P99_nm"])

    out_dir = PROJECT_ROOT / "results" / "phase4a_neighbor_shift_summary"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "cnn_metrics_vs_neighbor_shift.png"

    fig, ax = plt.subplots(figsize=(8.8, 5.0), constrained_layout=True)
    ax.plot(shifts, rmse, marker="o", linewidth=2.0, color="#4e79a7", label="RMSE")
    ax.plot(shifts, mae, marker="s", linewidth=2.0, color="#f28e2b", label="MAE")
    ax.plot(shifts, p95, marker="^", linewidth=2.0, color="#59a14f", label="P95")
    ax.plot(shifts, p99, marker="d", linewidth=2.0, color="#e15759", label="P99")

    ax.set_title("CNN metrics vs neighbor-shift range")
    ax.set_xlabel("Neighbor shift half-range (nm)")
    ax.set_ylabel("Error (nm)")
    ax.set_xticks(shifts, [f"+/-{s:.2f}" for s in shifts])
    ax.grid(True, alpha=0.25)
    ax.legend()

    for xs, ys in [(shifts, rmse), (shifts, mae), (shifts, p95), (shifts, p99)]:
        for x, y in zip(xs, ys):
            ax.text(x, y, f"{y:.4f}", fontsize=8, ha="center", va="bottom")

    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Saved figure: {out_path}")


if __name__ == "__main__":
    main()
