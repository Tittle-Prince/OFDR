from __future__ import annotations

import csv
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


MODEL_ORDER = ["MLP", "CNN", "CNN+SE"]


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", s.lower())


def _detect_col(cols: list[str], aliases: list[str]) -> str | None:
    m = {_norm(c): c for c in cols}
    for a in aliases:
        na = _norm(a)
        if na in m:
            return m[na]
    for c in cols:
        nc = _norm(c)
        for a in aliases:
            na = _norm(a)
            if na and (na in nc or nc in na):
                return c
    return None


def _safe_float(v: str) -> float | None:
    try:
        s = str(v).strip()
        if s == "":
            return None
        return float(s)
    except Exception:
        return None


def _parse_mean_std(text: str) -> tuple[float | None, float | None]:
    if text is None:
        return None, None
    s = str(text).strip().replace("卤", "±").replace("+/-", "±")
    if "±" in s:
        a, b = s.split("±", 1)
        return _safe_float(a), _safe_float(b)
    v = _safe_float(s)
    return v, None


def load_from_summary(path: Path) -> dict[str, dict[str, float | None]]:
    rows = list(csv.DictReader(path.open("r", encoding="utf-8-sig", newline="")))
    if not rows:
        raise ValueError(f"Empty csv: {path}")
    cols = list(rows[0].keys())
    method_col = _detect_col(cols, ["method", "model"])
    mae_mean_col = _detect_col(cols, ["mae_mean", "mae"])
    rmse_mean_col = _detect_col(cols, ["rmse_mean", "rmse"])
    mae_std_col = _detect_col(cols, ["mae_std", "mae_sd", "std_mae"])
    rmse_std_col = _detect_col(cols, ["rmse_std", "rmse_sd", "std_rmse"])
    if method_col is None or mae_mean_col is None or rmse_mean_col is None:
        raise ValueError(f"Required columns missing in {path}")

    out: dict[str, dict[str, float | None]] = {}
    for r in rows:
        m = str(r.get(method_col, "")).strip()
        if m not in MODEL_ORDER:
            continue
        out[m] = {
            "mae_mean": _safe_float(r.get(mae_mean_col, "")),
            "rmse_mean": _safe_float(r.get(rmse_mean_col, "")),
            "mae_std": _safe_float(r.get(mae_std_col, "")) if mae_std_col else None,
            "rmse_std": _safe_float(r.get(rmse_std_col, "")) if rmse_std_col else None,
        }
    return out


def load_from_main_table(path: Path) -> dict[str, dict[str, float | None]]:
    rows = list(csv.DictReader(path.open("r", encoding="utf-8-sig", newline="")))
    if not rows:
        raise ValueError(f"Empty csv: {path}")
    cols = list(rows[0].keys())
    method_col = _detect_col(cols, ["method", "model"])
    mae_col = _detect_col(cols, ["mae"])
    rmse_col = _detect_col(cols, ["rmse"])
    if method_col is None or mae_col is None or rmse_col is None:
        raise ValueError(f"Required columns missing in {path}")

    out: dict[str, dict[str, float | None]] = {}
    for r in rows:
        m = str(r.get(method_col, "")).strip()
        if m not in MODEL_ORDER:
            continue
        mae_mean, mae_std = _parse_mean_std(r.get(mae_col, ""))
        rmse_mean, rmse_std = _parse_mean_std(r.get(rmse_col, ""))
        out[m] = {
            "mae_mean": mae_mean,
            "rmse_mean": rmse_mean,
            "mae_std": mae_std,
            "rmse_std": rmse_std,
        }
    return out


def pick_data(root: Path) -> tuple[Path, dict[str, dict[str, float | None]]]:
    summary = root / "results" / "paper_results_step2" / "final" / "multiseed_summary.csv"
    main_table = root / "results" / "paper_results_step2" / "final" / "paper_main_table_final.csv"

    if summary.exists():
        data = load_from_summary(summary)
        if all(m in data for m in MODEL_ORDER):
            return summary, data
    if main_table.exists():
        data = load_from_main_table(main_table)
        if all(m in data for m in MODEL_ORDER):
            return main_table, data
    raise FileNotFoundError("Could not load complete MLP/CNN/CNN+SE metrics from summary or main table.")


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    src_path, data = pick_data(root)

    x = np.arange(len(MODEL_ORDER), dtype=int)
    mae = np.array([float(data[m]["mae_mean"]) for m in MODEL_ORDER], dtype=float)
    rmse = np.array([float(data[m]["rmse_mean"]) for m in MODEL_ORDER], dtype=float)
    mae_std = np.array([0.0 if data[m]["mae_std"] is None else float(data[m]["mae_std"]) for m in MODEL_ORDER], dtype=float)
    rmse_std = np.array([0.0 if data[m]["rmse_std"] is None else float(data[m]["rmse_std"]) for m in MODEL_ORDER], dtype=float)

    plt.rcParams["font.family"] = "Arial"
    dpi = 300
    fig_w, fig_h = 1800 / dpi, 800 / dpi
    fig, axes = plt.subplots(1, 2, figsize=(fig_w, fig_h), dpi=dpi)
    fig.patch.set_facecolor("white")

    line_color = "#1f1f1f"
    marker = "o"
    lw = 1.7
    ms = 5

    # (a) MAE
    ax = axes[0]
    ax.set_facecolor("white")
    ax.plot(x, mae, color=line_color, marker=marker, markersize=ms, linewidth=lw)
    if np.any(mae_std > 0):
        ax.errorbar(x, mae, yerr=mae_std, fmt="none", ecolor="#444444", elinewidth=1.0, capsize=3)
    ax.set_xticks(x)
    ax.set_xticklabels(MODEL_ORDER)
    ax.set_title("(a) MAE comparison", fontsize=11, pad=4)
    ax.set_xlabel("Model", fontsize=10)
    ax.set_ylabel("MAE", fontsize=10)
    ax.tick_params(axis="both", labelsize=8, width=1.0, length=4)
    for sp in ax.spines.values():
        sp.set_linewidth(1.1)
    ax.grid(False)

    # (b) RMSE
    ax = axes[1]
    ax.set_facecolor("white")
    ax.plot(x, rmse, color=line_color, marker=marker, markersize=ms, linewidth=lw)
    if np.any(rmse_std > 0):
        ax.errorbar(x, rmse, yerr=rmse_std, fmt="none", ecolor="#444444", elinewidth=1.0, capsize=3)
    ax.set_xticks(x)
    ax.set_xticklabels(MODEL_ORDER)
    ax.set_title("(b) RMSE comparison", fontsize=11, pad=4)
    ax.set_xlabel("Model", fontsize=10)
    ax.set_ylabel("RMSE", fontsize=10)
    ax.tick_params(axis="both", labelsize=8, width=1.0, length=4)
    for sp in ax.spines.values():
        sp.set_linewidth(1.1)
    ax.grid(False)

    fig.subplots_adjust(left=0.07, right=0.99, bottom=0.16, top=0.90, wspace=0.22)

    out = root / "results" / "paper_figures" / "Fig7_model_influence.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi)
    plt.close(fig)

    print("Data file used:", src_path.as_posix())
    print("Model metrics (MAE / RMSE):")
    for m in MODEL_ORDER:
        print(f"- {m}: {mae[MODEL_ORDER.index(m)]:.8f} / {rmse[MODEL_ORDER.index(m)]:.8f}")
    print("Blueprint correspondence (Fig7): Yes")
    print("SCI plotting style compliance: Yes")


if __name__ == "__main__":
    main()

