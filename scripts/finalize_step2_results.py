from __future__ import annotations

import csv
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_SEED_METRICS = PROJECT_ROOT / "results" / "paper_results_step2" / "raw" / "seed_metrics.csv"
INPUT_PAPER_TABLE = PROJECT_ROOT / "results" / "paper_results_step2" / "tables" / "paper_main_table.csv"
OUT_DIR = PROJECT_ROOT / "results" / "paper_results_step2" / "final"

ORDERED_METHODS = [
    "Cross-correlation",
    "Parametric fitting",
    "MLP",
    "CNN",
    "CNN+SE",
]


def normalize_key(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", s.lower())


def load_csv(path: Path) -> list[dict[str, str]]:
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)


def detect_column(cols: list[str], candidates: list[str]) -> str | None:
    norm_map = {normalize_key(c): c for c in cols}
    for cand in candidates:
        n = normalize_key(cand)
        if n in norm_map:
            return norm_map[n]
    # fallback fuzzy contains
    for c in cols:
        nc = normalize_key(c)
        for cand in candidates:
            n = normalize_key(cand)
            if n and (n in nc or nc in n):
                return c
    return None


def safe_float(v: Any) -> float | None:
    try:
        if v is None:
            return None
        s = str(v).strip()
        if s == "":
            return None
        return float(s)
    except Exception:
        return None


def parse_seed(v: str) -> tuple[Any, str]:
    s = str(v).strip()
    if re.fullmatch(r"[+-]?\d+", s):
        return int(s), s
    if re.fullmatch(r"[+-]?(\d+(\.\d*)?|\.\d+)", s):
        f = float(s)
        if abs(f - int(f)) < 1e-12:
            return int(f), s
        return f, s
    return s, s


def map_method(raw_method: str) -> str:
    s = normalize_key(raw_method)

    # Cross-correlation aliases
    if any(k in s for k in ["crosscorr", "crosscorrelation", "xcorr", "correlation"]):
        return "Cross-correlation"
    # Parametric fitting aliases
    if any(k in s for k in ["paramfit", "parametricfit", "curvefit", "fitting"]) or ("param" in s and "fit" in s):
        return "Parametric fitting"
    # MLP
    if "mlp" in s:
        return "MLP"
    # CNN+SE before CNN
    if "cnn" in s and ("se" in s or "squeeze" in s):
        return "CNN+SE"
    if s in {"cnn", "cnnbaseline", "baselinecnn"} or ("cnn" in s and "se" not in s):
        return "CNN"

    # fallback: keep as-is
    return raw_method.strip()


def summarize_check(seed_rows: list[dict[str, Any]], method_col: str, seed_col: str, mae_col: str, rmse_col: str, r2_col: str) -> dict[str, Any]:
    methods_raw = sorted({str(r[method_col]).strip() for r in seed_rows})
    seeds_raw = sorted({str(r[seed_col]).strip() for r in seed_rows}, key=lambda x: parse_seed(x)[0])
    counts = Counter(map_method(str(r[method_col])) for r in seed_rows)

    missing = 0
    for r in seed_rows:
        if any(safe_float(r[c]) is None for c in [mae_col, rmse_col, r2_col]):
            missing += 1

    return {
        "columns": list(seed_rows[0].keys()) if seed_rows else [],
        "methods_raw": methods_raw,
        "seeds_raw": seeds_raw,
        "counts_mapped": dict(counts),
        "metric_cols_detected": {"mae": mae_col, "rmse": rmse_col, "r2": r2_col},
        "missing_metric_rows": missing,
    }


def write_runs_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "seed", "mae", "rmse", "r2"])
        for r in rows:
            w.writerow([r["method"], r["seed"], r["mae"], r["rmse"], r["r2"]])


def method_sort_key(method: str) -> tuple[int, str]:
    if method in ORDERED_METHODS:
        return (ORDERED_METHODS.index(method), method)
    return (len(ORDERED_METHODS), method)


def compute_summary(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in runs:
        grouped[r["method"]].append(r)

    methods_sorted = sorted(grouped.keys(), key=method_sort_key)
    out = []
    for m in methods_sorted:
        arr_mae = np.array([float(x["mae"]) for x in grouped[m]], dtype=float)
        arr_rmse = np.array([float(x["rmse"]) for x in grouped[m]], dtype=float)
        arr_r2 = np.array([float(x["r2"]) for x in grouped[m]], dtype=float)
        out.append(
            {
                "method": m,
                "mae_mean": float(np.mean(arr_mae)),
                "mae_std": float(np.std(arr_mae)),
                "rmse_mean": float(np.mean(arr_rmse)),
                "rmse_std": float(np.std(arr_rmse)),
                "r2_mean": float(np.mean(arr_r2)),
                "r2_std": float(np.std(arr_r2)),
            }
        )
    return out


def write_summary_csv(path: Path, summary: list[dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "mae_mean", "mae_std", "rmse_mean", "rmse_std", "r2_mean", "r2_std"])
        for r in summary:
            w.writerow([r["method"], r["mae_mean"], r["mae_std"], r["rmse_mean"], r["rmse_std"], r["r2_mean"], r["r2_std"]])


def format_mean_std(mean: float, std: float) -> str:
    # Keep readability for small values while staying compact for paper tables.
    if abs(mean) < 0.1:
        return f"{mean:.4f}±{std:.4f}"
    return f"{mean:.3f}±{std:.3f}"


def write_paper_main_table_final(path: Path, summary: list[dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "MAE", "RMSE", "R2"])
        for r in summary:
            w.writerow(
                [
                    r["method"],
                    format_mean_std(r["mae_mean"], r["mae_std"]),
                    format_mean_std(r["rmse_mean"], r["rmse_std"]),
                    format_mean_std(r["r2_mean"], r["r2_std"]),
                ]
            )


def compare_with_reference(summary: list[dict[str, Any]], ref_rows: list[dict[str, str]]) -> tuple[bool, list[str]]:
    if not ref_rows:
        return False, ["paper_main_table.csv is empty or unreadable"]
    cols = list(ref_rows[0].keys())
    method_col = detect_column(cols, ["method", "model", "method_name", "algo"])
    rmse_mean_col = detect_column(cols, ["rmse_mean", "RMSE_mean"])
    mae_mean_col = detect_column(cols, ["mae_mean", "MAE_mean"])
    r2_mean_col = detect_column(cols, ["r2_mean", "R2_mean", "R²_mean"])
    if not all([method_col, rmse_mean_col, mae_mean_col, r2_mean_col]):
        return False, ["paper_main_table.csv has incompatible columns for numeric cross-check"]

    ref = {}
    for r in ref_rows:
        m = map_method(str(r[method_col]))
        rm = safe_float(r[rmse_mean_col])
        ma = safe_float(r[mae_mean_col])
        r2 = safe_float(r[r2_mean_col])
        if None in (rm, ma, r2):
            continue
        ref[m] = (ma, rm, r2)

    notes = []
    consistent = True
    for r in summary:
        m = r["method"]
        if m not in ref:
            notes.append(f"{m}: missing in paper_main_table.csv")
            consistent = False
            continue
        ma0, rm0, r20 = ref[m]
        d_ma = abs(r["mae_mean"] - ma0)
        d_rm = abs(r["rmse_mean"] - rm0)
        d_r2 = abs(r["r2_mean"] - r20)
        if d_ma > 1e-10 or d_rm > 1e-10 or d_r2 > 1e-10:
            notes.append(f"{m}: differs from paper_main_table.csv (mae {d_ma:.3e}, rmse {d_rm:.3e}, r2 {d_r2:.3e})")
            consistent = False
    if consistent:
        notes.append("paper_main_table.csv matches recomputed summary from seed_metrics.csv")
    return consistent, notes


def write_readme(
    path: Path,
    check: dict[str, Any],
    method_map: dict[str, str],
    runs_path: Path,
    summary_path: Path,
    paper_final_path: Path,
    reference_diff_notes: list[str],
    summary: list[dict[str, Any]],
) -> None:
    seed_list = check["seeds_raw"]
    counts = check["counts_mapped"]
    has_missing = check["missing_metric_rows"] > 0

    best_mae = min(summary, key=lambda x: x["mae_mean"])
    best_rmse = min(summary, key=lambda x: x["rmse_mean"])
    d = {r["method"]: r for r in summary}
    cnn_vs_mlp = None
    if "CNN" in d and "MLP" in d:
        cnn_vs_mlp = d["CNN"]["rmse_mean"] < d["MLP"]["rmse_mean"]
    se_vs_cnn = None
    if "CNN+SE" in d and "CNN" in d:
        se_vs_cnn = d["CNN+SE"]["rmse_mean"] < d["CNN"]["rmse_mean"]

    lines = []
    lines.append("Step2 Results Finalization README")
    lines.append("")
    lines.append("1) Task description")
    lines.append("This task finalizes existing Step2 results by reading and reshaping existing files only.")
    lines.append("No experiment was re-run, and no raw result file was modified.")
    lines.append("")
    lines.append("2) Input files")
    lines.append(f"- {INPUT_SEED_METRICS.as_posix()}")
    lines.append(f"- {INPUT_PAPER_TABLE.as_posix()}")
    lines.append("")
    lines.append("3) Output files")
    lines.append(f"- {runs_path.as_posix()}")
    lines.append(f"- {summary_path.as_posix()}")
    lines.append(f"- {paper_final_path.as_posix()}")
    lines.append(f"- {path.as_posix()}")
    lines.append("")
    lines.append("4) Original seed list")
    lines.append("- " + ", ".join(seed_list))
    lines.append("")
    lines.append("5) Method name mapping")
    for raw, mapped in sorted(method_map.items(), key=lambda x: x[0].lower()):
        lines.append(f"- {raw} -> {mapped}")
    lines.append("")
    lines.append("6) Output file usage")
    lines.append("- multiseed_runs.csv: standardized per-seed result table")
    lines.append("- multiseed_summary.csv: mean/std summary by method")
    lines.append("- paper_main_table_final.csv: paper-ready compact main table (mean±std)")
    lines.append("- README_results.txt: traceability and mapping notes")
    lines.append("")
    lines.append("7) Data check summary")
    lines.append(f"- Detected columns in seed_metrics.csv: {check['columns']}")
    lines.append(f"- Raw method count: {len(check['methods_raw'])}")
    lines.append(f"- Mapped method count: {len(counts)}")
    for m in ORDERED_METHODS:
        if m in counts:
            lines.append(f"  - {m}: {counts[m]} rows")
    extra_methods = [m for m in counts.keys() if m not in ORDERED_METHODS]
    for m in extra_methods:
        lines.append(f"  - {m}: {counts[m]} rows")
    lines.append(f"- Missing/invalid metric rows: {check['missing_metric_rows']}")
    lines.append("")
    lines.append("8) Cross-reference against paper_main_table.csv")
    for n in reference_diff_notes:
        lines.append(f"- {n}")
    if any("differs" in n for n in reference_diff_notes):
        lines.append("- Final summary uses seed_metrics.csv as authoritative source.")
    lines.append("")
    lines.append("9) Short conclusions")
    lines.append(f"- Lowest MAE mean: {best_mae['method']} ({best_mae['mae_mean']:.6f})")
    lines.append(f"- Lowest RMSE mean: {best_rmse['method']} ({best_rmse['rmse_mean']:.6f})")
    if cnn_vs_mlp is not None:
        lines.append(f"- CNN better than MLP by RMSE: {'Yes' if cnn_vs_mlp else 'No'}")
    if se_vs_cnn is not None:
        lines.append(f"- CNN+SE better than CNN by RMSE: {'Yes' if se_vs_cnn else 'No'}")

    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    runs_out = OUT_DIR / "multiseed_runs.csv"
    summary_out = OUT_DIR / "multiseed_summary.csv"
    paper_final_out = OUT_DIR / "paper_main_table_final.csv"
    readme_out = OUT_DIR / "README_results.txt"

    if not INPUT_SEED_METRICS.exists():
        raise FileNotFoundError(f"Missing input file: {INPUT_SEED_METRICS}")
    if not INPUT_PAPER_TABLE.exists():
        raise FileNotFoundError(f"Missing input file: {INPUT_PAPER_TABLE}")

    seed_rows = load_csv(INPUT_SEED_METRICS)
    paper_rows = load_csv(INPUT_PAPER_TABLE)
    if not seed_rows:
        raise ValueError("seed_metrics.csv is empty")

    cols = list(seed_rows[0].keys())
    method_col = detect_column(cols, ["method", "model", "method_name", "algo", "algorithm"])
    seed_col = detect_column(cols, ["seed", "random_seed", "run_seed", "trial_seed"])
    mae_col = detect_column(cols, ["mae", "MAE", "mae_nm", "mean_absolute_error"])
    rmse_col = detect_column(cols, ["rmse", "RMSE", "rmse_nm", "root_mean_square_error", "rms"])
    r2_col = detect_column(cols, ["r2", "R2", "R²", "rsquared", "r_square"])

    missing_cols = [name for name, c in [("method", method_col), ("seed", seed_col), ("mae", mae_col), ("rmse", rmse_col), ("r2", r2_col)] if c is None]
    if missing_cols:
        raise ValueError(f"Could not detect required columns in seed_metrics.csv: {missing_cols}")

    check = summarize_check(seed_rows, method_col, seed_col, mae_col, rmse_col, r2_col)

    # Build standardized runs
    method_map: dict[str, str] = {}
    runs = []
    skipped = 0
    for r in seed_rows:
        raw_method = str(r[method_col]).strip()
        mapped = map_method(raw_method)
        method_map[raw_method] = mapped
        mae = safe_float(r[mae_col])
        rmse = safe_float(r[rmse_col])
        r2 = safe_float(r[r2_col])
        if None in (mae, rmse, r2):
            skipped += 1
            continue
        seed_sortable, seed_raw = parse_seed(str(r[seed_col]))
        runs.append(
            {
                "method": mapped,
                "seed": seed_raw if isinstance(seed_sortable, str) else seed_sortable,
                "_seed_sort": seed_sortable,
                "mae": float(mae),
                "rmse": float(rmse),
                "r2": float(r2),
            }
        )

    runs.sort(key=lambda x: (method_sort_key(x["method"]), x["_seed_sort"]))
    for r in runs:
        r.pop("_seed_sort", None)

    summary = compute_summary(runs)
    consistent, diff_notes = compare_with_reference(summary, paper_rows)
    if not consistent and not diff_notes:
        diff_notes = ["Found differences vs paper_main_table.csv; summary still based on seed_metrics.csv."]

    write_runs_csv(runs_out, runs)
    write_summary_csv(summary_out, summary)
    write_paper_main_table_final(paper_final_out, summary)
    write_readme(readme_out, check, method_map, runs_out, summary_out, paper_final_out, diff_notes, summary)

    # Required terminal output
    print("Data check summary:")
    print(f"- seed_metrics columns: {check['columns']}")
    print(f"- methods(raw): {check['methods_raw']}")
    print(f"- seeds(raw): {check['seeds_raw']}")
    print(f"- metric columns detected: {check['metric_cols_detected']}")
    print("- rows per mapped method:")
    for m in sorted(check["counts_mapped"].keys(), key=method_sort_key):
        print(f"  {m}: {check['counts_mapped'][m]}")
    print(f"- missing/invalid metric rows: {check['missing_metric_rows']}")
    if skipped > 0:
        print(f"- skipped rows during normalization: {skipped}")

    print("\nGenerated files:")
    print(f"- {runs_out.as_posix()}")
    print(f"- {summary_out.as_posix()}")
    print(f"- {paper_final_out.as_posix()}")
    print(f"- {readme_out.as_posix()}")

    print("\nmultiseed_summary.csv preview:")
    with open(summary_out, "r", encoding="utf-8") as f:
        print(f.read().strip())

    print("\npaper_main_table_final.csv preview:")
    with open(paper_final_out, "r", encoding="utf-8") as f:
        print(f.read().strip())

    best = min(summary, key=lambda x: x["rmse_mean"])["method"] if summary else "N/A"
    print(
        "\nShort summary: finalization completed without rerunning experiments; "
        f"best RMSE method is {best}. Ready to proceed to paper figure generation."
    )


if __name__ == "__main__":
    main()

