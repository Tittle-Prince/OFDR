from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sweep multi-frame fusion over single-frame predictions")
    p.add_argument(
        "--pred",
        type=str,
        default="results/phase4_array_se_hard/dualinput_fusion/predictions.npz",
        help="NPZ with keys y_true and pred_bias_corrected (or pred_raw)",
    )
    p.add_argument(
        "--out-dir",
        type=str,
        default="results/phase4_array_se_hard/dualinput_fusion",
    )
    return p.parse_args()


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def centered_ma(x: np.ndarray, n: int) -> np.ndarray:
    k = np.ones(n, dtype=np.float64) / float(n)
    return np.convolve(x, k, mode="same")


def ema(x: np.ndarray, n: int) -> np.ndarray:
    alpha = 2.0 / float(n + 1)
    out = np.empty_like(x)
    out[0] = x[0]
    for i in range(1, len(x)):
        out[i] = alpha * x[i] + (1.0 - alpha) * out[i - 1]
    return out


def kalman_1d(x: np.ndarray, q_scale: float, r: float) -> np.ndarray:
    q = float(r * q_scale)
    out = np.empty_like(x)
    xhat = float(x[0])
    p = float(r)
    out[0] = xhat
    for i in range(1, len(x)):
        p = p + q
        k = p / (p + r)
        xhat = xhat + k * (float(x[i]) - xhat)
        p = (1.0 - k) * p
        out[i] = xhat
    return out


def main() -> None:
    args = parse_args()
    pred_path = Path(args.pred)
    if not pred_path.exists():
        raise FileNotFoundError(f"Prediction file not found: {pred_path}")
    d = np.load(pred_path)
    if "y_true" not in d:
        raise KeyError("pred npz must contain y_true")
    if "pred_bias_corrected" in d:
        pred_key = "pred_bias_corrected"
    elif "pred_raw" in d:
        pred_key = "pred_raw"
    else:
        raise KeyError("pred npz must contain pred_bias_corrected or pred_raw")

    y = d["y_true"].astype(np.float64).reshape(-1)
    p = d[pred_key].astype(np.float64).reshape(-1)
    # Build a smooth trajectory proxy by sorting labels.
    order = np.argsort(y)
    ys = y[order]
    ps = p[order]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "multiframe_fusion_sweep.csv"
    out_json = out_dir / "multiframe_fusion_summary.json"

    windows = [3, 5, 7, 9, 15, 21, 31, 41, 61, 81, 101]
    q_scales = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2]
    base_rmse = rmse(ys, ps)

    rows: list[tuple[str, int, float, float]] = []
    for n in windows:
        p_ma = centered_ma(ps, n)
        rows.append(("centered_ma", n, np.nan, rmse(ys, p_ma)))
        p_ema = ema(ps, n)
        rows.append(("ema", n, np.nan, rmse(ys, p_ema)))

    r = float(np.var(ps - ys) + 1e-12)
    for qs in q_scales:
        p_kf = kalman_1d(ps, q_scale=qs, r=r)
        rows.append(("kalman", -1, qs, rmse(ys, p_kf)))

    with open(out_csv, "w", encoding="utf-8") as f:
        f.write("method,window,q_scale,rmse_nm\n")
        f.write(f"single,1,,{base_rmse:.8f}\n")
        for m, n, qs, r0 in rows:
            w_str = "" if n < 0 else str(n)
            q_str = "" if np.isnan(qs) else f"{qs:.6g}"
            f.write(f"{m},{w_str},{q_str},{r0:.8f}\n")

    best = min(rows, key=lambda t: t[3])
    summary = {
        "pred_key": pred_key,
        "single_rmse_nm": base_rmse,
        "best_method": best[0],
        "best_window": None if best[1] < 0 else best[1],
        "best_q_scale": None if np.isnan(best[2]) else best[2],
        "best_rmse_nm": best[3],
        "improvement_percent": (base_rmse - best[3]) / max(1e-12, base_rmse) * 100.0,
        "note": "Trajectory built by sorting y_true; this is a smooth-ramp proxy, not real time-series.",
    }
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"single RMSE: {base_rmse:.6f} nm")
    print(
        f"best fusion: method={summary['best_method']}, "
        f"window={summary['best_window']}, q_scale={summary['best_q_scale']}, "
        f"rmse={summary['best_rmse_nm']:.6f} nm"
    )
    print(f"saved: {out_csv}")


if __name__ == "__main__":
    main()

