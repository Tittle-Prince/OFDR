from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from phase2.baselines import estimate_center_by_parametric_fit
from phase2.common import load_config, load_phase2_data, save_method_outputs, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase2 baseline: Parametric fitting")
    parser.add_argument("--config", type=str, default="config/phase2.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(int(cfg["phase2"]["seed"]))
    data = load_phase2_data(cfg)

    lambda_b0 = float(cfg["phase2"]["lambda_b0_nm"])
    fit_window_points = int(cfg["parametric_fitting"]["fit_window_points"])
    baseline_percentile = float(cfg["parametric_fitting"]["baseline_percentile"])

    y_true = data.y_dlambda[data.idx_test]
    y_pred = np.zeros_like(y_true)

    for i, idx in enumerate(data.idx_test):
        center = estimate_center_by_parametric_fit(
            data.wavelengths,
            data.x[idx],
            fit_window_points=fit_window_points,
            baseline_percentile=baseline_percentile,
        )
        y_pred[i] = center - lambda_b0

    metrics = save_method_outputs(cfg, "parametric_fitting", y_true, y_pred)
    print(
        "Parametric fitting | "
        f"RMSE={metrics['rmse']:.6f} nm | MAE={metrics['mae']:.6f} nm | R2={metrics['r2']:.6f}"
    )


if __name__ == "__main__":
    main()

