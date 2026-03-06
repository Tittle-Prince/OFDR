from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from phase2.baselines import estimate_shift_by_cross_correlation, gaussian_spectrum, normalize_minmax
from phase2.common import load_config, load_phase2_data, save_method_outputs, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase2 baseline: Cross-correlation")
    parser.add_argument("--config", type=str, default="config/phase2.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(int(cfg["phase2"]["seed"]))
    data = load_phase2_data(cfg)

    step_nm = float(data.wavelengths[1] - data.wavelengths[0])
    lambda_b0 = float(cfg["phase2"]["lambda_b0_nm"])
    ccfg = cfg["cross_correlation"]
    ref = gaussian_spectrum(
        data.wavelengths,
        center_nm=lambda_b0,
        sigma_nm=float(ccfg["reference_sigma_nm"]),
        amplitude=float(ccfg["reference_amplitude"]),
        baseline=float(ccfg["reference_baseline"]),
    )
    if bool(ccfg["normalize_reference"]):
        ref = normalize_minmax(ref)

    y_true = data.y_dlambda[data.idx_test]
    y_pred = np.zeros_like(y_true)
    for i, idx in enumerate(data.idx_test):
        y_pred[i] = estimate_shift_by_cross_correlation(ref, data.x[idx], step_nm)

    metrics = save_method_outputs(cfg, "cross_correlation", y_true, y_pred)
    print(
        "Cross-correlation | "
        f"RMSE={metrics['rmse']:.6f} nm | MAE={metrics['mae']:.6f} nm | R2={metrics['r2']:.6f}"
    )


if __name__ == "__main__":
    main()
