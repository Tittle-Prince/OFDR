from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from phase3.common import load_config, load_dataset, resolve_project_path, set_seed
from phase3.data_builder import build_dataset_b


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare Dataset_B for Phase3 robustness evaluation")
    parser.add_argument("--config", type=str, default="config/phase3.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(int(cfg["phase3"]["seed"]))

    src_path = resolve_project_path(cfg["phase3"]["source_dataset_a_path"])
    dst_path = resolve_project_path(cfg["phase3"]["dataset_b_path"])

    if not src_path.exists():
        raise FileNotFoundError(
            f"Source Dataset_A not found: {src_path}\n"
            "Run `python src\\phase1_pipeline.py --config config\\phase1.yaml --regenerate` first."
        )

    source_data = load_dataset(src_path)
    rng = np.random.default_rng(int(cfg["phase3"]["seed"]))
    dataset_b = build_dataset_b(source_data, cfg, rng)

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        dst_path,
        X=dataset_b.x,
        Y_dlambda=dataset_b.y_dlambda,
        Y_dT=dataset_b.y_dt,
        wavelengths=dataset_b.wavelengths,
        idx_train=dataset_b.idx_train,
        idx_val=dataset_b.idx_val,
        idx_test=dataset_b.idx_test,
    )
    print(f"Saved Dataset_B to: {dst_path}")
    print(f"X shape: {dataset_b.x.shape}, test size: {len(dataset_b.idx_test)}")


if __name__ == "__main__":
    main()

