from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from phase3.common import load_config
from phase3.runner import run_single_method


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase3 model: CNN + SE")
    parser.add_argument("--config", type=str, default="config/phase3.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    run_single_method(cfg, method_key="cnn_se", display_name="CNN + SE")


if __name__ == "__main__":
    main()

