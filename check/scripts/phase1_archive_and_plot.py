from __future__ import annotations

import runpy
from pathlib import Path


if __name__ == "__main__":
    target = Path(__file__).resolve().parents[2] / "scripts" / "maintenance" / "phase1_archive_and_plot.py"
    runpy.run_path(str(target), run_name="__main__")

