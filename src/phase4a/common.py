from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_config(config_path: str | Path) -> dict:
    p = Path(config_path)
    if not p.is_absolute():
        p = project_root() / p
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_project_path(path_str: str) -> Path:
    return project_root() / path_str


def set_seed(seed: int) -> None:
    np.random.seed(seed)

