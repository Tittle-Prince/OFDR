from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import yaml


@dataclass
class Phase2Data:
    x: np.ndarray
    y_dlambda: np.ndarray
    y_dt: np.ndarray
    wavelengths: np.ndarray
    idx_train: np.ndarray
    idx_val: np.ndarray
    idx_test: np.ndarray


def get_project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_project_path(path_str: str) -> Path:
    return get_project_root() / path_str


def load_config(config_path: str | Path) -> dict:
    path = Path(config_path)
    if not path.is_absolute():
        path = get_project_root() / path
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_phase2_data(cfg: dict) -> Phase2Data:
    data_path = resolve_project_path(cfg["phase2"]["data_path"])
    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset file not found: {data_path}\n"
            "Run `python src\\phase1_pipeline.py --config config\\phase1.yaml --regenerate` first."
        )

    data = np.load(data_path)
    required = ["X", "Y_dlambda", "Y_dT", "wavelengths", "idx_train", "idx_val", "idx_test"]
    missing = [k for k in required if k not in data]
    if missing:
        raise KeyError(f"Dataset missing keys: {missing}")

    return Phase2Data(
        x=data["X"].astype(np.float32),
        y_dlambda=data["Y_dlambda"].astype(np.float32),
        y_dt=data["Y_dT"].astype(np.float32),
        wavelengths=data["wavelengths"].astype(np.float32),
        idx_train=data["idx_train"].astype(np.int64),
        idx_val=data["idx_val"].astype(np.int64),
        idx_test=data["idx_test"].astype(np.int64),
    )


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2) + 1e-12
    return float(1.0 - ss_res / ss_tot)


def metrics_dict(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "rmse": rmse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "r2": r2(y_true, y_pred),
    }


def method_dir(cfg: dict, method_name: str) -> Path:
    out_dir = resolve_project_path(cfg["phase2"]["results_dir"]) / method_name
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def save_method_outputs(
    cfg: dict,
    method_name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    extra_metrics: dict | None = None,
) -> dict:
    out_dir = method_dir(cfg, method_name)
    metrics = metrics_dict(y_true, y_pred)
    if extra_metrics:
        metrics.update(extra_metrics)

    np.savez(out_dir / "predictions.npz", y_true=y_true, y_pred=y_pred)
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    return metrics

