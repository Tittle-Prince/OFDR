from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import yaml


@dataclass
class Phase3Data:
    x: np.ndarray
    y_dlambda: np.ndarray
    y_dt: np.ndarray
    wavelengths: np.ndarray
    idx_train: np.ndarray
    idx_val: np.ndarray
    idx_test: np.ndarray


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_project_path(path_str: str) -> Path:
    return project_root() / path_str


def load_config(config_path: str | Path) -> dict:
    p = Path(config_path)
    if not p.is_absolute():
        p = project_root() / p
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_dataset(path: Path) -> Phase3Data:
    data = np.load(path)
    needed = ["X", "Y_dlambda", "Y_dT", "wavelengths", "idx_train", "idx_val", "idx_test"]
    missing = [k for k in needed if k not in data]
    if missing:
        raise KeyError(f"Dataset missing keys: {missing}")

    return Phase3Data(
        x=data["X"].astype(np.float32),
        y_dlambda=data["Y_dlambda"].astype(np.float32),
        y_dt=data["Y_dT"].astype(np.float32),
        wavelengths=data["wavelengths"].astype(np.float32),
        idx_train=data["idx_train"].astype(np.int64),
        idx_val=data["idx_val"].astype(np.int64),
        idx_test=data["idx_test"].astype(np.int64),
    )


def load_dataset_b(cfg: dict) -> Phase3Data:
    data_b_path = resolve_project_path(cfg["phase3"]["dataset_b_path"])
    if not data_b_path.exists():
        raise FileNotFoundError(
            f"Dataset_B not found: {data_b_path}\n"
            "Run `python src\\phase3\\run_prepare_dataset_b.py --config config\\phase3.yaml` first."
        )
    return load_dataset(data_b_path)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2) + 1e-12
    return float(1.0 - ss_res / ss_tot)


def metrics_dict(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {"rmse": rmse(y_true, y_pred), "mae": mae(y_true, y_pred), "r2": r2(y_true, y_pred)}


def method_dir(cfg: dict, method_key: str) -> Path:
    out = resolve_project_path(cfg["phase3"]["results_dir"]) / method_key
    out.mkdir(parents=True, exist_ok=True)
    return out


def save_method_outputs(cfg: dict, method_key: str, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    out = method_dir(cfg, method_key)
    metrics = metrics_dict(y_true, y_pred)
    np.savez(out / "predictions.npz", y_true=y_true, y_pred=y_pred)
    with open(out / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    return metrics

