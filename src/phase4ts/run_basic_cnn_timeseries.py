from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from phase3.common import metrics_dict, set_seed
from phase3.models import build_model
from phase3.train_utils import make_loaders, predict, train_model
from phase4a.common import load_config, resolve_project_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run basic CNN on current phase4ts dataset")
    p.add_argument("--config", type=str, default="config/phase4_array_se_hard_ts.yaml")
    p.add_argument("--input-mode", type=str, default="local", choices=["local", "dual"])
    p.add_argument("--model-key", type=str, default="cnn_baseline", choices=["cnn_baseline", "cnn_se"])
    p.add_argument("--run-tag", type=str, default="basic_local_cnn")
    return p.parse_args()


def _p95(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.percentile(np.abs(y_true - y_pred), 95))


def _p99(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.percentile(np.abs(y_true - y_pred), 99))


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    seed = int(cfg["phase4ts"]["seed"])
    set_seed(seed + 909)

    data_path = resolve_project_path(cfg["phase4ts"]["dataset_path"])
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset missing: {data_path}")
    d = np.load(data_path)
    need = ["X_local", "X_total", "Y_dlambda_target", "idx_train", "idx_val", "idx_test"]
    miss = [k for k in need if k not in d]
    if miss:
        raise KeyError(f"Dataset missing keys: {miss}")

    x_local = d["X_local"].astype(np.float32)
    x_total = d["X_total"].astype(np.float32)
    y = d["Y_dlambda_target"].astype(np.float32)
    idx_train = d["idx_train"].astype(np.int64)
    idx_val = d["idx_val"].astype(np.int64)
    idx_test = d["idx_test"].astype(np.int64)

    if args.input_mode == "dual":
        x = np.concatenate([x_local, x_total], axis=1).astype(np.float32)
    else:
        x = x_local

    tr_cfg = cfg.get("train_single", cfg.get("train", {}))
    batch_size = int(tr_cfg.get("batch_size", 256))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader = make_loaders(x, y, idx_train, idx_val, batch_size=batch_size)
    model = build_model(args.model_key, input_dim=x.shape[1]).to(device)
    model = train_model(model, train_loader, val_loader, tr_cfg, device)

    y_true = y[idx_test]
    y_pred = predict(model, x, idx_test, device).astype(np.float32)
    m = metrics_dict(y_true, y_pred)
    p95 = _p95(y_true, y_pred)
    p99 = _p99(y_true, y_pred)

    out_dir = resolve_project_path(cfg["phase4ts"]["results_dir"]) / "basic_cnn" / str(args.run_tag)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "metrics_table.csv", "w", encoding="utf-8") as f:
        f.write("Method,Input,RMSE_nm,MAE_nm,P95_nm,P99_nm,R2\n")
        f.write(
            f"{args.model_key},{args.input_mode},{m['rmse']:.8f},{m['mae']:.8f},{p95:.8f},{p99:.8f},{m['r2']:.8f}\n"
        )
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "method": args.model_key,
                "input_mode": args.input_mode,
                "rmse": float(m["rmse"]),
                "mae": float(m["mae"]),
                "p95": float(p95),
                "p99": float(p99),
                "r2": float(m["r2"]),
            },
            f,
            indent=2,
        )
    np.savez(out_dir / "predictions.npz", y_true=y_true.astype(np.float32), y_pred=y_pred.astype(np.float32))

    print(
        f"{args.model_key} ({args.input_mode}) | RMSE={m['rmse']:.6f} | MAE={m['mae']:.6f} | "
        f"P95={p95:.6f} | P99={p99:.6f} | R2={m['r2']:.6f}"
    )
    print(f"Saved: {out_dir / 'metrics_table.csv'}")


if __name__ == "__main__":
    main()

