from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from phase3.common import metrics_dict, set_seed
from phase3.models import build_model
from phase3.train_utils import make_loaders, predict, train_model
from phase4a.common import load_config, resolve_project_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Hardest-case CNN training with configurable tail-aware loss/hard weighting")
    p.add_argument("--config", type=str, default="config/phase4a_shift004_linewidth_l3_method_tailaware.yaml")
    p.add_argument("--epochs-override", type=int, default=None, help="Optional quick override for smoke tests.")
    p.add_argument("--eval-only", action="store_true", help="Skip training and evaluate from existing checkpoint.")
    return p.parse_args()


def _p95(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.percentile(np.abs(y_true - y_pred), 95))


def _p99(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.percentile(np.abs(y_true - y_pred), 99))


def _metric_row(name: str, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    m = metrics_dict(y_true, y_pred)
    return {
        "Method": name,
        "RMSE_nm": float(m["rmse"]),
        "MAE_nm": float(m["mae"]),
        "P95_nm": _p95(y_true, y_pred),
        "P99_nm": _p99(y_true, y_pred),
        "R2": float(m["r2"]),
    }


def _count_params(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters()))


def train_save_reload_predict(
    x: np.ndarray,
    y: np.ndarray,
    idx_train: np.ndarray,
    idx_val: np.ndarray,
    idx_test: np.ndarray,
    cfg_train: dict,
    seed: int,
    ckpt_path: Path,
) -> tuple[np.ndarray, int]:
    set_seed(seed)
    train_loader, val_loader = make_loaders(
        x,
        y,
        idx_train,
        idx_val,
        batch_size=int(cfg_train["batch_size"]),
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model("cnn_baseline", input_dim=x.shape[1]).to(device)
    model = train_model(model, train_loader, val_loader, cfg_train, device)
    param_count = _count_params(model)

    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), ckpt_path)

    # Reload once to verify checkpoint compatibility.
    reloaded = build_model("cnn_baseline", input_dim=x.shape[1]).to(device)
    state = torch.load(ckpt_path, map_location=device)
    reloaded.load_state_dict(state)
    pred = predict(reloaded, x, idx_test, device).astype(np.float32)
    return pred, param_count


def load_predict(x: np.ndarray, idx_test: np.ndarray, ckpt_path: Path) -> tuple[np.ndarray, int]:
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found for eval-only mode: {ckpt_path}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model("cnn_baseline", input_dim=x.shape[1]).to(device)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    pred = predict(model, x, idx_test, device).astype(np.float32)
    return pred, _count_params(model)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    data_path = resolve_project_path(cfg["phase4a"]["dataset_path"])
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")
    d = np.load(data_path)
    needed = ["X_local", "Y_dlambda_target", "idx_train", "idx_val", "idx_test"]
    miss = [k for k in needed if k not in d]
    if miss:
        raise KeyError(f"Dataset missing keys: {miss}")

    x = d["X_local"].astype(np.float32)
    y = d["Y_dlambda_target"].astype(np.float32)
    idx_train = d["idx_train"].astype(np.int64)
    idx_val = d["idx_val"].astype(np.int64)
    idx_test = d["idx_test"].astype(np.int64)
    y_true = y[idx_test].astype(np.float32)

    train_cfg = dict(cfg["train"])
    if args.epochs_override is not None:
        train_cfg["epochs"] = int(args.epochs_override)
        train_cfg["patience"] = max(1, min(int(train_cfg["patience"]), int(args.epochs_override)))

    mcfg = cfg.get("method_enhance", {})
    method_name = str(mcfg.get("method_label", "CNN"))
    seed = int(cfg["phase4a"]["seed"]) + int(mcfg.get("seed_cnn_offset", 21))

    out_dir = resolve_project_path(cfg["phase4a"]["results_dir"]) / str(mcfg.get("results_subdir", "method_enhance_tailaware"))
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = out_dir / "model_cnn.pt"

    if args.eval_only:
        pred, params = load_predict(x=x, idx_test=idx_test, ckpt_path=ckpt)
    else:
        pred, params = train_save_reload_predict(
            x=x,
            y=y,
            idx_train=idx_train,
            idx_val=idx_val,
            idx_test=idx_test,
            cfg_train=train_cfg,
            seed=seed,
            ckpt_path=ckpt,
        )

    row = _metric_row(method_name, y_true, pred)

    with open(out_dir / "metrics_table.csv", "w", encoding="utf-8") as f:
        f.write("Method,RMSE_nm,MAE_nm,P95_nm,P99_nm,R2\n")
        f.write(
            f"{row['Method']},{row['RMSE_nm']:.8f},{row['MAE_nm']:.8f},"
            f"{row['P95_nm']:.8f},{row['P99_nm']:.8f},{row['R2']:.8f}\n"
        )

    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "metrics": row,
                "train_config": train_cfg,
                "params": params,
                "checkpoint": str(ckpt),
                "config": str(args.config),
                "epochs_override": args.epochs_override,
                "eval_only": bool(args.eval_only),
            },
            f,
            indent=2,
        )

    np.savez(
        out_dir / "predictions.npz",
        y_true=y_true,
        pred_cnn=pred.astype(np.float32),
    )

    print(
        f"{row['Method']:<14} RMSE={row['RMSE_nm']:.6f} | MAE={row['MAE_nm']:.6f} | "
        f"P95={row['P95_nm']:.6f} | P99={row['P99_nm']:.6f} | R2={row['R2']:.6f}"
    )
    print(f"Params: CNN={params}")
    print(f"Saved: {out_dir / 'metrics_table.csv'}")
    print(f"Saved: {ckpt}")


if __name__ == "__main__":
    main()
