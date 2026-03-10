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
from phase3.train_utils import make_loaders, train_model
from phase4a.common import load_config, resolve_project_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Dual-input neural fusion: [X_local, X_total]")
    p.add_argument("--config", type=str, default="config/phase4_array_se_hard.yaml")
    return p.parse_args()


def load_dataset(cfg: dict) -> dict:
    data_path = resolve_project_path(cfg["phase4a"]["dataset_path"])
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset missing: {data_path}")
    d = np.load(data_path)
    needed = ["X_local", "X_total", "Y_dlambda_target", "idx_train", "idx_val", "idx_test"]
    miss = [k for k in needed if k not in d]
    if miss:
        raise KeyError(f"Dataset missing keys: {miss}")
    x_local = d["X_local"].astype(np.float32)
    x_total = d["X_total"].astype(np.float32)
    x_cat = np.concatenate([x_local, x_total], axis=1).astype(np.float32)
    return {
        "x": x_cat,
        "y": d["Y_dlambda_target"].astype(np.float32),
        "idx_train": d["idx_train"].astype(np.int64),
        "idx_val": d["idx_val"].astype(np.int64),
        "idx_test": d["idx_test"].astype(np.int64),
    }


def predict_idx(model: torch.nn.Module, x: np.ndarray, idx: np.ndarray, device: torch.device) -> np.ndarray:
    xt = torch.tensor(x[idx], dtype=torch.float32, device=device)
    model.eval()
    with torch.no_grad():
        pred = model(xt).cpu().numpy()
    return pred.astype(np.float32)


def rmse_to_required_frames(y_true: np.ndarray, y_pred: np.ndarray, target_rmse_nm: float) -> float | None:
    e = (y_pred - y_true).astype(np.float64)
    bias = float(np.mean(e))
    var = float(np.var(e))
    t2 = float(target_rmse_nm) ** 2
    if t2 <= bias * bias:
        return None
    return float(var / (t2 - bias * bias))


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    ds = load_dataset(cfg)

    set_seed(int(cfg["phase4a"]["seed"]) + 401)
    train_loader, val_loader = make_loaders(
        ds["x"],
        ds["y"],
        ds["idx_train"],
        ds["idx_val"],
        batch_size=int(cfg["train"]["batch_size"]),
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model("cnn_se", input_dim=ds["x"].shape[1]).to(device)
    model = train_model(model, train_loader, val_loader, cfg["train"], device)

    y_val = ds["y"][ds["idx_val"]]
    p_val = predict_idx(model, ds["x"], ds["idx_val"], device)
    bias_val = float(np.mean(y_val - p_val))

    y_test = ds["y"][ds["idx_test"]]
    p_test_raw = predict_idx(model, ds["x"], ds["idx_test"], device)
    p_test_bc = (p_test_raw + bias_val).astype(np.float32)

    m_raw = metrics_dict(y_test, p_test_raw)
    m_bc = metrics_dict(y_test, p_test_bc)

    need_raw = rmse_to_required_frames(y_test, p_test_raw, target_rmse_nm=0.007)
    need_bc = rmse_to_required_frames(y_test, p_test_bc, target_rmse_nm=0.007)

    out_dir = resolve_project_path(cfg["phase4a"]["results_dir"]) / "dualinput_fusion"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "metrics_table.csv", "w", encoding="utf-8") as f:
        f.write("Method,RMSE_nm,MAE_nm,R2\n")
        f.write(f"DualInput-CNNSE(raw),{m_raw['rmse']:.8f},{m_raw['mae']:.8f},{m_raw['r2']:.8f}\n")
        f.write(f"DualInput-CNNSE(bias-corrected),{m_bc['rmse']:.8f},{m_bc['mae']:.8f},{m_bc['r2']:.8f}\n")

    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "raw": m_raw,
                "bias_corrected": m_bc,
                "val_bias_nm": bias_val,
                "required_frames_for_0p007nm_rmse_raw": need_raw,
                "required_frames_for_0p007nm_rmse_bias_corrected": need_bc,
            },
            f,
            indent=2,
        )

    np.savez(
        out_dir / "predictions.npz",
        y_true=y_test.astype(np.float32),
        pred_raw=p_test_raw.astype(np.float32),
        pred_bias_corrected=p_test_bc.astype(np.float32),
        val_bias=np.array([bias_val], dtype=np.float32),
    )

    print(f"DualInput-CNNSE(raw)          RMSE={m_raw['rmse']:.6f} | MAE={m_raw['mae']:.6f} | R2={m_raw['r2']:.6f}")
    print(f"DualInput-CNNSE(bias-correct) RMSE={m_bc['rmse']:.6f} | MAE={m_bc['mae']:.6f} | R2={m_bc['r2']:.6f}")
    print(f"Val bias correction: {bias_val:+.6f} nm")
    print(f"Frames needed for 0.007nm RMSE (raw): {need_raw}")
    print(f"Frames needed for 0.007nm RMSE (bias-corrected): {need_bc}")
    print(f"Saved: {out_dir / 'metrics_table.csv'}")


if __name__ == "__main__":
    main()

