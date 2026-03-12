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
    p = argparse.ArgumentParser(description="Method enhancement experiment: CNN vs CNN+BiLSTM")
    p.add_argument("--config", type=str, default="config/phase4a_shift004_linewidth_l3_method_bilstm.yaml")
    p.add_argument("--epochs-override", type=int, default=None, help="Optional quick override for smoke tests.")
    p.add_argument("--eval-only", action="store_true", help="Skip training and evaluate from existing checkpoints.")
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
    model_key: str,
    model_kwargs: dict[str, int | bool],
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
    model = build_model(model_key, input_dim=x.shape[1], **model_kwargs).to(device)
    model = train_model(model, train_loader, val_loader, cfg_train, device)
    param_count = _count_params(model)

    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), ckpt_path)

    # Reload once to verify checkpoint compatibility for the new model path.
    reloaded = build_model(model_key, input_dim=x.shape[1], **model_kwargs).to(device)
    state = torch.load(ckpt_path, map_location=device)
    reloaded.load_state_dict(state)
    pred = predict(reloaded, x, idx_test, device).astype(np.float32)
    return pred, param_count


def load_predict(
    x: np.ndarray,
    idx_test: np.ndarray,
    model_key: str,
    model_kwargs: dict[str, int | bool],
    ckpt_path: Path,
) -> tuple[np.ndarray, int]:
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found for eval-only mode: {ckpt_path}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(model_key, input_dim=x.shape[1], **model_kwargs).to(device)
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
    model_cfg = cfg.get("model", {})
    if str(model_cfg.get("name", "cnn_bilstm")) != "cnn_bilstm":
        raise ValueError("This script only supports model.name=cnn_bilstm")

    bilstm_kwargs = {
        "bilstm_hidden": int(model_cfg.get("bilstm_hidden", 64)),
        "bilstm_layers": int(model_cfg.get("bilstm_layers", 1)),
        "bidirectional": bool(model_cfg.get("bidirectional", True)),
    }

    seed_base = int(cfg["phase4a"]["seed"])
    seed_cnn = seed_base + int(mcfg.get("seed_cnn_offset", 21))
    seed_bilstm = seed_base + int(mcfg.get("seed_bilstm_offset", 41))

    out_dir = resolve_project_path(cfg["phase4a"]["results_dir"]) / str(mcfg.get("results_subdir", "method_enhance_cnn_vs_bilstm"))
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_cnn = out_dir / "model_cnn.pt"
    ckpt_bilstm = out_dir / "model_cnn_bilstm.pt"

    if args.eval_only:
        pred_cnn, cnn_params = load_predict(
            x=x,
            idx_test=idx_test,
            model_key="cnn_baseline",
            model_kwargs={},
            ckpt_path=ckpt_cnn,
        )
        pred_bilstm, bilstm_params = load_predict(
            x=x,
            idx_test=idx_test,
            model_key="cnn_bilstm",
            model_kwargs=bilstm_kwargs,
            ckpt_path=ckpt_bilstm,
        )
    else:
        pred_cnn, cnn_params = train_save_reload_predict(
            x=x,
            y=y,
            idx_train=idx_train,
            idx_val=idx_val,
            idx_test=idx_test,
            cfg_train=train_cfg,
            seed=seed_cnn,
            model_key="cnn_baseline",
            model_kwargs={},
            ckpt_path=ckpt_cnn,
        )
        pred_bilstm, bilstm_params = train_save_reload_predict(
            x=x,
            y=y,
            idx_train=idx_train,
            idx_val=idx_val,
            idx_test=idx_test,
            cfg_train=train_cfg,
            seed=seed_bilstm,
            model_key="cnn_bilstm",
            model_kwargs=bilstm_kwargs,
            ckpt_path=ckpt_bilstm,
        )

    row_cnn = _metric_row("CNN", y_true, pred_cnn)
    row_bilstm = _metric_row("CNN+BiLSTM", y_true, pred_bilstm)
    rows = [row_cnn, row_bilstm]

    with open(out_dir / "metrics_table.csv", "w", encoding="utf-8") as f:
        f.write("Method,RMSE_nm,MAE_nm,P95_nm,P99_nm,R2\n")
        for r in rows:
            f.write(
                f"{r['Method']},{r['RMSE_nm']:.8f},{r['MAE_nm']:.8f},{r['P95_nm']:.8f},{r['P99_nm']:.8f},{r['R2']:.8f}\n"
            )

    rmse_gain_pct = 100.0 * (row_cnn["RMSE_nm"] - row_bilstm["RMSE_nm"]) / (row_cnn["RMSE_nm"] + 1e-12)
    mae_gain_pct = 100.0 * (row_cnn["MAE_nm"] - row_bilstm["MAE_nm"]) / (row_cnn["MAE_nm"] + 1e-12)
    p95_gain_pct = 100.0 * (row_cnn["P95_nm"] - row_bilstm["P95_nm"]) / (row_cnn["P95_nm"] + 1e-12)
    p99_gain_pct = 100.0 * (row_cnn["P99_nm"] - row_bilstm["P99_nm"]) / (row_cnn["P99_nm"] + 1e-12)

    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "cnn": row_cnn,
                "cnn_bilstm": row_bilstm,
                "gain_percent_vs_cnn": {
                    "rmse": rmse_gain_pct,
                    "mae": mae_gain_pct,
                    "p95": p95_gain_pct,
                    "p99": p99_gain_pct,
                },
                "model_info": {
                    "cnn_params": cnn_params,
                    "cnn_bilstm_params": bilstm_params,
                    "bilstm_config": bilstm_kwargs,
                },
                "checkpoints": {
                    "cnn": str(ckpt_cnn),
                    "cnn_bilstm": str(ckpt_bilstm),
                },
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
        pred_cnn=pred_cnn.astype(np.float32),
        pred_cnn_bilstm=pred_bilstm.astype(np.float32),
    )

    print(
        f"CNN          RMSE={row_cnn['RMSE_nm']:.6f} | MAE={row_cnn['MAE_nm']:.6f} | "
        f"P95={row_cnn['P95_nm']:.6f} | P99={row_cnn['P99_nm']:.6f} | R2={row_cnn['R2']:.6f}"
    )
    print(
        f"CNN+BiLSTM   RMSE={row_bilstm['RMSE_nm']:.6f} | MAE={row_bilstm['MAE_nm']:.6f} | "
        f"P95={row_bilstm['P95_nm']:.6f} | P99={row_bilstm['P99_nm']:.6f} | R2={row_bilstm['R2']:.6f}"
    )
    print(
        f"Gain vs CNN: RMSE={rmse_gain_pct:+.2f}% | MAE={mae_gain_pct:+.2f}% | "
        f"P95={p95_gain_pct:+.2f}% | P99={p99_gain_pct:+.2f}%"
    )
    print(f"Params: CNN={cnn_params} | CNN+BiLSTM={bilstm_params}")
    print(f"Saved: {out_dir / 'metrics_table.csv'}")
    print(f"Saved: {ckpt_cnn}")
    print(f"Saved: {ckpt_bilstm}")


if __name__ == "__main__":
    main()
