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
from phase3.train_utils import make_loaders, predict, train_model
from phase4a.common import load_config, resolve_project_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Method enhancement experiment: CNN vs lightweight CNN+SE")
    p.add_argument("--config", type=str, default="config/phase4a_shift004_linewidth_l3_method_se.yaml")
    return p.parse_args()


class SEBlock1D(nn.Module):
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        hidden = max(4, channels // reduction)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _ = x.shape
        w = self.pool(x).view(b, c)
        w = self.fc(w).view(b, c, 1)
        return x * w


class LightSEConvRegressor(nn.Module):
    """
    Lightweight SE variant:
    - Same baseline backbone
    - Only one SE block after high-level conv block
    """

    def __init__(self, input_dim: int, use_light_se: bool, se_reduction: int = 8):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )
        self.block2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )
        self.block3 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.ReLU(),
        )
        self.se = SEBlock1D(128, reduction=se_reduction) if use_light_se else nn.Identity()

        with torch.no_grad():
            dummy = torch.zeros(1, 1, input_dim)
            feat = self._forward_features(dummy)
            feat_dim = int(np.prod(feat.shape[1:]))

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feat_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def _forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.se(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)
        x = self._forward_features(x)
        return self.head(x).squeeze(-1)


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


def train_eval(
    x: np.ndarray,
    y: np.ndarray,
    idx_train: np.ndarray,
    idx_val: np.ndarray,
    idx_test: np.ndarray,
    cfg_train: dict,
    seed: int,
    use_light_se: bool,
    se_reduction: int,
) -> tuple[np.ndarray, dict]:
    set_seed(seed)
    train_loader, val_loader = make_loaders(
        x,
        y,
        idx_train,
        idx_val,
        batch_size=int(cfg_train["batch_size"]),
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LightSEConvRegressor(
        input_dim=x.shape[1],
        use_light_se=use_light_se,
        se_reduction=se_reduction,
    ).to(device)
    model = train_model(model, train_loader, val_loader, cfg_train, device)
    pred = predict(model, x, idx_test, device).astype(np.float32)
    y_true = y[idx_test].astype(np.float32)
    metrics = _metric_row("CNN+LightSE" if use_light_se else "CNN", y_true, pred)
    return pred, metrics


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

    mcfg = cfg.get("method_enhance", {})
    se_reduction = int(mcfg.get("se_reduction", 8))
    seed_base = int(cfg["phase4a"]["seed"])
    seed_cnn = seed_base + int(mcfg.get("seed_cnn_offset", 21))
    seed_se = seed_base + int(mcfg.get("seed_lightse_offset", 31))

    pred_cnn, met_cnn = train_eval(
        x=x,
        y=y,
        idx_train=idx_train,
        idx_val=idx_val,
        idx_test=idx_test,
        cfg_train=cfg["train"],
        seed=seed_cnn,
        use_light_se=False,
        se_reduction=se_reduction,
    )
    pred_se, met_se = train_eval(
        x=x,
        y=y,
        idx_train=idx_train,
        idx_val=idx_val,
        idx_test=idx_test,
        cfg_train=cfg["train"],
        seed=seed_se,
        use_light_se=True,
        se_reduction=se_reduction,
    )

    y_true = y[idx_test].astype(np.float32)
    rows = [met_cnn, met_se]

    out_dir = resolve_project_path(cfg["phase4a"]["results_dir"]) / str(mcfg.get("results_subdir", "method_enhance_cnn_vs_lightse"))
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "metrics_table.csv", "w", encoding="utf-8") as f:
        f.write("Method,RMSE_nm,MAE_nm,P95_nm,P99_nm,R2\n")
        for r in rows:
            f.write(
                f"{r['Method']},{r['RMSE_nm']:.8f},{r['MAE_nm']:.8f},{r['P95_nm']:.8f},{r['P99_nm']:.8f},{r['R2']:.8f}\n"
            )

    rmse_gain_pct = 100.0 * (met_cnn["RMSE_nm"] - met_se["RMSE_nm"]) / (met_cnn["RMSE_nm"] + 1e-12)
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "cnn": met_cnn,
                "cnn_lightse": met_se,
                "rmse_gain_percent_vs_cnn": rmse_gain_pct,
                "config": str(args.config),
            },
            f,
            indent=2,
        )

    np.savez(
        out_dir / "predictions.npz",
        y_true=y_true,
        pred_cnn=pred_cnn.astype(np.float32),
        pred_cnn_lightse=pred_se.astype(np.float32),
    )

    print(
        f"CNN          RMSE={met_cnn['RMSE_nm']:.6f} | MAE={met_cnn['MAE_nm']:.6f} | "
        f"P95={met_cnn['P95_nm']:.6f} | P99={met_cnn['P99_nm']:.6f} | R2={met_cnn['R2']:.6f}"
    )
    print(
        f"CNN+LightSE  RMSE={met_se['RMSE_nm']:.6f} | MAE={met_se['MAE_nm']:.6f} | "
        f"P95={met_se['P95_nm']:.6f} | P99={met_se['P99_nm']:.6f} | R2={met_se['R2']:.6f}"
    )
    print(f"RMSE gain vs CNN: {rmse_gain_pct:+.2f}%")
    print(f"Saved: {out_dir / 'metrics_table.csv'}")


if __name__ == "__main__":
    main()

