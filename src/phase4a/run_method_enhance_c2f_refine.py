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
from phase3.train_utils import make_loaders, predict
from phase4a.common import load_config, resolve_project_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare CNN vs Coarse-to-Fine residual refinement module")
    p.add_argument("--config", type=str, default="config/phase4a_shift004_linewidth_l3_method_c2f.yaml")
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


class RefineHead(nn.Module):
    def __init__(self, window_points: int, c1: int, c2: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, c1, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(c1, c2, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(c2, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x_win: torch.Tensor) -> torch.Tensor:
        # x_win: [B, W]
        return self.net(x_win.unsqueeze(1)).squeeze(-1)


class CoarseToFineResidualRefiner(nn.Module):
    """
    Coarse-to-fine "外挂式"模块:
    1) coarse CNN 给出粗预测
    2) 以 coarse 预测中心从输入谱截取更窄 refinement window
    3) 轻量 refine head 预测残差
    4) final = coarse + residual
    """

    def __init__(
        self,
        input_dim: int,
        wl_start_nm: float,
        wl_step_nm: float,
        lambda0_nm: float,
        refine_window_points: int,
        c1: int,
        c2: int,
        hidden: int,
    ):
        super().__init__()
        self.coarse = build_model("cnn_baseline", input_dim=input_dim)
        self.refine = RefineHead(refine_window_points, c1=c1, c2=c2, hidden=hidden)
        self.input_dim = int(input_dim)
        self.wl_start_nm = float(wl_start_nm)
        self.wl_step_nm = float(wl_step_nm)
        self.lambda0_nm = float(lambda0_nm)
        self.refine_window_points = int(refine_window_points)

    def _extract_refine_window(self, x: torch.Tensor, center_idx: torch.Tensor) -> torch.Tensor:
        # x: [B, N], center_idx: [B]
        b, n = x.shape
        w = int(self.refine_window_points)
        half = w // 2
        offsets = torch.arange(w, device=x.device, dtype=torch.long) - half
        idx = center_idx[:, None] + offsets[None, :]
        idx = torch.clamp(idx, 0, n - 1)
        x_win = torch.gather(x, 1, idx)
        return x_win

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        coarse = self.coarse(x)
        # coarse 是 delta_lambda，先变回 wavelength 再映射到输入索引
        coarse_wl = self.lambda0_nm + coarse
        coarse_idx = torch.round((coarse_wl - self.wl_start_nm) / self.wl_step_nm).long()
        x_win = self._extract_refine_window(x, coarse_idx)
        delta = self.refine(x_win)
        final = coarse + delta
        return coarse, delta, final


def train_baseline(
    x: np.ndarray,
    y: np.ndarray,
    idx_train: np.ndarray,
    idx_val: np.ndarray,
    cfg_train: dict,
    seed: int,
) -> nn.Module:
    set_seed(seed)
    train_loader, val_loader = make_loaders(x, y, idx_train, idx_val, batch_size=int(cfg_train["batch_size"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model("cnn_baseline", input_dim=x.shape[1]).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=float(cfg_train["lr"]), weight_decay=float(cfg_train["weight_decay"]))
    loss_fn = nn.MSELoss()
    best_val = float("inf")
    best_state = None
    stale = 0
    epochs = int(cfg_train["epochs"])
    patience = int(cfg_train["patience"])

    for ep in range(1, epochs + 1):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()

        model.eval()
        vals = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                pv = model(xb)
                vals.append(float(loss_fn(pv, yb).item()))
        vm = float(np.mean(vals))
        if ep % 5 == 0 or ep == 1:
            print(f"[CNN] Epoch {ep:03d} | val_mse={vm:.6f}")
        if vm < best_val:
            best_val = vm
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                print(f"[CNN] Early stop at epoch {ep}, best val_mse={best_val:.6f}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def train_c2f(
    model: CoarseToFineResidualRefiner,
    x: np.ndarray,
    y: np.ndarray,
    idx_train: np.ndarray,
    idx_val: np.ndarray,
    cfg_train: dict,
    alpha_coarse: float,
    seed: int,
) -> CoarseToFineResidualRefiner:
    set_seed(seed)
    train_loader, val_loader = make_loaders(x, y, idx_train, idx_val, batch_size=int(cfg_train["batch_size"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    opt = torch.optim.Adam(model.parameters(), lr=float(cfg_train["lr"]), weight_decay=float(cfg_train["weight_decay"]))
    loss_fn = nn.MSELoss()
    best_val = float("inf")
    best_state = None
    stale = 0
    epochs = int(cfg_train["epochs"])
    patience = int(cfg_train["patience"])

    for ep in range(1, epochs + 1):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad()
            coarse, _, final = model(xb)
            loss_final = loss_fn(final, yb)
            loss_coarse = loss_fn(coarse, yb)
            loss = loss_final + float(alpha_coarse) * loss_coarse
            loss.backward()
            opt.step()

        model.eval()
        vals = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                _, _, pf = model(xb)
                vals.append(float(loss_fn(pf, yb).item()))
        vm = float(np.mean(vals))
        if ep % 5 == 0 or ep == 1:
            print(f"[C2F] Epoch {ep:03d} | val_mse={vm:.6f}")
        if vm < best_val:
            best_val = vm
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                print(f"[C2F] Early stop at epoch {ep}, best val_mse={best_val:.6f}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def predict_c2f(model: CoarseToFineResidualRefiner, x: np.ndarray, idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    xt = torch.tensor(x[idx], dtype=torch.float32, device=device)
    model.eval()
    with torch.no_grad():
        coarse, _, final = model(xt)
    return coarse.cpu().numpy().astype(np.float32), final.cpu().numpy().astype(np.float32)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    data_path = resolve_project_path(cfg["phase4a"]["dataset_path"])
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset missing: {data_path}")
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

    wl_start = float(cfg.get("array", {}).get("wavelength_start_nm", 1549.0))
    wl_end = float(cfg.get("array", {}).get("wavelength_end_nm", 1551.0))
    n_points = int(cfg.get("array", {}).get("num_points", x.shape[1]))
    wl_step = (wl_end - wl_start) / float(max(1, n_points - 1))
    lambda0 = float(cfg.get("array", {}).get("lambda0_nm", 1550.0))

    mcfg = cfg.get("method_c2f", {})
    seed_base = int(cfg["phase4a"]["seed"])
    seed_cnn = seed_base + int(mcfg.get("seed_cnn_offset", 21))
    seed_c2f = seed_base + int(mcfg.get("seed_c2f_offset", 41))

    # Stage-1 baseline CNN
    cnn = train_baseline(
        x=x,
        y=y,
        idx_train=idx_train,
        idx_val=idx_val,
        cfg_train=cfg["train"],
        seed=seed_cnn,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pred_cnn = predict(cnn.to(device), x, idx_test, device).astype(np.float32)
    y_true = y[idx_test].astype(np.float32)

    # Stage-1 + Stage-2 C2F refine
    c2f = CoarseToFineResidualRefiner(
        input_dim=x.shape[1],
        wl_start_nm=wl_start,
        wl_step_nm=wl_step,
        lambda0_nm=lambda0,
        refine_window_points=int(mcfg.get("refine_window_points", 96)),
        c1=int(mcfg.get("refine_channels_1", 16)),
        c2=int(mcfg.get("refine_channels_2", 32)),
        hidden=int(mcfg.get("refine_hidden", 32)),
    )
    c2f = train_c2f(
        model=c2f,
        x=x,
        y=y,
        idx_train=idx_train,
        idx_val=idx_val,
        cfg_train=cfg["train"],
        alpha_coarse=float(mcfg.get("alpha_coarse", 0.2)),
        seed=seed_c2f,
    )
    pred_coarse, pred_final = predict_c2f(c2f, x, idx_test)

    row_cnn = _metric_row("CNN", y_true, pred_cnn)
    row_c2f_coarse = _metric_row("C2F-Coarse", y_true, pred_coarse)
    row_c2f_final = _metric_row("CNN+C2FRefine", y_true, pred_final)
    rows = [row_cnn, row_c2f_coarse, row_c2f_final]

    out_dir = resolve_project_path(cfg["phase4a"]["results_dir"]) / str(mcfg.get("results_subdir", "method_enhance_c2f_refine"))
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "metrics_table.csv", "w", encoding="utf-8") as f:
        f.write("Method,RMSE_nm,MAE_nm,P95_nm,P99_nm,R2\n")
        for r in rows:
            f.write(
                f"{r['Method']},{r['RMSE_nm']:.8f},{r['MAE_nm']:.8f},{r['P95_nm']:.8f},{r['P99_nm']:.8f},{r['R2']:.8f}\n"
            )

    rmse_gain = 100.0 * (row_cnn["RMSE_nm"] - row_c2f_final["RMSE_nm"]) / (row_cnn["RMSE_nm"] + 1e-12)
    p95_gain = 100.0 * (row_cnn["P95_nm"] - row_c2f_final["P95_nm"]) / (row_cnn["P95_nm"] + 1e-12)
    p99_gain = 100.0 * (row_cnn["P99_nm"] - row_c2f_final["P99_nm"]) / (row_cnn["P99_nm"] + 1e-12)

    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "cnn": row_cnn,
                "c2f_coarse": row_c2f_coarse,
                "c2f_final": row_c2f_final,
                "gain_percent_vs_cnn": {
                    "rmse": rmse_gain,
                    "p95": p95_gain,
                    "p99": p99_gain,
                },
                "config": str(args.config),
            },
            f,
            indent=2,
        )

    np.savez(
        out_dir / "predictions.npz",
        y_true=y_true.astype(np.float32),
        pred_cnn=pred_cnn.astype(np.float32),
        pred_c2f_coarse=pred_coarse.astype(np.float32),
        pred_c2f_final=pred_final.astype(np.float32),
    )

    print(
        f"CNN           RMSE={row_cnn['RMSE_nm']:.6f} | MAE={row_cnn['MAE_nm']:.6f} | "
        f"P95={row_cnn['P95_nm']:.6f} | P99={row_cnn['P99_nm']:.6f}"
    )
    print(
        f"C2F-Coarse    RMSE={row_c2f_coarse['RMSE_nm']:.6f} | MAE={row_c2f_coarse['MAE_nm']:.6f} | "
        f"P95={row_c2f_coarse['P95_nm']:.6f} | P99={row_c2f_coarse['P99_nm']:.6f}"
    )
    print(
        f"CNN+C2FRefine RMSE={row_c2f_final['RMSE_nm']:.6f} | MAE={row_c2f_final['MAE_nm']:.6f} | "
        f"P95={row_c2f_final['P95_nm']:.6f} | P99={row_c2f_final['P99_nm']:.6f}"
    )
    print(f"Gain vs CNN: RMSE={rmse_gain:+.2f}% | P95={p95_gain:+.2f}% | P99={p99_gain:+.2f}%")
    print(f"Saved: {out_dir / 'metrics_table.csv'}")


if __name__ == "__main__":
    main()

