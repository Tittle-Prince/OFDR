from __future__ import annotations

import argparse
import csv
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
    p = argparse.ArgumentParser(description="Method enhancement experiment: CNN + Coarse-to-Fine residual refine V2")
    p.add_argument("--config", type=str, default="config/phase4a_shift004_linewidth_l3_method_c2f_v2.yaml")
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


def _count_params(model: nn.Module, trainable_only: bool = False) -> int:
    params = model.parameters()
    if trainable_only:
        return int(sum(p.numel() for p in params if p.requires_grad))
    return int(sum(p.numel() for p in params))


class RefineHead(nn.Module):
    def __init__(self, c1: int, c2: int, hidden: int):
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


class CoarseToFineRefineV2(nn.Module):
    """
    Stable plug-in C2F-V2:
    - coarse predictor: frozen baseline CNN (not a separately trained coarse head)
    - refinement: local residual prediction around coarse-centered window
    - residual clip: delta = clip * tanh(raw_delta)
    """

    def __init__(
        self,
        coarse_model: nn.Module,
        wl_start_nm: float,
        wl_step_nm: float,
        lambda0_nm: float,
        refine_window_points: int,
        residual_clip_nm: float,
        c1: int,
        c2: int,
        hidden: int,
    ):
        super().__init__()
        self.coarse = coarse_model
        for p in self.coarse.parameters():
            p.requires_grad = False
        self.refine = RefineHead(c1=c1, c2=c2, hidden=hidden)
        self.wl_start_nm = float(wl_start_nm)
        self.wl_step_nm = float(wl_step_nm)
        self.lambda0_nm = float(lambda0_nm)
        self.refine_window_points = int(refine_window_points)
        self.residual_clip_nm = float(residual_clip_nm)

    def _extract_refine_window(self, x: torch.Tensor, center_idx: torch.Tensor) -> torch.Tensor:
        # x: [B, N], center_idx: [B]
        _, n = x.shape
        w = int(self.refine_window_points)
        half = w // 2
        offsets = torch.arange(w, device=x.device, dtype=torch.long) - half
        idx = center_idx[:, None] + offsets[None, :]
        idx = torch.clamp(idx, 0, n - 1)
        x_win = torch.gather(x, 1, idx)
        return x_win

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Keep coarse predictor in eval mode for deterministic behavior.
        self.coarse.eval()
        with torch.no_grad():
            coarse = self.coarse(x)
        coarse_wl = self.lambda0_nm + coarse
        coarse_idx = torch.round((coarse_wl - self.wl_start_nm) / self.wl_step_nm).long()
        x_win = self._extract_refine_window(x, coarse_idx)
        raw_delta = self.refine(x_win)
        if self.residual_clip_nm > 0:
            delta = self.residual_clip_nm * torch.tanh(raw_delta)
        else:
            delta = raw_delta
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
    train_loader, val_loader = make_loaders(
        x=x,
        y=y,
        idx_train=idx_train,
        idx_val=idx_val,
        batch_size=int(cfg_train["batch_size"]),
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model("cnn_baseline", input_dim=x.shape[1]).to(device)
    model = train_model(model, train_loader, val_loader, cfg_train, device)
    return model


def train_refine_v2(
    model: CoarseToFineRefineV2,
    x: np.ndarray,
    y: np.ndarray,
    idx_train: np.ndarray,
    idx_val: np.ndarray,
    cfg_train: dict,
    seed: int,
) -> CoarseToFineRefineV2:
    set_seed(seed)
    train_loader, val_loader = make_loaders(
        x=x,
        y=y,
        idx_train=idx_train,
        idx_val=idx_val,
        batch_size=int(cfg_train["batch_size"]),
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    # Train only the plug-in refine branch.
    opt = torch.optim.Adam(
        model.refine.parameters(),
        lr=float(cfg_train["lr"]),
        weight_decay=float(cfg_train["weight_decay"]),
    )
    loss_fn = nn.MSELoss()

    best_val = float("inf")
    best_state = None
    stale = 0
    epochs = int(cfg_train["epochs"])
    patience = int(cfg_train["patience"])

    for ep in range(1, epochs + 1):
        model.train()
        model.coarse.eval()
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad()
            _, _, final = model(xb)
            loss = loss_fn(final, yb)
            loss.backward()
            opt.step()

        model.eval()
        vals = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                _, _, final = model(xb)
                vals.append(float(loss_fn(final, yb).item()))
        val_mse = float(np.mean(vals))
        if ep % 5 == 0 or ep == 1:
            print(f"[RefineV2] Epoch {ep:03d} | val_mse={val_mse:.6f}")

        if val_mse < best_val:
            best_val = val_mse
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                print(f"[RefineV2] Early stop at epoch {ep}, best val_mse={best_val:.6f}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def predict_c2f_v2(model: CoarseToFineRefineV2, x: np.ndarray, idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    xt = torch.tensor(x[idx], dtype=torch.float32, device=device)
    model.eval()
    with torch.no_grad():
        coarse, _, final = model(xt)
    return coarse.cpu().numpy().astype(np.float32), final.cpu().numpy().astype(np.float32)


def _maybe_load_old_c2f_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            try:
                rows.append(
                    {
                        "Method": f"Old-{r['Method']}",
                        "RMSE_nm": float(r["RMSE_nm"]),
                        "MAE_nm": float(r["MAE_nm"]),
                        "P95_nm": float(r["P95_nm"]),
                        "P99_nm": float(r["P99_nm"]),
                        "R2": float(r["R2"]),
                    }
                )
            except (KeyError, ValueError):
                continue
    return rows


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
    y_true = y[idx_test].astype(np.float32)

    array_cfg = cfg.get("array", {})
    wl_start = float(array_cfg.get("wavelength_start_nm", 1549.0))
    wl_end = float(array_cfg.get("wavelength_end_nm", 1551.0))
    n_points = int(array_cfg.get("num_points", x.shape[1]))
    wl_step = (wl_end - wl_start) / float(max(1, n_points - 1))
    lambda0 = float(array_cfg.get("lambda0_nm", 1550.0))

    model_cfg = cfg.get("model", {})
    mcfg = cfg.get("method_c2f_v2", {})
    use_baseline_as_coarse = bool(model_cfg.get("use_baseline_cnn_as_coarse", True))
    if not use_baseline_as_coarse:
        raise ValueError("This V2 script requires model.use_baseline_cnn_as_coarse=true")
    refine_window = int(model_cfg.get("refine_window_size", 128))
    residual_clip_nm = float(model_cfg.get("residual_clip_nm", 0.03))

    seed_base = int(cfg["phase4a"]["seed"])
    seed_cnn = seed_base + int(mcfg.get("seed_cnn_offset", 21))
    seed_refine = seed_base + int(mcfg.get("seed_refine_offset", 51))

    # Stage-1: baseline CNN as stable coarse predictor.
    cnn = train_baseline(
        x=x,
        y=y,
        idx_train=idx_train,
        idx_val=idx_val,
        cfg_train=cfg["train"],
        seed=seed_cnn,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cnn = cnn.to(device)
    pred_cnn = predict(cnn, x, idx_test, device).astype(np.float32)

    # Stage-2: plug-in residual refine branch.
    c2f_v2 = CoarseToFineRefineV2(
        coarse_model=cnn,
        wl_start_nm=wl_start,
        wl_step_nm=wl_step,
        lambda0_nm=lambda0,
        refine_window_points=refine_window,
        residual_clip_nm=residual_clip_nm,
        c1=int(mcfg.get("refine_channels_1", 16)),
        c2=int(mcfg.get("refine_channels_2", 32)),
        hidden=int(mcfg.get("refine_hidden", 32)),
    )
    c2f_v2 = train_refine_v2(
        model=c2f_v2,
        x=x,
        y=y,
        idx_train=idx_train,
        idx_val=idx_val,
        cfg_train=cfg["train"],
        seed=seed_refine,
    )
    pred_c2f_coarse, pred_c2f_final = predict_c2f_v2(c2f_v2, x, idx_test)

    row_cnn = _metric_row("CNN", y_true, pred_cnn)
    row_v2_coarse = _metric_row("C2Fv2-Coarse(BaselineCNN)", y_true, pred_c2f_coarse)
    row_v2_final = _metric_row("CNN+C2FRefineV2", y_true, pred_c2f_final)
    rows = [row_cnn, row_v2_coarse, row_v2_final]

    if bool(mcfg.get("compare_with_old_c2f", False)):
        old_path = resolve_project_path(str(mcfg.get("old_c2f_metrics_table", "")))
        rows.extend(_maybe_load_old_c2f_rows(old_path))

    out_dir = resolve_project_path(cfg["phase4a"]["results_dir"]) / str(mcfg.get("results_subdir", "method_enhance_c2f_refine_v2"))
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "metrics_table.csv", "w", encoding="utf-8") as f:
        f.write("Method,RMSE_nm,MAE_nm,P95_nm,P99_nm,R2\n")
        for r in rows:
            f.write(
                f"{r['Method']},{r['RMSE_nm']:.8f},{r['MAE_nm']:.8f},{r['P95_nm']:.8f},{r['P99_nm']:.8f},{r['R2']:.8f}\n"
            )

    rmse_gain = 100.0 * (row_cnn["RMSE_nm"] - row_v2_final["RMSE_nm"]) / (row_cnn["RMSE_nm"] + 1e-12)
    p95_gain = 100.0 * (row_cnn["P95_nm"] - row_v2_final["P95_nm"]) / (row_cnn["P95_nm"] + 1e-12)
    p99_gain = 100.0 * (row_cnn["P99_nm"] - row_v2_final["P99_nm"]) / (row_cnn["P99_nm"] + 1e-12)

    model_info = {
        "coarse_model_total_params": _count_params(cnn, trainable_only=False),
        "coarse_model_trainable_params": _count_params(cnn, trainable_only=True),
        "refine_model_total_params": _count_params(c2f_v2.refine, trainable_only=False),
        "refine_model_trainable_params": _count_params(c2f_v2.refine, trainable_only=True),
        "combined_total_params": _count_params(c2f_v2, trainable_only=False),
        "combined_trainable_params": _count_params(c2f_v2, trainable_only=True),
        "refine_window_size": refine_window,
        "residual_clip_nm": residual_clip_nm,
    }

    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "cnn": row_cnn,
                "c2f_v2_coarse": row_v2_coarse,
                "c2f_v2_final": row_v2_final,
                "gain_percent_vs_cnn": {
                    "rmse": rmse_gain,
                    "p95": p95_gain,
                    "p99": p99_gain,
                },
                "model_info": model_info,
                "config": str(args.config),
            },
            f,
            indent=2,
        )

    np.savez(
        out_dir / "predictions.npz",
        y_true=y_true.astype(np.float32),
        pred_cnn=pred_cnn.astype(np.float32),
        pred_c2f_v2_coarse=pred_c2f_coarse.astype(np.float32),
        pred_c2f_v2_final=pred_c2f_final.astype(np.float32),
    )

    print(
        f"CNN                       RMSE={row_cnn['RMSE_nm']:.6f} | MAE={row_cnn['MAE_nm']:.6f} | "
        f"P95={row_cnn['P95_nm']:.6f} | P99={row_cnn['P99_nm']:.6f} | R2={row_cnn['R2']:.6f}"
    )
    print(
        f"C2Fv2-Coarse(BaselineCNN) RMSE={row_v2_coarse['RMSE_nm']:.6f} | MAE={row_v2_coarse['MAE_nm']:.6f} | "
        f"P95={row_v2_coarse['P95_nm']:.6f} | P99={row_v2_coarse['P99_nm']:.6f} | R2={row_v2_coarse['R2']:.6f}"
    )
    print(
        f"CNN+C2FRefineV2           RMSE={row_v2_final['RMSE_nm']:.6f} | MAE={row_v2_final['MAE_nm']:.6f} | "
        f"P95={row_v2_final['P95_nm']:.6f} | P99={row_v2_final['P99_nm']:.6f} | R2={row_v2_final['R2']:.6f}"
    )
    print(f"Gain vs CNN: RMSE={rmse_gain:+.2f}% | P95={p95_gain:+.2f}% | P99={p99_gain:+.2f}%")
    print(
        f"Params: coarse={model_info['coarse_model_total_params']} | refine={model_info['refine_model_total_params']} | "
        f"trainable(combined)={model_info['combined_trainable_params']}"
    )
    print(f"Refine window={refine_window} | residual_clip_nm={residual_clip_nm:.4f}")
    print(f"Saved: {out_dir / 'metrics_table.csv'}")


if __name__ == "__main__":
    main()

