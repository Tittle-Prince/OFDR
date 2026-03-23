from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

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


def _tail_aware_per_sample_loss(pred: torch.Tensor, target: torch.Tensor, tau: float, lambda_tail: float) -> torch.Tensor:
    abs_err = torch.abs(pred - target)
    tail = torch.clamp(abs_err - float(tau), min=0.0)
    return abs_err + float(lambda_tail) * tail.pow(2)


def _percentile(arr: np.ndarray, q: float) -> float:
    return float(np.percentile(arr, q))


def _build_offline_hard_weights(
    coarse_pred: np.ndarray,
    y_true: np.ndarray,
    hard_cfg: dict,
) -> tuple[np.ndarray, dict]:
    """
    Build stable offline weights from frozen coarse-model errors instead of
    using noisy batch-wise online hard mining.
    """
    abs_err = np.abs(coarse_pred.astype(np.float32) - y_true.astype(np.float32))
    enabled = bool(hard_cfg.get("enabled", False))
    alpha = float(hard_cfg.get("alpha", 0.0))
    tau = hard_cfg.get("tau", None)
    top_frac = float(hard_cfg.get("top_fraction", 0.0))

    if not enabled or alpha <= 0.0:
        weights = np.ones_like(abs_err, dtype=np.float32)
        return weights, {
            "enabled": False,
            "alpha": alpha,
            "tau": None if tau is None else float(tau),
            "top_fraction": top_frac,
            "threshold_used": None,
            "hard_ratio": 0.0,
        }

    threshold = None
    hard_mask = np.zeros_like(abs_err, dtype=bool)

    if tau is not None:
        threshold = float(tau)
        hard_mask |= abs_err > threshold

    if top_frac > 0.0:
        q = max(0.0, min(100.0, 100.0 * (1.0 - top_frac)))
        q_thr = _percentile(abs_err, q)
        threshold = max(float(threshold), float(q_thr)) if threshold is not None else float(q_thr)
        hard_mask |= abs_err >= q_thr

    if threshold is None:
        weights = np.ones_like(abs_err, dtype=np.float32)
        return weights, {
            "enabled": False,
            "alpha": alpha,
            "tau": None,
            "top_fraction": top_frac,
            "threshold_used": None,
            "hard_ratio": 0.0,
        }

    weights = np.ones_like(abs_err, dtype=np.float32)
    weights[hard_mask] += alpha
    return weights.astype(np.float32), {
        "enabled": True,
        "alpha": alpha,
        "tau": None if tau is None else float(tau),
        "top_fraction": top_frac,
        "threshold_used": float(threshold),
        "hard_ratio": float(hard_mask.mean()),
    }


def _compute_hard_threshold(abs_err: np.ndarray, cfg: dict) -> tuple[float | None, np.ndarray]:
    tau = cfg.get("tau", None)
    top_frac = float(cfg.get("top_fraction", 0.0))
    threshold = float(tau) if tau is not None else None
    hard_mask = np.zeros_like(abs_err, dtype=bool)
    if tau is not None:
        hard_mask |= abs_err > float(tau)
    if top_frac > 0.0:
        q = max(0.0, min(100.0, 100.0 * (1.0 - top_frac)))
        q_thr = _percentile(abs_err, q)
        threshold = max(float(threshold), float(q_thr)) if threshold is not None else float(q_thr)
        hard_mask |= abs_err >= q_thr
    return threshold, hard_mask


def _build_explicit_aux_targets(
    y_target: np.ndarray,
    neighbor_delta_lambdas_nm: np.ndarray,
    linewidth_scales: np.ndarray,
    target_index: int,
) -> np.ndarray:
    if target_index <= 0 or target_index >= neighbor_delta_lambdas_nm.shape[1] - 1:
        raise ValueError("target_index must have immediate left/right neighbors for explicit modeling.")
    left_rel = neighbor_delta_lambdas_nm[:, target_index - 1] - y_target
    right_rel = neighbor_delta_lambdas_nm[:, target_index + 1] - y_target
    target_sigma = linewidth_scales[:, target_index]
    return np.stack([left_rel, right_rel, target_sigma], axis=1).astype(np.float32)


def _standardize_targets(train_targets: np.ndarray, all_targets: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = train_targets.mean(axis=0, keepdims=True)
    std = train_targets.std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    z = (all_targets - mean) / std
    return z.astype(np.float32), mean.astype(np.float32), std.astype(np.float32)


def _make_refine_loaders(
    x: np.ndarray,
    y: np.ndarray,
    idx_train: np.ndarray,
    idx_val: np.ndarray,
    train_weights: np.ndarray,
    batch_size: int,
    aux_targets_train: np.ndarray | None = None,
    aux_targets_val: np.ndarray | None = None,
    risk_targets_train: np.ndarray | None = None,
    risk_targets_val: np.ndarray | None = None,
) -> tuple[DataLoader, DataLoader]:
    train_tensors = [
        torch.tensor(x[idx_train], dtype=torch.float32),
        torch.tensor(y[idx_train], dtype=torch.float32),
        torch.tensor(train_weights, dtype=torch.float32),
    ]
    val_tensors = [
        torch.tensor(x[idx_val], dtype=torch.float32),
        torch.tensor(y[idx_val], dtype=torch.float32),
    ]
    if aux_targets_train is not None and aux_targets_val is not None:
        train_tensors.append(torch.tensor(aux_targets_train, dtype=torch.float32))
        val_tensors.append(torch.tensor(aux_targets_val, dtype=torch.float32))
    if risk_targets_train is not None and risk_targets_val is not None:
        train_tensors.append(torch.tensor(risk_targets_train, dtype=torch.float32))
        val_tensors.append(torch.tensor(risk_targets_val, dtype=torch.float32))
    train_ds = TensorDataset(*train_tensors)
    val_ds = TensorDataset(*val_tensors)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


class RefineHead(nn.Module):
    def __init__(
        self,
        c1: int,
        c2: int,
        hidden: int,
        explicit_aux_dim: int = 0,
        use_risk_gate: bool = False,
        risk_gate_bias_init: float = -2.0,
        risk_gate_power: float = 1.0,
    ):
        super().__init__()
        self.explicit_aux_dim = int(explicit_aux_dim)
        self.use_risk_gate = bool(use_risk_gate)
        self.features = nn.Sequential(
            nn.Conv1d(1, c1, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(c1, c2, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(c2, hidden),
            nn.ReLU(),
        )
        if self.explicit_aux_dim > 0:
            self.aux_head = nn.Linear(hidden, self.explicit_aux_dim)
            self.delta_head = nn.Sequential(
                nn.Linear(hidden + self.explicit_aux_dim, hidden),
                nn.ReLU(),
                nn.Linear(hidden, 1),
            )
        else:
            self.aux_head = None
            self.delta_head = nn.Linear(hidden, 1)
        if self.use_risk_gate:
            gate_in_dim = hidden + (self.explicit_aux_dim if self.explicit_aux_dim > 0 else 0)
            self.risk_head = nn.Linear(gate_in_dim, 1)
            nn.init.constant_(self.risk_head.bias, float(risk_gate_bias_init))
        else:
            self.risk_head = None

    def forward(self, x_win: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        # x_win: [B, W]
        feat = self.features(x_win.unsqueeze(1))
        aux = self.aux_head(feat) if self.aux_head is not None else None
        delta_in = torch.cat([feat, aux], dim=1) if aux is not None else feat
        delta = self.delta_head(delta_in).squeeze(-1)
        risk_logit = self.risk_head(delta_in).squeeze(-1) if self.risk_head is not None else None
        return delta, aux, risk_logit


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
        explicit_aux_dim: int = 0,
        use_risk_gate: bool = False,
        risk_gate_bias_init: float = -2.0,
        risk_gate_power: float = 1.0,
    ):
        super().__init__()
        self.coarse = coarse_model
        for p in self.coarse.parameters():
            p.requires_grad = False
        self.refine = RefineHead(
            c1=c1,
            c2=c2,
            hidden=hidden,
            explicit_aux_dim=explicit_aux_dim,
            use_risk_gate=use_risk_gate,
            risk_gate_bias_init=risk_gate_bias_init,
        )
        self.wl_start_nm = float(wl_start_nm)
        self.wl_step_nm = float(wl_step_nm)
        self.lambda0_nm = float(lambda0_nm)
        self.refine_window_points = int(refine_window_points)
        self.residual_clip_nm = float(residual_clip_nm)
        self.risk_gate_power = float(risk_gate_power)

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

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        # Keep coarse predictor in eval mode for deterministic behavior.
        self.coarse.eval()
        with torch.no_grad():
            coarse = self.coarse(x)
        coarse_wl = self.lambda0_nm + coarse
        coarse_idx = torch.round((coarse_wl - self.wl_start_nm) / self.wl_step_nm).long()
        x_win = self._extract_refine_window(x, coarse_idx)
        raw_delta, aux_pred, risk_logit = self.refine(x_win)
        if self.residual_clip_nm > 0:
            delta = self.residual_clip_nm * torch.tanh(raw_delta)
        else:
            delta = raw_delta
        if risk_logit is not None:
            gate = torch.sigmoid(risk_logit)
            gate_eff = gate.pow(self.risk_gate_power) if self.risk_gate_power != 1.0 else gate
            delta = gate_eff * delta
        else:
            gate = None
        final = coarse + delta
        return coarse, delta, final, aux_pred, gate


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
    coarse_pred_train: np.ndarray | None = None,
    coarse_pred_val: np.ndarray | None = None,
    refine_train_cfg: dict | None = None,
    explicit_aux_all: np.ndarray | None = None,
) -> CoarseToFineRefineV2:
    set_seed(seed)
    refine_train_cfg = dict(refine_train_cfg or {})
    tail_cfg = dict(refine_train_cfg.get("tail_loss", {}))
    offline_hard_cfg = dict(refine_train_cfg.get("offline_hard_weighting", {}))
    explicit_cfg = dict(refine_train_cfg.get("explicit_modeling", {}))
    risk_gate_cfg = dict(refine_train_cfg.get("risk_gate", {}))
    use_tail_refine = bool(refine_train_cfg.get("use_tail_refine", False))
    use_explicit = bool(explicit_cfg.get("enabled", False))
    use_risk_gate = bool(risk_gate_cfg.get("enabled", False))
    batch_size = int(cfg_train["batch_size"])
    aux_targets_train = None
    aux_targets_val = None
    risk_targets_train = None
    risk_targets_val = None
    aux_stats = {"enabled": False}
    risk_stats = {"enabled": False}

    if use_tail_refine:
        if coarse_pred_train is None:
            raise ValueError("coarse_pred_train is required when use_tail_refine=true")
        train_weights, hard_stats = _build_offline_hard_weights(
            coarse_pred=coarse_pred_train,
            y_true=y[idx_train],
            hard_cfg=offline_hard_cfg,
        )
        if use_explicit:
            if explicit_aux_all is None:
                raise ValueError("explicit_aux_all is required when explicit_modeling.enabled=true")
            explicit_z_all, aux_mean, aux_std = _standardize_targets(
                train_targets=explicit_aux_all[idx_train],
                all_targets=explicit_aux_all,
            )
            aux_targets_train = explicit_z_all[idx_train]
            aux_targets_val = explicit_z_all[idx_val]
            aux_stats = {
                "enabled": True,
                "target_names": ["neighbor_left_rel_nm", "neighbor_right_rel_nm", "target_linewidth_scale"],
                "mean": aux_mean.squeeze(0).tolist(),
                "std": aux_std.squeeze(0).tolist(),
                "lambda_explicit": float(explicit_cfg.get("lambda_explicit", 0.05)),
            }
        if use_risk_gate:
            if coarse_pred_val is None:
                raise ValueError("coarse_pred_val is required when risk_gate.enabled=true")
            train_abs_err = np.abs(coarse_pred_train.astype(np.float32) - y[idx_train].astype(np.float32))
            val_abs_err = np.abs(coarse_pred_val.astype(np.float32) - y[idx_val].astype(np.float32))
            threshold, train_mask = _compute_hard_threshold(train_abs_err, risk_gate_cfg)
            if threshold is None:
                raise ValueError("risk_gate requires tau and/or top_fraction to define hard samples")
            val_mask = val_abs_err >= float(threshold)
            risk_targets_train = train_mask.astype(np.float32)
            risk_targets_val = val_mask.astype(np.float32)
            risk_stats = {
                "enabled": True,
                "threshold_used": float(threshold),
                "train_hard_ratio": float(train_mask.mean()),
                "val_hard_ratio": float(val_mask.mean()),
                "lambda_gate": float(risk_gate_cfg.get("lambda_gate", 0.05)),
            }
        train_loader, val_loader = _make_refine_loaders(
            x=x,
            y=y,
            idx_train=idx_train,
            idx_val=idx_val,
            train_weights=train_weights,
            batch_size=batch_size,
            aux_targets_train=aux_targets_train,
            aux_targets_val=aux_targets_val,
            risk_targets_train=risk_targets_train,
            risk_targets_val=risk_targets_val,
        )
        print(
            "[RefineV2-Tail] "
            f"offline_hard_enabled={hard_stats['enabled']} | hard_ratio={hard_stats['hard_ratio']:.3f} | "
            f"threshold={hard_stats['threshold_used']}"
        )
        if use_risk_gate:
            print(
                "[RefineV2-Gate] "
                f"threshold={risk_stats['threshold_used']:.6f} | "
                f"train_hard_ratio={risk_stats['train_hard_ratio']:.3f} | val_hard_ratio={risk_stats['val_hard_ratio']:.3f}"
            )
    else:
        train_loader, val_loader = make_loaders(
            x=x,
            y=y,
            idx_train=idx_train,
            idx_val=idx_val,
            batch_size=batch_size,
        )
        hard_stats = {
            "enabled": False,
            "alpha": 0.0,
            "tau": None,
            "top_fraction": 0.0,
            "threshold_used": None,
            "hard_ratio": 0.0,
        }
        risk_stats = {"enabled": False}

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
        train_losses = []
        train_main_losses = []
        train_aux_losses = []
        train_gate_losses = []
        train_abs_errs: list[np.ndarray] = []
        for batch in train_loader:
            if use_tail_refine:
                xb = batch[0]
                yb = batch[1]
                wb = batch[2]
                next_idx = 3
                if use_explicit:
                    auxb = batch[next_idx].to(device)
                    next_idx += 1
                else:
                    auxb = None
                if use_risk_gate:
                    riskb = batch[next_idx].to(device)
                else:
                    riskb = None
                wb = wb.to(device)
            else:
                xb, yb = batch
                wb = None
                auxb = None
                riskb = None
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad()
            _, _, final, aux_pred, gate = model(xb)
            if use_tail_refine:
                per_sample = _tail_aware_per_sample_loss(
                    pred=final,
                    target=yb,
                    tau=float(tail_cfg.get("tau", 0.01)),
                    lambda_tail=float(tail_cfg.get("lambda_tail", 2.0)),
                )
                if wb is not None:
                    per_sample = per_sample * wb
                loss_main = per_sample.mean()
                if use_explicit and auxb is not None and aux_pred is not None:
                    loss_aux = nn.functional.mse_loss(aux_pred, auxb)
                else:
                    loss_aux = torch.zeros((), device=device)
                if use_risk_gate and riskb is not None and gate is not None:
                    gate_logit = torch.logit(torch.clamp(gate, 1e-4, 1.0 - 1e-4))
                    loss_gate = nn.functional.binary_cross_entropy_with_logits(gate_logit, riskb)
                else:
                    loss_gate = torch.zeros((), device=device)
                loss = (
                    loss_main
                    + float(explicit_cfg.get("lambda_explicit", 0.05)) * loss_aux
                    + float(risk_gate_cfg.get("lambda_gate", 0.05)) * loss_gate
                )
                train_abs_errs.append(torch.abs(final - yb).detach().cpu().numpy())
            else:
                loss_main = loss_fn(final, yb)
                loss_aux = torch.zeros((), device=device)
                loss_gate = torch.zeros((), device=device)
                loss = loss_main
            loss.backward()
            opt.step()
            train_losses.append(float(loss.item()))
            train_main_losses.append(float(loss_main.item()))
            train_aux_losses.append(float(loss_aux.item()))
            train_gate_losses.append(float(loss_gate.item()))

        model.eval()
        vals = []
        val_main_vals = []
        val_aux_vals = []
        val_gate_vals = []
        val_abs_errs: list[np.ndarray] = []
        with torch.no_grad():
            for batch in val_loader:
                xb, yb = batch[:2]
                next_idx = 2
                auxb = batch[next_idx].to(device) if (use_explicit and len(batch) > next_idx) else None
                if use_explicit and len(batch) > next_idx:
                    next_idx += 1
                riskb = batch[next_idx].to(device) if (use_risk_gate and len(batch) > next_idx) else None
                xb = xb.to(device)
                yb = yb.to(device)
                _, _, final, aux_pred, gate = model(xb)
                if use_tail_refine:
                    val_main = _tail_aware_per_sample_loss(
                        pred=final,
                        target=yb,
                        tau=float(tail_cfg.get("tau", 0.01)),
                        lambda_tail=float(tail_cfg.get("lambda_tail", 2.0)),
                    ).mean()
                    if use_explicit and auxb is not None and aux_pred is not None:
                        val_aux = nn.functional.mse_loss(aux_pred, auxb)
                    else:
                        val_aux = torch.zeros((), device=device)
                    if use_risk_gate and riskb is not None and gate is not None:
                        gate_logit = torch.logit(torch.clamp(gate, 1e-4, 1.0 - 1e-4))
                        val_gate = nn.functional.binary_cross_entropy_with_logits(gate_logit, riskb)
                    else:
                        val_gate = torch.zeros((), device=device)
                    val_loss = (
                        val_main
                        + float(explicit_cfg.get("lambda_explicit", 0.05)) * val_aux
                        + float(risk_gate_cfg.get("lambda_gate", 0.05)) * val_gate
                    )
                    vals.append(float(val_loss.item()))
                    val_main_vals.append(float(val_main.item()))
                    val_aux_vals.append(float(val_aux.item()))
                    val_gate_vals.append(float(val_gate.item()))
                    val_abs_errs.append(torch.abs(final - yb).cpu().numpy())
                else:
                    vals.append(float(loss_fn(final, yb).item()))
        val_mse = float(np.mean(vals))
        monitor_val = float(np.mean(val_main_vals)) if (use_tail_refine and use_explicit and val_main_vals) else val_mse
        if ep % 5 == 0 or ep == 1:
            if use_tail_refine and val_abs_errs:
                train_abs = np.concatenate(train_abs_errs) if train_abs_errs else np.array([], dtype=np.float32)
                val_abs = np.concatenate(val_abs_errs)
                train_p95 = float(np.percentile(train_abs, 95)) if train_abs.size else float("nan")
                val_p95 = float(np.percentile(val_abs, 95))
                val_p99 = float(np.percentile(val_abs, 99))
                train_loss = float(np.mean(train_losses)) if train_losses else float("nan")
                train_main = float(np.mean(train_main_losses)) if train_main_losses else float("nan")
                train_aux = float(np.mean(train_aux_losses)) if train_aux_losses else 0.0
                train_gate = float(np.mean(train_gate_losses)) if train_gate_losses else 0.0
                val_main = float(np.mean(val_main_vals)) if val_main_vals else float("nan")
                val_aux = float(np.mean(val_aux_vals)) if val_aux_vals else 0.0
                val_gate = float(np.mean(val_gate_vals)) if val_gate_vals else 0.0
                if use_explicit:
                    print(
                        f"[RefineV2-ExplicitTail] Epoch {ep:03d} | train_main={train_main:.6f} | train_aux={train_aux:.6f} | "
                        f"train_gate={train_gate:.6f} | train_total={train_loss:.6f} | train_p95={train_p95:.6f} | "
                        f"val_main={val_main:.6f} | val_aux={val_aux:.6f} | val_gate={val_gate:.6f} | val_total={val_mse:.6f} | "
                        f"val_p95={val_p95:.6f} | val_p99={val_p99:.6f}"
                    )
                else:
                    print(
                        f"[RefineV2-Tail] Epoch {ep:03d} | train_loss={train_loss:.6f} | "
                        f"train_p95={train_p95:.6f} | val_tail={val_mse:.6f} | "
                        f"val_p95={val_p95:.6f} | val_p99={val_p99:.6f}"
                    )
            else:
                print(f"[RefineV2] Epoch {ep:03d} | val_mse={val_mse:.6f}")

        if monitor_val < best_val:
            best_val = monitor_val
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                metric_name = "val_main" if (use_tail_refine and use_explicit) else "val_mse"
                print(f"[RefineV2] Early stop at epoch {ep}, best {metric_name}={best_val:.6f}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model._offline_hard_stats = hard_stats
    model._explicit_aux_stats = aux_stats
    model._risk_gate_stats = risk_stats
    return model


def predict_c2f_v2(model: CoarseToFineRefineV2, x: np.ndarray, idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    xt = torch.tensor(x[idx], dtype=torch.float32, device=device)
    model.eval()
    with torch.no_grad():
        coarse, _, final, _, _ = model(xt)
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
    target_index = int(d["target_index"][0]) if "target_index" in d else int(cfg.get("array", {}).get("target_index", 2))

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
    refine_train_cfg = dict(mcfg.get("refine_training", {}))
    explicit_cfg = dict(refine_train_cfg.get("explicit_modeling", {}))
    risk_gate_cfg = dict(refine_train_cfg.get("risk_gate", {}))
    explicit_aux_all = None
    if bool(explicit_cfg.get("enabled", False)):
        need_aux_keys = ["neighbor_delta_lambdas_nm", "linewidth_scales"]
        miss_aux = [k for k in need_aux_keys if k not in d]
        if miss_aux:
            raise KeyError(f"Dataset missing keys for explicit modeling: {miss_aux}")
        explicit_aux_all = _build_explicit_aux_targets(
            y_target=y,
            neighbor_delta_lambdas_nm=d["neighbor_delta_lambdas_nm"].astype(np.float32),
            linewidth_scales=d["linewidth_scales"].astype(np.float32),
            target_index=target_index,
        )

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
    pred_cnn_train = predict(cnn, x, idx_train, device).astype(np.float32)
    pred_cnn_val = predict(cnn, x, idx_val, device).astype(np.float32)

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
        explicit_aux_dim=int(explicit_aux_all.shape[1]) if explicit_aux_all is not None else 0,
        use_risk_gate=bool(risk_gate_cfg.get("enabled", False)),
        risk_gate_bias_init=float(risk_gate_cfg.get("bias_init", -2.0)),
        risk_gate_power=float(risk_gate_cfg.get("power", 1.0)),
    )
    c2f_v2 = train_refine_v2(
        model=c2f_v2,
        x=x,
        y=y,
        idx_train=idx_train,
        idx_val=idx_val,
        cfg_train=cfg["train"],
        seed=seed_refine,
        coarse_pred_train=pred_cnn_train,
        coarse_pred_val=pred_cnn_val,
        refine_train_cfg=refine_train_cfg,
        explicit_aux_all=explicit_aux_all,
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
        "refine_training": refine_train_cfg,
        "offline_hard_stats": getattr(c2f_v2, "_offline_hard_stats", {}),
        "explicit_aux_stats": getattr(c2f_v2, "_explicit_aux_stats", {}),
        "risk_gate_stats": getattr(c2f_v2, "_risk_gate_stats", {}),
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
