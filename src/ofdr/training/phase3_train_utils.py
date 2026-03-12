from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


def make_loaders(
    x: np.ndarray,
    y: np.ndarray,
    idx_train: np.ndarray,
    idx_val: np.ndarray,
    batch_size: int,
) -> tuple[DataLoader, DataLoader]:
    x_train = torch.tensor(x[idx_train], dtype=torch.float32)
    y_train = torch.tensor(y[idx_train], dtype=torch.float32)
    x_val = torch.tensor(x[idx_val], dtype=torch.float32)
    y_val = torch.tensor(y[idx_val], dtype=torch.float32)

    train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(x_val, y_val), batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


def _compute_per_sample_loss(pred: torch.Tensor, target: torch.Tensor, loss_cfg: dict) -> torch.Tensor:
    """
    Return per-sample loss values without reduction.

    Supported losses:
    - mse (default): e^2
    - tail_aware_l1: |e| + lambda_tail * max(0, |e| - tau)^2
      This keeps the base L1 error while adding extra penalty only on large-error
      samples, so optimization pays more attention to tail behavior (e.g. P95).
    """
    err = pred - target
    name = str(loss_cfg.get("name", "mse")).lower()
    if name == "mse":
        return err.pow(2)
    if name == "tail_aware_l1":
        tau = float(loss_cfg.get("tau", 0.01))
        lambda_tail = float(loss_cfg.get("lambda_tail", 3.0))
        abs_err = torch.abs(err)
        tail = torch.clamp(abs_err - tau, min=0.0)
        return abs_err + lambda_tail * tail.pow(2)
    raise ValueError(f"Unsupported loss.name: {name}")


def _apply_hard_weighting(per_sample_loss: torch.Tensor, abs_err: torch.Tensor, hard_cfg: dict) -> torch.Tensor:
    """
    Lightweight batch-wise hard sample weighting:
      weight_i = 1 + alpha * I(|e_i| > tau_hard)
    Used only in training. It increases gradient contribution from high-error
    samples to prioritize tail metrics (especially P95), rather than only mean error.
    """
    if not bool(hard_cfg.get("enabled", False)):
        return per_sample_loss

    tau_hard = float(hard_cfg.get("tau", 0.01))
    alpha = float(hard_cfg.get("alpha", 1.5))
    weights = 1.0 + alpha * (abs_err > tau_hard).to(per_sample_loss.dtype)
    return per_sample_loss * weights


def train_model(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, cfg_train: dict, device: torch.device) -> nn.Module:
    lr = float(cfg_train["lr"])
    wd = float(cfg_train["weight_decay"])
    epochs = int(cfg_train["epochs"])
    patience = int(cfg_train["patience"])
    loss_cfg = dict(cfg_train.get("loss", {}))
    hard_cfg = dict(cfg_train.get("hard_weighting", {}))

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    criterion = nn.MSELoss()

    best_val = float("inf")
    best_state = None
    stale = 0

    for epoch in range(1, epochs + 1):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            per_sample = _compute_per_sample_loss(pred, yb, loss_cfg=loss_cfg)
            abs_err = torch.abs(pred - yb)
            per_sample = _apply_hard_weighting(per_sample, abs_err=abs_err, hard_cfg=hard_cfg)
            loss = per_sample.mean()
            loss.backward()
            optimizer.step()

        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                val_losses.append(float(criterion(model(xb), yb).item()))
        val_mse = float(np.mean(val_losses))
        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:03d} | val_mse={val_mse:.6f}")

        if val_mse < best_val:
            best_val = val_mse
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                print(f"Early stop at epoch {epoch}, best val_mse={best_val:.6f}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def predict(model: nn.Module, x: np.ndarray, idx_test: np.ndarray, device: torch.device) -> np.ndarray:
    xt = torch.tensor(x[idx_test], dtype=torch.float32, device=device)
    model.eval()
    with torch.no_grad():
        pred = model(xt).cpu().numpy()
    return pred.astype(np.float32)

