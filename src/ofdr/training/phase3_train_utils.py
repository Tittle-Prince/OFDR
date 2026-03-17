from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


def _standardize_per_sample(v: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    mu = v.mean(axis=1, keepdims=True)
    std = v.std(axis=1, keepdims=True) + eps
    return (v - mu) / std


def _build_feature_input(x: np.ndarray, use_d1: bool, use_d2: bool) -> np.ndarray:
    """
    Build channel-first spectral features [B, C, L]:
      - channel 0 always raw spectrum
      - optional channel for first derivative (per-sample standardized)
      - optional channel for second derivative (per-sample standardized)
    """
    feats: list[np.ndarray] = [x]
    if use_d1 or use_d2:
        dx = np.gradient(x, axis=1)
        if use_d1:
            feats.append(_standardize_per_sample(dx))
        if use_d2:
            d2x = np.gradient(dx, axis=1)
            feats.append(_standardize_per_sample(d2x))
    return np.stack(feats, axis=1).astype(np.float32)


def _robust_minmax(v: np.ndarray, p_low: float = 5.0, p_high: float = 95.0, eps: float = 1e-8) -> np.ndarray:
    lo = float(np.percentile(v, p_low))
    hi = float(np.percentile(v, p_high))
    if hi - lo < eps:
        return np.zeros_like(v, dtype=np.float32)
    z = (v - lo) / (hi - lo)
    return np.clip(z, 0.0, 1.0).astype(np.float32)


def _compute_overlap_score(x: np.ndarray) -> np.ndarray:
    """
    Build a pseudo overlap score in [0, 1] from each spectrum using:
      1) effective width
      2) asymmetry (|skewness|)
      3) shoulder-energy ratio

    Intuition:
    - Overlap/broadening often spreads energy and increases effective width.
    - Neighbor interaction can break left-right symmetry (higher asymmetry).
    - Strong overlap introduces shoulder energy away from the central peak.
    """
    x_pos = x - x.min(axis=1, keepdims=True)
    energy = x_pos.sum(axis=1, keepdims=True) + 1e-8
    p = x_pos / energy

    n = x.shape[1]
    pos = np.linspace(-1.0, 1.0, n, dtype=np.float32)[None, :]

    mu = (p * pos).sum(axis=1)
    centered = pos - mu[:, None]

    var = (p * (centered**2)).sum(axis=1)
    width = np.sqrt(np.maximum(var, 1e-8))

    skew_abs = np.abs((p * (centered**3)).sum(axis=1) / (width**3 + 1e-8))

    center_mask = (np.abs(pos) <= 0.33).astype(np.float32)
    shoulder_mask = (np.abs(pos) >= 0.66).astype(np.float32)
    center_e = (p * center_mask).sum(axis=1)
    shoulder_e = (p * shoulder_mask).sum(axis=1)
    shoulder_ratio = shoulder_e / (center_e + 1e-8)

    width_n = _robust_minmax(width)
    skew_n = _robust_minmax(skew_abs)
    shoulder_n = _robust_minmax(shoulder_ratio)

    score = 0.45 * width_n + 0.35 * skew_n + 0.20 * shoulder_n
    return np.clip(score, 0.0, 1.0).astype(np.float32)


def make_loaders(
    x: np.ndarray,
    y: np.ndarray,
    idx_train: np.ndarray,
    idx_val: np.ndarray,
    batch_size: int,
    data_cfg: dict | None = None,
) -> tuple[DataLoader, DataLoader]:
    cfg = dict(data_cfg or {})
    deriv_cfg = dict(cfg.get("derivative_input", {}))
    derivative_enabled = bool(deriv_cfg.get("enabled", False))
    use_d1 = bool(deriv_cfg.get("use_d1", derivative_enabled))
    use_d2 = bool(deriv_cfg.get("use_d2", derivative_enabled))
    aux_enabled = bool(dict(cfg.get("aux_head", {})).get("enabled", False))

    x_use = x.astype(np.float32)
    x_feat = _build_feature_input(x_use, use_d1=use_d1, use_d2=use_d2) if (use_d1 or use_d2) else x_use

    x_train = torch.tensor(x_feat[idx_train], dtype=torch.float32)
    y_train = torch.tensor(y[idx_train], dtype=torch.float32)
    x_val = torch.tensor(x_feat[idx_val], dtype=torch.float32)
    y_val = torch.tensor(y[idx_val], dtype=torch.float32)

    if aux_enabled:
        aux_all = _compute_overlap_score(x_use)
        aux_train = torch.tensor(aux_all[idx_train], dtype=torch.float32)
        aux_val = torch.tensor(aux_all[idx_val], dtype=torch.float32)
        train_ds = TensorDataset(x_train, y_train, aux_train)
        val_ds = TensorDataset(x_val, y_val, aux_val)
    else:
        train_ds = TensorDataset(x_train, y_train)
        val_ds = TensorDataset(x_val, y_val)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
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


def _split_model_output(output: torch.Tensor | tuple[torch.Tensor, torch.Tensor] | list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor | None]:
    if isinstance(output, (tuple, list)):
        if len(output) == 0:
            raise ValueError("Model output tuple/list is empty.")
        main = output[0]
        aux = output[1] if len(output) > 1 else None
        return main, aux
    return output, None


def _train_stage(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    aux_criterion: nn.Module,
    epochs: int,
    patience: int,
    loss_cfg: dict,
    hard_cfg: dict,
    aux_enabled: bool,
    lambda_aux: float,
    epoch_offset: int = 0,
) -> tuple[dict[str, torch.Tensor] | None, float]:
    best_val = float("inf")
    best_state = None
    stale = 0

    for epoch in range(1, epochs + 1):
        model.train()
        train_main_losses: list[float] = []
        train_aux_losses: list[float] = []
        train_total_losses: list[float] = []
        for batch in train_loader:
            if len(batch) == 2:
                xb, yb = batch
                aux_target = None
            elif len(batch) == 3:
                xb, yb, aux_target = batch
            else:
                raise ValueError(f"Unexpected batch structure with {len(batch)} tensors.")

            xb = xb.to(device)
            yb = yb.to(device)
            if aux_target is not None:
                aux_target = aux_target.to(device)

            optimizer.zero_grad()
            output = model(xb)
            pred_main, pred_aux = _split_model_output(output)

            per_sample = _compute_per_sample_loss(pred_main, yb, loss_cfg=loss_cfg)
            abs_err = torch.abs(pred_main - yb)
            per_sample = _apply_hard_weighting(per_sample, abs_err=abs_err, hard_cfg=hard_cfg)
            loss_main = per_sample.mean()

            if aux_enabled and aux_target is not None and pred_aux is not None:
                loss_aux = aux_criterion(pred_aux, aux_target)
                loss = loss_main + lambda_aux * loss_aux
            else:
                loss_aux = torch.zeros((), device=device)
                loss = loss_main

            loss.backward()
            optimizer.step()
            train_main_losses.append(float(loss_main.item()))
            train_aux_losses.append(float(loss_aux.item()))
            train_total_losses.append(float(loss.item()))

        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                xb = batch[0].to(device)
                yb = batch[1].to(device)
                pred_main, _ = _split_model_output(model(xb))
                val_losses.append(float(criterion(pred_main, yb).item()))
        val_mse = float(np.mean(val_losses))
        full_epoch = epoch_offset + epoch
        if full_epoch % 5 == 0 or full_epoch == 1:
            train_main = float(np.mean(train_main_losses)) if train_main_losses else float("nan")
            train_total = float(np.mean(train_total_losses)) if train_total_losses else float("nan")
            if aux_enabled:
                train_aux = float(np.mean(train_aux_losses)) if train_aux_losses else float("nan")
                print(
                    f"Epoch {full_epoch:03d} | train_main={train_main:.6f} | train_aux={train_aux:.6f} | "
                    f"train_total={train_total:.6f} | val_mse={val_mse:.6f}"
                )
            else:
                print(
                    f"Epoch {full_epoch:03d} | train_main={train_main:.6f} | "
                    f"train_total={train_total:.6f} | val_mse={val_mse:.6f}"
                )

        if val_mse < best_val:
            best_val = val_mse
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                print(f"Early stop at epoch {full_epoch}, best val_mse={best_val:.6f}")
                break

    return best_state, best_val


def train_model(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, cfg_train: dict, device: torch.device) -> nn.Module:
    lr = float(cfg_train["lr"])
    wd = float(cfg_train["weight_decay"])
    epochs = int(cfg_train["epochs"])
    patience = int(cfg_train["patience"])
    loss_cfg = dict(cfg_train.get("loss", {}))
    hard_cfg = dict(cfg_train.get("hard_weighting", {}))
    aux_cfg = dict(cfg_train.get("aux_loss", {}))
    aux_enabled = bool(aux_cfg.get("enabled", False))
    lambda_aux = float(aux_cfg.get("lambda_aux", 0.1))
    finetune_cfg = dict(cfg_train.get("finetune", {}))

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    criterion = nn.MSELoss()
    aux_criterion = nn.MSELoss()
    best_state, best_val = _train_stage(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        optimizer=optimizer,
        criterion=criterion,
        aux_criterion=aux_criterion,
        epochs=epochs,
        patience=patience,
        loss_cfg=loss_cfg,
        hard_cfg=hard_cfg,
        aux_enabled=aux_enabled,
        lambda_aux=lambda_aux,
        epoch_offset=0,
    )

    if best_state is not None:
        model.load_state_dict(best_state)

    if bool(finetune_cfg.get("enabled", False)):
        ft_epochs = int(finetune_cfg.get("epochs", 10))
        ft_patience = int(finetune_cfg.get("patience", min(5, ft_epochs)))
        ft_lr = float(finetune_cfg.get("lr", lr * 0.2))
        ft_loss_cfg = dict(finetune_cfg.get("loss", loss_cfg))
        ft_hard_cfg = dict(finetune_cfg.get("hard_weighting", hard_cfg))
        ft_optimizer = torch.optim.Adam(model.parameters(), lr=ft_lr, weight_decay=wd)
        ft_state, ft_best_val = _train_stage(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            optimizer=ft_optimizer,
            criterion=criterion,
            aux_criterion=aux_criterion,
            epochs=ft_epochs,
            patience=ft_patience,
            loss_cfg=ft_loss_cfg,
            hard_cfg=ft_hard_cfg,
            aux_enabled=aux_enabled,
            lambda_aux=lambda_aux,
            epoch_offset=epochs,
        )
        if ft_state is not None and ft_best_val <= best_val:
            best_state = ft_state
            best_val = ft_best_val

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def predict(
    model: nn.Module,
    x: np.ndarray,
    idx_test: np.ndarray,
    device: torch.device,
    data_cfg: dict | None = None,
) -> np.ndarray:
    cfg = dict(data_cfg or {})
    deriv_cfg = dict(cfg.get("derivative_input", {}))
    derivative_enabled = bool(deriv_cfg.get("enabled", False))
    use_d1 = bool(deriv_cfg.get("use_d1", derivative_enabled))
    use_d2 = bool(deriv_cfg.get("use_d2", derivative_enabled))
    x_use = x.astype(np.float32)
    x_feat = _build_feature_input(x_use, use_d1=use_d1, use_d2=use_d2) if (use_d1 or use_d2) else x_use
    xt = torch.tensor(x_feat[idx_test], dtype=torch.float32, device=device)
    model.eval()
    with torch.no_grad():
        pred_main, _ = _split_model_output(model(xt))
        pred = pred_main.cpu().numpy()
    return pred.astype(np.float32)

