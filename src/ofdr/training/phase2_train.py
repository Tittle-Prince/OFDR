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


def train_regressor(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, cfg_train: dict, device: torch.device) -> nn.Module:
    lr = float(cfg_train["lr"])
    wd = float(cfg_train["weight_decay"])
    epochs = int(cfg_train["epochs"])
    patience = int(cfg_train["patience"])

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    criterion = nn.MSELoss()

    best_val = float("inf")
    best_state = None
    bad_epochs = 0

    for epoch in range(1, epochs + 1):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                val_losses.append(float(criterion(model(xb), yb).item()))
        mean_val = float(np.mean(val_losses))

        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:03d} | val_mse={mean_val:.6f}")

        if mean_val < best_val:
            best_val = mean_val
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
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

