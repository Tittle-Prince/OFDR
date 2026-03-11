from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


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


class ConvRegressor(nn.Module):
    def __init__(self, input_dim: int, use_dilation: bool, use_se: bool):
        super().__init__()
        d2 = 2 if use_dilation else 1
        pad2 = 4 if use_dilation else 2

        self.block1 = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )

        b2_layers: list[nn.Module] = [
            nn.Conv1d(32, 64, kernel_size=5, padding=pad2, dilation=d2),
            nn.ReLU(),
        ]
        if use_se:
            b2_layers.append(SEBlock1D(64))
        b2_layers.append(nn.MaxPool1d(2))
        self.block2 = nn.Sequential(*b2_layers)

        b3_layers: list[nn.Module] = [
            nn.Conv1d(64, 128, kernel_size=5, padding=pad2, dilation=d2),
            nn.ReLU(),
        ]
        if use_se:
            b3_layers.append(SEBlock1D(128))
        self.block3 = nn.Sequential(*b3_layers)

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
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)
        x = self._forward_features(x)
        return self.head(x).squeeze(-1)


def build_model(method_key: str, input_dim: int) -> nn.Module:
    if method_key == "cnn_baseline":
        return ConvRegressor(input_dim=input_dim, use_dilation=False, use_se=False)
    if method_key == "cnn_dilated":
        return ConvRegressor(input_dim=input_dim, use_dilation=True, use_se=False)
    if method_key == "cnn_se":
        return ConvRegressor(input_dim=input_dim, use_dilation=False, use_se=True)
    if method_key == "cnn_dilated_se":
        return ConvRegressor(input_dim=input_dim, use_dilation=True, use_se=True)
    raise ValueError(f"Unknown method_key: {method_key}")

