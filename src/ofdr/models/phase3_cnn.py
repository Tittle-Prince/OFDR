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
    def __init__(self, input_dim: int, use_dilation: bool, use_se: bool, in_channels: int = 1):
        super().__init__()
        d2 = 2 if use_dilation else 1
        pad2 = 4 if use_dilation else 2

        self.block1 = nn.Sequential(
            nn.Conv1d(int(in_channels), 32, kernel_size=7, padding=3),
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
            dummy = torch.zeros(1, int(in_channels), input_dim)
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
        if x.dim() == 2:
            x = x.unsqueeze(1)
        elif x.dim() != 3:
            raise ValueError(f"ConvRegressor expects [B,L] or [B,C,L], got shape {tuple(x.shape)}")
        x = self._forward_features(x)
        return self.head(x).squeeze(-1)


class CNNBiLSTMRegressor(nn.Module):
    def __init__(
        self,
        input_dim: int,
        in_channels: int = 1,
        bilstm_hidden: int = 64,
        bilstm_layers: int = 1,
        bidirectional: bool = True,
    ):
        super().__init__()
        if int(bilstm_layers) != 1:
            raise ValueError("CNNBiLSTMRegressor only supports bilstm_layers=1 for this minimal implementation.")

        # Reuse the existing baseline CNN feature extractor to keep a strict incremental comparison.
        base_cnn = ConvRegressor(input_dim=input_dim, use_dilation=False, use_se=False, in_channels=int(in_channels))
        self.block1 = base_cnn.block1
        self.block2 = base_cnn.block2
        self.block3 = base_cnn.block3

        with torch.no_grad():
            dummy = torch.zeros(1, int(in_channels), input_dim)
            feat = self._forward_features(dummy)  # [1, C, L]
            feat_channels = int(feat.shape[1])

        self.bilstm = nn.LSTM(
            input_size=feat_channels,
            hidden_size=int(bilstm_hidden),
            num_layers=1,
            batch_first=True,
            bidirectional=bool(bidirectional),
        )
        lstm_out_dim = int(bilstm_hidden) * (2 if bool(bidirectional) else 1)
        self.head = nn.Sequential(
            nn.Linear(lstm_out_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def _forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input: [B, N] -> CNN features: [B, C, L]
        if x.dim() == 2:
            x = x.unsqueeze(1)
        elif x.dim() != 3:
            raise ValueError(f"CNNBiLSTMRegressor expects [B,L] or [B,C,L], got shape {tuple(x.shape)}")
        feat = self._forward_features(x)
        # BiLSTM expects [B, seq_len, feature_dim].
        seq = feat.transpose(1, 2)
        seq_out, _ = self.bilstm(seq)
        # Mean pooling over sequence is shape-stable for variable spectrum lengths.
        pooled = seq_out.mean(dim=1)
        return self.head(pooled).squeeze(-1)


class NADNetV1(nn.Module):
    """
    Neighborhood-Aware Derivative Network (v1).

    - Local branch: small kernels for fine peak-shape details.
    - Context branch: larger/dilated kernels for neighborhood overlap context.
    - Optional aux head predicts overlap_score in [0, 1].
    """

    def __init__(
        self,
        input_dim: int,
        in_channels: int = 3,
        base_channels: int = 32,
        local_kernel: int = 3,
        context_kernel: int = 9,
        context_dilation: int = 2,
        use_aux_head: bool = True,
    ):
        super().__init__()
        _ = input_dim  # kept for interface consistency
        bc = int(base_channels)
        lk = int(local_kernel)
        ck = int(context_kernel)
        cd = int(context_dilation)
        context_pad = ((ck - 1) // 2) * cd

        self.local_branch = nn.Sequential(
            nn.Conv1d(int(in_channels), bc, kernel_size=lk, padding=lk // 2, bias=False),
            nn.BatchNorm1d(bc),
            nn.ReLU(inplace=True),
            nn.Conv1d(bc, bc * 2, kernel_size=lk, padding=lk // 2, bias=False),
            nn.BatchNorm1d(bc * 2),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
        )
        self.context_branch = nn.Sequential(
            nn.Conv1d(int(in_channels), bc, kernel_size=ck, padding=context_pad, dilation=cd, bias=False),
            nn.BatchNorm1d(bc),
            nn.ReLU(inplace=True),
            nn.Conv1d(bc, bc * 2, kernel_size=5, padding=4, dilation=2, bias=False),
            nn.BatchNorm1d(bc * 2),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
        )
        self.shared = nn.Sequential(
            nn.Conv1d(bc * 4, bc * 4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(bc * 4),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Conv1d(bc * 4, bc * 4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(bc * 4),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(16),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, int(in_channels), int(input_dim))
            feat = self._forward_shared(dummy)
            feat_dim = int(np.prod(feat.shape[1:]))
        self.main_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feat_dim, bc * 4),
            nn.ReLU(inplace=True),
            nn.Linear(bc * 4, 1),
        )
        self.use_aux_head = bool(use_aux_head)
        if self.use_aux_head:
            self.aux_head = nn.Sequential(
                nn.Flatten(),
                nn.Linear(feat_dim, bc * 2),
                nn.ReLU(inplace=True),
                nn.Linear(bc * 2, 1),
                nn.Sigmoid(),
            )
        else:
            self.aux_head = None

    def _forward_shared(self, x: torch.Tensor) -> torch.Tensor:
        local_feat = self.local_branch(x)
        context_feat = self.context_branch(x)
        fused = torch.cat([local_feat, context_feat], dim=1)
        return self.shared(fused)

    def forward(self, x: torch.Tensor) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if x.dim() == 2:
            x = x.unsqueeze(1)
        elif x.dim() != 3:
            raise ValueError(f"NADNetV1 expects [B,L] or [B,C,L], got shape {tuple(x.shape)}")

        shared_feat = self._forward_shared(x)

        main = self.main_head(shared_feat).squeeze(-1)
        if not self.use_aux_head:
            return main
        aux = self.aux_head(shared_feat).squeeze(-1)
        return main, aux


def build_model(method_key: str, input_dim: int, **kwargs: int | bool) -> nn.Module:
    in_channels = int(kwargs.get("in_channels", 1))
    if method_key == "cnn_baseline":
        return ConvRegressor(input_dim=input_dim, use_dilation=False, use_se=False, in_channels=in_channels)
    if method_key == "cnn_dilated":
        return ConvRegressor(input_dim=input_dim, use_dilation=True, use_se=False, in_channels=in_channels)
    if method_key == "cnn_se":
        return ConvRegressor(input_dim=input_dim, use_dilation=False, use_se=True, in_channels=in_channels)
    if method_key == "cnn_dilated_se":
        return ConvRegressor(input_dim=input_dim, use_dilation=True, use_se=True, in_channels=in_channels)
    if method_key == "cnn_bilstm":
        return CNNBiLSTMRegressor(
            input_dim=input_dim,
            in_channels=in_channels,
            bilstm_hidden=int(kwargs.get("bilstm_hidden", 64)),
            bilstm_layers=int(kwargs.get("bilstm_layers", 1)),
            bidirectional=bool(kwargs.get("bidirectional", True)),
        )
    if method_key == "nad_net_v1":
        return NADNetV1(
            input_dim=input_dim,
            in_channels=in_channels,
            base_channels=int(kwargs.get("base_channels", 32)),
            local_kernel=int(kwargs.get("local_kernel", 3)),
            context_kernel=int(kwargs.get("context_kernel", 9)),
            context_dilation=int(kwargs.get("context_dilation", 2)),
            use_aux_head=bool(kwargs.get("use_aux_head", True)),
        )
    raise ValueError(f"Unknown method_key: {method_key}")

