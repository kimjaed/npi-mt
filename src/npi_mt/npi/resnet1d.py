# src/npi_mt/npi/resnet1d.py
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    """Residual block for 1D conv feature extraction."""
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, dropout: float = 0.0):
        super().__init__()
        pad = kernel_size // 2

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=pad)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, padding=pad)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.drop = nn.Dropout(dropout)

        self.skip = nn.Conv1d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.skip(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.drop(out)
        out = self.bn2(self.conv2(out))
        out = F.relu(out + identity)
        return out


class ResNetResidual1D(nn.Module):
    """
    ResNet-1D mapping MT channels -> residual correction in log-resistivity.

    Input shape:  (B, C, F)   where typically C=2 (log_appres, phase)
    Output shape: (B, Z)      (Z = number of depth layers)
    """
    def __init__(
        self,
        n_freqs: int | None = None,
        n_layers: int | None = None,
        dropout: float = 0.2,
        *,
        # Back-compat (old code)
        input_channels: int = 2,
        output_depth: int | None = None,
        dropout_rate: float | None = None,
    ):
        super().__init__()

        # back-compat mapping
        if output_depth is not None and n_layers is None:
            n_layers = output_depth
        if dropout_rate is not None:
            dropout = dropout_rate

        if n_layers is None:
            raise TypeError("Must provide n_layers (or output_depth).")
        self.n_layers = int(n_layers)

        # If you want strict F checking, pass n_freqs. Otherwise allow any F.
        self.n_freqs = None if n_freqs is None else int(n_freqs)

        self.input_channels = int(input_channels)

        self.initial_conv = nn.Conv1d(input_channels, 32, kernel_size=3, padding=1)

        self.res_block1 = ResBlock(32, 64, dropout=dropout)
        self.res_block2 = ResBlock(64, 128, dropout=dropout)
        self.res_block3 = ResBlock(128, 64, dropout=dropout)
        self.res_block4 = ResBlock(64, 32, dropout=dropout)

        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)

        self.fc = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, n_layers),
        )

        self._init_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,C,F)
        if x.ndim != 3 or x.shape[1] != self.input_channels:
            raise ValueError(f"Expected x shape (B,{self.input_channels},F), got {tuple(x.shape)}")
        if self.n_freqs is not None and x.shape[2] != self.n_freqs:
            raise ValueError(f"Expected F={self.n_freqs}, got {x.shape[2]}")

        x = F.relu(self.initial_conv(x))
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.res_block3(x)
        x = self.res_block4(x)
        x = self.global_avg_pool(x).squeeze(-1)  # (B,32)
        return self.fc(x)          # (B,Z)

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)


def freeze_bn_stats(module: nn.Module) -> None:
    """
    Use running stats for BatchNorm during fine-tuning (common when batch size is tiny).
    Keeps affine params trainable.
    """
    if isinstance(module, nn.BatchNorm1d):
        module.eval()
        for p in module.parameters():
            p.requires_grad = True
