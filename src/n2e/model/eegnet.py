"""EEGNet-8,2 — field-standard single-subject decoder (baseline A2).

Reference: Lawhern et al. 2018. Flatten dim is derived by a dry forward pass so the
conv/pool arithmetic can't drift out of sync with the config.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class EEGNet(nn.Module):
    def __init__(
        self,
        n_channels: int,
        n_times: int,
        n_classes: int = 2,
        F1: int = 8,
        D: int = 2,
        F2: int = 16,
        kernel: int = 125,
        dropout: float = 0.25,
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(1, F1, (1, kernel), padding=(0, kernel // 2), bias=False)
        self.bn1 = nn.BatchNorm2d(F1)
        self.depthwise = nn.Conv2d(F1, F1 * D, (n_channels, 1), groups=F1, bias=False)
        self.bn2 = nn.BatchNorm2d(F1 * D)
        self.pool1 = nn.AvgPool2d((1, 4))
        self.drop1 = nn.Dropout(dropout)
        self.sep_depth = nn.Conv2d(F1 * D, F1 * D, (1, 16), padding=(0, 8), groups=F1 * D, bias=False)
        self.sep_point = nn.Conv2d(F1 * D, F2, (1, 1), bias=False)
        self.bn3 = nn.BatchNorm2d(F2)
        self.pool2 = nn.AvgPool2d((1, 8))
        self.drop2 = nn.Dropout(dropout)
        with torch.no_grad():
            flat = self._features(torch.zeros(1, 1, n_channels, n_times)).flatten(1).shape[1]
        self.fc = nn.Linear(flat, n_classes)

    def _features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.bn1(self.conv1(x))
        x = F.elu(self.bn2(self.depthwise(x)))
        x = self.drop1(self.pool1(x))
        x = F.elu(self.bn3(self.sep_point(self.sep_depth(x))))
        x = self.drop2(self.pool2(x))
        return x

    def forward(self, x: torch.Tensor, coords=None) -> torch.Tensor:
        # x: (B, C, T); coords accepted for a uniform call signature, ignored.
        x = x.unsqueeze(1)
        return self.fc(self._features(x).flatten(1))


def build_eegnet(cfg, n_channels: int, n_times: int, n_classes: int = 2) -> EEGNet:
    e = cfg.decode.eegnet
    kernel = int(round(float(e.kernel_s) * float(cfg.data.sfreq)))
    return EEGNet(
        n_channels=n_channels,
        n_times=n_times,
        n_classes=n_classes,
        F1=int(e.F1),
        D=int(e.D),
        F2=int(e.F2),
        kernel=kernel,
        dropout=float(e.dropout),
    )
