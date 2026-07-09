"""Decoding head + full decode model (shared architecture across A1 and B)."""
from __future__ import annotations

import torch
import torch.nn as nn

from .encoder import BrainEncoder


class DecodeModel(nn.Module):
    """BrainEncoder + linear head. Identical architecture for A1 (random init) and
    B (SSL-pretrained). Head is mean-pool -> Linear(d, n_classes)."""

    def __init__(self, cfg, n_classes: int = 2):
        super().__init__()
        self.encoder = BrainEncoder(cfg)
        self.head = nn.Linear(int(cfg.encoder.d_model), n_classes)

    def forward(self, x, coords) -> torch.Tensor:
        return self.head(self.encoder(x, coords))

    def freeze_encoder(self) -> None:
        for p in self.encoder.parameters():
            p.requires_grad_(False)

    def load_encoder_state(self, state_dict: dict) -> None:
        self.encoder.load_state_dict(state_dict)
