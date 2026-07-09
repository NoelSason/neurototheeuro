"""Transformer encoder over (channel x time-patch) tokens."""
from __future__ import annotations

import torch
import torch.nn as nn

from .tokenizer import PatchTokenizer


class BrainEncoder(nn.Module):
    """Patch tokenizer + pre-norm transformer, with mean-pooled readout."""

    def __init__(self, cfg):
        super().__init__()
        self.tok = PatchTokenizer(cfg)
        d = int(cfg.tokenizer.d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d,
            nhead=int(cfg.encoder.n_heads),
            dim_feedforward=d * int(cfg.encoder.mlp_ratio),
            dropout=float(cfg.encoder.dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            layer, int(cfg.encoder.depth), enable_nested_tensor=False
        )
        self.norm = nn.LayerNorm(d)
        self.pool = str(cfg.encoder.pool)
        self.d = d

    def embed(self, x, coords):
        """Return ``content, pos, targets, C, P`` from the tokenizer."""
        return self.tok(x, coords)

    def encode(self, tokens: torch.Tensor) -> torch.Tensor:
        """Run the transformer + final norm over pre-built tokens (B, N, d)."""
        return self.norm(self.transformer(tokens))

    def forward(self, x, coords) -> torch.Tensor:
        """Pooled representation (B, d) for downstream heads."""
        content, pos, _, _, _ = self.embed(x, coords)
        h = self.encode(content + pos)
        if self.pool == "cls":
            return h[:, 0]
        return h.mean(dim=1)
