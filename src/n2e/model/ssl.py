"""Masked-patch reconstruction SSL (Model B pretraining objective).

Random per-token masking; masked tokens are replaced by a learned mask embedding
(positional encoding retained); a light MLP decoder reconstructs the masked patches'
normalized values; loss = MSE on masked patches only. Latent-space prediction is the
noted v2.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import BrainEncoder


class MaskedPatchSSL(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.encoder = BrainEncoder(cfg)
        d = int(cfg.tokenizer.d_model)
        patch_len = self.encoder.tok.patch_len
        self.mask_ratio = float(cfg.ssl.mask_ratio)
        self.mask_scheme = str(cfg.ssl.mask_scheme)
        self.mask_emb = nn.Parameter(torch.zeros(d))
        nn.init.normal_(self.mask_emb, std=0.02)
        hidden = int(cfg.ssl.decoder_hidden)
        self.decoder = nn.Sequential(
            nn.Linear(d, hidden), nn.GELU(), nn.Linear(hidden, patch_len)
        )

    def _sample_mask(self, B: int, N: int, C: int, P: int, device) -> torch.Tensor:
        if self.mask_scheme == "channel":  # mask whole channels
            m = torch.rand(B, C, 1, device=device) < self.mask_ratio
            return m.expand(B, C, P).reshape(B, N)
        if self.mask_scheme == "timepatch":  # mask whole time-patches (all channels)
            m = torch.rand(B, 1, P, device=device) < self.mask_ratio
            return m.expand(B, C, P).reshape(B, N)
        return torch.rand(B, N, device=device) < self.mask_ratio  # random (default)

    def forward(self, x: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        content, pos, targets, C, P = self.encoder.embed(x, coords)
        B, N, d = content.shape
        mask = self._sample_mask(B, N, C, P, x.device)
        if mask.sum() == 0:
            mask[:, 0] = True
        tokens = torch.where(mask.unsqueeze(-1), self.mask_emb.view(1, 1, d), content) + pos
        h = self.encoder.encode(tokens)
        pred = self.decoder(h)  # (B, N, patch_len)
        return F.mse_loss(pred[mask], targets[mask])
