"""Coordinate-aware patch tokenizer.

One token per (scalp-channel, time-patch): linear patch-embed + summed positional
encoding = 3-D CapTrak coordinate (Fourier->MLP, montage-portable) + learned
channel-id embedding (non-portable) + sinusoidal temporal-index encoding.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


class CoordEncoding(nn.Module):
    """3-D electrode coordinate -> d-dim embedding via Fourier features + MLP."""

    def __init__(self, d_model: int, bands: int = 16, coord_scale: float = 10.0):
        super().__init__()
        self.coord_scale = coord_scale
        self.register_buffer("freqs", 2.0 ** torch.arange(bands) * math.pi)
        self.mlp = nn.Sequential(
            nn.Linear(3 * 2 * bands, d_model), nn.GELU(), nn.Linear(d_model, d_model)
        )

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        # coords: (B, C, 3) in metres (~±0.1). Scale to ~unit before Fourier.
        scaled = coords * self.coord_scale
        ang = scaled.unsqueeze(-1) * self.freqs  # (B, C, 3, bands)
        feats = torch.cat([ang.sin(), ang.cos()], dim=-1)  # (B, C, 3, 2*bands)
        feats = feats.flatten(start_dim=-2)  # (B, C, 6*bands)
        return self.mlp(feats)  # (B, C, d)


class PatchTokenizer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.patch_len = int(round(float(cfg.tokenizer.patch_s) * float(cfg.data.sfreq)))
        self.stride = int(round(float(cfg.tokenizer.patch_stride_s) * float(cfg.data.sfreq)))
        self.d = int(cfg.tokenizer.d_model)
        self.proj = nn.Linear(self.patch_len, self.d)
        self.coord = CoordEncoding(self.d, int(cfg.tokenizer.coord_fourier_bands))
        self.use_chan_id = bool(cfg.tokenizer.use_channel_id)
        if self.use_chan_id:
            self.chan_emb = nn.Embedding(int(cfg.data.n_scalp_channels), self.d)

    def _temporal(self, n_patches: int, device) -> torch.Tensor:
        pos = torch.arange(n_patches, device=device, dtype=torch.float32).unsqueeze(1)
        idx = torch.arange(self.d, device=device, dtype=torch.float32).unsqueeze(0)
        angle = pos / (10000.0 ** (2 * (idx // 2) / self.d))
        pe = torch.zeros(n_patches, self.d, device=device)
        pe[:, 0::2] = torch.sin(angle[:, 0::2])
        pe[:, 1::2] = torch.cos(angle[:, 1::2])
        return pe

    def forward(self, x: torch.Tensor, coords: torch.Tensor):
        """x: (B, C, T), coords: (B, C, 3).

        Returns ``content, pos, targets, C, P`` where content/pos are (B, N, d),
        targets are the raw patch values (B, N, patch_len), N = C * P.
        """
        B, C, _ = x.shape
        patches = x.unfold(-1, self.patch_len, self.stride)  # (B, C, P, patch_len)
        P = patches.shape[2]
        content = self.proj(patches)  # (B, C, P, d)

        coord_e = self.coord(coords)  # (B, C, d)
        pos = coord_e.unsqueeze(2).expand(B, C, P, self.d).clone()
        pos = pos + self._temporal(P, x.device).view(1, 1, P, self.d)
        if self.use_chan_id:
            ids = torch.arange(C, device=x.device)
            pos = pos + self.chan_emb(ids).view(1, C, 1, self.d)

        content = content.reshape(B, C * P, self.d)
        pos = pos.reshape(B, C * P, self.d)
        targets = patches.reshape(B, C * P, self.patch_len)
        return content, pos, targets, C, P
