"""Self-supervised masked-patch pretraining -> Model B encoder weights."""
from __future__ import annotations

import math

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from ..model.ssl import MaskedPatchSSL
from ..utils.logutil import get_logger

log = get_logger()


def _cosine_warmup(base_lr: float, total_steps: int, warmup_frac: float):
    warmup = max(1, int(total_steps * warmup_frac))

    def lr_at(step: int) -> float:
        if step < warmup:
            return base_lr * step / warmup
        prog = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * base_lr * (1 + math.cos(math.pi * prog))

    return lr_at


def pretrain_encoder(crops: np.ndarray, coords: np.ndarray, cfg, device: str = "cpu", seed: int = 0) -> dict:
    """SSL-pretrain on ``crops`` (N,C,Tc) with per-crop ``coords`` (N,C,3).

    Returns the encoder ``state_dict`` (CPU tensors) for downstream Model B.
    """
    if len(crops) == 0:
        raise ValueError("no SSL crops provided for pretraining")
    torch.manual_seed(seed)
    o = cfg.ssl.optim
    model = MaskedPatchSSL(cfg).to(device)
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=float(o.lr),
        betas=tuple(float(b) for b in o.betas),
        weight_decay=float(o.weight_decay),
    )
    loader = DataLoader(
        TensorDataset(torch.from_numpy(crops), torch.from_numpy(coords)),
        batch_size=int(o.batch_size),
        shuffle=True,
    )
    epochs = int(o.epochs)
    lr_at = _cosine_warmup(float(o.lr), epochs * max(1, len(loader)), float(o.warmup_frac))
    step = 0
    model.train()
    for ep in range(epochs):
        total, nb = 0.0, 0
        for xb, cb in loader:
            for g in opt.param_groups:
                g["lr"] = lr_at(step)
            xb, cb = xb.to(device), cb.to(device)
            opt.zero_grad()
            loss = model(xb, cb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(o.grad_clip))
            opt.step()
            step += 1
            total += loss.item()
            nb += 1
        if ep == 0 or (ep + 1) % max(1, epochs // 5) == 0:
            log.info("[SSL] epoch %d/%d recon_mse=%.4f", ep + 1, epochs, total / max(1, nb))
    return {k: v.detach().cpu() for k, v in model.encoder.state_dict().items()}
