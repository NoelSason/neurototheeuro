"""Downstream adaptation + evaluation for all model variants (emits result rows).

Every variant sees the SAME target-subject budget slice and the SAME fixed 30% test
trials. The head is identical across A1 and B. Rows follow the frozen results.csv
schema (context columns are stamped from ``ctx``).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader, TensorDataset

from ..data.epoch import SubjectArrays
from ..model.eegnet import build_eegnet
from ..model.heads import DecodeModel
from ..experiment.splits import assert_no_leakage, draw_budget, target_test_split


def _metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    y_pred = (y_prob >= 0.5).astype(int)
    out = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }
    try:
        out["auc"] = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else float("nan")
    except ValueError:
        out["auc"] = float("nan")
    return out


def _fit(model, opt, X, C, y, epochs, batch_size, device, seed):
    torch.manual_seed(seed)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(X), torch.from_numpy(C), torch.from_numpy(y).long()),
        batch_size=batch_size,
        shuffle=True,
    )
    model.train()
    for _ in range(epochs):
        for xb, cb, yb in loader:
            xb, cb, yb = xb.to(device), cb.to(device), yb.to(device)
            opt.zero_grad()
            loss = F.cross_entropy(model(xb, cb), yb)
            loss.backward()
            opt.step()
    return model


@torch.no_grad()
def _predict(model, X, C, device) -> np.ndarray:
    model.eval()
    logits = model(torch.from_numpy(X).to(device), torch.from_numpy(C).to(device))
    return torch.softmax(logits, dim=1)[:, 1].cpu().numpy()


def evaluate_target(
    target: SubjectArrays,
    cfg,
    budget_pct: float,
    seed: int,
    ctx: dict,
    device: str = "cpu",
    pretrained_state: dict | None = None,
    xsub_state: dict | None = None,
    shuffle_labels: bool = False,
    models: set | None = None,
) -> list[dict]:
    X, y, coords = target.epochs, target.labels, target.coords
    pool_idx, test_idx = target_test_split(y, float(cfg.experiment.target_test_frac), target.subject)
    train_idx = draw_budget(pool_idx, y, budget_pct, seed)
    assert_no_leakage(train_idx, test_idx)

    Xtr, ytr = X[train_idx], y[train_idx].copy()
    Xte, yte = X[test_idx], y[test_idx]
    if shuffle_labels:  # permutation sanity: destroy the train label mapping
        ytr = np.random.default_rng(seed).permutation(ytr)
    Ctr = np.repeat(coords[None], len(Xtr), axis=0)
    Cte = np.repeat(coords[None], len(Xte), axis=0)

    ft = cfg.decode.finetune
    lp = cfg.decode.linear_probe
    en = cfg.decode.eegnet
    requested = set(models) if models is not None else set(cfg.experiment.models)
    rows: list[dict] = []

    def emit(model_name, adaptation, probs):
        for metric, value in _metrics(yte, probs).items():
            rows.append(
                {
                    **ctx,
                    "hidden_subject": target.subject,
                    "seed": seed,
                    "target_budget_pct": budget_pct,
                    "model": model_name,
                    "adaptation": adaptation,
                    "metric": metric,
                    "value": float(value),
                    "n_train_trials": int(len(train_idx)),
                    "n_test_trials": int(len(test_idx)),
                    "class_pos_frac": float(np.mean(yte)) if len(yte) else float("nan"),
                }
            )

    def finetune_groups(model):
        return [
            {"params": model.encoder.parameters(), "lr": float(ft.encoder_lr)},
            {"params": model.head.parameters(), "lr": float(ft.head_lr)},
        ]

    # A1: matched transformer, random init, end-to-end
    if "A1_matched" in requested:
        m = DecodeModel(cfg).to(device)
        opt = torch.optim.AdamW(finetune_groups(m), weight_decay=float(ft.weight_decay))
        _fit(m, opt, Xtr, Ctr, ytr, int(ft.epochs), int(ft.batch_size), device, seed)
        emit("A1_matched", "from_scratch", _predict(m, Xte, Cte, device))

    # A2: EEGNet from scratch
    if "A2_eegnet" in requested:
        m = build_eegnet(cfg, X.shape[1], X.shape[2]).to(device)
        opt = torch.optim.AdamW(m.parameters(), lr=float(en.lr))
        _fit(m, opt, Xtr, Ctr, ytr, int(en.epochs), int(en.batch_size), device, seed)
        emit("A2_eegnet", "from_scratch", _predict(m, Xte, Cte, device))

    # B: SSL-pretrained encoder -> finetune (headline) + linear probe (rep read)
    if "B_ssl" in requested and pretrained_state is not None:
        m = DecodeModel(cfg).to(device)
        m.load_encoder_state(pretrained_state)
        opt = torch.optim.AdamW(finetune_groups(m), weight_decay=float(ft.weight_decay))
        _fit(m, opt, Xtr, Ctr, ytr, int(ft.epochs), int(ft.batch_size), device, seed)
        emit("B_ssl", "finetune", _predict(m, Xte, Cte, device))

        mp = DecodeModel(cfg).to(device)
        mp.load_encoder_state(pretrained_state)
        mp.freeze_encoder()
        opt = torch.optim.AdamW(mp.head.parameters(), lr=float(lp.lr))
        _fit(mp, opt, Xtr, Ctr, ytr, int(lp.epochs), int(lp.batch_size), device, seed)
        emit("B_ssl", "linearprobe", _predict(mp, Xte, Cte, device))

    # X-sub supervised anchor: supervised-pretrained encoder -> light finetune
    if "Xsub_sup" in requested and xsub_state is not None:
        m = DecodeModel(cfg).to(device)
        m.load_encoder_state(xsub_state)
        opt = torch.optim.AdamW(finetune_groups(m), weight_decay=float(ft.weight_decay))
        _fit(m, opt, Xtr, Ctr, ytr, int(ft.epochs), int(ft.batch_size), device, seed)
        emit("Xsub_sup", "finetune", _predict(m, Xte, Cte, device))

    return rows


def supervised_pretrain_encoder(study_arrays: list[SubjectArrays], cfg, device: str = "cpu", seed: int = 0) -> dict:
    """Train a DecodeModel supervised on pooled study epochs; return its encoder state
    (Model X-sub anchor)."""
    X = np.concatenate([a.epochs for a in study_arrays], axis=0)
    y = np.concatenate([a.labels for a in study_arrays], axis=0)
    C = np.concatenate([np.repeat(a.coords[None], len(a.epochs), axis=0) for a in study_arrays], axis=0)
    ft = cfg.decode.finetune
    m = DecodeModel(cfg).to(device)
    opt = torch.optim.AdamW(
        [
            {"params": m.encoder.parameters(), "lr": float(ft.encoder_lr)},
            {"params": m.head.parameters(), "lr": float(ft.head_lr)},
        ],
        weight_decay=float(ft.weight_decay),
    )
    _fit(m, opt, X.astype(np.float32), C.astype(np.float32), y, int(ft.epochs), int(ft.batch_size), device, seed)
    return {k: v.detach().cpu() for k, v in m.encoder.state_dict().items()}
