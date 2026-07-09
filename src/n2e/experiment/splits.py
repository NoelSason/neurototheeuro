"""Rotating-LOSO splits: fixed per-subject test partition, stratified budget draws.

The 30% target test partition is seeded *only* by the subject id, so it is identical
across budgets, seeds, and models — every comparison is on the same held-out trials.
Budget draws are stratified and seeded by the experiment seed.
"""
from __future__ import annotations

import hashlib

import numpy as np
from sklearn.model_selection import train_test_split


def stable_seed(*parts) -> int:
    """Deterministic seed from arbitrary parts (unlike salted ``hash()``)."""
    s = "_".join(str(p) for p in parts)
    return int(hashlib.md5(s.encode()).hexdigest(), 16) % (2**31)


def target_test_split(labels: np.ndarray, test_frac: float, subject: str):
    """Return ``(pool_idx, test_idx)`` — a fixed stratified split keyed by subject."""
    idx = np.arange(len(labels))
    seed = stable_seed(subject, "test")
    pool_idx, test_idx = train_test_split(
        idx, test_size=test_frac, stratify=labels, random_state=seed
    )
    return np.sort(pool_idx), np.sort(test_idx)


def draw_budget(pool_idx: np.ndarray, labels: np.ndarray, budget_pct: float, seed: int) -> np.ndarray:
    """Stratified subsample of ``pool_idx`` at ``budget_pct``% (>=1 per class)."""
    if budget_pct >= 100:
        return np.sort(pool_idx)
    rng = np.random.default_rng(seed)
    y = labels[pool_idx]
    chosen = []
    for cls in np.unique(y):
        cls_idx = pool_idx[y == cls]
        k = max(1, int(round(len(cls_idx) * budget_pct / 100.0)))
        chosen.append(rng.choice(cls_idx, size=min(k, len(cls_idx)), replace=False))
    return np.sort(np.concatenate(chosen))


def study_pool_subjects(decodable: list[str], target: str, pool_size: int) -> list[str]:
    """Pick ``pool_size`` study subjects (excluding target), deterministically."""
    others = [s for s in decodable if s != target]
    if pool_size >= len(others):
        return sorted(others)
    rng = np.random.default_rng(stable_seed(target, "pool", pool_size))
    return sorted(rng.choice(others, size=pool_size, replace=False).tolist())


def assert_no_leakage(train_idx: np.ndarray, test_idx: np.ndarray) -> None:
    overlap = np.intersect1d(train_idx, test_idx)
    if overlap.size:
        raise AssertionError(f"train/test overlap on {overlap.size} trials: {overlap[:5]}")
