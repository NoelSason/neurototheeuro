"""Filesystem helpers and canonical repo paths."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

# src/n2e/utils/io.py -> parents[3] == repo root
_REPO_ROOT = Path(__file__).resolve().parents[3]


def repo_root() -> Path:
    return _REPO_ROOT


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_npz(path: str | Path, **arrays: Any) -> None:
    ensure_dir(Path(path).parent)
    np.savez_compressed(path, **arrays)


def load_npz(path: str | Path):
    return np.load(path, allow_pickle=True)


def save_json(path: str | Path, obj: Any) -> None:
    ensure_dir(Path(path).parent)
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=2, default=str)


def load_json(path: str | Path) -> Any:
    with open(path) as fh:
        return json.load(fh)
