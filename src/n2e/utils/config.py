"""Config loader: YAML with optional ``_base`` inheritance + deep merge.

The frozen spec lives in ``configs/spec_v1.yaml``; overrides (``smoke.yaml``,
``spec_v1_patch0p5.yaml``) set ``_base`` and only restate what changes.
"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


class DotDict(dict):
    """dict with recursive attribute access (``cfg.encoder.depth``)."""

    def __getattr__(self, key: str) -> Any:
        try:
            return self[key]
        except KeyError as exc:  # pragma: no cover - trivial
            raise AttributeError(key) from exc

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value


def _wrap(obj: Any) -> Any:
    if isinstance(obj, dict):
        return DotDict({k: _wrap(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_wrap(v) for v in obj]
    return obj


def deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge ``override`` onto ``base`` (override wins)."""
    out = copy.deepcopy(base)
    for key, val in override.items():
        if key in out and isinstance(out[key], dict) and isinstance(val, dict):
            out[key] = deep_merge(out[key], val)
        else:
            out[key] = copy.deepcopy(val)
    return out


def _load_raw(path: Path) -> dict:
    path = Path(path)
    with open(path) as fh:
        raw = yaml.safe_load(fh) or {}
    base_name = raw.pop("_base", None)
    if base_name is None:
        return raw
    base = _load_raw(path.parent / base_name)
    return deep_merge(base, raw)


def load_config(path: str | Path) -> DotDict:
    """Load a config file, resolving ``_base`` inheritance, as a DotDict."""
    return _wrap(_load_raw(Path(path)))
