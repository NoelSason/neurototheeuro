"""Lightweight stdout logger + git provenance."""
from __future__ import annotations

import logging
import subprocess
import sys

from .io import repo_root


def get_logger(name: str = "n2e", level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(name)s | %(message)s", "%H:%M:%S")
        )
        logger.addHandler(handler)
        logger.setLevel(level)
        logger.propagate = False
    return logger


def git_sha(short: bool = True) -> str:
    try:
        sha = (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=repo_root(), stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
        return sha[:7] if short else sha
    except Exception:
        return "nogit"
