"""Fetch real ds005273 signals from OpenNeuro (git-annex) + data-integrity report.

The local ``ds005273/`` in this repo carries BIDS metadata only and is not
annex-linked to OpenNeuro, so real signals are pulled from OpenNeuro's public
git-annex mirror. ``git-annex`` (system package, S3-enabled) is the transport;
``datalad`` is not required.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import pandas as pd

from . import codebook

OPENNEURO_GIT = "https://github.com/OpenNeuroDatasets/{ds}.git"


def clone_openneuro(dataset: str, dest: str | Path) -> Path:
    dest = Path(dest)
    if (dest / ".git").exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "clone", OPENNEURO_GIT.format(ds=dataset), str(dest)], check=True)
    return dest


def annex_get(bids_root: str | Path, rel_paths: list[str]) -> None:
    subprocess.run(["git", "-C", str(bids_root), "annex", "get", *rel_paths], check=True)


def fetch_subjects(dataset: str, dest: str | Path, subjects: list[str], task: str = "task-1") -> Path:
    """Clone the OpenNeuro mirror (once) and ``annex get`` each subject's triplet."""
    root = clone_openneuro(dataset, dest)
    for sub in subjects:
        triplet = [f"{sub}/eeg/{sub}_{task}_eeg.{ext}" for ext in ("vhdr", "vmrk", "eeg")]
        annex_get(root, triplet)
    return root


def integrity_report(bids_root: str | Path, subjects: list[str], task: str = "task-1") -> pd.DataFrame:
    """Load each subject's raw + events and report channels/sfreq/duration/events."""
    import mne

    rows = []
    for sub in subjects:
        eeg = Path(bids_root) / sub / "eeg"
        vhdr = eeg / f"{sub}_{task}_eeg.vhdr"
        rec: dict = {"subject": sub}
        try:
            raw = mne.io.read_raw_brainvision(vhdr, preload=False, verbose="ERROR")
            rec.update(
                loaded=True,
                n_channels=int(raw.info["nchan"]),
                sfreq=float(raw.info["sfreq"]),
                duration_s=round(float(raw.n_times / raw.info["sfreq"]), 1),
            )
        except Exception as exc:  # pragma: no cover - depends on fetched data
            rec.update(loaded=False, error=str(exc)[:120])
            rows.append(rec)
            continue
        ev_path = eeg / f"{sub}_{task}_events.tsv"
        if ev_path.exists():
            ev = codebook.read_events(ev_path)
            rec.update(n_events=int(len(ev)), n_stim=int(ev["scode"].notna().sum()))
        else:
            rec.update(n_events=0, n_stim=0, note="no events.tsv (SSL-only subject)")
        rows.append(rec)
    return pd.DataFrame(rows)
