"""Fetch real ds005273 signals from OpenNeuro (git-annex) + data-integrity report.

The local ``ds005273/`` in this repo carries BIDS metadata only and is not
annex-linked to OpenNeuro, so real signals are pulled from OpenNeuro's public
git-annex mirror. ``git-annex`` (system package, S3-enabled) is the transport;
``datalad`` is not required.
"""
from __future__ import annotations

import json
import os
import subprocess
import tempfile
import warnings
from collections.abc import Collection
from pathlib import Path

import pandas as pd

from . import codebook

OPENNEURO_GIT = "https://github.com/OpenNeuroDatasets/{ds}.git"
DEFAULT_OPENNEURO_VERSION = "1.0.0"
DEFAULT_SSL_ONLY_SUBJECTS = ("sub-17",)


def _git_output(root: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(root), *args], stderr=subprocess.DEVNULL, text=True
    ).strip()


def clone_openneuro(
    dataset: str,
    dest: str | Path,
    version: str = DEFAULT_OPENNEURO_VERSION,
) -> Path:
    dest = Path(dest)
    if (dest / ".git").exists():
        if version:
            expected = _git_output(dest, "rev-parse", f"{version}^{{commit}}")
            actual = _git_output(dest, "rev-parse", "HEAD")
            if actual != expected:
                raise RuntimeError(
                    f"existing {dataset} checkout is {actual[:12]}, expected tag "
                    f"{version} ({expected[:12]})"
                )
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["git", "clone"]
    if version:
        cmd += ["--branch", version, "--single-branch"]
    cmd += [OPENNEURO_GIT.format(ds=dataset), str(dest)]
    subprocess.run(cmd, check=True)
    return dest


def annex_get(bids_root: str | Path, rel_paths: list[str], jobs: int = 1) -> None:
    cmd = ["git", "-C", str(bids_root), "annex", "get"]
    if jobs > 1:
        cmd += ["--jobs", str(jobs)]
    subprocess.run([*cmd, *rel_paths], check=True)


def fetch_subjects(
    dataset: str,
    dest: str | Path,
    subjects: list[str],
    task: str = "task-1",
    version: str = DEFAULT_OPENNEURO_VERSION,
    jobs: int = 1,
    strict: bool = False,
) -> Path:
    """Clone the pinned mirror and fetch each subject's available BrainVision files.

    The pinned source has known omissions: ``sub-17`` and ``sub-37`` lack markers,
    while ``sub-41`` lacks both marker and signal. Available files are fetched and
    omissions are preserved for the integrity report; ``strict=True`` aborts on a
    missing header or signal.
    """
    if jobs < 1:
        raise ValueError("jobs must be >= 1")
    root = clone_openneuro(dataset, dest, version=version)
    to_fetch = []
    for sub in subjects:
        triplet = [f"{sub}/eeg/{sub}_{task}_eeg.{ext}" for ext in ("vhdr", "vmrk", "eeg")]
        present = [p for p in triplet if os.path.lexists(root / p)]
        required = {triplet[0], triplet[2]}
        missing_required = sorted(required - set(present))
        to_fetch.extend(present)
        if triplet[1] not in present:
            warnings.warn(f"{sub} has no source marker file: {triplet[1]}", stacklevel=2)
        if missing_required:
            message = f"{sub} is missing required source files: {', '.join(missing_required)}"
            if strict:
                raise FileNotFoundError(message)
            warnings.warn(message, stacklevel=2)
    if not to_fetch:
        raise FileNotFoundError("none of the requested source files exist in the dataset")
    annex_get(root, to_fetch, jobs=jobs)
    return root


def _header_reference(text: str, key: str) -> str:
    prefix = f"{key}="
    for line in text.splitlines():
        if line.startswith(prefix):
            return line[len(prefix) :].strip()
    raise ValueError(f"BrainVision header has no {key} entry")


def _raw_metadata(vhdr: Path) -> dict:
    """Load BrainVision metadata, injecting an empty marker only when source lacks it."""
    import mne

    header = vhdr.read_text(encoding="utf-8-sig")
    marker_name = _header_reference(header, "MarkerFile")
    marker = vhdr.parent / marker_name
    marker_is_source_file = os.path.lexists(marker)
    if marker_is_source_file and not marker.exists():
        raise FileNotFoundError(f"marker annex content is not materialized: {marker}")
    marker_shim = not marker_is_source_file

    if not marker_shim:
        raw = mne.io.read_raw_brainvision(vhdr, preload=False, verbose="ERROR")
        return {
            "n_channels": int(raw.info["nchan"]),
            "sfreq": float(raw.info["sfreq"]),
            "duration_s": round(float(raw.n_times / raw.info["sfreq"]), 1),
            "marker_shim": False,
        }

    data_name = _header_reference(header, "DataFile")
    data = vhdr.parent / data_name
    if not data.exists():
        raise FileNotFoundError(data)
    with tempfile.TemporaryDirectory(prefix="n2e-brainvision-") as tmp:
        tmp_root = Path(tmp)
        tmp_vhdr = tmp_root / vhdr.name
        tmp_vhdr.write_text(header, encoding="utf-8")
        os.symlink(data.resolve(), tmp_root / data_name)
        (tmp_root / marker_name).write_text(
            "Brain Vision Data Exchange Marker File, Version 1.0\n\n"
            "[Common Infos]\n"
            "Codepage=UTF-8\n"
            f"DataFile={data_name}\n\n"
            "[Marker Infos]\n",
            encoding="utf-8",
        )
        raw = mne.io.read_raw_brainvision(tmp_vhdr, preload=False, verbose="ERROR")
        return {
            "n_channels": int(raw.info["nchan"]),
            "sfreq": float(raw.info["sfreq"]),
            "duration_s": round(float(raw.n_times / raw.info["sfreq"]), 1),
            "marker_shim": True,
        }


def integrity_report(
    bids_root: str | Path,
    subjects: list[str],
    task: str = "task-1",
    ssl_only: Collection[str] = (),
) -> pd.DataFrame:
    """Load each subject's raw + events and report channels/sfreq/duration/events."""
    ssl_only = set(ssl_only)
    rows = []
    for sub in subjects:
        eeg = Path(bids_root) / sub / "eeg"
        vhdr = eeg / f"{sub}_{task}_eeg.vhdr"
        vmrk = eeg / f"{sub}_{task}_eeg.vmrk"
        binary = eeg / f"{sub}_{task}_eeg.eeg"
        rec: dict = {
            "subject": sub,
            "source_has_vhdr": os.path.lexists(vhdr),
            "source_has_vmrk": os.path.lexists(vmrk),
            "source_has_eeg": os.path.lexists(binary),
            "has_vhdr": vhdr.exists(),
            "has_vmrk": vmrk.exists(),
            "has_eeg": binary.exists(),
        }
        rec["source_triplet_complete"] = bool(
            rec["source_has_vhdr"] and rec["source_has_vmrk"] and rec["source_has_eeg"]
        )
        json_path = eeg / f"{sub}_{task}_eeg.json"
        try:
            rec["expected_duration_s"] = round(
                float(json.loads(json_path.read_text())["RecordingDuration"]), 1
            )
        except Exception as exc:
            rec.update(duration_match=False, metadata_error=str(exc)[:120])
        try:
            meta = _raw_metadata(vhdr)
            rec.update(loaded=True, **meta)
        except Exception as exc:  # pragma: no cover - depends on fetched data
            rec.update(loaded=False, error=str(exc)[:120])
        if rec.get("loaded") and "expected_duration_s" in rec:
            rec.update(
                duration_delta_s=round(
                    rec["duration_s"] - rec["expected_duration_s"], 1
                ),
                duration_match=abs(
                    rec["duration_s"] - rec["expected_duration_s"]
                )
                <= 0.1,
            )
        else:
            rec["duration_match"] = False
        ev_path = eeg / f"{sub}_{task}_events.tsv"
        if ev_path.exists():
            try:
                ev = codebook.read_events(ev_path)
                frequencies = codebook.scode_frequencies(ev)
                n_events = int(len(ev))
                n_stim = int(ev["scode"].notna().sum())
                n_absent = int(frequencies.get(70, 0))
                n_present = int(frequencies.get(71, 0))
                rec.update(
                    n_events=n_events,
                    n_stim=n_stim,
                    n_presence_absent=n_absent,
                    n_presence_present=n_present,
                    presence_anchors_ok=n_absent > 0 and n_present > 0,
                )
                if rec.get("marker_shim"):
                    rec["note"] = (
                        "source .vmrk absent, loaded with empty-marker shim; "
                        "labels from events.tsv"
                    )
                rec["events_ok"] = bool(
                    n_events > 0
                    and n_stim > 0
                    and rec["presence_anchors_ok"]
                )
            except Exception as exc:
                rec.update(
                    n_events=0,
                    n_stim=0,
                    events_ok=False,
                    events_error=str(exc)[:120],
                )
        else:
            if sub in ssl_only:
                note = "no events.tsv (declared SSL-only subject)"
                if rec.get("marker_shim"):
                    note += "; source .vmrk absent, loaded with empty-marker shim"
                rec.update(n_events=0, n_stim=0, events_ok=True, note=note)
            else:
                rec.update(
                    n_events=0,
                    n_stim=0,
                    events_ok=False,
                    note="missing required events.tsv for decodable subject",
                )
        rec["integrity_ok"] = bool(
            rec.get("loaded")
            and rec["has_vhdr"]
            and rec["has_eeg"]
            and rec.get("n_channels") == 63
            and rec.get("sfreq") == 1000.0
            and rec.get("duration_match")
            and rec.get("events_ok")
        )
        # The current training loader still reads BrainVision files directly. A
        # source-missing marker can be inspected here, but Opus must share this
        # shim with preprocessing before such a subject is execution-ready.
        rec["pipeline_ready"] = bool(rec["integrity_ok"] and not rec.get("marker_shim"))
        rows.append(rec)
    return pd.DataFrame(rows)
