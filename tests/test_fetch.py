from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from n2e.data import fetch


def _source_entry(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.symlink_to(path.parent / "annex-object-does-not-need-to-exist")


def test_fetch_subjects_allows_missing_marker_for_ssl_only(tmp_path, monkeypatch):
    root = tmp_path / "dataset"
    _source_entry(root / "sub-17/eeg/sub-17_task-1_eeg.vhdr")
    _source_entry(root / "sub-17/eeg/sub-17_task-1_eeg.eeg")
    calls = []

    monkeypatch.setattr(fetch, "clone_openneuro", lambda *args, **kwargs: root)
    monkeypatch.setattr(
        fetch,
        "annex_get",
        lambda got_root, paths, jobs=1: calls.append((got_root, paths, jobs)),
    )

    with pytest.warns(UserWarning, match="sub-17 has no source marker file"):
        fetch.fetch_subjects("ds005273", root, ["sub-17"], jobs=4)

    assert calls == [
        (
            root,
            [
                "sub-17/eeg/sub-17_task-1_eeg.vhdr",
                "sub-17/eeg/sub-17_task-1_eeg.eeg",
            ],
            4,
        )
    ]


def test_fetch_subjects_rejects_missing_required_binary(tmp_path, monkeypatch):
    root = tmp_path / "dataset"
    _source_entry(root / "sub-17/eeg/sub-17_task-1_eeg.vhdr")
    monkeypatch.setattr(fetch, "clone_openneuro", lambda *args, **kwargs: root)

    with pytest.warns(UserWarning, match="sub-17 has no source marker file"):
        with pytest.raises(FileNotFoundError, match="missing required source files"):
            fetch.fetch_subjects("ds005273", root, ["sub-17"], strict=True)


def test_fetch_subjects_reports_missing_required_binary_and_continues(tmp_path, monkeypatch):
    root = tmp_path / "dataset"
    header = root / "sub-41/eeg/sub-41_task-1_eeg.vhdr"
    _source_entry(header)
    calls = []
    monkeypatch.setattr(fetch, "clone_openneuro", lambda *args, **kwargs: root)
    monkeypatch.setattr(
        fetch,
        "annex_get",
        lambda got_root, paths, jobs=1: calls.append((got_root, paths, jobs)),
    )

    with pytest.warns(UserWarning) as caught:
        fetch.fetch_subjects("ds005273", root, ["sub-41"], jobs=2)

    messages = [str(w.message) for w in caught]
    assert any("sub-41 has no source marker file" in message for message in messages)
    assert any("sub-41 is missing required source files" in message for message in messages)
    assert calls == [(root, ["sub-41/eeg/sub-41_task-1_eeg.vhdr"], 2)]


def test_raw_metadata_uses_empty_marker_shim(tmp_path):
    eeg = tmp_path / "sub-17_task-1_eeg.eeg"
    np.arange(10, dtype=np.float32).tofile(eeg)
    vhdr = tmp_path / "sub-17_task-1_eeg.vhdr"
    vhdr.write_text(
        "Brain Vision Data Exchange Header File Version 1.0\n"
        "[Common Infos]\n"
        "Codepage=UTF-8\n"
        f"DataFile={eeg.name}\n"
        "MarkerFile=sub-17_task-1_eeg.vmrk\n"
        "DataFormat=BINARY\n"
        "DataOrientation=MULTIPLEXED\n"
        "NumberOfChannels=1\n"
        "SamplingInterval=1000\n"
        "[Binary Infos]\n"
        "BinaryFormat=IEEE_FLOAT_32\n"
        "[Channel Infos]\n"
        "Ch1=Cz,,1,µV\n",
        encoding="utf-8",
    )

    meta = fetch._raw_metadata(vhdr)

    assert meta == {
        "n_channels": 1,
        "sfreq": 1000.0,
        "duration_s": 0.0,
        "marker_shim": True,
    }


def test_raw_metadata_rejects_unmaterialized_marker_annex_object(tmp_path):
    eeg = tmp_path / "sample.eeg"
    np.arange(10, dtype=np.float32).tofile(eeg)
    marker = tmp_path / "sample.vmrk"
    marker.symlink_to(tmp_path / "missing-annex-object")
    vhdr = tmp_path / "sample.vhdr"
    vhdr.write_text(
        "Brain Vision Data Exchange Header File Version 1.0\n"
        "[Common Infos]\n"
        "DataFile=sample.eeg\n"
        "MarkerFile=sample.vmrk\n",
        encoding="utf-8",
    )

    with pytest.raises(FileNotFoundError, match="marker annex content is not materialized"):
        fetch._raw_metadata(vhdr)


def test_integrity_report_requires_explicit_ssl_only_subject(tmp_path, monkeypatch):
    eeg_dir = tmp_path / "sub-17/eeg"
    eeg_dir.mkdir(parents=True)
    (eeg_dir / "sub-17_task-1_eeg.vhdr").write_text("placeholder", encoding="utf-8")
    (eeg_dir / "sub-17_task-1_eeg.eeg").write_bytes(b"placeholder")
    (eeg_dir / "sub-17_task-1_eeg.json").write_text(
        '{"RecordingDuration": 10.0}', encoding="utf-8"
    )
    monkeypatch.setattr(
        fetch,
        "_raw_metadata",
        lambda path: {
            "n_channels": 63,
            "sfreq": 1000.0,
            "duration_s": 10.0,
            "marker_shim": True,
        },
    )

    undeclared = fetch.integrity_report(tmp_path, ["sub-17"]).iloc[0]
    declared = fetch.integrity_report(
        tmp_path, ["sub-17"], ssl_only={"sub-17"}
    ).iloc[0]

    assert not bool(undeclared["events_ok"])
    assert not bool(undeclared["integrity_ok"])
    assert bool(declared["events_ok"])
    assert bool(declared["integrity_ok"])
    assert not bool(declared["pipeline_ready"])


def test_integrity_report_keeps_metadata_and_events_when_signal_is_missing(tmp_path):
    eeg_dir = tmp_path / "sub-41/eeg"
    eeg_dir.mkdir(parents=True)
    (eeg_dir / "sub-41_task-1_eeg.json").write_text(
        '{"RecordingDuration": 123.4}', encoding="utf-8"
    )
    (eeg_dir / "sub-41_task-1_events.tsv").write_text(
        "onset\tduration\ttrial_type\tvalue\tsample\n"
        "0.0\t0.001\tStimulus/S 74\t1\t0\n",
        encoding="utf-8",
    )

    report = fetch.integrity_report(tmp_path, ["sub-41"]).iloc[0]

    assert not bool(report["loaded"])
    assert not bool(report["integrity_ok"])
    assert report["expected_duration_s"] == 123.4
    assert report["n_events"] == 1
    assert report["n_stim"] == 1
    assert not bool(report["presence_anchors_ok"])
    assert not bool(report["events_ok"])


def test_fetch_cli_persists_failed_report_and_requires_override(tmp_path):
    script = Path(__file__).resolve().parents[1] / "scripts/fetch_data.py"
    report = tmp_path / "integrity.csv"
    base = [
        sys.executable,
        str(script),
        "--dest",
        str(tmp_path),
        "--subjects",
        "sub-41",
        "--report-only",
        "--ssl-only",
        "--report",
        str(report),
    ]

    failed = subprocess.run(base, text=True, capture_output=True)

    assert failed.returncode == 1
    assert report.exists()
    assert "integrity gate failed" in failed.stderr

    allowed = subprocess.run(
        [*base, "--allow-integrity-failures"], text=True, capture_output=True
    )

    assert allowed.returncode == 0
    assert report.exists()
