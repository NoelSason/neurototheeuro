"""MNE preprocessing: band-pass, notch, resample, 60-ch select, CAR, robust-z stats.

Everything stays in MNE's native Volts internally; the µV rejection threshold is
converted at epoch time. Robust-z statistics are computed on the continuous CAR'd
recording (unsupervised, no train/test leakage) and applied per-epoch downstream.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from . import codebook
from .recording import SubjectRecording


@dataclass
class Preprocessed:
    subject: str
    raw: object                 # mne Raw, CAR'd, at cfg.data.sfreq, native Volts
    events: pd.DataFrame
    scalp_names: list[str]
    coords: np.ndarray          # (n_ch, 3) metres, aligned to scalp_names
    norm_median: np.ndarray     # (n_ch, 1) Volts
    norm_iqr: np.ndarray        # (n_ch, 1) Volts
    synthetic: bool = False


def read_electrode_coords(electrodes_tsv: str | Path) -> dict[str, Optional[tuple]]:
    df = pd.read_csv(electrodes_tsv, sep="\t")
    coords: dict[str, Optional[tuple]] = {}
    for _, row in df.iterrows():
        try:
            coords[str(row["name"])] = (float(row["x"]), float(row["y"]), float(row["z"]))
        except (ValueError, TypeError):  # 'n/a' -> no coordinate (EOG1/EOG2/FCz)
            coords[str(row["name"])] = None
    return coords


def load_real_subject(bids_root: str | Path, subject: str, cfg) -> SubjectRecording:
    """Load a ds005273 subject: BrainVision raw + events.tsv + CapTrak electrodes."""
    import mne

    eeg_dir = Path(bids_root) / subject / "eeg"
    vhdr = eeg_dir / f"{subject}_task-1_eeg.vhdr"
    raw = mne.io.read_raw_brainvision(vhdr, preload=True, verbose="ERROR")
    events = codebook.read_events(eeg_dir / f"{subject}_task-1_events.tsv")
    coords = read_electrode_coords(eeg_dir / f"{subject}_space-CapTrak_electrodes.tsv")
    return SubjectRecording(
        subject=subject,
        raw=raw,
        events=events,
        coords=coords,
        sfreq_orig=float(raw.info["sfreq"]),
        synthetic=False,
    )


def preprocess_recording(rec: SubjectRecording, cfg) -> Preprocessed:
    raw = rec.raw.copy()
    lo, hi = cfg.data.bandpass
    raw.filter(float(lo), float(hi), fir_design="firwin", verbose="ERROR")

    # Notch at line frequency + first harmonic (redundant under the 40 Hz low-pass,
    # applied for spec fidelity; harmless).
    nyq = float(rec.sfreq_orig) / 2.0
    notch = [f for f in (float(cfg.data.line_freq), 2 * float(cfg.data.line_freq)) if f < nyq]
    if notch:
        raw.notch_filter(notch, verbose="ERROR")

    raw.resample(float(cfg.data.sfreq), verbose="ERROR")

    drop = [c for c in cfg.data.drop_channels if c in raw.ch_names]
    if drop:
        raw.drop_channels(drop)

    raw.set_eeg_reference("average", projection=False, verbose="ERROR")

    data = raw.get_data()  # (n_ch, n_times) Volts, CAR'd
    median = np.median(data, axis=1, keepdims=True)
    q75, q25 = np.percentile(data, [75, 25], axis=1, keepdims=True)
    iqr = (q75 - q25) / 1.349
    iqr[iqr == 0] = 1.0

    scalp_names = list(raw.ch_names)
    coords = np.array(
        [rec.coords.get(n) if rec.coords.get(n) is not None else (0.0, 0.0, 0.0) for n in scalp_names],
        dtype=float,
    )
    return Preprocessed(
        subject=rec.subject,
        raw=raw,
        events=rec.events,
        scalp_names=scalp_names,
        coords=coords,
        norm_median=median,
        norm_iqr=iqr,
        synthetic=rec.synthetic,
    )
