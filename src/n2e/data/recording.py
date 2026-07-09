"""Shared container for a single subject's raw recording + events + coords."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd


@dataclass
class SubjectRecording:
    """One subject: continuous raw, ds005273-style events, and channel coordinates.

    - ``raw``: an ``mne.io.BaseRaw`` at the original sampling rate.
    - ``events``: DataFrame with columns ``onset, duration, trial_type, sample``
      (``scode`` is derived downstream by :func:`n2e.data.codebook.read_events`).
    - ``coords``: ``{channel_name: (x, y, z) in metres}`` or ``None`` for channels
      without CapTrak coordinates (EOG1/EOG2/FCz).
    """

    subject: str
    raw: object
    events: pd.DataFrame
    coords: dict[str, Optional[tuple[float, float, float]]]
    sfreq_orig: float
    synthetic: bool = False
