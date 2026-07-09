"""Deterministic synthetic ds005273-like recordings.

Mirrors the real montage (63 labelled channels incl. EOG1/EOG2/FCz) and trigger
structure (anchor S70/S71 + class S74/S75) so the *entire* real pipeline
(preprocess -> epoch -> tokenize -> SSL -> decode -> stats) runs offline for the
smoke run and unit tests. A posterior ERP whose sign flips with the label makes the
contrast genuinely (linearly) decodable, so the smoke can assert "signal is learnable"
while label-permutation collapses to chance.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .recording import SubjectRecording

# Real ds005273 channel order (EOG1/EOG2/FCz are the non-scalp / reference channels).
CH63 = [
    "Fp1", "Fz", "F3", "F7", "FT9", "FC5", "FC1", "C3", "T7", "EOG1", "CP5", "CP1",
    "Pz", "P3", "P7", "O1", "Oz", "O2", "P4", "P8", "EOG2", "CP6", "CP2", "C4", "T8",
    "FT10", "FC6", "FC2", "F4", "F8", "Fp2", "AF7", "AF3", "AFz", "F1", "F5", "FT7",
    "FC3", "C1", "C5", "TP7", "CP3", "P1", "P5", "PO7", "PO3", "POz", "PO4", "PO8",
    "P6", "P2", "CPz", "CP4", "TP8", "C6", "C2", "FC4", "FT8", "F6", "AF8", "AF4",
    "F2", "FCz",
]
NON_SCALP = {"EOG1", "EOG2", "FCz"}
POSTERIOR = {"O1", "Oz", "O2", "POz", "PO3", "PO4", "PO7", "PO8", "P1", "P2", "Pz", "P3", "P4"}


def _coord_for(idx: int) -> tuple[float, float, float]:
    """Deterministic point on the upper hemisphere (~0.09 m), stable per channel index."""
    rng = np.random.default_rng(1000 + idx)
    theta = rng.uniform(0.0, np.pi / 2)  # elevation
    phi = rng.uniform(0.0, 2 * np.pi)
    r = 0.09
    return (
        float(r * np.sin(theta) * np.cos(phi)),
        float(r * np.sin(theta) * np.sin(phi)),
        float(r * np.cos(theta)),
    )


def make_synthetic_subject(
    subject: str,
    cfg,
    seed: int = 0,
    n_trials: int = 100,
    trial_len_s: float = 3.0,
    anchor_offset_s: float = 1.0,
    effect_uv: float = 8.0,
    noise_uv: float = 15.0,
) -> SubjectRecording:
    import mne

    sfreq = float(cfg.data.sfreq_orig)
    line = float(cfg.data.line_freq)
    rng = np.random.default_rng(seed)
    ch_names = list(CH63)
    n_ch = len(ch_names)
    n_times = int(n_trials * trial_len_s * sfreq)
    t = np.arange(n_times) / sfreq

    # Background: white noise + per-channel-phase alpha (10 Hz) and line noise
    data = rng.normal(0.0, noise_uv, size=(n_ch, n_times)).astype(np.float32)
    phase_a = rng.uniform(0, 2 * np.pi, size=(n_ch, 1))
    phase_l = rng.uniform(0, 2 * np.pi, size=(n_ch, 1))
    data += (5.0 * np.sin(2 * np.pi * 10.0 * t[None, :] + phase_a)).astype(np.float32)
    data += (3.0 * np.sin(2 * np.pi * line * t[None, :] + phase_l)).astype(np.float32)

    # Labels (balanced) and a posterior ERP bump at +300 ms whose sign encodes the label
    labels = rng.integers(0, 2, size=n_trials)
    post_idx = [i for i, n in enumerate(ch_names) if n in POSTERIOR]
    erp_t = np.arange(-0.1, 0.5, 1.0 / sfreq)
    erp = np.exp(-0.5 * ((erp_t - 0.3) / 0.05) ** 2).astype(np.float32)

    rows: list[tuple[float, str]] = []
    for k in range(n_trials):
        base = k * trial_len_s
        anchor_on = base + anchor_offset_s
        label = int(labels[k])
        start = int((anchor_on - 0.1) * sfreq)
        stop = start + len(erp)
        if stop <= n_times:
            amp = effect_uv if label == 1 else -effect_uv
            for ci in post_idx:
                data[ci, start:stop] += amp * erp
        anchor_code = 71 if label == 1 else 70    # present vs absent (primary contrast)
        class_code = 75 if label == 1 else 74      # orientation-family marker
        rows.append((base + 0.5, "Stimulus/S 6"))
        rows.append((anchor_on, f"Stimulus/S {anchor_code}"))
        rows.append((anchor_on + 0.3, f"Stimulus/S {class_code}"))
        # present trials: emit a "seen" response marker on ~half (non-degenerate seen_unseen)
        if label == 1 and rng.random() < 0.5:
            rows.append((anchor_on + 0.5, "Stimulus/S 59"))
        rows.append((anchor_on + 0.8, "Stimulus/S 1"))

    events = pd.DataFrame(rows, columns=["onset", "trial_type"])
    events["duration"] = 0.001
    events["sample"] = (events["onset"] * sfreq).astype(int)
    events = events[["onset", "duration", "trial_type", "sample"]]

    coords = {n: (None if n in NON_SCALP else _coord_for(i)) for i, n in enumerate(ch_names)}
    info = mne.create_info(ch_names, sfreq, ch_types="eeg")
    raw = mne.io.RawArray(data * 1e-6, info, verbose="ERROR")  # µV -> V (MNE convention)
    return SubjectRecording(
        subject=subject, raw=raw, events=events, coords=coords, sfreq_orig=sfreq, synthetic=True
    )
