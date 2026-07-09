"""Build decoding epochs + SSL crops (normalized) with per-channel coordinates."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import codebook
from .codebook import Contrast
from .preprocess import Preprocessed
from ..utils.io import ensure_dir


@dataclass
class SubjectArrays:
    subject: str
    epochs: np.ndarray      # (n_epochs, n_ch, n_time) robust-z normalized
    labels: np.ndarray      # (n_epochs,) int
    crops: np.ndarray       # (n_crops, n_ch, n_time_crop) normalized (SSL, unlabeled)
    coords: np.ndarray      # (n_ch, 3) metres
    ch_names: list[str]
    contrast: str
    provisional: bool
    balance: dict
    synthetic: bool = False


def _make_crops(data_z: np.ndarray, crop_len: int) -> np.ndarray:
    n_ch, n_times = data_z.shape
    n = n_times // crop_len
    if n <= 0:
        return np.zeros((0, n_ch, crop_len), dtype=np.float32)
    trimmed = data_z[:, : n * crop_len].reshape(n_ch, n, crop_len).transpose(1, 0, 2)
    return np.ascontiguousarray(trimmed, dtype=np.float32)


def build_subject_arrays(pp: Preprocessed, contrast: Contrast, cfg, make_crops: bool = True) -> SubjectArrays:
    sfreq = float(cfg.data.sfreq)
    tmin, tmax = float(cfg.data.decode_tmin), float(cfg.data.decode_tmax)
    bmin, bmax = [float(x) for x in cfg.data.decode_baseline]

    data = pp.raw.get_data()  # (n_ch, n_times) Volts
    n_ch, n_times = data.shape
    epoch_len = int(round((tmax - tmin) * sfreq))
    b0 = int(round((bmin - tmin) * sfreq))
    b1 = int(round((bmax - tmin) * sfreq))
    reject_v = float(cfg.data.epoch_reject_ptp_uv) * 1e-6

    trials = codebook.assign_trials(pp.events, contrast)
    epochs: list[np.ndarray] = []
    labels: list[int] = []
    for _, tr in trials[trials["valid"]].iterrows():
        start = int(round((float(tr["onset"]) + tmin) * sfreq))
        stop = start + epoch_len
        if start < 0 or stop > n_times:
            continue
        seg = data[:, start:stop].copy()
        seg = seg - seg[:, b0:b1].mean(axis=1, keepdims=True)  # baseline correction
        ptp = seg.max(axis=1) - seg.min(axis=1)
        if np.any(ptp > reject_v):  # amplitude-threshold rejection (µV)
            continue
        seg = (seg - pp.norm_median) / pp.norm_iqr  # robust-z (leak-free continuous stats)
        epochs.append(seg.astype(np.float32))
        labels.append(int(tr["label"]))

    epochs_arr = np.stack(epochs) if epochs else np.zeros((0, n_ch, epoch_len), np.float32)
    labels_arr = np.asarray(labels, dtype=int)

    crops = np.zeros((0, n_ch, int(cfg.data.ssl_crop_s * sfreq)), np.float32)
    if make_crops:
        data_z = (data - pp.norm_median) / pp.norm_iqr
        crops = _make_crops(data_z, int(cfg.data.ssl_crop_s * sfreq))

    return SubjectArrays(
        subject=pp.subject,
        epochs=epochs_arr,
        labels=labels_arr,
        crops=crops,
        coords=pp.coords.astype(np.float32),
        ch_names=list(pp.scalp_names),
        contrast=contrast.name,
        provisional=contrast.provisional,
        balance=codebook.balance_report(trials),
        synthetic=pp.synthetic,
    )


def save_subject_arrays(arr: SubjectArrays, cache_dir: str | Path) -> Path:
    ensure_dir(cache_dir)
    path = Path(cache_dir) / f"{arr.subject}_{arr.contrast}.npz"
    np.savez_compressed(
        path,
        epochs=arr.epochs,
        labels=arr.labels,
        crops=arr.crops,
        coords=arr.coords,
        ch_names=np.array(arr.ch_names),
        meta=np.array(
            [arr.subject, arr.contrast, int(arr.provisional), int(arr.synthetic)], dtype=object
        ),
        balance=np.array([arr.balance], dtype=object),
    )
    return path


def load_subject_arrays(path: str | Path) -> SubjectArrays:
    z = np.load(path, allow_pickle=True)
    subject, contrast, provisional, synthetic = z["meta"]
    return SubjectArrays(
        subject=str(subject),
        epochs=z["epochs"],
        labels=z["labels"],
        crops=z["crops"],
        coords=z["coords"],
        ch_names=[str(c) for c in z["ch_names"]],
        contrast=str(contrast),
        provisional=bool(int(provisional)),
        balance=dict(z["balance"][0]),
        synthetic=bool(int(synthetic)),
    )
