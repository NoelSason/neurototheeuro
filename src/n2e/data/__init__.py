"""Data layer: fetch, codebook parsing, preprocessing, epoching."""
from . import codebook, epoch, fetch, preprocess, synthetic
from .codebook import CONTRASTS, Contrast, assign_trials, balance_report, read_events
from .epoch import SubjectArrays, build_subject_arrays, load_subject_arrays, save_subject_arrays
from .preprocess import Preprocessed, load_real_subject, preprocess_recording
from .recording import SubjectRecording
from .synthetic import make_synthetic_subject

__all__ = [
    "codebook",
    "epoch",
    "fetch",
    "preprocess",
    "synthetic",
    "CONTRASTS",
    "Contrast",
    "assign_trials",
    "balance_report",
    "read_events",
    "SubjectArrays",
    "build_subject_arrays",
    "load_subject_arrays",
    "save_subject_arrays",
    "Preprocessed",
    "load_real_subject",
    "preprocess_recording",
    "SubjectRecording",
    "make_synthetic_subject",
]
