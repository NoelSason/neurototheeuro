"""Codebook parsing must use trial_type (not value) and recover the real S-codes."""
from pathlib import Path

import pandas as pd

from n2e.data import codebook

REPO = Path(__file__).resolve().parents[1]
EVENTS = REPO / "ds005273/sub-10/eeg/sub-10_task-1_events.tsv"


def test_parse_scode_handles_spacing_and_non_stimulus():
    assert codebook.parse_scode("Stimulus/S  6") == 6
    assert codebook.parse_scode("Stimulus/S 70") == 70
    assert codebook.parse_scode("New Segment/") is None
    assert codebook.parse_scode(float("nan")) is None


def test_read_events_drops_value_and_recovers_frequencies():
    df = codebook.read_events(EVENTS)
    assert "value" not in df.columns  # guarded: value is a non-portable per-file remap
    assert "scode" in df.columns
    freqs = codebook.scode_frequencies(df)
    # exact counts from the ground-truth analysis of sub-10
    assert freqs[70] == 298 and freqs[71] == 600
    assert freqs[1] == 897 and freqs[74] == 299 and freqs[75] == 299


def test_value_column_is_not_the_scode():
    raw = pd.read_csv(EVENTS, sep="\t")
    row = raw[raw["trial_type"].str.contains("S 70", na=False)].iloc[0]
    # value 16 maps to S70 in sub-10 -> proves value != S-code
    assert int(row["value"]) == 16
    assert codebook.parse_scode(row["trial_type"]) == 70


def test_presence_contrast_matches_paper_ratio():
    df = codebook.read_events(EVENTS)
    trials = codebook.assign_trials(df, codebook.CONTRASTS["presence"])
    rep = codebook.balance_report(trials)
    assert rep["n_valid"] == 898  # 600 present (S71) + 298 absent (S70)
    assert rep["class_counts"] == {1: 600, 0: 298}
    assert abs(rep["pos_frac"] - 600 / 898) < 1e-6  # ~0.668 present (paper: 66.7%)


def test_seen_unseen_anchors_on_present_trials_only():
    df = codebook.read_events(EVENTS)
    trials = codebook.assign_trials(df, codebook.CONTRASTS["seen_unseen"])
    rep = codebook.balance_report(trials)
    assert rep["n_valid"] == 600  # only present trials are eligible
    assert set(rep["class_counts"]) == {0, 1}  # both seen and unseen present
    assert 0.1 < rep["pos_frac"] < 0.6  # minority reported seen (near-threshold)
