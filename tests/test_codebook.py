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


def test_assign_trials_balanced_primary_produces_two_classes():
    df = codebook.read_events(EVENTS)
    trials = codebook.assign_trials(df, codebook.CONTRASTS["balanced_primary"])
    rep = codebook.balance_report(trials)
    assert rep["n_valid"] > 0
    assert set(rep["class_counts"]).issubset({0, 1})
    assert 0.2 < rep["pos_frac"] < 0.8  # de-risk contrast is reasonably balanced
