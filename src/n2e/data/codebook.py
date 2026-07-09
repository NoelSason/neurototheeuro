"""Trigger-code parsing and decoding-contrast definitions for ds005273.

CRITICAL correctness rule
-------------------------
The trigger code lives in the ``trial_type`` string as ``"Stimulus/S NN"``.
The ``value`` column is a **per-file dense remap** (e.g. in sub-10 ``value 2`` == S1,
``value 16`` == S70) and is **NOT comparable across subjects**. Never derive labels
from ``value``. :func:`read_events` parses ``trial_type`` and drops ``value`` to
prevent accidental downstream use.

Observed S-code frequencies (sub-10, 9,310 events, ~897 trials)
--------------------------------------------------------------
Structural (≈1 per trial): S1 897, S4 897, S5 897, S6 898, S7 898, S10 897.
"50s" family: S50 521, S51 77, S55 445, S56 293, S57 7, S58 1, S59 152, S63 32.
"70s" family: S70 298, S71 600, S74 299, S75 299, S76 598, S77 300.

Notes for whoever holds the paper codebook:
- ``S70 + S71 = 898`` (exactly one per trial) → a clean binary condition, 1:2 split.
- ``S74 + S75 + S77 = 898`` (≈299 each) → a balanced 3-way condition.
- The seen/unseen semantics are UNKNOWN here and must be confirmed from the ds005273
  publication. Until then both contrasts below are marked ``provisional=True`` and
  every result row / Table 4 carries that flag. TODO(paper): fill the real mapping.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

_SCODE_RE = re.compile(r"S\s*(\d+)")


def parse_scode(trial_type: object) -> Optional[int]:
    """Extract the integer NN from a ``"Stimulus/S NN"`` ``trial_type`` string."""
    if not isinstance(trial_type, str):
        return None
    match = _SCODE_RE.search(trial_type)
    return int(match.group(1)) if match else None


def read_events(path: str | Path) -> pd.DataFrame:
    """Read a BIDS ``*_events.tsv``; add ``scode`` from ``trial_type``; drop ``value``."""
    df = pd.read_csv(path, sep="\t")
    df = df.copy()
    df["scode"] = df["trial_type"].map(parse_scode)
    if "value" in df.columns:  # guard: value is a non-portable per-file remap
        df = df.drop(columns=["value"])
    return df


def scode_frequencies(events: pd.DataFrame) -> dict[int, int]:
    valid = events.dropna(subset=["scode"])
    return {int(k): int(v) for k, v in valid["scode"].astype(int).value_counts().items()}


@dataclass(frozen=True)
class Contrast:
    """A binary (or n-ary) decoding contrast defined over trigger codes.

    - ``anchor_codes``: stimulus-onset codes to time-lock epochs on (one per trial).
    - ``class_codes``: ordered mapping ``{class_name: (codes,)}``; label index = order.
      A trial is assigned a class if exactly one class's codes appear in its window
      ``[anchor, next_anchor)``; otherwise it is invalid (dropped).
    - ``provisional``: True until the code→semantics mapping is confirmed from the paper.
    """

    name: str
    anchor_codes: tuple[int, ...]
    class_codes: dict[str, tuple[int, ...]]
    provisional: bool = True
    note: str = ""

    @property
    def class_names(self) -> list[str]:
        return list(self.class_codes.keys())


CONTRASTS: dict[str, Contrast] = {
    # De-risk contrast: maximally balanced, reliably-coded. 74 vs 75 are ~equal
    # frequency (299 vs 299 in sub-10). Semantics pending paper.
    "balanced_primary": Contrast(
        name="balanced_primary",
        anchor_codes=(70, 71),
        class_codes={"cond_A": (74,), "cond_B": (75,)},
        provisional=True,
        note=(
            "PROVISIONAL: S74 vs S75 chosen for near-equal frequency (299/299). "
            "Class semantics unknown pending the ds005273 codebook."
        ),
    ),
    # Headline contrast (reported second). Placeholder mapping — DO NOT report as a
    # scientific seen/unseen result until confirmed from the paper.
    "seen_unseen": Contrast(
        name="seen_unseen",
        anchor_codes=(70, 71),
        class_codes={"class0": (70,), "class1": (71,)},
        provisional=True,
        note=(
            "PROVISIONAL/STUB: S70 vs S71 placeholder (298/600). The true seen/unseen "
            "mapping MUST be confirmed from the paper. TODO(paper)."
        ),
    ),
}


def assign_trials(events: pd.DataFrame, contrast: Contrast) -> pd.DataFrame:
    """Segment events into trials and assign each a class label for ``contrast``.

    Returns a DataFrame with columns ``onset`` (s), ``sample`` (orig sfreq),
    ``anchor_scode``, ``label`` (0-based; -1 = invalid), ``valid``.
    """
    ev = events.copy()
    if "scode" not in ev.columns:  # synthetic / raw events -> derive from trial_type
        ev["scode"] = ev["trial_type"].map(parse_scode)
    ev = ev.dropna(subset=["scode"]).copy()
    ev["scode"] = ev["scode"].astype(int)
    ev = ev.sort_values("onset").reset_index(drop=True)

    anchors = ev[ev["scode"].isin(contrast.anchor_codes)].reset_index(drop=True)
    onsets = anchors["onset"].to_numpy(dtype=float)
    samples = anchors["sample"].to_numpy() if "sample" in anchors else np.zeros(len(anchors))
    next_onset = np.append(onsets[1:], np.inf) if len(onsets) else onsets

    code_to_label: dict[int, int] = {}
    for label, cname in enumerate(contrast.class_names):
        for code in contrast.class_codes[cname]:
            code_to_label[code] = label

    ev_onsets = ev["onset"].to_numpy(dtype=float)
    ev_codes = ev["scode"].to_numpy()
    labels = np.full(len(anchors), -1, dtype=int)
    for i in range(len(anchors)):
        window = (ev_onsets >= onsets[i]) & (ev_onsets < next_onset[i])
        present = {code_to_label[c] for c in ev_codes[window].tolist() if c in code_to_label}
        if len(present) == 1:
            labels[i] = present.pop()

    out = pd.DataFrame(
        {
            "onset": onsets,
            "sample": np.asarray(samples).astype(int),
            "anchor_scode": anchors["scode"].to_numpy() if len(anchors) else np.array([], int),
            "label": labels,
        }
    )
    out["valid"] = out["label"] >= 0
    return out


def balance_report(trials: pd.DataFrame) -> dict:
    """Per-class counts + positive fraction over valid trials (feeds Table 4)."""
    valid = trials[trials["valid"]]
    counts = {int(k): int(v) for k, v in valid["label"].value_counts().items()}
    n = int(len(valid))
    pos = counts.get(1, 0)
    return {
        "n_total": int(len(trials)),
        "n_valid": n,
        "class_counts": counts,
        "pos_frac": float(pos / n) if n else float("nan"),
    }
