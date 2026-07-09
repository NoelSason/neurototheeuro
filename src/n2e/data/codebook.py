"""Trigger-code parsing and decoding-contrast definitions for ds005273.

CRITICAL correctness rule
-------------------------
The trigger code lives in the ``trial_type`` string as ``"Stimulus/S NN"``.
The ``value`` column is a **per-file dense remap** (e.g. in sub-10 ``value 2`` == S1,
``value 16`` == S70) and is **NOT comparable across subjects**. Never derive labels
from ``value``. :func:`read_events` parses ``trial_type`` and drops ``value``.

Paradigm (recovered from the paper — codebook is NOT in the OpenNeuro repo)
--------------------------------------------------------------------------
Rodríguez-San Esteban, González-López, Chica, *Scientific Reports* 2025 (PMC11885812):
Gabor **orientation discrimination** (clockwise/counter, keys k/m) + a **binary
seen/unseen visibility report** (keys d/c) collected on *every* trial. Target
**present on 66.7%** of trials, **absent on 33.3%** (40/20 per 60-trial block; 600
experimental trials, plus localizer blocks → ~900 total markers/subject). Stimulus
50 ms, mask at +300 ms. Decodable per the paper: **stimulus presence/absence** and
**subjective awareness (seen/unseen)**; **orientation was NOT decodable**.

Trigger interpretation (from the paper's proportions + a marker contingency analysis
of sub-10; see git history / codebook notes)
- **Presence (CONFIRMED by the 67/33 ratio):** ``S71`` = present (600), ``S70`` = absent (298).
- Stimulus-locked balanced 3-way ``S74/S75/S77`` (~299 each, ~1/3 within *both* present
  and absent) = an unknown experimental factor (cue/contrast/SOA); ``S76`` = a 2/3 marker.
  These are NOT orientation (they occur on absent trials too).
- Response-family (~1.8 s post-stimulus): ``S59`` (present 151, absent 1) and ``S51``
  (present 77, absent 0) are **present-exclusive** → "seen/aware" candidates, but
  ambiguous with "orientation-correct" (also present-only). ``S50/S55/S56`` occur on
  both present and absent trials. TODO(confirm): disambiguate seen vs orientation-correct
  from the two-response temporal order once signals are fetched, or from the authors' script.

Observed S-code frequencies (sub-10): S1/4/5/6/7/10 ≈ 897 (structural, 1/trial);
S50 521, S51 77, S55 445, S56 293, S57 7, S58 1, S59 152, S63 32;
S70 298, S71 600, S74 299, S75 299, S76 598, S77 300.
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
    """A decoding contrast defined over trigger codes.

    - ``anchor_codes``: codes that mark an *eligible* trial for this contrast (the epoch
      is time-locked to the anchor). e.g. presence anchors on {70,71}; seen/unseen anchors
      only on present trials {71}.
    - ``boundary_codes``: codes that delimit trials (default = ``anchor_codes``). A trial's
      code window is ``[boundary, next_boundary)``. Set wider than ``anchor_codes`` when
      eligible trials are a subset (so windows aren't stretched across skipped trials).
    - ``class_codes``: ordered ``{class_name: (codes,)}``; label index = order. A trial is
      labelled by the single class whose codes appear in its window.
    - ``default_label``: label for an eligible trial whose window matches *no* class code
      (e.g. "unseen" = a present trial with no seen marker). ``None`` -> drop such trials.
    - ``provisional``: True until the code→semantics mapping is confirmed from paper/script.
    """

    name: str
    anchor_codes: tuple[int, ...]
    class_codes: dict[str, tuple[int, ...]]
    boundary_codes: Optional[tuple[int, ...]] = None
    default_label: Optional[int] = None
    provisional: bool = True
    note: str = ""

    @property
    def class_names(self) -> list[str]:
        return list(self.class_codes.keys())


CONTRASTS: dict[str, Contrast] = {
    # Primary de-risk contrast: stimulus present vs absent. Confirmed — the 600/298
    # split is exactly the paper's 66.7%/33.3% present/absent ratio. 2:1 imbalanced;
    # balanced-accuracy + AUC handle it, and budget draws are class-stratified.
    "presence": Contrast(
        name="presence",
        anchor_codes=(70, 71),
        class_codes={"absent": (70,), "present": (71,)},
        provisional=False,
        note="CONFIRMED: S71=present (66.7%), S70=absent (33.3%) — matches the paper.",
    ),
    # Headline contrast: subjective awareness on present trials. PROVISIONAL — the
    # present-exclusive response codes are seen/aware candidates but ambiguous with
    # orientation-correct. Every result carries the provisional flag until confirmed.
    "seen_unseen": Contrast(
        name="seen_unseen",
        anchor_codes=(71,),            # present trials only
        boundary_codes=(70, 71),       # but bound windows by all trial starts
        class_codes={"unseen": (), "seen": (51, 57, 58, 59)},
        default_label=0,               # present trial with no "seen" marker -> unseen
        provisional=True,
        note=(
            "PROVISIONAL: 'seen' = present trial carrying a present-exclusive response "
            "code {51,57,58,59}; 'unseen' = present trial without one. These codes are "
            "ambiguous with orientation-correct — confirm before reporting. TODO(paper/script)."
        ),
    ),
}


def assign_trials(events: pd.DataFrame, contrast: Contrast) -> pd.DataFrame:
    """Segment events into trials and assign each eligible trial a class label.

    Returns a DataFrame with columns ``onset`` (s), ``sample`` (orig sfreq),
    ``anchor_scode``, ``label`` (0-based; -1 = invalid/dropped), ``valid``.
    """
    ev = events.copy()
    if "scode" not in ev.columns:  # synthetic / raw events -> derive from trial_type
        ev["scode"] = ev["trial_type"].map(parse_scode)
    ev = ev.dropna(subset=["scode"]).copy()
    ev["scode"] = ev["scode"].astype(int)
    ev = ev.sort_values("onset").reset_index(drop=True)

    boundary_codes = contrast.boundary_codes or contrast.anchor_codes
    bnd = ev[ev["scode"].isin(boundary_codes)].reset_index(drop=True)
    b_onset = bnd["onset"].to_numpy(dtype=float)
    b_sample = bnd["sample"].to_numpy() if "sample" in bnd else np.zeros(len(bnd))
    b_code = bnd["scode"].to_numpy()
    next_onset = np.append(b_onset[1:], np.inf) if len(b_onset) else b_onset

    code_to_label: dict[int, int] = {}
    for label, cname in enumerate(contrast.class_names):
        for code in contrast.class_codes[cname]:
            code_to_label[code] = label

    ev_onsets = ev["onset"].to_numpy(dtype=float)
    ev_codes = ev["scode"].to_numpy()
    onsets, samples, anchors, labels = [], [], [], []
    for i in range(len(bnd)):
        if b_code[i] not in contrast.anchor_codes:  # trial not eligible for this contrast
            continue
        window = (ev_onsets >= b_onset[i]) & (ev_onsets < next_onset[i])
        present = {code_to_label[c] for c in ev_codes[window].tolist() if c in code_to_label}
        if len(present) == 1:
            label = present.pop()
        elif len(present) == 0 and contrast.default_label is not None:
            label = contrast.default_label
        else:
            label = -1  # ambiguous (>1 class) or unmatched with no default
        onsets.append(b_onset[i])
        samples.append(int(b_sample[i]))
        anchors.append(int(b_code[i]))
        labels.append(label)

    out = pd.DataFrame(
        {
            "onset": np.asarray(onsets, dtype=float),
            "sample": np.asarray(samples, dtype=int),
            "anchor_scode": np.asarray(anchors, dtype=int),
            "label": np.asarray(labels, dtype=int),
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
