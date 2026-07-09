# neurototheeuro — Experiment Brief (GPT-5.6-Sol / Codex hand-off)

> Copy-paste this to the GPT-5.6-Sol (Codex) agent to run the Step-1 de-risking
> experiment. This track is assigned to **GPT-5.6-Sol**, not the Fable/Opus/Sonnet
> lane. Keep it fully self-contained — no coupling to CortexBridge or TRIBE v2.

---

## Architect decisions (locked 2026-07-09, Fable 5) — apply these on top

These refine (do NOT re-scope) the tasks below. Full rationale + the design-question
catalog + dataset shortlist live in the architect's direction doc.

1. **Headline result = data-efficiency curve.** Primary reported result is how little
   of the NEW subject's labeled data Model B needs to match Model A — sweep the target
   budget (5/10/25/50/100% of that subject's trials) for both A and B and plot both
   curves. Pretraining-breadth (study pool = 1/4/14/28 subjects) is the secondary
   ablation; raw fixed-budget A-vs-B is the sanity anchor.
2. **Tokenizer = coordinate-aware patch-embed** (no VQ in v1). Per-channel ~1 s temporal
   patches → linear embed → add channel-id + 3D electrode-coordinate (CapTrak) positional
   encoding. SSL = masked-patch reconstruction (latent-space prediction is the noted v2).
   Decoding head stays tiny and **identical** across A and B; start with a linear probe.
3. **First decoding label = cleanest balanced contrast, then seen-vs-unseen.** The label
   is NOT a column — reconstruct it from the BrainVision trigger codebook (`value` S1…S77).
   De-risk the pipeline on the most balanced reliably-coded contrast first (document the
   exact codes), then report the headline seen-vs-unseen.
4. **Scope = ds005273 only for Step 1.** Do not pool other datasets yet. Step-2 pool
   (vetted): add **THINGS-EEG1 (ds003825)** first (64 ch @ 1 kHz, CC0, closest rig), then
   **HBN-EEG (ds005505–ds005516, ~1,900+ subj)** for pretraining scale.

**Splits/stats:** rotating leave-N-subjects-out (not one fixed set of 5), matched label
budgets for A and B, ≥5 seeds, Wilcoxon signed-rank paired test across hidden subjects.
**Metric:** balanced accuracy + AUC.

**Data caveats:** `ds005648` is on GitHub but NOT in this local checkout (only `ds005273/`
present). No `participants.tsv` locally. Signals still need `datalad get`.

---

You are GPT-5.6-Sol (the Codex lane) working in the repo at ~/Desktop/NeuroToTheEuro.

ASSIGNMENT CHECK (do this first): Read CLAUDE.md and README.md in this repo.
Per the company model-orchestration workflow and Canvascope's master plan, the
"fetch real EEG signals + Model A vs B de-risk" track is assigned to GPT-5.6-Sol
— that's you. Confirm you're the right lane, then proceed. This is an
independent research track: keep it 100% self-contained. Do NOT touch, import
from, or reference the CortexBridge repo or TRIBE v2 in any way.

=== CONTEXT ===
Thesis (see README.md): can one self-supervised model learn a transferable
"language of the brain" that decodes a NEW person's brain with little/no
retraining — the LLM pretraining trick applied to neural signals?

The whole point of this task is ONE cheap, public-data de-risking experiment
(README "Step 1"): show that a model which studied MANY people's brains reads a
NEW held-out person's brain better than a model that only ever saw that one
person. No hardware, no patients.

Current repo state (verified):
- ds005273/ is a DataLad dataset ("Neural representation of consciously seen and
  unseen information", CC0, DOI 10.18112/openneuro.ds005273.v1.0.0).
- 33 subjects, EEG in BIDS format, SAME task, SAME rig — ideal for cross-subject
  transfer.
- ONLY metadata is currently on disk (channels.tsv, events.tsv, *_eeg.json,
  electrodes, coordsystem). The actual signal binaries are NOT fetched.
- Raw format is BrainVision: each run is a triplet
  sub-XX_task-1_eeg.vhdr (header) + .eeg (binary) + .vmrk (markers).
  Brain Products, 63 EEG channels, 1000 Hz, continuous, ~5866 s/subject.
- There is NO Python scaffold yet (no venv/requirements/pyproject).

=== TASK 1 — Fetch the real EEG signals ===
- Set up a reproducible Python env at the repo root (venv or conda) with a
  pinned requirements/pyproject. Core deps: mne, mne-bids, datalad, numpy,
  scipy, scikit-learn, torch. Do NOT commit the venv or the fetched binaries
  (.gitignore them).
- Use DataLad to pull the actual data: `datalad get` the .eeg/.vhdr/.vmrk
  triplets for all 33 subjects (the dataset is already a datalad dataset here;
  the sibling is OpenNeuro ds005273). Verify each subject's .eeg binary is
  materialized (real bytes, not an annex pointer) and loads in MNE via
  mne.io.read_raw_brainvision.
- Sanity-check the load: 63 channels, 1000 Hz, expected duration, events parse
  from the .vmrk / events.tsv. Write a short data-integrity report
  (subjects loaded, channels, sfreq, #events, any failures).

=== TASK 2 — Preprocessing + epoching ===
- Build a preprocessing pipeline (MNE): band-pass (e.g. 0.1–40 Hz), notch 50 Hz
  (this is 50 Hz line freq, Brain Products EU rig), re-reference, resample to a
  sane rate (e.g. 250 Hz) to cut compute, epoch around the task events.
- Read the events.json/events.tsv to pick the decoding label for the
  de-risking experiment — the natural target is the consciousness/visibility
  contrast (consciously SEEN vs UNSEEN, or stimulus condition). Document exactly
  which column/codes you use as the label and why.
- Save preprocessed epochs + labels per subject to disk (e.g. .fif or .npy),
  reproducibly, so the experiment step doesn't re-preprocess every run.

=== TASK 3 — The Model A vs Model B de-risking experiment ===
Hold out a small set of "hidden" test subjects (e.g. 5 of 33). The rest are the
"study" pool. Fixed decoding architecture (a compact EEG encoder — e.g. a small
temporal CNN / EEGNet-style or a shallow transformer over channels×time — pick
one and keep it IDENTICAL across A and B).

- Model A (baseline / single-brain): for each hidden subject, train the encoder
  + decoding head FROM SCRATCH on only a small labeled slice of THAT subject,
  score on that same subject's held-out trials.
- Model B (pretrained / many-brains): first pretrain the encoder on the study
  pool (the other ~28 subjects). Prefer self-supervised pretraining (masked
  reconstruction of EEG segments — the LLM trick, no labels) since that's the
  thesis; then attach + fine-tune the SAME small decoding head on the SAME small
  labeled slice of the hidden subject. Score on the same held-out trials as A.
- Also run a simpler supervised cross-subject baseline as a sanity anchor
  (train supervised on the study pool, light fine-tune on the hidden subject).
- COMPARE A vs B on the hidden subjects with the same decoding metric
  (accuracy / AUC / balanced accuracy for the seen-vs-unseen contrast). Use
  matched data budgets and multiple seeds; report mean ± spread and a
  significance test across the hidden subjects. If B > A, the core idea has
  signal. If not, report that honestly.

=== DELIVERABLES ===
- Reproducible scripts (fetch → preprocess → pretrain → experiment) runnable
  end-to-end from a documented command sequence.
- A results file: the A-vs-B table (per hidden subject + aggregate), the metric,
  the significance test, and a plain-English verdict on whether cross-subject
  pretraining helped.
- Update this repo's own plan/tracker (or create one if none exists) with what
  was done, exact commands, decisions, gotchas, and next steps — enough that a
  fresh agent can resume.

=== GUARDRAILS ===
- Keep everything in THIS repo; zero coupling to CortexBridge/TRIBE v2.
- CC0 data — fine to use, but don't over-claim. This experiment de-risks ONE
  question (does multi-brain pretraining transfer on this dataset); it is not
  proof of a general brain foundation model. Frame results at that altitude.
- Don't commit fetched binaries or the venv. Do commit code, configs, and the
  results/report.
- Grind it out and be exhaustive across all 33 subjects and multiple seeds —
  that thoroughness is exactly this lane's job.
