# Step-1 pipeline — run guide

Implements the frozen v1 spec (`configs/spec_v1.yaml`; full rationale in the
architect plan) as a reproducible package under `src/n2e`. The Step-1 de-risk asks
whether an SSL-pretrained encoder (**Model B**) decodes a *new* subject better, and
with less of their data, than single-subject baselines (**A1** matched transformer,
**A2** EEGNet).

## Layout
```
configs/        spec_v1.yaml (source of truth) + smoke.yaml + spec_v1_patch0p5.yaml
src/n2e/data    fetch, codebook, preprocess, epoch, synthetic
src/n2e/model   tokenizer, encoder, ssl, eegnet, heads
src/n2e/train   pretrain (SSL), decode (A1/A2/B/Xsub adapt+eval)
src/n2e/experiment  splits, run_grid, stats, tables
scripts/        run_experiment.py, fetch_data.py, analyze.py
tests/          codebook / splits / stats
results/        committed CSVs, tables, verdict (never signal binaries)
```

## Environment (once)
```bash
/opt/homebrew/bin/python3.12 -m venv .venv          # torch/mne need 3.12, not 3.14
.venv/bin/pip install -U pip
.venv/bin/pip install -e .            # or: pip install numpy scipy scikit-learn pandas pyyaml mne mne-bids torch pytest
```

## Phase 0 — smoke + tests (no download; proves plumbing + output shape)
```bash
.venv/bin/python -m pytest tests/ -q
.venv/bin/python scripts/run_experiment.py --config configs/smoke.yaml \
    --source synthetic --n-synth-subjects 4 --out results/smoke
```
Runs the entire pipeline (preprocess → tokenize → SSL → A1/A2/B/Xsub → stats/tables)
on a deterministic synthetic generator that mirrors the real 63-ch montage + trigger
structure. Emits `results/smoke/results.csv` in the frozen schema plus `TABLES.md`,
`T1–T4*.csv`, `verdict.txt`, `primary_test.json`. Smoke numbers are **plumbing only**
(EEGNet fits the synthetic ERP; raw-reconstruction SSL has little to learn in noise, so
Model B does not "win" here — that is the correct null for this synthetic setup).

## Phase 1 — real data (the grind)
1. **Fetch** from OpenNeuro (git-annex; `datalad` not required):
   ```bash
   .venv/bin/python scripts/fetch_data.py --dest data/ds005273 \
       --subjects sub-10 sub-11 ... sub-42        # all 33; ~hundreds of MB each
   ```
   Writes `data/integrity_report.csv` (channels / sfreq / duration / #events per subject).
2. **Run the grid** (rotating LOSO × budget × seed; `sub-17` is SSL-only, no events):
   ```bash
   .venv/bin/python scripts/run_experiment.py --config configs/spec_v1.yaml --source real \
       --bids-root data/ds005273 \
       --decodable sub-10 sub-11 sub-12 ... sub-42   # 32 decodable (omit sub-17) \
       --ssl-only sub-17 \
       --out results/ds005273_balanced_primary --device cpu
   ```
3. **Patch-length sensitivity** (pre-registered): rerun with `--config configs/spec_v1_patch0p5.yaml`.
4. **Re-analyze** an existing CSV without recompute: `scripts/analyze.py --results <csv> --config configs/spec_v1.yaml`.

## Output shape (frozen)
`results.csv` columns: `dataset, contrast, hidden_subject, seed, study_pool_size,
target_budget_pct, model, adaptation, metric, value, n_train_trials, n_test_trials,
class_pos_frac, git_sha, spec_version`. All tables are reductions of it. Primary
endpoint = Wilcoxon signed-rank, `B_ssl`(finetune) vs `A2_eegnet`, balanced_primary,
25% budget, across held-out subjects (α=0.05, one-sided).

## Correctness notes baked into the code
- **Label = `trial_type` `Stimulus/S NN`, never `value`** (value is a per-file remap;
  see `data/codebook.py`). Verified against the real events in `tests/test_codebook.py`.
- Encoder uses the **60 coordinate-bearing scalp channels**; EOG1/EOG2/FCz are dropped.
- **`seen_unseen` needs the ds005273 paper.** Runs today on `balanced_primary` (S74 vs
  S75, balanced); the code→semantics map is a one-line edit in `data/codebook.py`
  (`CONTRASTS`) once confirmed. All provisional results carry a `provisional` flag.

## Known Phase-1 considerations
- **Memory:** `run_grid` stacks study-pool SSL crops in RAM (fine for the smoke; ~tens of
  GB for all 32 subjects). For the full grind, swap in a memmap/streaming crop dataset.
- **SSL objective:** v1 reconstructs raw µV; on noisy EEG this is hard (latent-space
  prediction is the noted v2). The machinery is verified to converge on predictable signal.
- **Compute:** CPU is fine for the smoke; use `--device` and the full SSL/finetune epoch
  counts in `spec_v1.yaml` for the real grid (long-running).
