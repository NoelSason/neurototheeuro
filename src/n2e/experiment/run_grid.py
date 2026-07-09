"""Orchestrate the rotating-LOSO x budget x seed grid -> results.csv + tables.

For each held-out target subject: SSL-pretrain Model B on the study pool (once),
optionally supervised-pretrain the X-sub anchor, then adapt/evaluate every model at
each budget x seed. A secondary breadth ablation re-pretrains B at pool sizes
{1,4,14,28} at the primary budget.

Note (Phase-1 scale): the study-pool crops are stacked in memory here, which is fine
for the smoke but not for all 32 subjects (~tens of GB). The full grind should swap in
a memmap/streaming crop dataset — see README_PIPELINE.md.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..data import (
    CONTRASTS,
    build_subject_arrays,
    load_real_subject,
    make_synthetic_subject,
    preprocess_recording,
)
from ..train.decode import evaluate_target, supervised_pretrain_encoder
from ..train.pretrain import pretrain_encoder
from ..utils.io import ensure_dir
from ..utils.logutil import get_logger, git_sha
from . import stats, tables
from .splits import stable_seed, study_pool_subjects

log = get_logger()

RESULTS_COLUMNS = [
    "dataset", "contrast", "hidden_subject", "seed", "study_pool_size", "target_budget_pct",
    "model", "adaptation", "metric", "value", "n_train_trials", "n_test_trials",
    "class_pos_frac", "git_sha", "spec_version",
]


def build_arrays(cfg, subjects, contrast, source="synthetic", bids_root=None, n_trials_synth=120):
    arrays = {}
    for i, s in enumerate(subjects):
        if source == "synthetic":
            rec = make_synthetic_subject(s, cfg, seed=1000 + i, n_trials=n_trials_synth)
        else:
            rec = load_real_subject(bids_root, s, cfg)
        pp = preprocess_recording(rec, cfg)
        arrays[s] = build_subject_arrays(pp, contrast, cfg, make_crops=True)
        log.info(
            "built %s: epochs=%s crops=%s balance=%s",
            s, arrays[s].epochs.shape, arrays[s].crops.shape, arrays[s].balance.get("class_counts"),
        )
    return arrays


def _stack_crops(arrs):
    xs, cs = [], []
    for a in arrs:
        if len(a.crops) == 0:
            continue
        xs.append(a.crops)
        cs.append(np.repeat(a.coords[None], len(a.crops), axis=0))
    if not xs:
        raise ValueError("no SSL crops in pretraining pool")
    return np.concatenate(xs, 0).astype(np.float32), np.concatenate(cs, 0).astype(np.float32)


def run(
    cfg,
    decodable,
    ssl_only,
    contrast_name,
    source="synthetic",
    device="cpu",
    out_dir="results/smoke",
    bids_root=None,
    do_breadth=True,
    n_trials_synth=120,
):
    contrast = CONTRASTS[contrast_name]
    arrays = build_arrays(cfg, list(decodable) + list(ssl_only), contrast, source, bids_root, n_trials_synth)
    ctx_base = {
        "dataset": cfg.data.dataset,
        "contrast": contrast_name,
        "git_sha": git_sha(),
        "spec_version": cfg.spec_version,
    }
    want = set(cfg.experiment.models)
    rows: list[dict] = []
    provenance: list[dict] = []

    for target in decodable:
        study = [s for s in decodable if s != target] + list(ssl_only)
        log.info("=== target %s | study pool %d subjects ===", target, len(study))
        state = None
        if "B_ssl" in want:
            crops, coords = _stack_crops([arrays[s] for s in study])
            state = pretrain_encoder(crops, coords, cfg, device, seed=stable_seed(target, "ssl"))
        xsub = None
        if "Xsub_sup" in want:
            labeled = [arrays[s] for s in study if len(arrays[s].epochs) > 0]
            xsub = supervised_pretrain_encoder(labeled, cfg, device, seed=stable_seed(target, "xsub"))

        for budget in cfg.experiment.budgets_pct:
            for seed in cfg.seeds:
                ctx = {**ctx_base, "study_pool_size": "full"}
                rows += evaluate_target(
                    arrays[target], cfg, budget, seed, ctx, device,
                    pretrained_state=state, xsub_state=xsub,
                )
        provenance.append(
            {
                "subject": target,
                "contrast": contrast_name,
                "provisional": contrast.provisional,
                "n_valid": arrays[target].balance["n_valid"],
                "class_counts": arrays[target].balance["class_counts"],
                "pos_frac": round(arrays[target].balance["pos_frac"], 3),
            }
        )

        if do_breadth and "B_ssl" in want:
            for size in cfg.experiment.breadth_pool_sizes:
                pool = study_pool_subjects(decodable, target, size)
                crops_b, coords_b = _stack_crops([arrays[s] for s in pool])
                state_b = pretrain_encoder(crops_b, coords_b, cfg, device, seed=stable_seed(target, "ssl", size))
                for seed in cfg.seeds:
                    ctx = {**ctx_base, "study_pool_size": str(size)}
                    rows += evaluate_target(
                        arrays[target], cfg, cfg.experiment.primary_budget_pct, seed, ctx, device,
                        pretrained_state=state_b, models={"B_ssl"},
                    )

    df = pd.DataFrame(rows).reindex(columns=RESULTS_COLUMNS)
    out = ensure_dir(out_dir)
    df.to_csv(Path(out) / "results.csv", index=False)
    agg = stats.aggregate_seeds(df)
    summary = tables.save_all(agg, cfg, out, provenance, contrast_name)
    log.info("VERDICT: %s", summary["verdict"])
    return df, summary
