"""Significance template: primary Wilcoxon detects a real effect and a null one."""
import numpy as np
import pandas as pd

from n2e.experiment import stats
from n2e.utils import load_config

CFG = load_config("configs/spec_v1.yaml")


def _agg(deltas, budget=25):
    rows = []
    for i, d in enumerate(deltas):
        subj = f"sub-{i:02d}"
        common = dict(
            dataset="x", contrast="balanced_primary", study_pool_size="full",
            target_budget_pct=budget, metric="balanced_accuracy", hidden_subject=subj,
        )
        rows.append({**common, "model": "A2_eegnet", "adaptation": "from_scratch", "value": 0.5})
        rows.append({**common, "model": "B_ssl", "adaptation": "finetune", "value": 0.5 + d})
    return pd.DataFrame(rows)


def test_primary_detects_positive_effect():
    res = stats.primary_test(_agg([0.08, 0.06, 0.05, 0.09, 0.07, 0.04, 0.10]), CFG)
    assert res["n_subjects"] == 7
    assert res["median_delta"] > 0
    assert res["p_value"] < 0.05 and res["significant"]
    assert res["rank_biserial"] > 0.9


def test_primary_null_effect_not_significant():
    res = stats.primary_test(_agg([0.01, -0.02, 0.0, -0.01, 0.02, -0.03, 0.01]), CFG)
    assert res["p_value"] > 0.05 and not res["significant"]


def test_holm_is_monotone_and_bounded():
    p = np.array([0.01, 0.04, 0.03, 0.2])
    adj = stats._holm(p)
    assert (adj >= p - 1e-12).all() and (adj <= 1.0).all()


def test_aggregate_seeds_averages():
    rows = [
        dict(dataset="x", contrast="c", hidden_subject="s", study_pool_size="full",
             target_budget_pct=25, model="B_ssl", adaptation="finetune", metric="auc",
             value=v, seed=seed)
        for seed, v in [(0, 0.4), (1, 0.6)]
    ]
    agg = stats.aggregate_seeds(pd.DataFrame(rows))
    assert len(agg) == 1 and abs(agg.iloc[0]["value"] - 0.5) < 1e-9
