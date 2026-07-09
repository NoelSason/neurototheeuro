"""Significance-test template: pre-registered Wilcoxon primary + secondary curve.

Unit of analysis = subject (seeds are averaged first). Primary endpoint is a single
pre-registered Wilcoxon signed-rank test (B_ssl finetune vs A2_eegnet, balanced
contrast, primary budget). Secondary tests across the budget family are Holm-corrected
and flagged exploratory.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import rankdata, wilcoxon

_KEYS = [
    "dataset",
    "contrast",
    "hidden_subject",
    "study_pool_size",
    "target_budget_pct",
    "model",
    "adaptation",
    "metric",
]


def aggregate_seeds(df: pd.DataFrame) -> pd.DataFrame:
    """Average over seeds -> one value per subject×condition (no pseudo-replication)."""
    return df.groupby(_KEYS, as_index=False)["value"].mean()


def _paired(agg, contrast, budget, metric, model_a, adapt_a, model_b, adapt_b, pool="full"):
    def series(model, adapt):
        sub = agg[
            (agg.contrast == contrast)
            & (agg.target_budget_pct == budget)
            & (agg.metric == metric)
            & (agg.model == model)
            & (agg.adaptation == adapt)
            & (agg.study_pool_size == pool)
        ]
        return sub.set_index("hidden_subject")["value"]

    a, b = series(model_a, adapt_a), series(model_b, adapt_b)
    common = sorted(set(a.index) & set(b.index))
    return np.array([a[s] for s in common]), np.array([b[s] for s in common]), common


def rank_biserial(delta: np.ndarray) -> float:
    d = delta[delta != 0]
    if len(d) == 0:
        return 0.0
    ranks = rankdata(np.abs(d))
    t_pos, t_neg = ranks[d > 0].sum(), ranks[d < 0].sum()
    total = t_pos + t_neg
    return float((t_pos - t_neg) / total) if total else 0.0


def bootstrap_ci(delta: np.ndarray, n: int = 10000, alpha: float = 0.05, seed: int = 0):
    """Percentile bootstrap CI of the mean paired delta (subject-level)."""
    if len(delta) == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    boot = rng.choice(delta, size=(n, len(delta)), replace=True).mean(axis=1)
    lo, hi = np.percentile(boot, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def _wilcoxon(b: np.ndarray, a: np.ndarray, alternative: str):
    delta = b - a
    nonzero = np.count_nonzero(delta)
    if nonzero == 0:
        return float("nan"), 1.0
    stat, p = wilcoxon(b, a, alternative=alternative, zero_method="wilcox")
    return float(stat), float(p)


def primary_test(agg: pd.DataFrame, cfg) -> dict:
    p = cfg.stats.primary
    a, b, subs = _paired(
        agg, p.contrast, p.budget_pct, p.metric, "A2_eegnet", "from_scratch", "B_ssl", "finetune"
    )
    delta = b - a
    stat, pval = _wilcoxon(b, a, p.alternative)
    lo, hi = bootstrap_ci(delta, n=int(cfg.stats.bootstrap_n))
    return {
        "endpoint": f"{p.metric} | {p.contrast} | budget {p.budget_pct}% | B_ssl(finetune) vs A2_eegnet",
        "test": "wilcoxon_signed_rank",
        "alternative": p.alternative,
        "n_subjects": len(subs),
        "median_A2_eegnet": float(np.median(a)) if len(a) else float("nan"),
        "median_B_ssl": float(np.median(b)) if len(b) else float("nan"),
        "median_delta": float(np.median(delta)) if len(delta) else float("nan"),
        "mean_delta": float(np.mean(delta)) if len(delta) else float("nan"),
        "wilcoxon_stat": stat,
        "p_value": pval,
        "alpha": float(p.alpha),
        "significant": bool(pval < float(p.alpha)) if pval == pval else False,
        "rank_biserial": rank_biserial(delta),
        "mean_delta_ci95": bootstrap_ci(delta, n=int(cfg.stats.bootstrap_n)),
    }


def secondary_curve(agg: pd.DataFrame, cfg, contrast: str | None = None) -> pd.DataFrame:
    """Per-budget Wilcoxon (B finetune vs A2), Holm-corrected across the budget family."""
    contrast = contrast or cfg.experiment.primary_contrast
    metric = cfg.stats.primary.metric
    rows = []
    for budget in cfg.experiment.budgets_pct:
        a, b, subs = _paired(
            agg, contrast, budget, metric, "A2_eegnet", "from_scratch", "B_ssl", "finetune"
        )
        if len(subs) == 0:
            continue
        stat, pval = _wilcoxon(b, a, cfg.stats.primary.alternative)
        rows.append(
            {
                "budget_pct": budget,
                "n_subjects": len(subs),
                "median_delta": float(np.median(b - a)),
                "rank_biserial": rank_biserial(b - a),
                "p_raw": pval,
            }
        )
    out = pd.DataFrame(rows)
    if len(out):
        out["p_holm"] = _holm(out["p_raw"].to_numpy())
        out["significant_holm"] = out["p_holm"] < float(cfg.stats.primary.alpha)
    return out


def _holm(pvals: np.ndarray) -> np.ndarray:
    m = len(pvals)
    order = np.argsort(pvals)
    adjusted = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        val = (m - rank) * pvals[idx]
        running = max(running, val)
        adjusted[idx] = min(running, 1.0)
    return adjusted


def data_efficiency_factor(agg: pd.DataFrame, cfg, contrast: str | None = None) -> dict:
    """Smallest B budget whose median matches A2@100%; factor = 100 / that budget."""
    contrast = contrast or cfg.experiment.primary_contrast
    metric = cfg.stats.primary.metric

    def median_at(model, adapt, budget):
        sub = agg[
            (agg.contrast == contrast)
            & (agg.target_budget_pct == budget)
            & (agg.metric == metric)
            & (agg.model == model)
            & (agg.adaptation == adapt)
            & (agg.study_pool_size == "full")
        ]
        return float(sub["value"].median()) if len(sub) else float("nan")

    reference = median_at("A2_eegnet", "from_scratch", 100)
    for budget in sorted(cfg.experiment.budgets_pct):
        if median_at("B_ssl", "finetune", budget) >= reference:
            return {
                "reference_A2_full": reference,
                "crossover_budget_pct": budget,
                "data_efficiency_factor": round(100.0 / budget, 2),
            }
    return {"reference_A2_full": reference, "crossover_budget_pct": None, "data_efficiency_factor": None}
