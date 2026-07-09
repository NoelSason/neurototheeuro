"""Build the frozen output tables (T1-T4) + verdict from aggregated results."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..utils.io import ensure_dir, save_json
from . import stats


def _to_md(df: pd.DataFrame) -> str:
    """Minimal, dependency-free markdown table (avoids the optional `tabulate` dep)."""
    if df is None or len(df) == 0:
        return "_(empty)_"
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    for _, row in df.iterrows():
        cells = ["" if pd.isna(row[c]) else str(row[c]) for c in df.columns]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _cell(series: pd.Series) -> str:
    if len(series) == 0:
        return "n/a"
    return f"{series.mean():.3f}±{series.std(ddof=0):.3f}"


def _pick(agg, contrast, budget, metric, model, adapt, pool="full") -> pd.Series:
    return agg[
        (agg.contrast == contrast)
        & (agg.target_budget_pct == budget)
        & (agg.metric == metric)
        & (agg.model == model)
        & (agg.adaptation == adapt)
        & (agg.study_pool_size == pool)
    ]["value"]


def table1_efficiency(agg, cfg, contrast=None) -> pd.DataFrame:
    contrast = contrast or cfg.experiment.primary_contrast
    metric = cfg.stats.primary.metric
    rows = []
    for budget in cfg.experiment.budgets_pct:
        a1 = _pick(agg, contrast, budget, metric, "A1_matched", "from_scratch")
        a2 = _pick(agg, contrast, budget, metric, "A2_eegnet", "from_scratch")
        bf = _pick(agg, contrast, budget, metric, "B_ssl", "finetune")
        bl = _pick(agg, contrast, budget, metric, "B_ssl", "linearprobe")
        rows.append(
            {
                "budget_pct": budget,
                "A1_matched": _cell(a1),
                "A2_eegnet": _cell(a2),
                "B_finetune": _cell(bf),
                "B_linprobe": _cell(bl),
                "delta_B_A2": round(bf.mean() - a2.mean(), 3) if len(bf) and len(a2) else np.nan,
            }
        )
    df = pd.DataFrame(rows)
    sec = stats.secondary_curve(agg, cfg, contrast)
    if len(sec):
        df = df.merge(sec[["budget_pct", "p_raw", "p_holm"]], on="budget_pct", how="left")
    return df


def table2_per_subject(agg, cfg, contrast=None) -> pd.DataFrame:
    contrast = contrast or cfg.experiment.primary_contrast
    metric = cfg.stats.primary.metric
    budget = cfg.stats.primary.budget_pct

    a1 = _series_by_subject(agg, contrast, budget, metric, "A1_matched", "from_scratch")
    a2 = _series_by_subject(agg, contrast, budget, metric, "A2_eegnet", "from_scratch")
    bf = _series_by_subject(agg, contrast, budget, metric, "B_ssl", "finetune")
    subs = sorted(set(a2.index) & set(bf.index))
    df = pd.DataFrame(
        {
            "hidden_subject": subs,
            "A1_matched": [a1.get(s, np.nan) for s in subs],
            "A2_eegnet": [a2[s] for s in subs],
            "B_finetune": [bf[s] for s in subs],
        }
    )
    df["delta_B_A2"] = df["B_finetune"] - df["A2_eegnet"]
    return df


def _series_by_subject(agg, contrast, budget, metric, model, adapt, pool="full") -> pd.Series:
    sub = agg[
        (agg.contrast == contrast)
        & (agg.target_budget_pct == budget)
        & (agg.metric == metric)
        & (agg.model == model)
        & (agg.adaptation == adapt)
        & (agg.study_pool_size == pool)
    ]
    return sub.set_index("hidden_subject")["value"]


def table3_breadth(agg, cfg, contrast=None) -> pd.DataFrame:
    contrast = contrast or cfg.experiment.primary_contrast
    metric = cfg.stats.primary.metric
    budget = cfg.experiment.primary_budget_pct
    a2_ref = _pick(agg, contrast, 100, metric, "A2_eegnet", "from_scratch").mean()
    rows = []
    for size in cfg.experiment.breadth_pool_sizes:
        b = _pick(agg, contrast, budget, metric, "B_ssl", "finetune", pool=str(size))
        rows.append(
            {
                "study_pool_size": size,
                "B_finetune_mean": round(b.mean(), 3) if len(b) else np.nan,
                "B_finetune_sd": round(b.std(ddof=0), 3) if len(b) else np.nan,
                "delta_vs_A2_full": round(b.mean() - a2_ref, 3) if len(b) else np.nan,
                "n_subjects": len(b),
            }
        )
    return pd.DataFrame(rows)


def table4_provenance(provenance: list[dict]) -> pd.DataFrame:
    if not provenance:
        return pd.DataFrame()
    return pd.DataFrame(provenance)


def verdict(primary: dict, eff: dict, contrast: str, provisional: bool) -> str:
    direction = "OUTPERFORMS" if primary["median_delta"] > 0 else "does NOT outperform"
    sig = "significant" if primary.get("significant") else "not significant"
    factor = eff.get("data_efficiency_factor")
    factor_txt = (
        f"~{factor}x less target data to match the from-scratch EEGNet baseline"
        if factor
        else "no crossover reached within the tested budgets"
    )
    prov = " [PROVISIONAL contrast — semantics pending the ds005273 paper]" if provisional else ""
    return (
        f"On '{contrast}' (n={primary['n_subjects']} held-out subjects): SSL-pretrained Model B "
        f"{direction} the single-subject baseline at the {primary['endpoint'].split('budget ')[1].split(' ')[0]} "
        f"budget (median Δbalanced-acc={primary['median_delta']:+.3f}, Wilcoxon p={primary['p_value']:.4g}, "
        f"{sig} at α={primary['alpha']}). Data-efficiency: {factor_txt}.{prov}"
    )


def save_all(agg, cfg, out_dir, provenance, contrast=None) -> dict:
    contrast = contrast or cfg.experiment.primary_contrast
    out = ensure_dir(out_dir)
    t1 = table1_efficiency(agg, cfg, contrast)
    t2 = table2_per_subject(agg, cfg, contrast)
    t3 = table3_breadth(agg, cfg, contrast)
    t4 = table4_provenance(provenance)
    primary = stats.primary_test(agg, cfg)
    eff = stats.data_efficiency_factor(agg, cfg, contrast)
    provisional = bool(provenance[0]["provisional"]) if provenance else True
    verdict_txt = verdict(primary, eff, contrast, provisional)

    t1.to_csv(out / "T1_data_efficiency.csv", index=False)
    t2.to_csv(out / "T2_per_subject.csv", index=False)
    t3.to_csv(out / "T3_breadth.csv", index=False)
    t4.to_csv(out / "T4_label_provenance.csv", index=False)
    save_json(out / "primary_test.json", {**primary, "data_efficiency": eff})
    (out / "verdict.txt").write_text(verdict_txt + "\n")

    with open(out / "TABLES.md", "w") as fh:
        fh.write(f"# Results — contrast: {contrast}\n\n")
        fh.write(f"**Verdict:** {verdict_txt}\n\n")
        fh.write("## T1 — Data-efficiency curve (headline)\n\n")
        fh.write(_to_md(t1) + "\n\n")
        fh.write(f"## T2 — Per-subject @ {cfg.stats.primary.budget_pct}% budget\n\n")
        fh.write(_to_md(t2.round(3)) + "\n\n")
        fh.write("## T3 — Pretraining-breadth ablation\n\n")
        fh.write(_to_md(t3) + "\n\n")
        fh.write("## T4 — Label provenance\n\n")
        fh.write(_to_md(t4) + "\n")

    return {"primary": primary, "eff": eff, "verdict": verdict_txt}
