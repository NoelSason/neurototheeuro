#!/usr/bin/env python
"""Recompute stats + tables from an existing results.csv (the '04_analyze' step).

  python scripts/analyze.py --results results/smoke/results.csv --config configs/smoke.yaml
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from n2e.experiment import stats, tables  # noqa: E402
from n2e.utils import load_config  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", required=True)
    ap.add_argument("--config", default="configs/spec_v1.yaml")
    ap.add_argument("--out", default=None)
    ap.add_argument("--contrast", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    df = pd.read_csv(args.results)
    agg = stats.aggregate_seeds(df)
    out = args.out or str(pathlib.Path(args.results).parent)
    contrast = args.contrast or cfg.experiment.primary_contrast
    summary = tables.save_all(agg, cfg, out, provenance=[], contrast=contrast)
    print(summary["verdict"])


if __name__ == "__main__":
    main()
