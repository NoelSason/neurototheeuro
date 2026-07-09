#!/usr/bin/env python
"""Run the Step-1 rotating-LOSO grid (build -> SSL pretrain -> decode -> stats/tables).

Examples
--------
Smoke (synthetic, seconds on CPU):
  python scripts/run_experiment.py --config configs/smoke.yaml --source synthetic \
      --n-synth-subjects 4 --out results/smoke

Full grind (real data, after fetch):
  python scripts/run_experiment.py --config configs/spec_v1.yaml --source real \
      --bids-root data/ds005273 --decodable sub-10 sub-11 ... --ssl-only sub-17 \
      --out results/ds005273_balanced_primary --device cpu
"""
from __future__ import annotations

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from n2e.experiment.run_grid import run  # noqa: E402
from n2e.utils import load_config, set_seed  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/spec_v1.yaml")
    ap.add_argument("--source", choices=["synthetic", "real"], default="synthetic")
    ap.add_argument("--contrast", default=None, help="default: cfg.experiment.primary_contrast")
    ap.add_argument("--decodable", nargs="*", default=None, help="subject ids (real source)")
    ap.add_argument("--ssl-only", nargs="*", default=[], help="SSL-only subjects, e.g. sub-17")
    ap.add_argument("--bids-root", default="ds005273")
    ap.add_argument("--out", default="results/run")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--no-breadth", action="store_true")
    ap.add_argument("--n-synth-subjects", type=int, default=6)
    ap.add_argument("--n-trials-synth", type=int, default=120)
    args = ap.parse_args()

    set_seed(0)
    cfg = load_config(args.config)
    contrast = args.contrast or cfg.experiment.primary_contrast
    if args.decodable:
        decodable = args.decodable
    elif args.source == "synthetic":
        decodable = [f"sub-s{i:02d}" for i in range(args.n_synth_subjects)]
    else:
        raise SystemExit("--decodable is required for --source real")

    run(
        cfg,
        decodable,
        args.ssl_only,
        contrast,
        source=args.source,
        device=args.device,
        out_dir=args.out,
        bids_root=args.bids_root,
        do_breadth=not args.no_breadth,
        n_trials_synth=args.n_trials_synth,
    )


if __name__ == "__main__":
    main()
