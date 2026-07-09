#!/usr/bin/env python
"""Fetch real ds005273 signals from OpenNeuro (git-annex) and report integrity.

  python scripts/fetch_data.py --dest data/ds005273 --subjects sub-10 sub-11
  python scripts/fetch_data.py --dest data/ds005273 --subjects sub-10 --report-only
"""
from __future__ import annotations

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from n2e.data.fetch import fetch_subjects, integrity_report  # noqa: E402
from n2e.utils.io import ensure_dir  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="ds005273")
    ap.add_argument("--dest", default="data/ds005273")
    ap.add_argument("--subjects", nargs="+", required=True)
    ap.add_argument("--report-only", action="store_true", help="skip fetch; just load + report")
    args = ap.parse_args()

    if not args.report_only:
        fetch_subjects(args.dataset, args.dest, args.subjects)
    report = integrity_report(args.dest, args.subjects)
    print(report.to_string(index=False))
    ensure_dir("data")
    report.to_csv("data/integrity_report.csv", index=False)
    print("\nwrote data/integrity_report.csv")


if __name__ == "__main__":
    main()
