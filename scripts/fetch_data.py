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

from n2e.data.fetch import (  # noqa: E402
    DEFAULT_OPENNEURO_VERSION,
    DEFAULT_SSL_ONLY_SUBJECTS,
    fetch_subjects,
    integrity_report,
)
from n2e.utils.io import ensure_dir  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="ds005273")
    ap.add_argument("--version", default=DEFAULT_OPENNEURO_VERSION)
    ap.add_argument("--dest", default="data/ds005273")
    ap.add_argument("--subjects", nargs="+", required=True)
    ap.add_argument("--jobs", type=int, default=4, help="parallel git-annex transfers")
    ap.add_argument("--strict", action="store_true", help="abort if a source subject lacks header or signal")
    ap.add_argument("--ssl-only", nargs="*", default=list(DEFAULT_SSL_ONLY_SUBJECTS))
    ap.add_argument("--report-only", action="store_true", help="skip fetch; just load + report")
    ap.add_argument("--report", default="results/ds005273_integrity.csv")
    ap.add_argument("--allow-integrity-failures", action="store_true")
    args = ap.parse_args()

    if not args.report_only:
        fetch_subjects(
            args.dataset,
            args.dest,
            args.subjects,
            version=args.version,
            jobs=args.jobs,
            strict=args.strict,
        )
    report = integrity_report(args.dest, args.subjects, ssl_only=args.ssl_only)
    print(report.to_string(index=False))
    report_path = pathlib.Path(args.report)
    ensure_dir(report_path.parent)
    report.to_csv(report_path, index=False)
    print(f"\nwrote {report_path}")
    failed = report.loc[~report["integrity_ok"].fillna(False), "subject"].tolist()
    not_ready = report.loc[~report["pipeline_ready"].fillna(False), "subject"].tolist()
    if (failed or not_ready) and not args.allow_integrity_failures:
        raise SystemExit(
            "integrity gate failed; "
            f"invalid={failed or 'none'}, pipeline_not_ready={not_ready or 'none'}"
        )


if __name__ == "__main__":
    main()
