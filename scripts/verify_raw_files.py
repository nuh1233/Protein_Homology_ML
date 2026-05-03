"""Verify that required CAFA 6 Kaggle raw files are available."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cafa6.io import (
    RAW_DIR,
    RAW_FILE_CHECK_REPORT,
    check_raw_files,
    format_raw_file_summary,
    write_raw_file_report,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=RAW_DIR,
        help="Directory containing the Kaggle raw files.",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=RAW_FILE_CHECK_REPORT,
        help="JSON report path to write.",
    )
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Exit with status 0 even when files are missing.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = check_raw_files(args.raw_dir)
    report_path = write_raw_file_report(report, args.report_path)

    print(format_raw_file_summary(report))
    print(f"Report written: {report_path}")

    if report["all_present"] or args.allow_missing:
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
