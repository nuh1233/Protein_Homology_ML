"""Create and validate a CAFA 6 Kaggle submission TSV."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cafa6.io import PROCESSED_DIR, REPORTS_DIR, write_json
from cafa6.submission import prepare_submission_frame, validate_submission_predictions, write_submission


PREDICTIONS_DIR = PROJECT_ROOT / "artifacts" / "predictions"
SUBMISSIONS_DIR = PROJECT_ROOT / "submissions"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, default=PREDICTIONS_DIR / "esm2_test_repaired.parquet")
    parser.add_argument("--test-sequences", type=Path, default=PROCESSED_DIR / "test_sequences.parquet")
    parser.add_argument("--go-terms", type=Path, default=PROCESSED_DIR / "go_terms.parquet")
    parser.add_argument("--top-k-per-branch", type=int, default=100)
    parser.add_argument("--top-k-total", type=int, default=0)
    parser.add_argument("--require-all-test-entries", action="store_true")
    parser.add_argument("--output", type=Path, default=SUBMISSIONS_DIR / "esm2_supervised_baseline.tsv")
    parser.add_argument("--report", type=Path, default=REPORTS_DIR / "submission_validation.json")
    return parser.parse_args()


def _check_inputs(paths: list[Path]) -> None:
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"Missing required input: {path}")


def make_submission(args: argparse.Namespace) -> dict[str, object]:
    _check_inputs([args.predictions, args.test_sequences, args.go_terms])

    predictions = pd.read_parquet(args.predictions)
    test_sequences = pd.read_parquet(args.test_sequences, columns=["entry_id"])
    terms = pd.read_parquet(args.go_terms, columns=["term", "branch"])

    top_k_total = args.top_k_total if args.top_k_total > 0 else None
    submission = prepare_submission_frame(
        predictions,
        test_entry_ids=test_sequences["entry_id"],
        valid_terms=terms,
        top_k_per_branch=args.top_k_per_branch,
        top_k_total=top_k_total,
    )
    validation = validate_submission_predictions(
        submission,
        test_entry_ids=test_sequences["entry_id"],
        valid_terms=terms,
        require_all_test_entries=args.require_all_test_entries,
    )

    write_submission(submission, args.output)
    report = {
        "inputs": {
            "predictions": str(args.predictions),
            "test_sequences": str(args.test_sequences),
            "go_terms": str(args.go_terms),
        },
        "outputs": {
            "submission": str(args.output),
            "report": str(args.report),
        },
        "parameters": {
            "top_k_per_branch": int(args.top_k_per_branch),
            "top_k_total": int(args.top_k_total),
            "require_all_test_entries": bool(args.require_all_test_entries),
        },
        "validation": validation,
    }
    write_json(report, args.report)
    return report


def main() -> int:
    args = parse_args()
    report = make_submission(args)
    validation = report["validation"]

    print("CAFA 6 submission written")
    print(f"- submission: {report['outputs']['submission']}")
    print(f"- report: {report['outputs']['report']}")
    print(f"- valid: {validation['valid']}")
    print(f"- rows: {validation['prediction_rows']}")
    print(f"- covered_test_entries: {validation['covered_test_entry_count']}/{validation['test_entry_count']}")
    return 0 if validation["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
