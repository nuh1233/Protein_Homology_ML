"""Score CAFA 6 validation predictions with branch-specific max F-measure."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cafa6.io import PROCESSED_DIR, REPORTS_DIR
from cafa6.metrics import make_thresholds, score_branch_fmaxes


DEFAULT_OUTPUT = REPORTS_DIR / "cv_scores.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True, help="Prediction table with entry_id, term, score.")
    parser.add_argument(
        "--truth",
        type=Path,
        default=PROCESSED_DIR / "train_terms_closure.parquet",
        help="Ontology-closed truth label table.",
    )
    parser.add_argument(
        "--go-terms",
        type=Path,
        default=PROCESSED_DIR / "go_terms.parquet",
        help="GO terms table used to infer prediction branches when needed.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="CSV score report output.")
    parser.add_argument("--threshold-step", type=float, default=0.01, help="Threshold grid step in [0, 1].")
    parser.add_argument(
        "--average",
        choices=["protein_macro", "micro"],
        default="protein_macro",
        help="F-measure averaging mode. protein_macro is the CAFA-style default.",
    )
    return parser.parse_args()


def read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".tsv", ".txt"}:
        return pd.read_csv(path, sep="\t")
    raise ValueError(f"Unsupported table format: {path}")


def main() -> int:
    args = parse_args()
    if not args.predictions.is_file():
        raise FileNotFoundError(f"Missing prediction table: {args.predictions}")
    if not args.truth.is_file():
        raise FileNotFoundError(f"Missing truth table: {args.truth}")

    predictions = read_table(args.predictions)
    truth = pd.read_parquet(args.truth)
    term_to_branch = pd.read_parquet(args.go_terms).loc[:, ["term", "branch"]] if args.go_terms.is_file() else None
    thresholds = make_thresholds(args.threshold_step)

    scores = score_branch_fmaxes(
        truth=truth,
        predictions=predictions,
        thresholds=thresholds,
        average=args.average,
        term_to_branch=term_to_branch,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    scores.to_csv(args.output, index=False)

    print("CAFA 6 prediction scores")
    print(f"- predictions: {args.predictions}")
    print(f"- truth: {args.truth}")
    print(f"- scores: {args.output}")
    print(f"- average: {args.average}")
    print(f"- mean_fmax: {scores['mean_fmax'].iloc[0]:.6f}")
    for row in scores.itertuples(index=False):
        print(
            f"- {row.branch}: fmax={row.fmax:.6f}, threshold={row.threshold:.4f}, "
            f"precision={row.precision:.6f}, recall={row.recall:.6f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
