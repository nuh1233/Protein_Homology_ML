"""Create deterministic protein-level or cluster-aware CAFA 6 folds."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cafa6.folds import make_fold_assignments, make_fold_report
from cafa6.io import PROCESSED_DIR, REPORTS_DIR, write_json, write_parquet


DEFAULT_OUTPUT = PROCESSED_DIR / "folds_clustered.parquet"
DEFAULT_REPORT = REPORTS_DIR / "fold_report.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train-sequences",
        type=Path,
        default=PROCESSED_DIR / "train_sequences.parquet",
        help="Canonical training sequence table.",
    )
    parser.add_argument(
        "--train-terms",
        type=Path,
        default=PROCESSED_DIR / "train_terms_closure.parquet",
        help="Ontology-closed training label table for fold summaries.",
    )
    parser.add_argument(
        "--cluster-file",
        type=Path,
        default=None,
        help="Optional file with entry_id and cluster_id columns. CSV, TSV, and parquet are supported.",
    )
    parser.add_argument("--n-folds", type=int, default=5, help="Number of folds to create.")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic fold seed.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Fold assignment parquet output.")
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT, help="Fold report JSON output.")
    return parser.parse_args()


def read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".csv":
        return pd.read_csv(path, dtype=str)
    if suffix in {".tsv", ".txt"}:
        return pd.read_csv(path, sep="\t", dtype=str)
    raise ValueError(f"Unsupported table format: {path}")


def main() -> int:
    args = parse_args()
    if not args.train_sequences.is_file():
        raise FileNotFoundError(f"Missing training sequences: {args.train_sequences}")

    sequences = pd.read_parquet(args.train_sequences)
    clusters = read_table(args.cluster_file) if args.cluster_file is not None else None
    train_terms = pd.read_parquet(args.train_terms) if args.train_terms.is_file() else None

    folds = make_fold_assignments(
        sequences=sequences,
        clusters=clusters,
        n_folds=args.n_folds,
        seed=args.seed,
    )
    report = make_fold_report(folds, train_terms=train_terms, n_folds=args.n_folds, seed=args.seed)

    write_parquet(folds, args.output)
    write_json(report, args.report)

    validation = report["validation"]
    print("CAFA 6 folds built")
    print(f"- folds: {args.output}")
    print(f"- report: {args.report}")
    print(f"- n_entries: {validation['n_entries']}")
    print(f"- n_clusters: {validation['n_clusters']}")
    print(f"- duplicate_entry_count: {validation['duplicate_entry_count']}")
    print(f"- cluster_leak_count: {validation['cluster_leak_count']}")
    print(f"- fold_sizes: {validation['fold_sizes']}")

    if validation["duplicate_entry_count"] or validation["cluster_leak_count"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
