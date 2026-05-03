"""Build GO ontology tables and closed CAFA 6 training labels."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cafa6.io import PROCESSED_DIR, RAW_DIR, REPORTS_DIR, write_json, write_parquet
from cafa6.ontology import build_go_tables, close_training_terms


ONTOLOGY_REPORT = REPORTS_DIR / "ontology_report.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--obo-path", type=Path, default=RAW_DIR / "go-basic.obo", help="Path to go-basic.obo.")
    parser.add_argument(
        "--train-terms-path",
        type=Path,
        default=PROCESSED_DIR / "train_terms_clean.parquet",
        help="Canonical direct training label table.",
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=PROCESSED_DIR,
        help="Directory for ontology parquet outputs.",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=ONTOLOGY_REPORT,
        help="JSON report path.",
    )
    return parser.parse_args()


def build_ontology(
    obo_path: Path,
    train_terms_path: Path,
    processed_dir: Path,
    report_path: Path,
) -> dict[str, object]:
    if not obo_path.is_file():
        raise FileNotFoundError(f"Missing GO OBO file: {obo_path}")
    if not train_terms_path.is_file():
        raise FileNotFoundError(f"Missing canonical training terms: {train_terms_path}")

    go_tables = build_go_tables(obo_path)
    train_terms = pd.read_parquet(train_terms_path)
    train_terms_closure, closure_report = close_training_terms(
        train_terms=train_terms,
        terms=go_tables.terms,
        ancestors=go_tables.ancestors,
        alt_ids=go_tables.alt_ids,
        all_terms=go_tables.all_terms,
    )

    outputs = {
        "go_terms": processed_dir / "go_terms.parquet",
        "go_edges": processed_dir / "go_edges.parquet",
        "go_ancestors": processed_dir / "go_ancestors.parquet",
        "train_terms_closure": processed_dir / "train_terms_closure.parquet",
    }
    write_parquet(go_tables.terms, outputs["go_terms"])
    write_parquet(go_tables.edges, outputs["go_edges"])
    write_parquet(go_tables.ancestors, outputs["go_ancestors"])
    write_parquet(train_terms_closure, outputs["train_terms_closure"])

    report = {
        "inputs": {
            "obo_path": str(obo_path),
            "train_terms_path": str(train_terms_path),
        },
        "outputs": {name: str(path) for name, path in outputs.items()},
        "ontology": go_tables.report,
        "closure": closure_report,
    }
    write_json(report, report_path)
    return report


def main() -> int:
    args = parse_args()
    report = build_ontology(args.obo_path, args.train_terms_path, args.processed_dir, args.report_path)

    print("CAFA 6 ontology tables built")
    for name, path in report["outputs"].items():
        print(f"- {name}: {path}")
    print(f"- ontology_report: {args.report_path}")

    ontology = report["ontology"]
    closure = report["closure"]
    print(f"- active_term_count: {ontology['active_term_count']}")
    print(f"- edge_count: {ontology['edge_count']}")
    print(f"- ancestor_row_count: {ontology['ancestor_row_count']}")
    print(f"- closed_annotation_rows: {closure['closed_annotation_rows']}")
    print(f"- missing_training_term_count: {closure['missing_training_term_count']}")
    print(f"- obsolete_training_term_count: {closure['obsolete_training_term_count']}")
    print(f"- branch_mismatch_training_term_count: {closure['branch_mismatch_training_term_count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
