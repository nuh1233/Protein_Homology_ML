"""Build canonical CAFA 6 sequence, label, and taxonomy tables."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cafa6.io import (
    BUILD_TABLES_REPORT,
    LABEL_FREQUENCY_REPORT,
    PROCESSED_DIR,
    RAW_DIR,
    SEQUENCE_SUMMARY_REPORT,
    check_raw_files,
    make_label_frequency,
    read_fasta,
    read_test_taxon_list,
    read_train_taxonomy,
    read_train_terms,
    summarize_sequences,
    write_json,
    write_parquet,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR, help="Directory containing raw Kaggle files.")
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=PROCESSED_DIR,
        help="Directory for canonical parquet outputs.",
    )
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=BUILD_TABLES_REPORT.parent,
        help="Directory for summary report outputs.",
    )
    return parser.parse_args()


def _require_unique_ids(frame, table_name: str) -> None:
    duplicate_count = int(frame["entry_id"].duplicated().sum())
    if duplicate_count:
        raise ValueError(f"{table_name} contains {duplicate_count} duplicate entry_id rows.")


def build_tables(raw_dir: Path, processed_dir: Path, reports_dir: Path) -> dict[str, object]:
    raw_check = check_raw_files(raw_dir)
    if not raw_check["all_present"]:
        missing = ", ".join(raw_check["missing"])
        raise FileNotFoundError(f"Required raw files are missing: {missing}")

    train_sequences = read_fasta(raw_dir / "train_sequences.fasta")
    test_sequences = read_fasta(raw_dir / "testsuperset.fasta")
    train_terms = read_train_terms(raw_dir / "train_terms.tsv")
    taxonomy_train = read_train_taxonomy(raw_dir / "train_taxonomy.tsv")
    test_taxon_lookup = read_test_taxon_list(raw_dir / "testsuperset-taxon-list.tsv")

    _require_unique_ids(train_sequences, "train_sequences")
    _require_unique_ids(test_sequences, "test_sequences")

    train_ids = set(train_sequences["entry_id"])
    labeled_ids = set(train_terms["entry_id"])
    missing_labeled_ids = sorted(labeled_ids.difference(train_ids))
    if missing_labeled_ids:
        preview = ", ".join(missing_labeled_ids[:10])
        raise ValueError(f"{len(missing_labeled_ids)} labeled EntryIDs are absent from train_sequences: {preview}")

    test_taxonomy = (
        test_sequences.loc[:, ["entry_id", "taxon_id"]]
        .merge(test_taxon_lookup, on="taxon_id", how="left")
        .sort_values("entry_id", kind="mergesort")
        .reset_index(drop=True)
    )

    train_sequence_ids = set(train_sequences["entry_id"])
    train_taxonomy_ids = set(taxonomy_train["entry_id"])
    missing_train_taxonomy = sorted(train_sequence_ids.difference(train_taxonomy_ids))
    extra_train_taxonomy = sorted(train_taxonomy_ids.difference(train_sequence_ids))
    missing_test_taxon_ids = sorted(test_taxonomy.loc[test_taxonomy["taxon_id"].isna(), "entry_id"].tolist())
    missing_test_species = sorted(
        test_taxonomy.loc[test_taxonomy["species"].isna() & test_taxonomy["taxon_id"].notna(), "taxon_id"]
        .drop_duplicates()
        .tolist()
    )

    label_frequency = make_label_frequency(train_terms)
    sequence_summary = summarize_sequences(train_sequences, test_sequences)

    outputs = {
        "train_sequences": processed_dir / "train_sequences.parquet",
        "test_sequences": processed_dir / "test_sequences.parquet",
        "train_terms_clean": processed_dir / "train_terms_clean.parquet",
        "taxonomy_train": processed_dir / "taxonomy_train.parquet",
        "taxonomy_test": processed_dir / "taxonomy_test.parquet",
    }

    write_parquet(train_sequences, outputs["train_sequences"])
    write_parquet(test_sequences, outputs["test_sequences"])
    write_parquet(train_terms, outputs["train_terms_clean"])
    write_parquet(taxonomy_train, outputs["taxonomy_train"])
    write_parquet(test_taxonomy, outputs["taxonomy_test"])

    reports_dir.mkdir(parents=True, exist_ok=True)
    label_frequency_path = reports_dir / LABEL_FREQUENCY_REPORT.name
    label_frequency.to_csv(label_frequency_path, index=False)
    sequence_summary_path = reports_dir / SEQUENCE_SUMMARY_REPORT.name
    write_json(sequence_summary, sequence_summary_path)

    report = {
        "outputs": {name: str(path) for name, path in outputs.items()},
        "reports": {
            "label_frequency": str(label_frequency_path),
            "sequence_summary": str(sequence_summary_path),
            "build_tables": str(reports_dir / BUILD_TABLES_REPORT.name),
        },
        "counts": {
            "train_sequences": int(len(train_sequences)),
            "test_sequences": int(len(test_sequences)),
            "train_terms_clean": int(len(train_terms)),
            "taxonomy_train": int(len(taxonomy_train)),
            "taxonomy_test": int(len(test_taxonomy)),
            "unique_terms": int(train_terms["term"].nunique()),
        },
        "branch_label_counts": {
            branch: int(count)
            for branch, count in train_terms.groupby("branch")["term"].nunique().sort_index().items()
        },
        "validation": {
            "labeled_entries_missing_from_train_sequences": missing_labeled_ids,
            "train_sequence_entries_missing_taxonomy": missing_train_taxonomy,
            "taxonomy_train_entries_not_in_train_sequences": extra_train_taxonomy,
            "test_entries_missing_taxon_id": missing_test_taxon_ids,
            "test_taxon_ids_missing_species_name": missing_test_species,
        },
    }
    write_json(report, reports_dir / BUILD_TABLES_REPORT.name)
    return report


def main() -> int:
    args = parse_args()
    report = build_tables(args.raw_dir, args.processed_dir, args.reports_dir)

    print("CAFA 6 canonical tables built")
    for name, path in report["outputs"].items():
        print(f"- {name}: {path}")
    for name, path in report["reports"].items():
        print(f"- {name}: {path}")

    validation = report["validation"]
    for name, values in validation.items():
        print(f"- {name}: {len(values)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
