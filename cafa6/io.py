"""Input/output helpers for the CAFA 6 project."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, Iterator

if TYPE_CHECKING:
    import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
REPORTS_DIR = PROJECT_ROOT / "artifacts" / "reports"
RAW_FILE_CHECK_REPORT = REPORTS_DIR / "raw_file_check.json"
SEQUENCE_SUMMARY_REPORT = REPORTS_DIR / "sequence_summary.json"
BUILD_TABLES_REPORT = REPORTS_DIR / "build_tables_report.json"
LABEL_FREQUENCY_REPORT = REPORTS_DIR / "label_frequency.csv"

REQUIRED_RAW_FILES: tuple[str, ...] = (
    "train_sequences.fasta",
    "train_terms.tsv",
    "train_taxonomy.tsv",
    "testsuperset.fasta",
    "testsuperset-taxon-list.tsv",
    "go-basic.obo",
    "IA.tsv",
    "sample_submission.tsv",
)

ASPECT_TO_BRANCH: dict[str, str] = {
    "F": "MF",
    "P": "BP",
    "C": "CC",
}

BRANCH_TO_ASPECT: dict[str, str] = {branch: aspect for aspect, branch in ASPECT_TO_BRANCH.items()}

_OX_PATTERN = re.compile(r"(?:^|\s)OX=(\d+)(?:\s|$)")


@dataclass(frozen=True)
class RawFileStatus:
    """Status for one expected raw input file."""

    file_name: str
    path: str
    exists: bool
    size_bytes: int | None


@dataclass(frozen=True)
class FastaRecord:
    """One parsed FASTA record."""

    entry_id: str
    sequence: str
    length: int
    header: str
    sequence_index: int
    taxon_id: str | None = None
    database: str | None = None
    uniprot_name: str | None = None


def check_raw_files(
    raw_dir: str | Path = RAW_DIR,
    required_files: Iterable[str] = REQUIRED_RAW_FILES,
) -> dict[str, object]:
    """Return a deterministic raw-file availability report."""

    raw_path = Path(raw_dir)
    statuses: list[RawFileStatus] = []

    for file_name in required_files:
        file_path = raw_path / file_name
        exists = file_path.is_file()
        size_bytes = file_path.stat().st_size if exists else None
        statuses.append(
            RawFileStatus(
                file_name=file_name,
                path=str(file_path),
                exists=exists,
                size_bytes=size_bytes,
            )
        )

    present = [status.file_name for status in statuses if status.exists]
    missing = [status.file_name for status in statuses if not status.exists]

    return {
        "raw_dir": str(raw_path),
        "required_count": len(statuses),
        "present_count": len(present),
        "missing_count": len(missing),
        "all_present": len(missing) == 0,
        "present": present,
        "missing": missing,
        "files": [asdict(status) for status in statuses],
    }


def write_raw_file_report(
    report: dict[str, object],
    output_path: str | Path = RAW_FILE_CHECK_REPORT,
) -> Path:
    """Write a raw-file report as stable, human-readable JSON."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def format_raw_file_summary(report: dict[str, object]) -> str:
    """Format a compact console summary for raw-file verification."""

    lines = [
        "CAFA 6 raw file check",
        f"Raw directory: {report['raw_dir']}",
        f"Required files: {report['required_count']}",
        f"Present: {report['present_count']}",
        f"Missing: {report['missing_count']}",
    ]

    missing = report.get("missing", [])
    if missing:
        lines.append("Missing files:")
        lines.extend(f"- {file_name}" for file_name in missing)
    else:
        lines.append("All required raw files are present.")

    return "\n".join(lines)


def parse_fasta_header(header: str) -> dict[str, str | None]:
    """Parse CAFA train/test FASTA header variants into normalized fields."""

    header = header.strip()
    if not header:
        raise ValueError("Encountered an empty FASTA header.")

    if "|" in header:
        parts = header.split("|", 2)
        if len(parts) < 2 or not parts[1]:
            raise ValueError(f"Unable to parse UniProt-style FASTA header: {header}")

        taxon_match = _OX_PATTERN.search(header)
        uniprot_name = None
        if len(parts) == 3:
            uniprot_name = parts[2].split(None, 1)[0]

        return {
            "entry_id": parts[1],
            "database": parts[0] or None,
            "uniprot_name": uniprot_name,
            "taxon_id": taxon_match.group(1) if taxon_match else None,
        }

    tokens = header.split()
    entry_id = tokens[0]
    taxon_id = tokens[1] if len(tokens) > 1 and tokens[1].isdigit() else None

    return {
        "entry_id": entry_id,
        "database": None,
        "uniprot_name": None,
        "taxon_id": taxon_id,
    }


def iter_fasta_records(path: str | Path) -> Iterator[FastaRecord]:
    """Yield parsed FASTA records from a file."""

    fasta_path = Path(path)
    header: str | None = None
    sequence_parts: list[str] = []
    sequence_index = 0

    with fasta_path.open("rt", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue

            if line.startswith(">"):
                if header is not None:
                    parsed = parse_fasta_header(header)
                    sequence = "".join(sequence_parts)
                    yield FastaRecord(
                        entry_id=str(parsed["entry_id"]),
                        sequence=sequence,
                        length=len(sequence),
                        header=header,
                        sequence_index=sequence_index,
                        taxon_id=parsed["taxon_id"],
                        database=parsed["database"],
                        uniprot_name=parsed["uniprot_name"],
                    )
                    sequence_index += 1

                header = line[1:]
                sequence_parts = []
                continue

            if header is None:
                raise ValueError(f"Sequence data before first FASTA header at line {line_number}: {fasta_path}")

            sequence_parts.append(line)

    if header is not None:
        parsed = parse_fasta_header(header)
        sequence = "".join(sequence_parts)
        yield FastaRecord(
            entry_id=str(parsed["entry_id"]),
            sequence=sequence,
            length=len(sequence),
            header=header,
            sequence_index=sequence_index,
            taxon_id=parsed["taxon_id"],
            database=parsed["database"],
            uniprot_name=parsed["uniprot_name"],
        )


def read_fasta(path: str | Path) -> pd.DataFrame:
    """Read a FASTA file into a normalized sequence table."""

    import pandas as pd

    records = [asdict(record) for record in iter_fasta_records(path)]
    columns = [
        "entry_id",
        "sequence",
        "length",
        "header",
        "sequence_index",
        "taxon_id",
        "database",
        "uniprot_name",
    ]
    return pd.DataFrame.from_records(records, columns=columns)


def read_train_terms(path: str | Path) -> pd.DataFrame:
    """Read and de-duplicate CAFA training labels."""

    import pandas as pd

    terms = pd.read_csv(path, sep="\t", dtype=str)
    terms = terms.rename(columns={"EntryID": "entry_id"})

    required_columns = {"entry_id", "term", "aspect"}
    missing_columns = required_columns.difference(terms.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Training terms file is missing required columns: {missing}")

    terms = terms.loc[:, ["entry_id", "term", "aspect"]].dropna()
    terms["aspect"] = terms["aspect"].str.strip()
    invalid_aspects = sorted(set(terms["aspect"]).difference(ASPECT_TO_BRANCH))
    if invalid_aspects:
        invalid = ", ".join(invalid_aspects)
        raise ValueError(f"Training terms contain unsupported aspect values: {invalid}")

    terms["branch"] = terms["aspect"].map(ASPECT_TO_BRANCH)
    terms = terms.drop_duplicates(["entry_id", "term", "aspect"])
    terms = terms.sort_values(["entry_id", "branch", "term"], kind="mergesort").reset_index(drop=True)
    return terms


def read_train_taxonomy(path: str | Path) -> pd.DataFrame:
    """Read headerless training taxonomy assignments."""

    import pandas as pd

    taxonomy = pd.read_csv(path, sep="\t", header=None, names=["entry_id", "taxon_id"], dtype=str)
    taxonomy = taxonomy.dropna(subset=["entry_id", "taxon_id"])
    taxonomy = taxonomy.drop_duplicates(["entry_id", "taxon_id"])

    conflicts = taxonomy.groupby("entry_id")["taxon_id"].nunique()
    conflicting_ids = conflicts[conflicts > 1].index.tolist()
    if conflicting_ids:
        preview = ", ".join(conflicting_ids[:10])
        raise ValueError(f"Training taxonomy has conflicting taxon IDs for entries: {preview}")

    return taxonomy.sort_values("entry_id", kind="mergesort").reset_index(drop=True)


def read_test_taxon_list(path: str | Path) -> pd.DataFrame:
    """Read the test taxon ID to species-name lookup."""

    import pandas as pd

    taxon_list = pd.read_csv(path, sep="\t", dtype=str)
    taxon_list = taxon_list.rename(columns={"ID": "taxon_id", "Species": "species"})

    required_columns = {"taxon_id", "species"}
    missing_columns = required_columns.difference(taxon_list.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Test taxon list is missing required columns: {missing}")

    taxon_list = taxon_list.loc[:, ["taxon_id", "species"]].dropna(subset=["taxon_id"])
    taxon_list = taxon_list.drop_duplicates("taxon_id")
    return taxon_list.sort_values("taxon_id", kind="mergesort").reset_index(drop=True)


def make_label_frequency(terms: pd.DataFrame) -> pd.DataFrame:
    """Compute branch-specific GO label frequencies."""

    frequency = (
        terms.groupby(["branch", "aspect", "term"], as_index=False)
        .agg(n_proteins=("entry_id", "nunique"), n_annotations=("entry_id", "size"))
        .sort_values(["branch", "n_proteins", "term"], ascending=[True, False, True], kind="mergesort")
        .reset_index(drop=True)
    )
    return frequency


def summarize_sequences(train_sequences: pd.DataFrame, test_sequences: pd.DataFrame) -> dict[str, object]:
    """Create compact sequence-count and length summaries."""

    def _summary(frame: pd.DataFrame) -> dict[str, object]:
        lengths = frame["length"]
        return {
            "n_sequences": int(len(frame)),
            "n_unique_entry_ids": int(frame["entry_id"].nunique()),
            "min_length": int(lengths.min()) if len(lengths) else 0,
            "mean_length": float(lengths.mean()) if len(lengths) else 0.0,
            "median_length": float(lengths.median()) if len(lengths) else 0.0,
            "max_length": int(lengths.max()) if len(lengths) else 0,
        }

    return {
        "train": _summary(train_sequences),
        "test": _summary(test_sequences),
    }


def write_json(data: dict[str, object], output_path: str | Path) -> Path:
    """Write a dictionary as stable, human-readable JSON."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def write_parquet(frame: pd.DataFrame, output_path: str | Path) -> Path:
    """Write a dataframe to parquet with stable parent-directory handling."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        frame.to_parquet(path, index=False)
    except ImportError as exc:
        raise RuntimeError("Writing parquet requires pyarrow or fastparquet to be installed.") from exc
    return path
