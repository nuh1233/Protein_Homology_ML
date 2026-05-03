from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from cafa6.io import (
    check_raw_files,
    format_raw_file_summary,
    make_label_frequency,
    parse_fasta_header,
    read_fasta,
    read_test_taxon_list,
    read_train_taxonomy,
    read_train_terms,
    summarize_sequences,
    write_json,
    write_raw_file_report,
)


def test_parse_uniprot_train_header() -> None:
    parsed = parse_fasta_header(
        "sp|A0A0C5B5G6|MOTSC_HUMAN Mitochondrial-derived peptide MOTS-c OS=Homo sapiens OX=9606 GN=MT-RNR1"
    )

    assert parsed["entry_id"] == "A0A0C5B5G6"
    assert parsed["database"] == "sp"
    assert parsed["uniprot_name"] == "MOTSC_HUMAN"
    assert parsed["taxon_id"] == "9606"


def test_parse_test_header() -> None:
    parsed = parse_fasta_header("A0A1B0GTW7 9606")

    assert parsed["entry_id"] == "A0A1B0GTW7"
    assert parsed["taxon_id"] == "9606"
    assert parsed["database"] is None


def test_read_fasta_multiline(tmp_path: Path) -> None:
    fasta_path = tmp_path / "example.fasta"
    fasta_path.write_text(
        ">sp|P1|P1_HUMAN Example OS=Homo sapiens OX=9606\n"
        "MAGA\n"
        "TTACA\n"
        ">Q2 10090\n"
        "MK\n",
        encoding="utf-8",
    )

    frame = read_fasta(fasta_path)

    assert frame.loc[0, "entry_id"] == "P1"
    assert frame.loc[0, "sequence"] == "MAGATTACA"
    assert frame.loc[0, "length"] == 9
    assert frame.loc[1, "entry_id"] == "Q2"
    assert frame.loc[1, "taxon_id"] == "10090"


def test_read_train_terms_deduplicates_and_maps_branch(tmp_path: Path) -> None:
    path = tmp_path / "terms.tsv"
    path.write_text(
        "EntryID\tterm\taspect\n"
        "P1\tGO:0001\tF\n"
        "P1\tGO:0001\tF\n"
        "P1\tGO:0002\tP\n"
        "P2\tGO:0003\tC\n",
        encoding="utf-8",
    )

    terms = read_train_terms(path)

    assert len(terms) == 3
    assert set(terms["branch"]) == {"MF", "BP", "CC"}

    frequency = make_label_frequency(terms)
    assert set(frequency.columns) == {"branch", "aspect", "term", "n_proteins", "n_annotations"}


def test_read_taxonomy_files(tmp_path: Path) -> None:
    train_taxonomy_path = tmp_path / "train_taxonomy.tsv"
    train_taxonomy_path.write_text("P1\t9606\nP1\t9606\nP2\t10090\n", encoding="utf-8")

    test_taxon_path = tmp_path / "testsuperset-taxon-list.tsv"
    test_taxon_path.write_text("ID\tSpecies\n9606\tHomo sapiens\n10090\tMus musculus\n", encoding="utf-8")

    train_taxonomy = read_train_taxonomy(train_taxonomy_path)
    test_taxon_list = read_test_taxon_list(test_taxon_path)

    assert train_taxonomy.to_dict("records") == [
        {"entry_id": "P1", "taxon_id": "9606"},
        {"entry_id": "P2", "taxon_id": "10090"},
    ]
    assert isinstance(test_taxon_list, pd.DataFrame)
    assert test_taxon_list.loc[0, "taxon_id"] == "10090"


def test_raw_file_report_helpers(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    (raw_dir / "present.tsv").write_text("x\n", encoding="utf-8")

    report = check_raw_files(raw_dir, required_files=["present.tsv", "missing.tsv"])
    report_path = write_raw_file_report(report, tmp_path / "report.json")
    summary = format_raw_file_summary(report)

    assert report["present"] == ["present.tsv"]
    assert report["missing"] == ["missing.tsv"]
    assert report["all_present"] is False
    assert report_path.is_file()
    assert "Missing files:" in summary
    assert "- missing.tsv" in summary


def test_raw_file_summary_all_present(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    (raw_dir / "file.tsv").write_text("x\n", encoding="utf-8")

    report = check_raw_files(raw_dir, required_files=["file.tsv"])

    assert "All required raw files are present." in format_raw_file_summary(report)


def test_parse_fasta_header_rejects_empty_and_malformed_uniprot() -> None:
    with pytest.raises(ValueError, match="empty FASTA header"):
        parse_fasta_header("")

    with pytest.raises(ValueError, match="UniProt-style"):
        parse_fasta_header("sp||bad")


def test_read_fasta_rejects_sequence_before_header(tmp_path: Path) -> None:
    fasta_path = tmp_path / "bad.fasta"
    fasta_path.write_text("MAGA\n", encoding="utf-8")

    with pytest.raises(ValueError, match="before first FASTA header"):
        list(read_fasta(fasta_path))


def test_read_train_terms_rejects_missing_and_invalid_aspect(tmp_path: Path) -> None:
    missing_path = tmp_path / "missing.tsv"
    missing_path.write_text("EntryID\tterm\nP1\tGO:1\n", encoding="utf-8")

    with pytest.raises(ValueError, match="missing required columns"):
        read_train_terms(missing_path)

    invalid_path = tmp_path / "invalid.tsv"
    invalid_path.write_text("EntryID\tterm\taspect\nP1\tGO:1\tX\n", encoding="utf-8")

    with pytest.raises(ValueError, match="unsupported aspect"):
        read_train_terms(invalid_path)


def test_taxonomy_parsers_reject_conflicts_and_missing_columns(tmp_path: Path) -> None:
    conflict_path = tmp_path / "taxonomy.tsv"
    conflict_path.write_text("P1\t9606\nP1\t10090\n", encoding="utf-8")

    with pytest.raises(ValueError, match="conflicting taxon IDs"):
        read_train_taxonomy(conflict_path)

    bad_test_path = tmp_path / "bad_test_taxa.tsv"
    bad_test_path.write_text("ID\tName\n9606\tHomo sapiens\n", encoding="utf-8")

    with pytest.raises(ValueError, match="missing required columns"):
        read_test_taxon_list(bad_test_path)


def test_sequence_summary_and_write_json(tmp_path: Path) -> None:
    train = pd.DataFrame(
        [
            {"entry_id": "P1", "length": 2},
            {"entry_id": "P2", "length": 4},
        ]
    )
    test = pd.DataFrame([{"entry_id": "T1", "length": 10}])

    summary = summarize_sequences(train, test)
    output_path = write_json(summary, tmp_path / "summary.json")

    assert summary["train"]["n_sequences"] == 2
    assert summary["train"]["mean_length"] == 3.0
    assert output_path.is_file()
