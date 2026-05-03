from __future__ import annotations

from pathlib import Path

import pandas as pd

from cafa6.ontology import build_go_tables, close_training_terms, parse_go_obo


def _write_obo(path: Path) -> None:
    path.write_text(
        "format-version: 1.2\n"
        "\n"
        "[Term]\n"
        "id: GO:0000001\n"
        "name: root process\n"
        "namespace: biological_process\n"
        "\n"
        "[Term]\n"
        "id: GO:0000002\n"
        "name: child process\n"
        "namespace: biological_process\n"
        "alt_id: GO:1234567\n"
        "is_a: GO:0000001 ! root process\n"
        "\n"
        "[Term]\n"
        "id: GO:0000003\n"
        "name: grandchild process\n"
        "namespace: biological_process\n"
        "is_a: GO:0000002 ! child process\n"
        "\n"
        "[Term]\n"
        "id: GO:0000004\n"
        "name: component root\n"
        "namespace: cellular_component\n"
        "\n"
        "[Term]\n"
        "id: GO:0000005\n"
        "name: component child\n"
        "namespace: cellular_component\n"
        "relationship: part_of GO:0000004 ! component root\n"
        "\n"
        "[Term]\n"
        "id: GO:9999999\n"
        "name: obsolete term\n"
        "namespace: biological_process\n"
        "is_obsolete: true\n",
        encoding="utf-8",
    )


def test_parse_go_obo_reads_terms_and_edges(tmp_path: Path) -> None:
    obo_path = tmp_path / "go-basic.obo"
    _write_obo(obo_path)

    terms, edges = parse_go_obo(obo_path)

    assert len(terms) == 6
    assert any(term.term == "GO:9999999" and term.is_obsolete for term in terms)
    assert {edge.relation for edge in edges} == {"is_a", "part_of"}


def test_build_go_tables_active_edges_and_ancestors(tmp_path: Path) -> None:
    obo_path = tmp_path / "go-basic.obo"
    _write_obo(obo_path)

    tables = build_go_tables(obo_path)

    assert "GO:9999999" not in set(tables.terms["term"])
    assert set(tables.alt_ids["alt_id"]) == {"GO:1234567"}
    assert set(tables.edges["relation"]) == {"is_a", "part_of"}

    ancestors = tables.ancestors.set_index(["term", "ancestor"])["distance"].to_dict()
    assert ancestors[("GO:0000003", "GO:0000002")] == 1
    assert ancestors[("GO:0000003", "GO:0000001")] == 2
    assert ancestors[("GO:0000005", "GO:0000004")] == 1


def test_close_training_terms_maps_alt_ids_and_reports_missing(tmp_path: Path) -> None:
    obo_path = tmp_path / "go-basic.obo"
    _write_obo(obo_path)
    tables = build_go_tables(obo_path)

    train_terms = pd.DataFrame(
        [
            {"entry_id": "P1", "term": "GO:0000003", "aspect": "P", "branch": "BP"},
            {"entry_id": "P2", "term": "GO:1234567", "aspect": "P", "branch": "BP"},
            {"entry_id": "P3", "term": "GO:9999999", "aspect": "P", "branch": "BP"},
            {"entry_id": "P4", "term": "GO:8888888", "aspect": "P", "branch": "BP"},
        ]
    )

    closed, report = close_training_terms(
        train_terms=train_terms,
        terms=tables.terms,
        ancestors=tables.ancestors,
        alt_ids=tables.alt_ids,
        all_terms=tables.all_terms,
    )

    p1_terms = set(closed.loc[closed["entry_id"] == "P1", "term"])
    assert p1_terms == {"GO:0000003", "GO:0000002", "GO:0000001"}

    p2_terms = set(closed.loc[closed["entry_id"] == "P2", "term"])
    assert p2_terms == {"GO:0000002", "GO:0000001"}

    assert report["alt_id_mapped_training_terms"] == ["GO:1234567"]
    assert report["obsolete_training_terms"] == ["GO:9999999"]
    assert report["missing_training_terms"] == ["GO:8888888"]
