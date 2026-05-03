from __future__ import annotations

import pandas as pd
import pytest

from cafa6.folds import (
    FoldConfig,
    assign_cluster_folds,
    make_fold_assignments,
    make_fold_report,
    make_length_bin,
    prepare_cluster_assignments,
    stable_hash_int,
    validate_fold_assignments,
)


def test_make_fold_assignments_keeps_clusters_together() -> None:
    sequences = pd.DataFrame(
        [
            {"entry_id": "P1", "length": 100, "taxon_id": "9606"},
            {"entry_id": "P2", "length": 110, "taxon_id": "9606"},
            {"entry_id": "P3", "length": 300, "taxon_id": "10090"},
            {"entry_id": "P4", "length": 900, "taxon_id": "10090"},
        ]
    )
    clusters = pd.DataFrame(
        [
            {"entry_id": "P1", "cluster_id": "C1"},
            {"entry_id": "P2", "cluster_id": "C1"},
            {"entry_id": "P3", "cluster_id": "C2"},
            {"entry_id": "P4", "cluster_id": "C3"},
        ]
    )

    folds = make_fold_assignments(sequences, clusters=clusters, n_folds=2, seed=7)

    assert folds.loc[folds["cluster_id"] == "C1", "fold"].nunique() == 1
    report = validate_fold_assignments(folds)
    assert report["duplicate_entry_count"] == 0
    assert report["cluster_leak_count"] == 0


def test_make_fold_assignments_is_deterministic() -> None:
    sequences = pd.DataFrame(
        [{"entry_id": f"P{i}", "length": 100 + i, "taxon_id": "9606"} for i in range(20)]
    )

    folds_a = make_fold_assignments(sequences, n_folds=5, seed=42)
    folds_b = make_fold_assignments(sequences, n_folds=5, seed=42)

    assert folds_a.loc[:, ["entry_id", "fold"]].equals(folds_b.loc[:, ["entry_id", "fold"]])


def test_make_fold_report_includes_branch_counts() -> None:
    sequences = pd.DataFrame(
        [
            {"entry_id": "P1", "length": 100, "taxon_id": "9606"},
            {"entry_id": "P2", "length": 120, "taxon_id": "9606"},
        ]
    )
    folds = make_fold_assignments(sequences, n_folds=2, seed=1)
    terms = pd.DataFrame(
        [
            {"entry_id": "P1", "term": "GO:1", "branch": "MF"},
            {"entry_id": "P2", "term": "GO:2", "branch": "BP"},
        ]
    )

    report = make_fold_report(folds, train_terms=terms, n_folds=2, seed=1)

    assert report["validation"]["n_entries"] == 2
    assert "unique_terms_by_fold_branch" in report


def test_singleton_fallback_and_length_bins() -> None:
    sequences = pd.DataFrame(
        [
            {"entry_id": "P1", "length": 50, "taxon_id": "9606"},
            {"entry_id": "P2", "length": 150, "taxon_id": "9606"},
            {"entry_id": "P3", "length": 300, "taxon_id": "9606"},
            {"entry_id": "P4", "length": 700, "taxon_id": "9606"},
            {"entry_id": "P5", "length": 1500, "taxon_id": "9606"},
            {"entry_id": "P6", "length": 2500, "taxon_id": "9606"},
        ]
    )

    assigned = prepare_cluster_assignments(sequences)

    assert assigned["cluster_source"].unique().tolist() == ["entry_id_singleton"]
    assert assigned["cluster_id"].tolist() == [f"singleton:P{i}" for i in range(1, 7)]
    assert [make_length_bin(length) for length in [50, 150, 300, 700, 1500, 2500]] == [
        "0000_0100",
        "0101_0200",
        "0201_0500",
        "0501_1000",
        "1001_2000",
        "2001_plus",
    ]


def test_missing_cluster_assignments_fall_back_to_singletons() -> None:
    sequences = pd.DataFrame(
        [
            {"entry_id": "P1", "length": 100, "taxon_id": "9606"},
            {"entry_id": "P2", "length": 100, "taxon_id": "9606"},
        ]
    )
    clusters = pd.DataFrame([{"entry_id": "P1", "cluster_id": 123}])

    assigned = prepare_cluster_assignments(sequences, clusters)

    assert assigned.loc[assigned["entry_id"] == "P1", "cluster_id"].item() == "123"
    assert assigned.loc[assigned["entry_id"] == "P2", "cluster_id"].item() == "singleton:P2"
    assert assigned.loc[assigned["entry_id"] == "P2", "cluster_source"].item() == "entry_id_singleton"


def test_fold_input_validation_errors() -> None:
    with pytest.raises(ValueError, match="entry_id"):
        prepare_cluster_assignments(pd.DataFrame([{"protein": "P1"}]))

    sequences = pd.DataFrame([{"entry_id": "P1", "length": 100}])
    with pytest.raises(ValueError, match="cluster_id"):
        prepare_cluster_assignments(sequences, pd.DataFrame([{"entry_id": "P1"}]))

    conflicting_clusters = pd.DataFrame(
        [
            {"entry_id": "P1", "cluster_id": "C1"},
            {"entry_id": "P1", "cluster_id": "C2"},
        ]
    )
    with pytest.raises(ValueError, match="multiple cluster IDs"):
        prepare_cluster_assignments(sequences, conflicting_clusters)

    clustered = prepare_cluster_assignments(sequences)
    with pytest.raises(ValueError, match="at least 2"):
        assign_cluster_folds(clustered, FoldConfig(n_folds=1, seed=42))


def test_validate_fold_assignments_detects_duplicate_entries_and_cluster_leak() -> None:
    folds = pd.DataFrame(
        [
            {"entry_id": "P1", "fold": 0, "cluster_id": "C1"},
            {"entry_id": "P1", "fold": 0, "cluster_id": "C1"},
            {"entry_id": "P2", "fold": 1, "cluster_id": "C1"},
        ]
    )

    report = validate_fold_assignments(folds)

    assert report["duplicate_entry_count"] == 1
    assert report["cluster_leak_count"] == 1
    assert report["leaking_clusters"] == ["C1"]


def test_validate_fold_assignments_requires_columns() -> None:
    with pytest.raises(ValueError, match="cluster_id"):
        validate_fold_assignments(pd.DataFrame([{"entry_id": "P1", "fold": 0}]))


def test_stable_hash_int_is_deterministic_and_seeded() -> None:
    assert stable_hash_int("cluster", seed=1) == stable_hash_int("cluster", seed=1)
    assert stable_hash_int("cluster", seed=1) != stable_hash_int("cluster", seed=2)
