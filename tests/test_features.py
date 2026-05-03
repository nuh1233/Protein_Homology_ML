from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

from cafa6.features import (
    align_rows,
    build_label_matrix,
    filter_trainable_labels,
    load_embedding_matrix,
    make_prediction_frame,
    select_branch_terms,
)


def test_load_embedding_matrix_aligns_manifest_order(tmp_path: Path) -> None:
    shard0 = tmp_path / "shard_00000.npy"
    shard1 = tmp_path / "shard_00001.npy"
    np.save(shard0, np.array([[1, 2], [3, 4]], dtype="float32"))
    np.save(shard1, np.array([[5, 6]], dtype="float32"))

    manifest = pd.DataFrame(
        [
            {
                "entry_id": "P1",
                "split": "train",
                "batch_id": "batch_00000",
                "row_index": 0,
                "row_index_in_batch": 0,
                "sequence_length": 10,
                "batch_path": str(tmp_path / "batch_00000.parquet"),
                "shard_path": str(shard0),
                "model_name": "test",
                "status": "complete",
                "completed_at": "",
            },
            {
                "entry_id": "P2",
                "split": "train",
                "batch_id": "batch_00000",
                "row_index": 1,
                "row_index_in_batch": 1,
                "sequence_length": 10,
                "batch_path": str(tmp_path / "batch_00000.parquet"),
                "shard_path": str(shard0),
                "model_name": "test",
                "status": "complete",
                "completed_at": "",
            },
            {
                "entry_id": "P3",
                "split": "train",
                "batch_id": "batch_00001",
                "row_index": 2,
                "row_index_in_batch": 0,
                "sequence_length": 10,
                "batch_path": str(tmp_path / "batch_00001.parquet"),
                "shard_path": str(shard1),
                "model_name": "test",
                "status": "complete",
                "completed_at": "",
            },
        ]
    )
    manifest_path = tmp_path / "manifest.csv"
    manifest.to_csv(manifest_path, index=False)

    loaded = load_embedding_matrix(manifest_path)

    assert loaded.entry_ids.tolist() == ["P1", "P2", "P3"]
    assert loaded.matrix.tolist() == [[1, 2], [3, 4], [5, 6]]


def test_align_rows_rejects_missing_entry() -> None:
    indices = align_rows(["P2", "P1"], ["P1", "P2"])
    assert indices.tolist() == [1, 0]

    try:
        align_rows(["P1"], ["P2"])
    except ValueError as exc:
        assert "Missing embedding rows" in str(exc)
    else:
        raise AssertionError("expected missing row error")


def test_select_branch_terms_and_label_matrix() -> None:
    terms = pd.DataFrame(
        [
            {"entry_id": "P1", "term": "GO:1", "branch": "MF"},
            {"entry_id": "P2", "term": "GO:1", "branch": "MF"},
            {"entry_id": "P1", "term": "GO:2", "branch": "MF"},
            {"entry_id": "P3", "term": "GO:3", "branch": "BP"},
        ]
    )

    selected = select_branch_terms(terms, branch="MF", min_count=1, max_labels=2)
    y = build_label_matrix(["P1", "P2", "P3"], terms, selected["term"], branch="MF")

    assert selected["term"].tolist() == ["GO:1", "GO:2"]
    assert y.toarray().tolist() == [[1, 1], [1, 0], [0, 0]]


def test_filter_trainable_labels_drops_all_zero_and_all_one() -> None:
    y = sparse.csr_matrix(
        [
            [1, 0, 1],
            [1, 0, 0],
            [1, 0, 0],
        ]
    )

    filtered, labels, keep = filter_trainable_labels(y, ["all_one", "all_zero", "usable"])

    assert labels == ["usable"]
    assert keep.tolist() == [2]
    assert filtered.toarray().tolist() == [[1], [0], [0]]


def test_make_prediction_frame_topk() -> None:
    frame = make_prediction_frame(
        entry_ids=["P1"],
        label_terms=["GO:1", "GO:2", "GO:3"],
        scores=np.array([[0.1, 0.9, 0.5]], dtype="float32"),
        branch="MF",
        top_k=2,
        min_score=0.2,
    )

    assert frame["term"].tolist() == ["GO:2", "GO:3"]
