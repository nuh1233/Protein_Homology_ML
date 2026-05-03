from __future__ import annotations

import pandas as pd
import pytest

from cafa6.calibration import clip_scores, normalize_scores_by_branch
from cafa6.homology import (
    add_hit_weights,
    aggregate_hit_labels,
    filter_oof_hits,
    filter_test_hits,
    make_oof_homology_predictions,
    make_test_homology_predictions,
    normalize_hit_table,
    prepare_training_labels,
    read_hit_table,
    summarize_homology_outputs,
)


def _terms() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"entry_id": "T1", "term": "GO:1", "branch": "MF"},
            {"entry_id": "T2", "term": "GO:1", "branch": "MF"},
            {"entry_id": "T2", "term": "GO:2", "branch": "BP"},
            {"entry_id": "Q1", "term": "GO:SELF", "branch": "CC"},
        ]
    )


def test_normalize_hit_table_supports_mmseqs_aliases() -> None:
    hits = pd.DataFrame(
        [
            {"qseqid": "Q1", "sseqid": "T1", "bitscore": 50, "pident": 80, "qcov": 90, "tcov": 70},
        ]
    )

    normalized = normalize_hit_table(hits)

    assert normalized.loc[0, "query_id"] == "Q1"
    assert normalized.loc[0, "target_id"] == "T1"
    assert normalized.loc[0, "bitscore"] == 50


def test_normalize_hit_table_requires_query_and_target() -> None:
    with pytest.raises(ValueError, match="query and target"):
        normalize_hit_table(pd.DataFrame([{"query_id": "Q1"}]))


def test_read_hit_table_infers_headerless_mmseqs_format(tmp_path) -> None:
    path = tmp_path / "hits.tsv"
    path.write_text("Q1\tT1\t80\t1e-50\t100\t90\t80\n", encoding="utf-8")

    hits = read_hit_table(path)

    assert hits.columns.tolist() == ["query_id", "target_id", "pident", "evalue", "bitscore", "qcov", "tcov"]
    assert hits.loc[0, "bitscore"] == 100


def test_add_hit_weights_uses_query_normalized_bitscore_and_coverage() -> None:
    hits = pd.DataFrame(
        [
            {"query_id": "Q1", "target_id": "T1", "bitscore": 100, "qcov": 100, "tcov": 100},
            {"query_id": "Q1", "target_id": "T2", "bitscore": 25, "qcov": 100, "tcov": 100},
        ]
    )

    weighted = add_hit_weights(hits).set_index("target_id")

    assert weighted.loc["T1", "hit_weight"] == 1.0
    assert weighted.loc["T2", "hit_weight"] == 0.25


def test_filter_oof_hits_excludes_self_and_same_fold() -> None:
    hits = pd.DataFrame(
        [
            {"query_id": "Q1", "target_id": "Q1", "score": 0.9},
            {"query_id": "Q1", "target_id": "T1", "score": 0.8},
            {"query_id": "Q1", "target_id": "T2", "score": 0.7},
        ]
    )
    folds = pd.DataFrame(
        [
            {"entry_id": "Q1", "fold": 0},
            {"entry_id": "T1", "fold": 0},
            {"entry_id": "T2", "fold": 1},
        ]
    )

    filtered = filter_oof_hits(hits, folds)

    assert filtered["target_id"].tolist() == ["T2"]


def test_filter_oof_hits_requires_fold_columns() -> None:
    with pytest.raises(ValueError, match="fold"):
        filter_oof_hits(pd.DataFrame([{"query_id": "Q1", "target_id": "T1"}]), pd.DataFrame([{"entry_id": "Q1"}]))


def test_aggregate_hit_labels_noisy_or_and_branch_outputs() -> None:
    hits = pd.DataFrame(
        [
            {"query_id": "Q1", "target_id": "T1", "hit_weight": 0.5},
            {"query_id": "Q1", "target_id": "T2", "hit_weight": 0.5},
        ]
    )

    predictions = aggregate_hit_labels(hits, _terms(), max_hits_per_query=None)
    score_go1 = predictions.loc[predictions["term"] == "GO:1", "score"].item()

    assert round(score_go1, 6) == 0.75
    assert set(predictions["branch"]) == {"MF", "BP"}
    assert predictions.loc[predictions["term"] == "GO:1", "n_hits"].item() == 2


def test_aggregate_hit_labels_limits_hits_and_handles_empty() -> None:
    hits = pd.DataFrame(
        [
            {"query_id": "Q1", "target_id": "T1", "hit_weight": 0.9},
            {"query_id": "Q1", "target_id": "T2", "hit_weight": 0.8},
        ]
    )

    predictions = aggregate_hit_labels(hits, _terms(), max_hits_per_query=1)
    assert set(predictions["term"]) == {"GO:1"}

    empty = aggregate_hit_labels(hits.iloc[0:0], _terms())
    assert empty.empty


def test_oof_and_test_prediction_helpers() -> None:
    hits = pd.DataFrame(
        [
            {"query_id": "Q1", "target_id": "T1", "score": 0.8},
            {"query_id": "Q1", "target_id": "T2", "score": 0.7},
        ]
    )
    folds = pd.DataFrame(
        [
            {"entry_id": "Q1", "fold": 0},
            {"entry_id": "T1", "fold": 1},
            {"entry_id": "T2", "fold": 1},
        ]
    )

    oof_hits, oof_predictions = make_oof_homology_predictions(hits, _terms(), folds, max_hits_per_query=None)
    test_hits, test_predictions = make_test_homology_predictions(hits, _terms(), max_hits_per_query=None)
    summary = summarize_homology_outputs(oof_hits, oof_predictions, test_hits, test_predictions)

    assert len(oof_hits) == 2
    assert set(oof_predictions["term"]) == {"GO:1", "GO:2"}
    assert summary["oof_prediction_rows"] == len(oof_predictions)
    assert summary["test_prediction_rows"] == len(test_predictions)


def test_filter_test_hits_can_exclude_or_allow_self_hits() -> None:
    hits = pd.DataFrame(
        [
            {"query_id": "Q1", "target_id": "Q1", "score": 0.9},
            {"query_id": "Q1", "target_id": "T1", "score": 0.8},
        ]
    )

    excluded = filter_test_hits(hits, _terms(), exclude_self_hits=True)
    allowed = filter_test_hits(hits, _terms(), exclude_self_hits=False)

    assert excluded["target_id"].tolist() == ["T1"]
    assert allowed["target_id"].tolist() == ["Q1", "T1"]


def test_prepare_training_labels_accepts_aspect_and_rejects_missing_branch() -> None:
    labels = pd.DataFrame([{"entry_id": "T1", "term": "GO:1", "aspect": "F"}])
    prepared = prepare_training_labels(labels)

    assert prepared.loc[0, "branch"] == "MF"

    with pytest.raises(ValueError, match="branch or aspect"):
        prepare_training_labels(pd.DataFrame([{"entry_id": "T1", "term": "GO:1"}]))


def test_calibration_helpers_clip_and_normalize() -> None:
    predictions = pd.DataFrame(
        [
            {"entry_id": "Q1", "term": "GO:1", "branch": "MF", "score": 2.0},
            {"entry_id": "Q2", "term": "GO:2", "branch": "MF", "score": 0.5},
            {"entry_id": "Q3", "term": "GO:3", "branch": "BP", "score": -1.0},
        ]
    )

    clipped = clip_scores(predictions)
    normalized = normalize_scores_by_branch(predictions)

    assert clipped["score"].tolist() == [1.0, 0.5, 0.0]
    assert normalized.loc[normalized["term"] == "GO:1", "score"].item() == 1.0
    assert normalized.loc[normalized["term"] == "GO:2", "score"].item() == 0.5

    with pytest.raises(ValueError, match="method"):
        normalize_scores_by_branch(predictions, method="rank")
