from __future__ import annotations

import pandas as pd
import pytest

from cafa6.metrics import make_thresholds, prepare_predictions, prepare_truth, score_branch_fmax, score_branch_fmaxes


def test_score_branch_fmaxes_reports_branches_separately() -> None:
    truth = pd.DataFrame(
        [
            {"entry_id": "P1", "term": "GO:MF1", "branch": "MF"},
            {"entry_id": "P2", "term": "GO:BP1", "branch": "BP"},
            {"entry_id": "P3", "term": "GO:CC1", "branch": "CC"},
        ]
    )
    predictions = pd.DataFrame(
        [
            {"entry_id": "P1", "term": "GO:MF1", "branch": "MF", "score": 0.9},
            {"entry_id": "P2", "term": "GO:BP1", "branch": "BP", "score": 0.8},
            {"entry_id": "P3", "term": "GO:CC1", "branch": "CC", "score": 0.7},
        ]
    )

    scores = score_branch_fmaxes(truth, predictions, thresholds=[0.5, 0.95])

    assert scores["branch"].tolist() == ["MF", "BP", "CC"]
    assert scores["fmax"].tolist() == [1.0, 1.0, 1.0]
    assert scores["mean_fmax"].iloc[0] == 1.0


def test_protein_macro_fmax_averages_precision_and_recall_by_protein() -> None:
    truth = pd.DataFrame(
        [
            {"entry_id": "P1", "term": "GO:1", "branch": "MF"},
            {"entry_id": "P1", "term": "GO:2", "branch": "MF"},
            {"entry_id": "P2", "term": "GO:1", "branch": "MF"},
        ]
    )
    predictions = pd.DataFrame(
        [
            {"entry_id": "P1", "term": "GO:1", "branch": "MF", "score": 0.9},
            {"entry_id": "P1", "term": "GO:2", "branch": "MF", "score": 0.8},
            {"entry_id": "P2", "term": "GO:BAD", "branch": "MF", "score": 0.7},
            {"entry_id": "P2", "term": "GO:1", "branch": "MF", "score": 0.6},
        ]
    )

    scores = score_branch_fmaxes(truth, predictions, thresholds=[0.6])
    mf_score = scores.loc[scores["branch"] == "MF"].iloc[0]

    assert round(mf_score.precision, 6) == 0.75
    assert round(mf_score.recall, 6) == 1.0
    assert round(mf_score.fmax, 6) == 0.857143


def test_prepare_predictions_uses_max_duplicate_score_and_term_mapping() -> None:
    predictions = pd.DataFrame(
        [
            {"entry_id": "P1", "term": "GO:1", "score": 0.2},
            {"entry_id": "P1", "term": "GO:1", "score": 0.9},
        ]
    )
    term_to_branch = {"GO:1": "MF"}

    prepared = prepare_predictions(predictions, term_to_branch=term_to_branch)

    assert prepared.to_dict("records") == [
        {"entry_id": "P1", "term": "GO:1", "branch": "MF", "score": 0.9}
    ]


def test_make_thresholds_includes_bounds() -> None:
    assert make_thresholds(0.5).tolist() == [0.0, 0.5, 1.0]


def test_prepare_truth_maps_aspect_to_branch() -> None:
    truth = pd.DataFrame(
        [
            {"entry_id": "P1", "term": "GO:1", "aspect": "F"},
            {"entry_id": "P1", "term": "GO:1", "aspect": "F"},
        ]
    )

    prepared = prepare_truth(truth)

    assert prepared.to_dict("records") == [{"entry_id": "P1", "term": "GO:1", "branch": "MF"}]


def test_prepare_truth_rejects_missing_or_bad_branch_columns() -> None:
    with pytest.raises(ValueError, match="entry_id"):
        prepare_truth(pd.DataFrame([{"term": "GO:1", "branch": "MF"}]))

    with pytest.raises(ValueError, match="unsupported aspect"):
        prepare_truth(pd.DataFrame([{"entry_id": "P1", "term": "GO:1", "aspect": "X"}]))

    with pytest.raises(ValueError, match="branch or aspect"):
        prepare_truth(pd.DataFrame([{"entry_id": "P1", "term": "GO:1"}]))


def test_prepare_predictions_maps_aspect_and_rejects_invalid_scores() -> None:
    predictions = pd.DataFrame([{"entry_id": "P1", "term": "GO:1", "aspect": "P", "score": 0.4}])

    prepared = prepare_predictions(predictions)

    assert prepared.loc[0, "branch"] == "BP"

    with pytest.raises(ValueError, match="score"):
        prepare_predictions(pd.DataFrame([{"entry_id": "P1", "term": "GO:1", "branch": "MF", "score": 1.2}]))

    with pytest.raises(ValueError, match="branch/aspect"):
        prepare_predictions(pd.DataFrame([{"entry_id": "P1", "term": "GO:1", "score": 0.2}]))

    with pytest.raises(ValueError, match="term"):
        prepare_predictions(pd.DataFrame([{"entry_id": "P1", "branch": "MF", "score": 0.2}]))


def test_micro_fmax_and_invalid_branch_average() -> None:
    truth = pd.DataFrame(
        [
            {"entry_id": "P1", "term": "GO:1", "branch": "MF"},
            {"entry_id": "P2", "term": "GO:2", "branch": "MF"},
        ]
    )
    predictions = pd.DataFrame(
        [
            {"entry_id": "P1", "term": "GO:1", "branch": "MF", "score": 0.9},
            {"entry_id": "P2", "term": "GO:BAD", "branch": "MF", "score": 0.8},
        ]
    )

    score = score_branch_fmax(truth, predictions, branch="MF", thresholds=[0.5], average="micro")

    assert score.precision == 0.5
    assert score.recall == 0.5
    assert score.fmax == 0.5

    with pytest.raises(ValueError, match="Unsupported branch"):
        score_branch_fmax(truth, predictions, branch="XX")

    with pytest.raises(ValueError, match="average"):
        score_branch_fmax(truth, predictions, branch="MF", average="bad")


def test_make_thresholds_rejects_bad_steps() -> None:
    with pytest.raises(ValueError, match="threshold step"):
        make_thresholds(0)

    with pytest.raises(ValueError, match="threshold step"):
        make_thresholds(2)
