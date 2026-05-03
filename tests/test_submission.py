from __future__ import annotations

import pandas as pd

from cafa6.ensemble import blend_prediction_frames, prune_top_k_by_group, repair_go_hierarchy
from cafa6.submission import prepare_submission_frame, validate_submission_predictions, write_submission
from scripts.calibrate_predictions import _repair_by_branch


def _terms() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"term": "GO:ROOT", "branch": "MF"},
            {"term": "GO:1", "branch": "MF"},
            {"term": "GO:2", "branch": "MF"},
        ]
    )


def test_repair_go_hierarchy_adds_ancestors_with_child_score() -> None:
    predictions = pd.DataFrame(
        [
            {"entry_id": "P1", "term": "GO:1", "branch": "MF", "score": 0.8},
            {"entry_id": "P1", "term": "GO:ROOT", "branch": "MF", "score": 0.2},
        ]
    )
    ancestors = pd.DataFrame(
        [
            {"term": "GO:1", "ancestor": "GO:ROOT", "term_branch": "MF", "ancestor_branch": "MF"},
        ]
    )

    repaired = repair_go_hierarchy(predictions, ancestors, terms=_terms()).set_index("term")

    assert repaired.loc["GO:1", "score"] == 0.8
    assert repaired.loc["GO:ROOT", "score"] == 0.8


def test_chunked_repair_by_branch_matches_entry_level_repair() -> None:
    predictions = pd.DataFrame(
        [
            {"entry_id": "P1", "term": "GO:1", "branch": "MF", "score": 0.8},
            {"entry_id": "P2", "term": "GO:2", "branch": "MF", "score": 0.7},
        ]
    )
    ancestors = pd.DataFrame(
        [
            {"term": "GO:1", "ancestor": "GO:ROOT", "term_branch": "MF", "ancestor_branch": "MF"},
            {"term": "GO:2", "ancestor": "GO:ROOT", "term_branch": "MF", "ancestor_branch": "MF"},
        ]
    )

    repaired = _repair_by_branch(predictions, ancestors, _terms(), top_k_per_branch=2, entry_chunk_size=1)

    assert set(repaired["term"]) == {"GO:1", "GO:2", "GO:ROOT"}
    assert repaired.loc[(repaired["entry_id"] == "P1") & (repaired["term"] == "GO:ROOT"), "score"].item() == 0.8
    assert repaired.loc[(repaired["entry_id"] == "P2") & (repaired["term"] == "GO:ROOT"), "score"].item() == 0.7


def test_prune_top_k_by_group_keeps_best_rows_per_branch() -> None:
    predictions = pd.DataFrame(
        [
            {"entry_id": "P1", "term": "GO:1", "branch": "MF", "score": 0.1},
            {"entry_id": "P1", "term": "GO:2", "branch": "MF", "score": 0.9},
        ]
    )

    pruned = prune_top_k_by_group(predictions, top_k=1)

    assert pruned["term"].tolist() == ["GO:2"]


def test_blend_prediction_frames_uses_weighted_mean_with_missing_as_zero() -> None:
    frame_a = pd.DataFrame([{"entry_id": "P1", "term": "GO:1", "branch": "MF", "score": 1.0}])
    frame_b = pd.DataFrame([{"entry_id": "P1", "term": "GO:1", "branch": "MF", "score": 0.0}])

    blended = blend_prediction_frames([frame_a, frame_b], weights=[3.0, 1.0])

    assert blended.loc[0, "score"] == 0.75


def test_prepare_and_validate_submission_frame(tmp_path) -> None:
    predictions = pd.DataFrame(
        [
            {"entry_id": "P1", "term": "GO:1", "branch": "MF", "score": 0.2},
            {"entry_id": "P1", "term": "GO:1", "branch": "MF", "score": 0.9},
            {"entry_id": "P1", "term": "GO:2", "branch": "MF", "score": 0.8},
            {"entry_id": "BAD", "term": "GO:1", "branch": "MF", "score": 0.7},
            {"entry_id": "P1", "term": "GO:BAD", "branch": "MF", "score": 0.6},
        ]
    )

    submission = prepare_submission_frame(
        predictions,
        test_entry_ids=["P1"],
        valid_terms=_terms(),
        top_k_per_branch=2,
    )
    validation = validate_submission_predictions(submission, ["P1"], _terms(), require_all_test_entries=True)
    output_path = write_submission(submission, tmp_path / "submission.tsv")

    assert submission["term"].tolist() == ["GO:1", "GO:2"]
    assert validation["valid"] is True
    assert output_path.read_text(encoding="utf-8").splitlines()[0] == "P1\tGO:1\t0.9"
