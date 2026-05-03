from __future__ import annotations

import pandas as pd
import pytest

from cafa6.calibration import (
    apply_branch_calibration,
    collapse_prediction_scores,
    fit_branch_calibration,
)


def test_collapse_prediction_scores_keeps_max_duplicate() -> None:
    predictions = pd.DataFrame(
        [
            {"entry_id": "P1", "term": "GO:1", "branch": "MF", "score": 0.1},
            {"entry_id": "P1", "term": "GO:1", "branch": "MF", "score": 0.9},
        ]
    )

    collapsed = collapse_prediction_scores(predictions)

    assert len(collapsed) == 1
    assert collapsed.loc[0, "score"] == 0.9


def test_fit_and_apply_branch_calibration_max_normalizes_scores() -> None:
    truth = pd.DataFrame(
        [
            {"entry_id": "P1", "term": "GO:1", "branch": "MF"},
            {"entry_id": "P2", "term": "GO:2", "branch": "MF"},
        ]
    )
    predictions = pd.DataFrame(
        [
            {"entry_id": "P1", "term": "GO:1", "branch": "MF", "score": 0.2},
            {"entry_id": "P2", "term": "GO:2", "branch": "MF", "score": 0.4},
        ]
    )

    calibration = fit_branch_calibration(predictions, truth, method="max")
    calibrated = apply_branch_calibration(predictions, calibration)

    assert calibration.loc[calibration["branch"] == "MF", "scale"].item() == 0.4
    assert calibrated["score"].tolist() == [0.5, 1.0]


def test_apply_branch_calibration_rejects_missing_branch_metadata() -> None:
    predictions = pd.DataFrame([{"entry_id": "P1", "term": "GO:1", "branch": "MF", "score": 0.2}])
    calibration = pd.DataFrame([{"branch": "BP", "method": "max", "scale": 1.0}])

    with pytest.raises(ValueError, match="Missing calibration"):
        apply_branch_calibration(predictions, calibration)
