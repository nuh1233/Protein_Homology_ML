"""Submission validation and writing utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from cafa6.calibration import clip_scores
from cafa6.ensemble import prune_top_k_by_group


SUBMISSION_COLUMNS: tuple[str, ...] = ("entry_id", "term", "score")


def _as_string_set(values: Iterable[str]) -> set[str]:
    return set(map(str, values))


def prepare_submission_frame(
    predictions: pd.DataFrame,
    test_entry_ids: Iterable[str] | None = None,
    valid_terms: Iterable[str] | pd.DataFrame | None = None,
    top_k_per_branch: int | None = None,
    top_k_total: int | None = None,
) -> pd.DataFrame:
    """Prepare GO predictions for Kaggle's headerless three-column TSV format."""

    required = {"entry_id", "term", "score"}
    missing = required.difference(predictions.columns)
    if missing:
        raise ValueError(f"predictions are missing required columns: {', '.join(sorted(missing))}")

    output = predictions.copy()
    if top_k_per_branch is not None and top_k_per_branch > 0:
        if "branch" not in output.columns:
            raise ValueError("top_k_per_branch requires a branch column.")
        output = prune_top_k_by_group(output, top_k=top_k_per_branch, group_columns=("entry_id", "branch"))

    output = clip_scores(output.loc[:, [column for column in ["entry_id", "term", "branch", "score"] if column in output.columns]])
    output["entry_id"] = output["entry_id"].astype(str)
    output["term"] = output["term"].astype(str)

    if test_entry_ids is not None:
        valid_entries = _as_string_set(test_entry_ids)
        output = output.loc[output["entry_id"].isin(valid_entries)]

    if valid_terms is not None:
        if isinstance(valid_terms, pd.DataFrame):
            valid_term_ids = _as_string_set(valid_terms["term"])
        else:
            valid_term_ids = _as_string_set(valid_terms)
        output = output.loc[output["term"].isin(valid_term_ids)]

    output = (
        output.groupby(["entry_id", "term"], as_index=False)
        .agg(score=("score", "max"))
        .sort_values(["entry_id", "score", "term"], ascending=[True, False, True], kind="mergesort")
        .reset_index(drop=True)
    )

    if top_k_total is not None and top_k_total > 0 and not output.empty:
        output = output.sort_values(["entry_id", "score", "term"], ascending=[True, False, True], kind="mergesort")
        ranks = output.groupby("entry_id").cumcount()
        output = output.loc[ranks < top_k_total].reset_index(drop=True)

    return output.loc[:, list(SUBMISSION_COLUMNS)]


def validate_submission_predictions(
    predictions: pd.DataFrame,
    test_entry_ids: Iterable[str],
    valid_terms: Iterable[str] | pd.DataFrame,
    require_all_test_entries: bool = False,
) -> dict[str, object]:
    """Return a validation report for a submission-like prediction frame."""

    missing_columns = sorted(set(SUBMISSION_COLUMNS).difference(predictions.columns))
    if missing_columns:
        return {
            "valid": False,
            "missing_columns": missing_columns,
            "prediction_rows": int(len(predictions)),
        }

    valid_entries = _as_string_set(test_entry_ids)
    valid_term_ids = _as_string_set(valid_terms["term"]) if isinstance(valid_terms, pd.DataFrame) else _as_string_set(valid_terms)

    output = predictions.loc[:, list(SUBMISSION_COLUMNS)].copy()
    output["entry_id"] = output["entry_id"].astype(str)
    output["term"] = output["term"].astype(str)
    output["score"] = pd.to_numeric(output["score"], errors="coerce")

    duplicate_count = int(output.duplicated(["entry_id", "term"]).sum())
    invalid_score_count = int((output["score"].isna() | (output["score"] < 0.0) | (output["score"] > 1.0)).sum())
    invalid_entry_count = int((~output["entry_id"].isin(valid_entries)).sum())
    invalid_term_count = int((~output["term"].isin(valid_term_ids)).sum())
    covered_entries = set(output.loc[output["entry_id"].isin(valid_entries), "entry_id"])
    missing_test_entries = sorted(valid_entries.difference(covered_entries))
    valid = (
        len(output) > 0
        and duplicate_count == 0
        and invalid_score_count == 0
        and invalid_entry_count == 0
        and invalid_term_count == 0
        and (not require_all_test_entries or not missing_test_entries)
    )

    return {
        "valid": bool(valid),
        "prediction_rows": int(len(output)),
        "unique_entry_ids": int(output["entry_id"].nunique()),
        "unique_terms": int(output["term"].nunique()),
        "duplicate_entry_term_count": duplicate_count,
        "invalid_score_count": invalid_score_count,
        "invalid_entry_count": invalid_entry_count,
        "invalid_term_count": invalid_term_count,
        "test_entry_count": int(len(valid_entries)),
        "covered_test_entry_count": int(len(covered_entries)),
        "missing_test_entry_count": int(len(missing_test_entries)),
        "missing_test_entry_preview": missing_test_entries[:20],
        "require_all_test_entries": bool(require_all_test_entries),
    }


def write_submission(frame: pd.DataFrame, output_path: str | Path) -> Path:
    """Write a headerless CAFA/Kaggle submission TSV."""

    missing = set(SUBMISSION_COLUMNS).difference(frame.columns)
    if missing:
        raise ValueError(f"submission frame is missing columns: {', '.join(sorted(missing))}")

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.loc[:, list(SUBMISSION_COLUMNS)].to_csv(path, sep="\t", header=False, index=False)
    return path
