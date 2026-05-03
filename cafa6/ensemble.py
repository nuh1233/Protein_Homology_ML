"""Prediction blending and GO hierarchy repair utilities."""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from cafa6.calibration import clip_scores, collapse_prediction_scores


PREDICTION_COLUMNS: tuple[str, ...] = ("entry_id", "term", "branch", "score")


def _require_prediction_columns(predictions: pd.DataFrame) -> None:
    missing = set(PREDICTION_COLUMNS).difference(predictions.columns)
    if missing:
        raise ValueError(f"predictions are missing required columns: {', '.join(sorted(missing))}")


def canonicalize_prediction_terms(
    predictions: pd.DataFrame,
    terms: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Return predictions with valid GO terms and canonical branch labels."""

    _require_prediction_columns(predictions)
    output = clip_scores(predictions.loc[:, list(PREDICTION_COLUMNS)])
    output["entry_id"] = output["entry_id"].astype(str)
    output["term"] = output["term"].astype(str)
    output["branch"] = output["branch"].astype(str)

    if terms is None:
        return collapse_prediction_scores(output)

    required = {"term", "branch"}
    missing = required.difference(terms.columns)
    if missing:
        raise ValueError(f"terms are missing required columns: {', '.join(sorted(missing))}")

    term_branch = terms.loc[:, ["term", "branch"]].drop_duplicates("term")
    output = output.merge(term_branch.rename(columns={"branch": "canonical_branch"}), on="term", how="inner")
    output["branch"] = output["canonical_branch"]
    output = output.drop(columns=["canonical_branch"])
    return collapse_prediction_scores(output)


def repair_go_hierarchy(
    predictions: pd.DataFrame,
    ancestors: pd.DataFrame,
    terms: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Add GO ancestor predictions and keep the maximum child-implied score."""

    base = canonicalize_prediction_terms(predictions, terms=terms)
    required = {"term", "ancestor"}
    missing = required.difference(ancestors.columns)
    if missing:
        raise ValueError(f"ancestors are missing required columns: {', '.join(sorted(missing))}")

    if base.empty:
        return base

    ancestor_columns = ["term", "ancestor"]
    if "ancestor_branch" in ancestors.columns:
        ancestor_columns.append("ancestor_branch")

    links = ancestors.loc[:, ancestor_columns].drop_duplicates()
    ancestor_predictions = base.merge(links, on="term", how="inner")
    if ancestor_predictions.empty:
        return base

    ancestor_predictions = ancestor_predictions.rename(columns={"term": "source_term", "ancestor": "term"})
    if "ancestor_branch" in ancestor_predictions.columns:
        ancestor_predictions["branch"] = ancestor_predictions["ancestor_branch"]
        ancestor_predictions = ancestor_predictions.drop(columns=["ancestor_branch"])

    ancestor_predictions = ancestor_predictions.loc[:, list(PREDICTION_COLUMNS)]
    repaired = pd.concat([base, ancestor_predictions], ignore_index=True)
    return collapse_prediction_scores(repaired)


def prune_top_k_by_group(
    predictions: pd.DataFrame,
    top_k: int | None,
    group_columns: tuple[str, ...] = ("entry_id", "branch"),
) -> pd.DataFrame:
    """Keep the highest scoring top-k rows inside each group."""

    if top_k is None or top_k <= 0 or predictions.empty:
        return predictions.reset_index(drop=True)

    missing = set(group_columns).difference(predictions.columns)
    if missing:
        raise ValueError(f"predictions are missing required group columns: {', '.join(sorted(missing))}")

    output = predictions.sort_values(
        list(group_columns) + ["score", "term"],
        ascending=[True] * len(group_columns) + [False, True],
        kind="mergesort",
    ).reset_index(drop=True)
    ranks = output.groupby(list(group_columns)).cumcount()
    return output.loc[ranks < top_k].reset_index(drop=True)


def blend_prediction_frames(
    frames: Sequence[pd.DataFrame],
    weights: Sequence[float] | None = None,
) -> pd.DataFrame:
    """Blend multiple prediction frames by weighted mean score."""

    if not frames:
        return pd.DataFrame(columns=list(PREDICTION_COLUMNS))
    if weights is None:
        weights = [1.0] * len(frames)
    if len(weights) != len(frames):
        raise ValueError("weights must have the same length as frames.")
    if any(weight < 0 for weight in weights):
        raise ValueError("weights must be non-negative.")
    total_weight = float(sum(weights))
    if total_weight <= 0.0:
        raise ValueError("At least one blend weight must be positive.")

    weighted_frames: list[pd.DataFrame] = []
    for frame, weight in zip(frames, weights, strict=True):
        prepared = canonicalize_prediction_terms(frame)
        prepared["weighted_score"] = prepared["score"] * float(weight)
        weighted_frames.append(prepared.loc[:, ["entry_id", "term", "branch", "weighted_score"]])

    stacked = pd.concat(weighted_frames, ignore_index=True)
    blended = (
        stacked.groupby(["entry_id", "term", "branch"], as_index=False)
        .agg(weighted_score=("weighted_score", "sum"))
        .rename(columns={"weighted_score": "score"})
    )
    blended["score"] = (blended["score"] / total_weight).clip(0.0, 1.0)
    return blended.sort_values(["entry_id", "branch", "score", "term"], ascending=[True, True, False, True], kind="mergesort").reset_index(drop=True)
