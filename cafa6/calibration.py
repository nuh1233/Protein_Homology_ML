"""Calibration utilities for branch-specific prediction scores."""

from __future__ import annotations

import pandas as pd

from cafa6.metrics import score_branch_fmaxes


def clip_scores(predictions: pd.DataFrame, score_column: str = "score") -> pd.DataFrame:
    """Return predictions with scores clipped to [0, 1]."""

    if score_column not in predictions.columns:
        raise ValueError(f"predictions must contain a {score_column!r} column.")

    output = predictions.copy()
    output[score_column] = pd.to_numeric(output[score_column], errors="raise").clip(0.0, 1.0)
    return output


def collapse_prediction_scores(
    predictions: pd.DataFrame,
    group_columns: tuple[str, ...] = ("entry_id", "term", "branch"),
    score_column: str = "score",
) -> pd.DataFrame:
    """Collapse duplicate prediction rows by maximum score."""

    missing = set(group_columns).difference(predictions.columns)
    if missing:
        raise ValueError(f"predictions are missing required columns: {', '.join(sorted(missing))}")
    if score_column not in predictions.columns:
        raise ValueError(f"predictions must contain a {score_column!r} column.")

    output = clip_scores(predictions.loc[:, list(group_columns) + [score_column]], score_column=score_column)
    if output.empty:
        return output

    output = (
        output.groupby(list(group_columns), as_index=False)
        .agg(**{score_column: (score_column, "max")})
        .sort_values(list(group_columns), kind="mergesort")
        .reset_index(drop=True)
    )
    return output


def normalize_scores_by_branch(
    predictions: pd.DataFrame,
    score_column: str = "score",
    branch_column: str = "branch",
    method: str = "max",
) -> pd.DataFrame:
    """Apply simple deterministic per-branch score normalization."""

    if method not in {"none", "max"}:
        raise ValueError("method must be one of: none, max.")

    output = clip_scores(predictions, score_column=score_column)
    if method == "none" or output.empty:
        return output
    if branch_column not in output.columns:
        raise ValueError(f"predictions must contain a {branch_column!r} column.")

    branch_max = output.groupby(branch_column)[score_column].transform("max")
    output[score_column] = output[score_column].where(branch_max <= 0.0, output[score_column] / branch_max)
    output[score_column] = output[score_column].clip(0.0, 1.0)
    return output


def fit_branch_calibration(
    oof_predictions: pd.DataFrame,
    truth: pd.DataFrame,
    method: str = "max",
    average: str = "protein_macro",
) -> pd.DataFrame:
    """Fit deterministic branch score calibration metadata from OOF predictions."""

    if method not in {"none", "max"}:
        raise ValueError("method must be one of: none, max.")
    if "branch" not in oof_predictions.columns:
        raise ValueError("oof_predictions must contain a branch column.")

    predictions = clip_scores(oof_predictions)
    score_stats = (
        predictions.groupby("branch", as_index=False)
        .agg(
            raw_score_min=("score", "min"),
            raw_score_max=("score", "max"),
            raw_score_mean=("score", "mean"),
            prediction_rows=("score", "size"),
        )
    )
    scores = score_branch_fmaxes(truth=truth, predictions=predictions, average=average)
    calibration = scores.merge(score_stats, on="branch", how="left")
    calibration["method"] = method
    calibration["scale"] = 1.0
    if method == "max":
        calibration["scale"] = calibration["raw_score_max"].where(calibration["raw_score_max"] > 0.0, 1.0)
    calibration["scale"] = calibration["scale"].fillna(1.0).astype(float)
    return calibration.sort_values("branch", kind="mergesort").reset_index(drop=True)


def apply_branch_calibration(
    predictions: pd.DataFrame,
    calibration: pd.DataFrame,
    score_column: str = "score",
    branch_column: str = "branch",
) -> pd.DataFrame:
    """Apply branch calibration metadata to a prediction table."""

    if branch_column not in predictions.columns:
        raise ValueError(f"predictions must contain a {branch_column!r} column.")
    required = {"branch", "method", "scale"}
    missing = required.difference(calibration.columns)
    if missing:
        raise ValueError(f"calibration is missing required columns: {', '.join(sorted(missing))}")

    output = clip_scores(predictions, score_column=score_column)
    methods = dict(zip(calibration["branch"], calibration["method"], strict=False))
    scales = dict(zip(calibration["branch"], calibration["scale"], strict=False))

    unknown = sorted(set(output[branch_column]).difference(methods))
    if unknown:
        raise ValueError(f"Missing calibration rows for branches: {unknown}")

    calibrated_scores = output[score_column].copy()
    for branch, method in methods.items():
        mask = output[branch_column] == branch
        if not mask.any():
            continue
        if method == "none":
            continue
        if method != "max":
            raise ValueError(f"Unsupported calibration method for branch {branch}: {method}")

        scale = float(scales[branch])
        if scale > 0.0:
            calibrated_scores.loc[mask] = calibrated_scores.loc[mask] / scale

    output[score_column] = calibrated_scores.clip(0.0, 1.0)
    return output
