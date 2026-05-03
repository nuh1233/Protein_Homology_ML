"""Branch-specific CAFA 6 scoring utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from cafa6.io import ASPECT_TO_BRANCH


BRANCHES: tuple[str, ...] = ("MF", "BP", "CC")


@dataclass(frozen=True)
class BranchScore:
    """Maximum F-measure details for one ontology branch."""

    branch: str
    fmax: float
    threshold: float
    precision: float
    recall: float
    n_truth_entries: int
    n_truth_terms: int
    n_prediction_rows: int


def make_thresholds(step: float = 0.01) -> np.ndarray:
    """Return deterministic thresholds in [0, 1]."""

    if step <= 0 or step > 1:
        raise ValueError("threshold step must be in the interval (0, 1].")

    n_steps = int(round(1.0 / step))
    thresholds = np.linspace(0.0, 1.0, n_steps + 1)
    return np.round(thresholds, 10)


def _ensure_branch_column(frame: pd.DataFrame, table_name: str) -> pd.DataFrame:
    output = frame.copy()
    if "branch" in output.columns:
        return output
    if "aspect" in output.columns:
        output["branch"] = output["aspect"].map(ASPECT_TO_BRANCH)
        if output["branch"].isna().any():
            bad = sorted(output.loc[output["branch"].isna(), "aspect"].dropna().unique().tolist())
            raise ValueError(f"{table_name} contains unsupported aspect values: {bad}")
        return output
    raise ValueError(f"{table_name} must contain either a branch or aspect column.")


def prepare_truth(truth: pd.DataFrame) -> pd.DataFrame:
    """Normalize a truth table to unique entry/term/branch rows."""

    required = {"entry_id", "term"}
    missing = required.difference(truth.columns)
    if missing:
        raise ValueError(f"truth is missing required columns: {', '.join(sorted(missing))}")

    output = _ensure_branch_column(truth, "truth")
    output = output.loc[:, ["entry_id", "term", "branch"]].dropna()
    output = output.loc[output["branch"].isin(BRANCHES)]
    output = output.drop_duplicates(["entry_id", "term", "branch"])
    return output.sort_values(["branch", "entry_id", "term"], kind="mergesort").reset_index(drop=True)


def prepare_predictions(
    predictions: pd.DataFrame,
    term_to_branch: pd.DataFrame | dict[str, str] | None = None,
) -> pd.DataFrame:
    """Normalize prediction rows and collapse duplicate scores by max."""

    required = {"entry_id", "term", "score"}
    missing = required.difference(predictions.columns)
    if missing:
        raise ValueError(f"predictions are missing required columns: {', '.join(sorted(missing))}")

    output = predictions.copy()
    if "branch" not in output.columns:
        if "aspect" in output.columns:
            output["branch"] = output["aspect"].map(ASPECT_TO_BRANCH)
        elif term_to_branch is not None:
            if isinstance(term_to_branch, pd.DataFrame):
                mapping = dict(zip(term_to_branch["term"], term_to_branch["branch"], strict=False))
            else:
                mapping = term_to_branch
            output["branch"] = output["term"].map(mapping)
        else:
            raise ValueError("predictions need a branch/aspect column or a term_to_branch mapping.")

    output = output.loc[:, ["entry_id", "term", "branch", "score"]].dropna()
    output = output.loc[output["branch"].isin(BRANCHES)]
    output["score"] = pd.to_numeric(output["score"], errors="raise")
    invalid_scores = output.loc[(output["score"] < 0.0) | (output["score"] > 1.0)]
    if not invalid_scores.empty:
        raise ValueError("prediction scores must be in the closed interval [0, 1].")

    output = (
        output.groupby(["entry_id", "term", "branch"], as_index=False)
        .agg(score=("score", "max"))
        .sort_values(["branch", "entry_id", "score", "term"], ascending=[True, True, False, True], kind="mergesort")
        .reset_index(drop=True)
    )
    return output


def _sets_by_entry(frame: pd.DataFrame, term_column: str = "term") -> dict[str, set[str]]:
    return frame.groupby("entry_id")[term_column].apply(lambda values: set(values)).to_dict()


def _protein_macro_at_threshold(
    truth_sets: dict[str, set[str]],
    prediction_frame: pd.DataFrame,
    threshold: float,
) -> tuple[float, float, float]:
    selected = prediction_frame.loc[prediction_frame["score"] >= threshold, ["entry_id", "term"]]
    prediction_sets = _sets_by_entry(selected) if not selected.empty else {}

    if prediction_sets:
        precision_values = []
        for entry_id, predicted_terms in prediction_sets.items():
            true_terms = truth_sets.get(entry_id, set())
            precision_values.append(len(predicted_terms.intersection(true_terms)) / len(predicted_terms))
        precision = float(np.mean(precision_values))
    else:
        precision = 0.0

    if truth_sets:
        recall_values = []
        for entry_id, true_terms in truth_sets.items():
            predicted_terms = prediction_sets.get(entry_id, set())
            recall_values.append(len(predicted_terms.intersection(true_terms)) / len(true_terms))
        recall = float(np.mean(recall_values))
    else:
        recall = 0.0

    f_measure = 0.0 if precision + recall == 0.0 else float(2.0 * precision * recall / (precision + recall))
    return precision, recall, f_measure


def _micro_at_threshold(
    truth_pairs: set[tuple[str, str]],
    prediction_frame: pd.DataFrame,
    threshold: float,
) -> tuple[float, float, float]:
    selected = prediction_frame.loc[prediction_frame["score"] >= threshold, ["entry_id", "term"]]
    predicted_pairs = set(map(tuple, selected.to_numpy())) if not selected.empty else set()

    true_positive = len(predicted_pairs.intersection(truth_pairs))
    precision = true_positive / len(predicted_pairs) if predicted_pairs else 0.0
    recall = true_positive / len(truth_pairs) if truth_pairs else 0.0
    f_measure = 0.0 if precision + recall == 0.0 else float(2.0 * precision * recall / (precision + recall))
    return float(precision), float(recall), f_measure


def score_branch_fmax(
    truth: pd.DataFrame,
    predictions: pd.DataFrame,
    branch: str,
    thresholds: Iterable[float] | None = None,
    average: str = "protein_macro",
) -> BranchScore:
    """Compute max F-measure for one ontology branch."""

    if branch not in BRANCHES:
        raise ValueError(f"Unsupported branch: {branch}")
    if average not in {"protein_macro", "micro"}:
        raise ValueError("average must be either 'protein_macro' or 'micro'.")

    truth_branch = prepare_truth(truth).loc[lambda frame: frame["branch"] == branch]
    prediction_branch = prepare_predictions(predictions).loc[lambda frame: frame["branch"] == branch]
    thresholds_array = np.array(list(thresholds) if thresholds is not None else make_thresholds(), dtype=float)

    if average == "protein_macro":
        truth_sets = _sets_by_entry(truth_branch)
        score_at_threshold = lambda threshold: _protein_macro_at_threshold(truth_sets, prediction_branch, threshold)
    else:
        truth_pairs = set(map(tuple, truth_branch.loc[:, ["entry_id", "term"]].to_numpy()))
        score_at_threshold = lambda threshold: _micro_at_threshold(truth_pairs, prediction_branch, threshold)

    best = BranchScore(
        branch=branch,
        fmax=0.0,
        threshold=float(thresholds_array[0]) if len(thresholds_array) else 0.0,
        precision=0.0,
        recall=0.0,
        n_truth_entries=int(truth_branch["entry_id"].nunique()),
        n_truth_terms=int(len(truth_branch)),
        n_prediction_rows=int(len(prediction_branch)),
    )

    for threshold in thresholds_array:
        precision, recall, f_measure = score_at_threshold(float(threshold))
        if f_measure > best.fmax:
            best = BranchScore(
                branch=branch,
                fmax=f_measure,
                threshold=float(threshold),
                precision=precision,
                recall=recall,
                n_truth_entries=int(truth_branch["entry_id"].nunique()),
                n_truth_terms=int(len(truth_branch)),
                n_prediction_rows=int(len(prediction_branch)),
            )

    return best


def score_branch_fmaxes(
    truth: pd.DataFrame,
    predictions: pd.DataFrame,
    thresholds: Iterable[float] | None = None,
    average: str = "protein_macro",
    term_to_branch: pd.DataFrame | dict[str, str] | None = None,
) -> pd.DataFrame:
    """Compute branch Fmax values and the official mean proxy."""

    truth_prepared = prepare_truth(truth)
    predictions_prepared = prepare_predictions(predictions, term_to_branch=term_to_branch)
    thresholds_array = np.array(list(thresholds) if thresholds is not None else make_thresholds(), dtype=float)

    rows = []
    for branch in BRANCHES:
        score = score_branch_fmax(
            truth_prepared,
            predictions_prepared,
            branch=branch,
            thresholds=thresholds_array,
            average=average,
        )
        rows.append(score.__dict__)

    scores = pd.DataFrame.from_records(rows)
    mean_fmax = float(scores["fmax"].mean()) if not scores.empty else 0.0
    scores["mean_fmax"] = mean_fmax
    scores["average"] = average
    return scores
