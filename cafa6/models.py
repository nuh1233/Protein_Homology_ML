"""Branch-specific supervised model utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import SGDClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from cafa6.features import make_prediction_frame


@dataclass
class BranchSupervisedModel:
    """A trained branch-specific multilabel classifier."""

    branch: str
    label_terms: list[str]
    estimator: object
    metadata: dict[str, object]


def make_sgd_ovr_classifier(
    alpha: float = 1e-4,
    max_iter: int = 1000,
    random_state: int = 42,
    n_jobs: int = -1,
) -> object:
    """Create a scaled one-vs-rest logistic SGD classifier."""

    base = SGDClassifier(
        loss="log_loss",
        alpha=alpha,
        max_iter=max_iter,
        tol=1e-3,
        average=True,
        class_weight="balanced",
        random_state=random_state,
    )
    return make_pipeline(
        StandardScaler(copy=True),
        OneVsRestClassifier(base, n_jobs=n_jobs),
    )


def train_branch_model(
    x_train: np.ndarray,
    y_train: sparse.csr_matrix,
    label_terms: Iterable[str],
    branch: str,
    alpha: float = 1e-4,
    max_iter: int = 1000,
    random_state: int = 42,
    n_jobs: int = -1,
) -> BranchSupervisedModel:
    """Train one branch-specific multilabel model."""

    labels = list(label_terms)
    if y_train.shape[1] != len(labels):
        raise ValueError("y_train column count must match label_terms.")
    if y_train.shape[1] == 0:
        raise ValueError(f"No trainable labels for branch {branch}.")

    estimator = make_sgd_ovr_classifier(
        alpha=alpha,
        max_iter=max_iter,
        random_state=random_state,
        n_jobs=n_jobs,
    )
    estimator.fit(x_train.astype("float32", copy=False), y_train)
    metadata = {
        "branch": branch,
        "n_samples": int(x_train.shape[0]),
        "n_features": int(x_train.shape[1]),
        "n_labels": int(len(labels)),
        "alpha": float(alpha),
        "max_iter": int(max_iter),
        "random_state": int(random_state),
    }
    return BranchSupervisedModel(branch=branch, label_terms=labels, estimator=estimator, metadata=metadata)


def predict_branch_scores(model: BranchSupervisedModel, x: np.ndarray) -> np.ndarray:
    """Predict dense probability scores for a branch model."""

    estimator = model.estimator
    if hasattr(estimator, "predict_proba"):
        scores = estimator.predict_proba(x.astype("float32", copy=False))
    elif hasattr(estimator, "decision_function"):
        raw = estimator.decision_function(x.astype("float32", copy=False))
        scores = 1.0 / (1.0 + np.exp(-raw))
    else:
        raise ValueError("Estimator must expose predict_proba or decision_function.")

    scores = np.asarray(scores, dtype="float32")
    if scores.ndim == 1:
        scores = scores.reshape(-1, 1)
    return np.clip(scores, 0.0, 1.0)


def predict_branch_topk(
    model: BranchSupervisedModel,
    x: np.ndarray,
    entry_ids: Iterable[str],
    top_k: int = 100,
    min_score: float = 0.0,
    batch_size: int = 4096,
) -> pd.DataFrame:
    """Predict branch scores in batches and return top-k long-form rows."""

    entries = list(map(str, entry_ids))
    frames: list[pd.DataFrame] = []
    for start in range(0, len(entries), batch_size):
        stop = min(start + batch_size, len(entries))
        scores = predict_branch_scores(model, x[start:stop])
        frame = make_prediction_frame(
            entry_ids=entries[start:stop],
            label_terms=model.label_terms,
            scores=scores,
            branch=model.branch,
            top_k=top_k,
            min_score=min_score,
        )
        if not frame.empty:
            frames.append(frame)

    if not frames:
        return pd.DataFrame(columns=["entry_id", "term", "branch", "score"])
    return pd.concat(frames, ignore_index=True)
