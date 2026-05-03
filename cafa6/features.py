"""Feature assembly utilities for CAFA 6 models."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy import sparse

from cafa6.embeddings import read_manifest


@dataclass(frozen=True)
class EmbeddingMatrix:
    """Aligned embedding matrix and row metadata."""

    entry_ids: np.ndarray
    matrix: np.ndarray
    manifest: pd.DataFrame


def load_embedding_matrix(manifest_path: str | Path) -> EmbeddingMatrix:
    """Load all shard embeddings described by a manifest into row-index order."""

    manifest = read_manifest(manifest_path)
    if manifest.empty:
        raise ValueError(f"Empty embedding manifest: {manifest_path}")

    incomplete = manifest.loc[manifest["status"] != "complete"]
    if not incomplete.empty:
        raise ValueError(f"Embedding manifest has {len(incomplete)} incomplete rows: {manifest_path}")

    manifest = manifest.sort_values("row_index", kind="mergesort").reset_index(drop=True)
    matrices: list[np.ndarray] = []
    expected_dim: int | None = None

    for _, batch_manifest in manifest.groupby("batch_id", sort=True):
        shard_path = Path(batch_manifest["shard_path"].iloc[0])
        if not shard_path.is_file():
            raise FileNotFoundError(f"Missing embedding shard: {shard_path}")

        shard = np.load(shard_path).astype("float32", copy=False)
        batch_manifest = batch_manifest.sort_values("row_index_in_batch", kind="mergesort")
        if shard.shape[0] != len(batch_manifest):
            raise ValueError(
                f"Shard row count mismatch for {shard_path}: shard={shard.shape[0]} manifest={len(batch_manifest)}"
            )
        if expected_dim is None:
            expected_dim = int(shard.shape[1])
        elif shard.shape[1] != expected_dim:
            raise ValueError(f"Shard dimension mismatch for {shard_path}: expected {expected_dim}, got {shard.shape[1]}")

        matrices.append(shard)

    matrix = np.vstack(matrices).astype("float32", copy=False)
    if matrix.shape[0] != len(manifest):
        raise ValueError(f"Loaded embedding row count mismatch: matrix={matrix.shape[0]} manifest={len(manifest)}")

    return EmbeddingMatrix(
        entry_ids=manifest["entry_id"].astype(str).to_numpy(),
        matrix=matrix,
        manifest=manifest,
    )


def align_rows(entry_ids: Iterable[str], target_entry_ids: Iterable[str]) -> np.ndarray:
    """Return row indices that align entry_ids to target_entry_ids order."""

    index_by_entry = {entry_id: index for index, entry_id in enumerate(map(str, entry_ids))}
    indices: list[int] = []
    missing: list[str] = []
    for entry_id in map(str, target_entry_ids):
        index = index_by_entry.get(entry_id)
        if index is None:
            missing.append(entry_id)
        else:
            indices.append(index)

    if missing:
        preview = ", ".join(missing[:10])
        raise ValueError(f"Missing embedding rows for {len(missing)} entry IDs: {preview}")
    return np.array(indices, dtype=int)


def select_branch_terms(
    train_terms: pd.DataFrame,
    branch: str,
    min_count: int = 25,
    max_labels: int | None = None,
) -> pd.DataFrame:
    """Select frequent labels for one branch, sorted by frequency then GO term."""

    required = {"entry_id", "term", "branch"}
    missing = required.difference(train_terms.columns)
    if missing:
        raise ValueError(f"train_terms is missing required columns: {', '.join(sorted(missing))}")

    terms = train_terms.loc[train_terms["branch"] == branch, ["entry_id", "term", "branch"]].drop_duplicates()
    frequency = (
        terms.groupby("term", as_index=False)
        .agg(n_proteins=("entry_id", "nunique"))
        .sort_values(["n_proteins", "term"], ascending=[False, True], kind="mergesort")
        .reset_index(drop=True)
    )
    frequency = frequency.loc[frequency["n_proteins"] >= min_count]
    if max_labels is not None and max_labels > 0:
        frequency = frequency.head(max_labels)
    frequency["branch"] = branch
    return frequency.loc[:, ["branch", "term", "n_proteins"]].reset_index(drop=True)


def build_label_matrix(
    entry_ids: Iterable[str],
    train_terms: pd.DataFrame,
    label_terms: Iterable[str],
    branch: str,
) -> sparse.csr_matrix:
    """Build a sparse multilabel indicator matrix in entry/label order."""

    entries = list(map(str, entry_ids))
    labels = list(map(str, label_terms))
    entry_index = {entry_id: index for index, entry_id in enumerate(entries)}
    label_index = {term: index for index, term in enumerate(labels)}

    terms = train_terms.loc[
        (train_terms["branch"] == branch) & (train_terms["term"].isin(label_index.keys())),
        ["entry_id", "term"],
    ].drop_duplicates()

    rows: list[int] = []
    cols: list[int] = []
    for row in terms.itertuples(index=False):
        entry_id = str(row.entry_id)
        if entry_id not in entry_index:
            continue
        rows.append(entry_index[entry_id])
        cols.append(label_index[str(row.term)])

    data = np.ones(len(rows), dtype=np.int8)
    return sparse.csr_matrix((data, (rows, cols)), shape=(len(entries), len(labels)), dtype=np.int8)


def filter_trainable_labels(
    y: sparse.csr_matrix,
    label_terms: Iterable[str],
    min_positive: int = 1,
) -> tuple[sparse.csr_matrix, list[str], np.ndarray]:
    """Drop labels that do not have both positive and negative examples."""

    labels = list(label_terms)
    positive_counts = np.asarray(y.sum(axis=0)).ravel()
    keep = (positive_counts >= min_positive) & (positive_counts < y.shape[0])
    keep_indices = np.where(keep)[0]
    kept_labels = [labels[index] for index in keep_indices]
    return y[:, keep_indices].tocsr(), kept_labels, keep_indices


def make_prediction_frame(
    entry_ids: Iterable[str],
    label_terms: Iterable[str],
    scores: np.ndarray,
    branch: str,
    top_k: int = 100,
    min_score: float = 0.0,
) -> pd.DataFrame:
    """Convert a dense score matrix to top-k long-form predictions."""

    entries = list(map(str, entry_ids))
    labels = np.array(list(label_terms), dtype=object)
    if scores.ndim != 2:
        raise ValueError("scores must be a 2D array.")
    if scores.shape != (len(entries), len(labels)):
        raise ValueError(f"score shape {scores.shape} does not match entries/labels {(len(entries), len(labels))}")

    if len(labels) == 0 or len(entries) == 0:
        return pd.DataFrame(columns=["entry_id", "term", "branch", "score"])

    k = min(top_k, len(labels)) if top_k and top_k > 0 else len(labels)
    rows: list[dict[str, object]] = []
    for row_index, entry_id in enumerate(entries):
        row_scores = scores[row_index]
        if k < len(labels):
            candidate_indices = np.argpartition(-row_scores, kth=k - 1)[:k]
            candidate_indices = candidate_indices[np.argsort(-row_scores[candidate_indices], kind="mergesort")]
        else:
            candidate_indices = np.argsort(-row_scores, kind="mergesort")

        for label_index in candidate_indices:
            score = float(row_scores[label_index])
            if score < min_score:
                continue
            rows.append(
                {
                    "entry_id": entry_id,
                    "term": str(labels[label_index]),
                    "branch": branch,
                    "score": score,
                }
            )

    return pd.DataFrame.from_records(rows, columns=["entry_id", "term", "branch", "score"])
