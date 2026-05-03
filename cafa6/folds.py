"""Protein-level and cluster-aware fold assignment utilities."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class FoldConfig:
    """Fold assignment configuration."""

    n_folds: int = 5
    seed: int = 42


def stable_hash_int(value: object, seed: int = 42) -> int:
    """Return a deterministic integer hash for fold ordering."""

    payload = f"{seed}:{value}".encode("utf-8")
    return int(hashlib.sha1(payload).hexdigest()[:16], 16)


def make_length_bin(length: int) -> str:
    """Map a sequence length to a coarse deterministic bin."""

    if length <= 100:
        return "0000_0100"
    if length <= 200:
        return "0101_0200"
    if length <= 500:
        return "0201_0500"
    if length <= 1000:
        return "0501_1000"
    if length <= 2000:
        return "1001_2000"
    return "2001_plus"


def prepare_cluster_assignments(
    sequences: pd.DataFrame,
    clusters: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Attach one cluster ID to each protein, using singleton clusters as fallback."""

    if "entry_id" not in sequences.columns:
        raise ValueError("sequences must contain an entry_id column.")

    base_columns = ["entry_id", "length", "taxon_id"]
    available_columns = [column for column in base_columns if column in sequences.columns]
    assigned = sequences.loc[:, available_columns].copy()
    assigned = assigned.drop_duplicates("entry_id")
    assigned["entry_id"] = assigned["entry_id"].astype(str)

    if clusters is None:
        assigned["cluster_id"] = "singleton:" + assigned["entry_id"].astype(str)
        assigned["cluster_source"] = "entry_id_singleton"
    else:
        missing = {"entry_id", "cluster_id"}.difference(clusters.columns)
        if missing:
            raise ValueError(f"clusters are missing required columns: {', '.join(sorted(missing))}")

        cluster_frame = clusters.loc[:, ["entry_id", "cluster_id"]].dropna().drop_duplicates()
        cluster_frame["entry_id"] = cluster_frame["entry_id"].astype(str)
        cluster_frame["cluster_id"] = cluster_frame["cluster_id"].astype(str)
        conflicts = cluster_frame.groupby("entry_id")["cluster_id"].nunique()
        conflicting_ids = conflicts[conflicts > 1].index.tolist()
        if conflicting_ids:
            preview = ", ".join(map(str, conflicting_ids[:10]))
            raise ValueError(f"Cluster file assigns multiple cluster IDs to entries: {preview}")

        assigned = assigned.merge(cluster_frame, on="entry_id", how="left")
        missing_cluster = assigned["cluster_id"].isna()
        assigned.loc[missing_cluster, "cluster_id"] = "singleton:" + assigned.loc[missing_cluster, "entry_id"].astype(str)
        assigned["cluster_source"] = "provided_cluster"
        assigned.loc[missing_cluster, "cluster_source"] = "entry_id_singleton"

    assigned["cluster_id"] = assigned["cluster_id"].astype(str)
    if "length" in assigned.columns:
        assigned["length_bin"] = assigned["length"].fillna(0).astype(int).map(make_length_bin)
    else:
        assigned["length_bin"] = "unknown"
    if "taxon_id" not in assigned.columns:
        assigned["taxon_id"] = "unknown"
    assigned["stratification_key"] = assigned["taxon_id"].fillna("unknown").astype(str) + "|" + assigned["length_bin"]
    return assigned.sort_values("entry_id", kind="mergesort").reset_index(drop=True)


def assign_cluster_folds(clustered_entries: pd.DataFrame, config: FoldConfig = FoldConfig()) -> pd.DataFrame:
    """Assign whole clusters to folds with deterministic size balancing."""

    if config.n_folds < 2:
        raise ValueError("n_folds must be at least 2.")

    group_sizes = (
        clustered_entries.groupby("cluster_id", as_index=False)
        .agg(n_entries=("entry_id", "nunique"))
        .assign(hash_key=lambda frame: frame["cluster_id"].map(lambda value: stable_hash_int(value, config.seed)))
        .sort_values(["n_entries", "hash_key", "cluster_id"], ascending=[False, True, True], kind="mergesort")
        .reset_index(drop=True)
    )

    fold_sizes = {fold: 0 for fold in range(config.n_folds)}
    fold_group_counts = {fold: 0 for fold in range(config.n_folds)}
    cluster_to_fold: dict[str, int] = {}

    for row in group_sizes.itertuples(index=False):
        fold = min(range(config.n_folds), key=lambda value: (fold_sizes[value], fold_group_counts[value], value))
        cluster_to_fold[str(row.cluster_id)] = int(fold)
        fold_sizes[fold] += int(row.n_entries)
        fold_group_counts[fold] += 1

    output = clustered_entries.copy()
    output["fold"] = output["cluster_id"].map(cluster_to_fold).astype(int)
    return output.sort_values(["fold", "entry_id"], kind="mergesort").reset_index(drop=True)


def make_fold_assignments(
    sequences: pd.DataFrame,
    clusters: pd.DataFrame | None = None,
    n_folds: int = 5,
    seed: int = 42,
) -> pd.DataFrame:
    """Create deterministic protein-level or cluster-aware folds."""

    clustered_entries = prepare_cluster_assignments(sequences, clusters)
    folds = assign_cluster_folds(clustered_entries, FoldConfig(n_folds=n_folds, seed=seed))
    columns = [
        "entry_id",
        "fold",
        "cluster_id",
        "cluster_source",
        "taxon_id",
        "length",
        "length_bin",
        "stratification_key",
    ]
    columns = [column for column in columns if column in folds.columns]
    return folds.loc[:, columns].sort_values("entry_id", kind="mergesort").reset_index(drop=True)


def validate_fold_assignments(folds: pd.DataFrame) -> dict[str, object]:
    """Validate protein and cluster leakage constraints."""

    required = {"entry_id", "fold", "cluster_id"}
    missing = required.difference(folds.columns)
    if missing:
        raise ValueError(f"folds are missing required columns: {', '.join(sorted(missing))}")

    duplicate_entry_count = int(folds["entry_id"].duplicated().sum())
    cluster_fold_counts = folds.groupby("cluster_id")["fold"].nunique()
    leaking_clusters = sorted(cluster_fold_counts[cluster_fold_counts > 1].index.astype(str).tolist())

    fold_sizes = {
        int(fold): int(count)
        for fold, count in folds.groupby("fold")["entry_id"].nunique().sort_index().items()
    }
    cluster_counts = {
        int(fold): int(count)
        for fold, count in folds.groupby("fold")["cluster_id"].nunique().sort_index().items()
    }

    return {
        "n_entries": int(folds["entry_id"].nunique()),
        "n_rows": int(len(folds)),
        "n_clusters": int(folds["cluster_id"].nunique()),
        "n_folds": int(folds["fold"].nunique()),
        "duplicate_entry_count": duplicate_entry_count,
        "cluster_leak_count": int(len(leaking_clusters)),
        "leaking_clusters": leaking_clusters,
        "fold_sizes": fold_sizes,
        "cluster_counts_by_fold": cluster_counts,
    }


def make_fold_report(
    folds: pd.DataFrame,
    train_terms: pd.DataFrame | None = None,
    n_folds: int = 5,
    seed: int = 42,
) -> dict[str, object]:
    """Build a compact fold validation and branch-distribution report."""

    validation = validate_fold_assignments(folds)
    report: dict[str, object] = {
        "n_folds_requested": int(n_folds),
        "seed": int(seed),
        "validation": validation,
    }

    if train_terms is not None and not train_terms.empty:
        label_counts = train_terms.merge(folds.loc[:, ["entry_id", "fold"]], on="entry_id", how="inner")
        if "branch" in label_counts.columns:
            by_branch = (
                label_counts.groupby(["fold", "branch"])["term"]
                .nunique()
                .reset_index(name="n_terms")
                .sort_values(["fold", "branch"], kind="mergesort")
            )
            report["unique_terms_by_fold_branch"] = [
                {"fold": int(row.fold), "branch": str(row.branch), "n_terms": int(row.n_terms)}
                for row in by_branch.itertuples(index=False)
            ]
            rows_by_branch = (
                label_counts.groupby(["fold", "branch"])["term"]
                .size()
                .reset_index(name="n_rows")
                .sort_values(["fold", "branch"], kind="mergesort")
            )
            report["annotation_rows_by_fold_branch"] = [
                {"fold": int(row.fold), "branch": str(row.branch), "n_rows": int(row.n_rows)}
                for row in rows_by_branch.itertuples(index=False)
            ]

    return report
