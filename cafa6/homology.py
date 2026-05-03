"""Homology hit aggregation and transfer utilities."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from cafa6.io import ASPECT_TO_BRANCH


QUERY_ALIASES: tuple[str, ...] = ("query_id", "query", "qseqid", "qid", "query_entry_id")
TARGET_ALIASES: tuple[str, ...] = ("target_id", "target", "sseqid", "subject_id", "sid", "target_entry_id")
SCORE_ALIASES: tuple[str, ...] = ("score", "hit_score", "weight")
BITSCORE_ALIASES: tuple[str, ...] = ("bitscore", "bits", "bit_score", "raw_score")
EVALUE_ALIASES: tuple[str, ...] = ("evalue", "eval")
PIDENT_ALIASES: tuple[str, ...] = ("pident", "identity", "percent_identity")
QCOV_ALIASES: tuple[str, ...] = ("qcov", "qcovs", "qcovhsp", "qcovus", "query_coverage")
TCOV_ALIASES: tuple[str, ...] = ("tcov", "scov", "scovs", "target_coverage", "subject_coverage")

STANDARD_HIT_COLUMNS: tuple[str, ...] = (
    "query_id",
    "target_id",
    "score",
    "bitscore",
    "evalue",
    "pident",
    "qcov",
    "tcov",
)


def _first_present(columns: pd.Index, aliases: tuple[str, ...]) -> str | None:
    column_lookup = {column.lower(): column for column in columns}
    for alias in aliases:
        if alias.lower() in column_lookup:
            return column_lookup[alias.lower()]
    return None


def read_hit_table(path: str | Path) -> pd.DataFrame:
    """Read a homology hit table from parquet, CSV, or TSV."""

    table_path = Path(path)
    suffix = table_path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(table_path)
    if suffix == ".csv":
        return pd.read_csv(table_path)
    if suffix in {".tsv", ".txt", ".m8"}:
        table = pd.read_csv(table_path, sep="\t")
        if _first_present(table.columns, QUERY_ALIASES) and _first_present(table.columns, TARGET_ALIASES):
            return table

        raw = pd.read_csv(table_path, sep="\t", header=None)
        if raw.shape[1] == 7:
            raw.columns = ["query_id", "target_id", "pident", "evalue", "bitscore", "qcov", "tcov"]
        elif raw.shape[1] >= 12:
            raw = raw.iloc[:, :12]
            raw.columns = [
                "query_id",
                "target_id",
                "pident",
                "alnlen",
                "mismatch",
                "gapopen",
                "qstart",
                "qend",
                "tstart",
                "tend",
                "evalue",
                "bitscore",
            ]
        elif raw.shape[1] >= 3:
            raw = raw.iloc[:, :3]
            raw.columns = ["query_id", "target_id", "score"]
        else:
            raise ValueError(f"Unable to infer headerless hit table columns: {table_path}")
        return raw
    raise ValueError(f"Unsupported hit table format: {table_path}")


def normalize_hit_table(hits: pd.DataFrame) -> pd.DataFrame:
    """Normalize common MMseqs/DIAMOND-style hit columns."""

    query_column = _first_present(hits.columns, QUERY_ALIASES)
    target_column = _first_present(hits.columns, TARGET_ALIASES)
    if query_column is None or target_column is None:
        raise ValueError("Hit table must contain query and target columns.")

    aliases = {
        "query_id": query_column,
        "target_id": target_column,
        "score": _first_present(hits.columns, SCORE_ALIASES),
        "bitscore": _first_present(hits.columns, BITSCORE_ALIASES),
        "evalue": _first_present(hits.columns, EVALUE_ALIASES),
        "pident": _first_present(hits.columns, PIDENT_ALIASES),
        "qcov": _first_present(hits.columns, QCOV_ALIASES),
        "tcov": _first_present(hits.columns, TCOV_ALIASES),
    }

    normalized = pd.DataFrame()
    normalized["query_id"] = hits[aliases["query_id"]]
    normalized["target_id"] = hits[aliases["target_id"]]

    for standard_name in STANDARD_HIT_COLUMNS[2:]:
        source_column = aliases[standard_name]
        if source_column is None:
            normalized[standard_name] = np.nan
        else:
            normalized[standard_name] = pd.to_numeric(hits[source_column], errors="coerce")

    normalized = normalized.dropna(subset=["query_id", "target_id"])
    normalized["query_id"] = normalized["query_id"].astype(str)
    normalized["target_id"] = normalized["target_id"].astype(str)
    normalized = normalized.loc[(normalized["query_id"] != "") & (normalized["target_id"] != "")]
    normalized = normalized.drop_duplicates().reset_index(drop=True)
    return normalized


def _as_fraction(values: pd.Series, default: float = 1.0) -> pd.Series:
    output = pd.to_numeric(values, errors="coerce").fillna(default).astype(float)
    output = output.where(output <= 1.0, output / 100.0)
    return output.clip(0.0, 1.0)


def add_hit_weights(hits: pd.DataFrame) -> pd.DataFrame:
    """Add bounded hit weights suitable for noisy-OR label aggregation."""

    normalized = normalize_hit_table(hits)
    weighted = normalized.copy()

    if weighted["score"].notna().any():
        base = weighted["score"].fillna(0.0).astype(float).clip(0.0, 1.0)
    elif weighted["bitscore"].notna().any():
        bitscore = weighted["bitscore"].fillna(0.0).clip(lower=0.0)
        query_max = bitscore.groupby(weighted["query_id"]).transform("max").replace(0.0, np.nan)
        base = (bitscore / query_max).fillna(0.0).clip(0.0, 1.0)
    elif weighted["pident"].notna().any():
        base = _as_fraction(weighted["pident"], default=0.0)
    elif weighted["evalue"].notna().any():
        evalue = weighted["evalue"].fillna(1.0).clip(lower=1e-300)
        base = (-np.log10(evalue) / 200.0).clip(0.0, 1.0)
    else:
        base = pd.Series(1.0, index=weighted.index)

    qcov = _as_fraction(weighted["qcov"], default=1.0)
    tcov = _as_fraction(weighted["tcov"], default=1.0)
    coverage = np.sqrt(qcov * tcov)

    weighted["hit_weight"] = (base * coverage).clip(0.0, 1.0)
    weighted = weighted.loc[weighted["hit_weight"] > 0.0].copy()
    weighted = weighted.sort_values(
        ["query_id", "hit_weight", "target_id"],
        ascending=[True, False, True],
        kind="mergesort",
    ).reset_index(drop=True)
    return weighted


def filter_oof_hits(hits: pd.DataFrame, folds: pd.DataFrame) -> pd.DataFrame:
    """Filter train-vs-train hits so validation labels cannot leak across folds."""

    if {"entry_id", "fold"}.difference(folds.columns):
        raise ValueError("folds must contain entry_id and fold columns.")

    weighted = add_hit_weights(hits)
    fold_table = folds.loc[:, ["entry_id", "fold"]].drop_duplicates("entry_id").copy()
    fold_table["entry_id"] = fold_table["entry_id"].astype(str)

    output = weighted.merge(
        fold_table.rename(columns={"entry_id": "query_id", "fold": "query_fold"}),
        on="query_id",
        how="inner",
    )
    output = output.merge(
        fold_table.rename(columns={"entry_id": "target_id", "fold": "target_fold"}),
        on="target_id",
        how="inner",
    )

    output = output.loc[output["query_id"] != output["target_id"]]
    output = output.loc[output["query_fold"] != output["target_fold"]]
    return output.sort_values(["query_id", "hit_weight", "target_id"], ascending=[True, False, True], kind="mergesort").reset_index(
        drop=True
    )


def filter_test_hits(hits: pd.DataFrame, train_terms: pd.DataFrame, exclude_self_hits: bool = True) -> pd.DataFrame:
    """Filter test-vs-train hits to targets that have training labels."""

    weighted = add_hit_weights(hits)
    labeled_targets = set(train_terms["entry_id"].astype(str))
    output = weighted.loc[weighted["target_id"].isin(labeled_targets)].copy()
    if exclude_self_hits:
        output = output.loc[output["query_id"] != output["target_id"]]
    return output.sort_values(["query_id", "hit_weight", "target_id"], ascending=[True, False, True], kind="mergesort").reset_index(
        drop=True
    )


def prepare_training_labels(train_terms: pd.DataFrame) -> pd.DataFrame:
    """Normalize training labels used as transfer targets."""

    required = {"entry_id", "term"}
    missing = required.difference(train_terms.columns)
    if missing:
        raise ValueError(f"train_terms is missing required columns: {', '.join(sorted(missing))}")

    labels = train_terms.copy()
    if "branch" not in labels.columns:
        if "aspect" not in labels.columns:
            raise ValueError("train_terms must contain branch or aspect.")
        labels["branch"] = labels["aspect"].map(ASPECT_TO_BRANCH)

    labels = labels.loc[:, ["entry_id", "term", "branch"]].dropna()
    labels["entry_id"] = labels["entry_id"].astype(str)
    labels = labels.drop_duplicates(["entry_id", "term", "branch"])
    return labels.sort_values(["entry_id", "branch", "term"], kind="mergesort").reset_index(drop=True)


def limit_hits_per_query(hits: pd.DataFrame, max_hits_per_query: int | None = 100) -> pd.DataFrame:
    """Keep the top weighted hits per query."""

    if max_hits_per_query is None or max_hits_per_query <= 0:
        return hits.copy()

    ranked = hits.sort_values(["query_id", "hit_weight", "target_id"], ascending=[True, False, True], kind="mergesort").copy()
    ranked["_hit_rank"] = ranked.groupby("query_id").cumcount()
    ranked = ranked.loc[ranked["_hit_rank"] < max_hits_per_query].drop(columns="_hit_rank")
    return ranked.reset_index(drop=True)


def aggregate_hit_labels(
    hits: pd.DataFrame,
    train_terms: pd.DataFrame,
    max_hits_per_query: int | None = 100,
    min_hit_weight: float = 0.0,
) -> pd.DataFrame:
    """Aggregate hit labels into query-term homology predictions."""

    if "hit_weight" not in hits.columns:
        weighted_hits = add_hit_weights(hits)
    else:
        weighted_hits = hits.copy()

    weighted_hits["hit_weight"] = pd.to_numeric(weighted_hits["hit_weight"], errors="coerce").fillna(0.0).clip(0.0, 1.0)
    weighted_hits = weighted_hits.loc[weighted_hits["hit_weight"] > min_hit_weight]
    weighted_hits = limit_hits_per_query(weighted_hits, max_hits_per_query=max_hits_per_query)
    if weighted_hits.empty:
        return pd.DataFrame(columns=["entry_id", "term", "branch", "score", "n_hits", "best_hit_weight"])

    labels = prepare_training_labels(train_terms)
    evidence = weighted_hits.merge(labels, left_on="target_id", right_on="entry_id", how="inner", suffixes=("_query", "_target"))
    if evidence.empty:
        return pd.DataFrame(columns=["entry_id", "term", "branch", "score", "n_hits", "best_hit_weight"])

    evidence["one_minus_weight"] = (1.0 - evidence["hit_weight"]).clip(1e-12, 1.0)
    grouped = evidence.groupby(["query_id", "term", "branch"], as_index=False).agg(
        log_miss=("one_minus_weight", lambda values: float(np.log(values).sum())),
        n_hits=("target_id", "nunique"),
        best_hit_weight=("hit_weight", "max"),
    )
    grouped["score"] = (1.0 - np.exp(grouped["log_miss"])).clip(0.0, 1.0)

    predictions = grouped.rename(columns={"query_id": "entry_id"})
    predictions = predictions.loc[:, ["entry_id", "term", "branch", "score", "n_hits", "best_hit_weight"]]
    predictions = predictions.sort_values(["entry_id", "branch", "score", "term"], ascending=[True, True, False, True], kind="mergesort")
    return predictions.reset_index(drop=True)


def make_oof_homology_predictions(
    hits: pd.DataFrame,
    train_terms: pd.DataFrame,
    folds: pd.DataFrame,
    max_hits_per_query: int | None = 100,
    min_hit_weight: float = 0.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create fold-safe OOF homology predictions."""

    filtered_hits = filter_oof_hits(hits, folds)
    predictions = aggregate_hit_labels(
        filtered_hits,
        train_terms=train_terms,
        max_hits_per_query=max_hits_per_query,
        min_hit_weight=min_hit_weight,
    )
    return filtered_hits, predictions


def make_test_homology_predictions(
    hits: pd.DataFrame,
    train_terms: pd.DataFrame,
    max_hits_per_query: int | None = 100,
    min_hit_weight: float = 0.0,
    exclude_self_hits: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create test homology predictions from train labels."""

    filtered_hits = filter_test_hits(hits, train_terms=train_terms, exclude_self_hits=exclude_self_hits)
    predictions = aggregate_hit_labels(
        filtered_hits,
        train_terms=train_terms,
        max_hits_per_query=max_hits_per_query,
        min_hit_weight=min_hit_weight,
    )
    return filtered_hits, predictions


def summarize_homology_outputs(
    oof_hits: pd.DataFrame,
    oof_predictions: pd.DataFrame,
    test_hits: pd.DataFrame,
    test_predictions: pd.DataFrame,
) -> dict[str, object]:
    """Create a compact homology transfer report."""

    def _prediction_counts(predictions: pd.DataFrame) -> dict[str, int]:
        if predictions.empty:
            return {}
        return {
            branch: int(count)
            for branch, count in predictions.groupby("branch")["term"].count().sort_index().items()
        }

    return {
        "oof_hit_rows": int(len(oof_hits)),
        "oof_query_count": int(oof_hits["query_id"].nunique()) if not oof_hits.empty else 0,
        "oof_prediction_rows": int(len(oof_predictions)),
        "oof_prediction_entries": int(oof_predictions["entry_id"].nunique()) if not oof_predictions.empty else 0,
        "oof_prediction_rows_by_branch": _prediction_counts(oof_predictions),
        "test_hit_rows": int(len(test_hits)),
        "test_query_count": int(test_hits["query_id"].nunique()) if not test_hits.empty else 0,
        "test_prediction_rows": int(len(test_predictions)),
        "test_prediction_entries": int(test_predictions["entry_id"].nunique()) if not test_predictions.empty else 0,
        "test_prediction_rows_by_branch": _prediction_counts(test_predictions),
        "score_range": {
            "oof_min": float(oof_predictions["score"].min()) if not oof_predictions.empty else None,
            "oof_max": float(oof_predictions["score"].max()) if not oof_predictions.empty else None,
            "test_min": float(test_predictions["score"].min()) if not test_predictions.empty else None,
            "test_max": float(test_predictions["score"].max()) if not test_predictions.empty else None,
        },
    }
