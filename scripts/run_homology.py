"""Prepare or aggregate CAFA 6 homology-transfer artifacts."""

from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cafa6.calibration import normalize_scores_by_branch
from cafa6.homology import (
    filter_oof_hits,
    filter_test_hits,
    limit_hits_per_query,
    prepare_training_labels,
    read_hit_table,
)
from cafa6.io import PROCESSED_DIR, RAW_DIR, REPORTS_DIR, write_json, write_parquet
from cafa6.ensemble import prune_top_k_by_group


RETRIEVAL_DIR = PROJECT_ROOT / "artifacts" / "retrieval"
DEFAULT_VALID_HITS = RETRIEVAL_DIR / "mmseqs_hits_valid.parquet"
DEFAULT_TEST_HITS = RETRIEVAL_DIR / "mmseqs_hits_test.parquet"
DEFAULT_OOF = RETRIEVAL_DIR / "homology_oof.parquet"
DEFAULT_TEST = RETRIEVAL_DIR / "homology_test.parquet"
DEFAULT_REPORT = REPORTS_DIR / "homology_report.json"
DEFAULT_PREPARE_REPORT = REPORTS_DIR / "homology_prepare_commands.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["aggregate", "prepare"], default="aggregate")
    parser.add_argument("--valid-hits", type=Path, default=DEFAULT_VALID_HITS, help="Train-vs-train hit table.")
    parser.add_argument("--test-hits", type=Path, default=DEFAULT_TEST_HITS, help="Test-vs-train hit table.")
    parser.add_argument(
        "--train-terms",
        type=Path,
        default=PROCESSED_DIR / "train_terms_closure.parquet",
        help="Ontology-closed training labels.",
    )
    parser.add_argument(
        "--folds",
        type=Path,
        default=PROCESSED_DIR / "folds_clustered.parquet",
        help="Fold assignments for OOF filtering.",
    )
    parser.add_argument("--valid-hits-output", type=Path, default=DEFAULT_VALID_HITS)
    parser.add_argument("--test-hits-output", type=Path, default=DEFAULT_TEST_HITS)
    parser.add_argument("--oof-output", type=Path, default=DEFAULT_OOF)
    parser.add_argument("--test-output", type=Path, default=DEFAULT_TEST)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--max-hits-per-query", type=int, default=100)
    parser.add_argument("--min-hit-weight", type=float, default=0.0)
    parser.add_argument(
        "--query-chunk-size",
        type=int,
        default=1000,
        help="Number of query proteins to aggregate at once. Lower this if RAM is tight.",
    )
    parser.add_argument(
        "--max-predictions-per-query-branch",
        type=int,
        default=100,
        help="Keep only top predictions per query and branch inside each aggregation chunk.",
    )
    parser.add_argument(
        "--normalize-branch-max",
        action="store_true",
        help="Scale each branch's prediction scores by its max score after aggregation.",
    )
    parser.add_argument(
        "--allow-test-self-hits",
        action="store_true",
        help="Allow test query IDs to transfer from same-ID train targets.",
    )
    parser.add_argument("--prepare-report", type=Path, default=DEFAULT_PREPARE_REPORT)
    return parser.parse_args()


def _empty_homology_prediction_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=["entry_id", "term", "branch", "score", "n_hits", "best_hit_weight"])


def _aggregate_weighted_hit_labels(
    hits: pd.DataFrame,
    labels: pd.DataFrame,
    max_hits_per_query: int | None = 100,
    min_hit_weight: float = 0.0,
) -> pd.DataFrame:
    weighted_hits = hits.copy()
    weighted_hits["hit_weight"] = pd.to_numeric(weighted_hits["hit_weight"], errors="coerce").fillna(0.0).clip(0.0, 1.0)
    weighted_hits = weighted_hits.loc[weighted_hits["hit_weight"] > min_hit_weight]
    weighted_hits = limit_hits_per_query(weighted_hits, max_hits_per_query=max_hits_per_query)
    if weighted_hits.empty:
        return _empty_homology_prediction_frame()

    evidence = weighted_hits.merge(labels, left_on="target_id", right_on="entry_id", how="inner", suffixes=("_query", "_target"))
    if evidence.empty:
        return _empty_homology_prediction_frame()

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


def _aggregate_homology_chunked(
    hits: pd.DataFrame,
    labels: pd.DataFrame,
    max_hits_per_query: int | None,
    min_hit_weight: float,
    query_chunk_size: int,
    max_predictions_per_query_branch: int | None,
    label: str,
) -> pd.DataFrame:
    if query_chunk_size <= 0:
        raise ValueError("query_chunk_size must be positive.")
    if hits.empty:
        return _empty_homology_prediction_frame()

    entry_codes, _ = pd.factorize(hits["query_id"], sort=False)
    chunk_ids = entry_codes // query_chunk_size
    n_chunks = int(chunk_ids.max() + 1) if len(chunk_ids) else 0
    print(
        f"Aggregating {label}: {hits['query_id'].nunique()} queries, {len(hits)} hits, {n_chunks} chunks",
        flush=True,
    )

    frames: list[pd.DataFrame] = []
    for chunk_index, (_, chunk_hits) in enumerate(hits.groupby(chunk_ids, sort=False), start=1):
        print(f"- {label} chunk {chunk_index}/{n_chunks}: {len(chunk_hits)} hits", flush=True)
        chunk_predictions = _aggregate_weighted_hit_labels(
            chunk_hits,
            labels=labels,
            max_hits_per_query=max_hits_per_query,
            min_hit_weight=min_hit_weight,
        )
        if not chunk_predictions.empty:
            chunk_predictions = prune_top_k_by_group(
                chunk_predictions,
                top_k=max_predictions_per_query_branch,
                group_columns=("entry_id", "branch"),
            )
            frames.append(chunk_predictions)

    if not frames:
        return _empty_homology_prediction_frame()
    return pd.concat(frames, ignore_index=True).reset_index(drop=True)


def _prediction_counts(predictions: pd.DataFrame) -> dict[str, int]:
    if predictions.empty:
        return {}
    return {branch: int(count) for branch, count in predictions.groupby("branch")["term"].count().sort_index().items()}


def _score_range(predictions: pd.DataFrame) -> tuple[float | None, float | None]:
    if predictions.empty:
        return None, None
    return float(predictions["score"].min()), float(predictions["score"].max())


def _validate_aggregate_inputs(args: argparse.Namespace) -> list[str]:
    required_paths = {
        "valid_hits": args.valid_hits,
        "test_hits": args.test_hits,
        "train_terms": args.train_terms,
        "folds": args.folds,
    }
    return [f"{name}: {path}" for name, path in required_paths.items() if not path.is_file()]


def _prepare_report() -> dict[str, object]:
    train_fasta = RAW_DIR / "train_sequences.fasta"
    test_fasta = RAW_DIR / "testsuperset.fasta"
    return {
        "purpose": "Prepare external MMseqs/DIAMOND homology search inputs. Commands are informational and not executed.",
        "expected_hit_columns": {
            "required": ["query_id", "target_id"],
            "recommended": ["bitscore", "evalue", "pident", "qcov", "tcov"],
            "aliases_supported": {
                "query_id": ["query", "qseqid", "qid", "query_entry_id"],
                "target_id": ["target", "sseqid", "subject_id", "sid", "target_entry_id"],
            },
        },
        "raw_fastas": {
            "train": str(train_fasta),
            "test": str(test_fasta),
        },
        "suggested_mmseqs_outputs": {
            "valid_hits": str(DEFAULT_VALID_HITS),
            "test_hits": str(DEFAULT_TEST_HITS),
        },
        "suggested_commands": [
            "mmseqs easy-search data/raw/train_sequences.fasta data/raw/train_sequences.fasta artifacts/retrieval/mmseqs_train_vs_train.tsv tmp_mmseqs --format-output query,target,pident,evalue,bits,qcov,tcov",
            "mmseqs easy-search data/raw/testsuperset.fasta data/raw/train_sequences.fasta artifacts/retrieval/mmseqs_test_vs_train.tsv tmp_mmseqs --format-output query,target,pident,evalue,bits,qcov,tcov",
            "python scripts/run_homology.py --mode aggregate --valid-hits artifacts/retrieval/mmseqs_train_vs_train.tsv --test-hits artifacts/retrieval/mmseqs_test_vs_train.tsv",
        ],
    }


def run_prepare(args: argparse.Namespace) -> int:
    report = _prepare_report()
    write_json(report, args.prepare_report)
    print("CAFA 6 homology preparation report written")
    print(f"- report: {args.prepare_report}")
    for command in report["suggested_commands"]:
        print(f"- {command}")
    return 0


def run_aggregate(args: argparse.Namespace) -> int:
    missing = _validate_aggregate_inputs(args)
    if missing:
        print("Missing required homology inputs:")
        for item in missing:
            print(f"- {item}")
        print("Run --mode prepare for suggested external search commands.")
        return 1

    train_terms = pd.read_parquet(args.train_terms)
    train_labels = prepare_training_labels(train_terms)
    folds = pd.read_parquet(args.folds)
    valid_hits_raw = read_hit_table(args.valid_hits)

    oof_hits = filter_oof_hits(valid_hits_raw, folds)
    del valid_hits_raw
    gc.collect()

    oof_predictions = _aggregate_homology_chunked(
        oof_hits,
        labels=train_labels,
        max_hits_per_query=args.max_hits_per_query,
        min_hit_weight=args.min_hit_weight,
        query_chunk_size=args.query_chunk_size,
        max_predictions_per_query_branch=args.max_predictions_per_query_branch,
        label="oof homology",
    )

    if args.normalize_branch_max:
        oof_predictions = normalize_scores_by_branch(oof_predictions, method="max")

    write_parquet(oof_hits, args.valid_hits_output)
    write_parquet(oof_predictions, args.oof_output)
    oof_min, oof_max = _score_range(oof_predictions)
    oof_summary = {
        "oof_hit_rows": int(len(oof_hits)),
        "oof_query_count": int(oof_hits["query_id"].nunique()) if not oof_hits.empty else 0,
        "oof_prediction_rows": int(len(oof_predictions)),
        "oof_prediction_entries": int(oof_predictions["entry_id"].nunique()) if not oof_predictions.empty else 0,
        "oof_prediction_rows_by_branch": _prediction_counts(oof_predictions),
        "oof_min": oof_min,
        "oof_max": oof_max,
    }
    del oof_hits, oof_predictions
    gc.collect()

    test_hits_raw = read_hit_table(args.test_hits)
    test_hits = filter_test_hits(test_hits_raw, train_terms=train_terms, exclude_self_hits=not args.allow_test_self_hits)
    del test_hits_raw
    gc.collect()

    test_predictions = _aggregate_homology_chunked(
        test_hits,
        labels=train_labels,
        max_hits_per_query=args.max_hits_per_query,
        min_hit_weight=args.min_hit_weight,
        query_chunk_size=args.query_chunk_size,
        max_predictions_per_query_branch=args.max_predictions_per_query_branch,
        label="test homology",
    )

    if args.normalize_branch_max:
        test_predictions = normalize_scores_by_branch(test_predictions, method="max")

    write_parquet(test_hits, args.test_hits_output)
    write_parquet(test_predictions, args.test_output)
    test_min, test_max = _score_range(test_predictions)
    test_summary = {
        "test_hit_rows": int(len(test_hits)),
        "test_query_count": int(test_hits["query_id"].nunique()) if not test_hits.empty else 0,
        "test_prediction_rows": int(len(test_predictions)),
        "test_prediction_entries": int(test_predictions["entry_id"].nunique()) if not test_predictions.empty else 0,
        "test_prediction_rows_by_branch": _prediction_counts(test_predictions),
        "test_min": test_min,
        "test_max": test_max,
    }

    report = {
        "inputs": {
            "valid_hits": str(args.valid_hits),
            "test_hits": str(args.test_hits),
            "train_terms": str(args.train_terms),
            "folds": str(args.folds),
        },
        "outputs": {
            "valid_hits": str(args.valid_hits_output),
            "test_hits": str(args.test_hits_output),
            "homology_oof": str(args.oof_output),
            "homology_test": str(args.test_output),
        },
        "parameters": {
            "max_hits_per_query": args.max_hits_per_query,
            "min_hit_weight": args.min_hit_weight,
            "query_chunk_size": args.query_chunk_size,
            "max_predictions_per_query_branch": args.max_predictions_per_query_branch,
            "normalize_branch_max": bool(args.normalize_branch_max),
            "exclude_test_self_hits": not args.allow_test_self_hits,
        },
        "summary": {
            **oof_summary,
            **test_summary,
            "score_range": {
                "oof_min": oof_summary["oof_min"],
                "oof_max": oof_summary["oof_max"],
                "test_min": test_summary["test_min"],
                "test_max": test_summary["test_max"],
            },
        },
    }
    write_json(report, args.report)

    print("CAFA 6 homology transfer built")
    for name, path in report["outputs"].items():
        print(f"- {name}: {path}")
    print(f"- report: {args.report}")
    for name, value in report["summary"].items():
        print(f"- {name}: {value}")
    return 0


def main() -> int:
    args = parse_args()
    if args.mode == "prepare":
        return run_prepare(args)
    return run_aggregate(args)


if __name__ == "__main__":
    raise SystemExit(main())
