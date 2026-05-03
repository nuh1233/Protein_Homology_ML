"""Blend homology transfer with supervised ESM predictions."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cafa6.calibration import clip_scores
from cafa6.ensemble import prune_top_k_by_group
from cafa6.io import PROCESSED_DIR, REPORTS_DIR, write_json, write_parquet
from cafa6.metrics import score_branch_fmaxes


PREDICTIONS_DIR = PROJECT_ROOT / "artifacts" / "predictions"
RETRIEVAL_DIR = PROJECT_ROOT / "artifacts" / "retrieval"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--homology-oof", type=Path, default=RETRIEVAL_DIR / "homology_oof.parquet")
    parser.add_argument("--homology-test", type=Path, default=RETRIEVAL_DIR / "homology_test.parquet")
    parser.add_argument("--supervised-oof", type=Path, default=PREDICTIONS_DIR / "esm2_oof_repaired.parquet")
    parser.add_argument("--supervised-test", type=Path, default=PREDICTIONS_DIR / "esm2_test_repaired.parquet")
    parser.add_argument("--truth", type=Path, default=PROCESSED_DIR / "train_terms_closure.parquet")
    parser.add_argument("--homology-scale", type=float, default=1.0)
    parser.add_argument("--supervised-scale", type=float, default=0.05)
    parser.add_argument("--supervised-top-k-per-branch", type=int, default=25)
    parser.add_argument("--test-entry-chunk-size", type=int, default=5000)
    parser.add_argument("--skip-oof", action="store_true", help="Reuse existing OOF blend and score files; build test only.")
    parser.add_argument("--output-oof", type=Path, default=PREDICTIONS_DIR / "ensemble_oof.parquet")
    parser.add_argument("--output-test", type=Path, default=PREDICTIONS_DIR / "ensemble_test.parquet")
    parser.add_argument("--score-output", type=Path, default=REPORTS_DIR / "ensemble_cv_scores.csv")
    parser.add_argument("--report", type=Path, default=REPORTS_DIR / "ensemble_report.json")
    return parser.parse_args()


def _check_inputs(paths: list[Path]) -> None:
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"Missing required input: {path}")


def _prepare_source(frame: pd.DataFrame, scale: float, top_k_per_branch: int | None = None) -> pd.DataFrame:
    required = {"entry_id", "term", "branch", "score"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"prediction source is missing columns: {', '.join(sorted(missing))}")

    output = frame.loc[:, ["entry_id", "term", "branch", "score"]].copy()
    output = clip_scores(output)
    if top_k_per_branch is not None and top_k_per_branch > 0:
        output = prune_top_k_by_group(output, top_k=top_k_per_branch, group_columns=("entry_id", "branch"))
    output["score"] = (output["score"] * float(scale)).clip(0.0, 1.0)
    return output


def _collapse_max_scores(frame: pd.DataFrame, entry_chunk_size: int | None = None) -> pd.DataFrame:
    if entry_chunk_size is None or entry_chunk_size <= 0:
        return (
            frame.groupby(["entry_id", "term", "branch"], as_index=False)
            .agg(score=("score", "max"))
            .sort_values(["entry_id", "branch", "score", "term"], ascending=[True, True, False, True], kind="mergesort")
            .reset_index(drop=True)
        )

    entry_codes, _ = pd.factorize(frame["entry_id"], sort=False)
    chunk_ids = entry_codes // entry_chunk_size
    n_chunks = int(chunk_ids.max() + 1) if len(chunk_ids) else 0
    frames: list[pd.DataFrame] = []
    for chunk_index in range(n_chunks):
        chunk = frame.loc[chunk_ids == chunk_index]
        print(f"- blending test chunk {chunk_index + 1}/{n_chunks}: {len(chunk)} rows", flush=True)
        collapsed = (
            chunk.groupby(["entry_id", "term", "branch"], as_index=False)
            .agg(score=("score", "max"))
            .sort_values(["entry_id", "branch", "score", "term"], ascending=[True, True, False, True], kind="mergesort")
            .reset_index(drop=True)
        )
        frames.append(collapsed)
    if not frames:
        return pd.DataFrame(columns=["entry_id", "term", "branch", "score"])
    return pd.concat(frames, ignore_index=True).reset_index(drop=True)


def _collapse_max_scores_to_parquet(frame: pd.DataFrame, output_path: Path, entry_chunk_size: int) -> int:
    if entry_chunk_size <= 0:
        raise ValueError("entry_chunk_size must be positive.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    entry_codes, _ = pd.factorize(frame["entry_id"], sort=False)
    chunk_ids = entry_codes // entry_chunk_size
    n_chunks = int(chunk_ids.max() + 1) if len(chunk_ids) else 0
    writer: pq.ParquetWriter | None = None
    total_rows = 0

    try:
        for chunk_index in range(n_chunks):
            chunk = frame.loc[chunk_ids == chunk_index]
            print(f"- blending test chunk {chunk_index + 1}/{n_chunks}: {len(chunk)} rows", flush=True)
            collapsed = (
                chunk.groupby(["entry_id", "term", "branch"], as_index=False)
                .agg(score=("score", "max"))
                .sort_values(["entry_id", "branch", "score", "term"], ascending=[True, True, False, True], kind="mergesort")
                .reset_index(drop=True)
            )
            table = pa.Table.from_pandas(collapsed, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(output_path, table.schema)
            writer.write_table(table)
            total_rows += len(collapsed)
            del chunk, collapsed, table
    finally:
        if writer is not None:
            writer.close()

    return total_rows


def _max_blend(
    homology: pd.DataFrame,
    supervised: pd.DataFrame,
    args: argparse.Namespace,
    entry_chunk_size: int | None = None,
) -> pd.DataFrame:
    homology_prepared = _prepare_source(homology, scale=args.homology_scale)
    supervised_prepared = _prepare_source(
        supervised,
        scale=args.supervised_scale,
        top_k_per_branch=args.supervised_top_k_per_branch,
    )
    output = pd.concat([homology_prepared, supervised_prepared], ignore_index=True)
    return _collapse_max_scores(output, entry_chunk_size=entry_chunk_size)


def _max_blend_to_parquet(
    homology: pd.DataFrame,
    supervised: pd.DataFrame,
    args: argparse.Namespace,
    output_path: Path,
    entry_chunk_size: int,
) -> int:
    homology_prepared = _prepare_source(homology, scale=args.homology_scale)
    supervised_prepared = _prepare_source(
        supervised,
        scale=args.supervised_scale,
        top_k_per_branch=args.supervised_top_k_per_branch,
    )
    output = pd.concat([homology_prepared, supervised_prepared], ignore_index=True)
    return _collapse_max_scores_to_parquet(output, output_path=output_path, entry_chunk_size=entry_chunk_size)


def blend_predictions(args: argparse.Namespace) -> dict[str, object]:
    _check_inputs([args.homology_test, args.supervised_test])

    if args.skip_oof:
        _check_inputs([args.output_oof, args.score_output])
        scores = pd.read_csv(args.score_output)
    else:
        _check_inputs([args.homology_oof, args.supervised_oof, args.truth])
        homology_oof = pd.read_parquet(args.homology_oof)
        supervised_oof = pd.read_parquet(args.supervised_oof)
        ensemble_oof = _max_blend(homology_oof, supervised_oof, args)
        write_parquet(ensemble_oof, args.output_oof)

        truth = pd.read_parquet(args.truth)
        scores = score_branch_fmaxes(truth=truth, predictions=ensemble_oof, average="protein_macro")
        args.score_output.parent.mkdir(parents=True, exist_ok=True)
        scores.to_csv(args.score_output, index=False)

        del homology_oof, supervised_oof, ensemble_oof, truth

    homology_test = pd.read_parquet(args.homology_test)
    supervised_test = pd.read_parquet(args.supervised_test)
    homology_test_rows = int(len(homology_test))
    supervised_test_rows = int(len(supervised_test))
    ensemble_test_rows = _max_blend_to_parquet(
        homology_test,
        supervised_test,
        args,
        output_path=args.output_test,
        entry_chunk_size=args.test_entry_chunk_size,
    )

    report = {
        "inputs": {
            "homology_oof": str(args.homology_oof),
            "homology_test": str(args.homology_test),
            "supervised_oof": str(args.supervised_oof),
            "supervised_test": str(args.supervised_test),
            "truth": str(args.truth),
        },
        "outputs": {
            "ensemble_oof": str(args.output_oof),
            "ensemble_test": str(args.output_test),
            "cv_scores": str(args.score_output),
            "report": str(args.report),
        },
        "parameters": {
            "homology_scale": float(args.homology_scale),
            "supervised_scale": float(args.supervised_scale),
            "supervised_top_k_per_branch": int(args.supervised_top_k_per_branch),
            "test_entry_chunk_size": int(args.test_entry_chunk_size),
            "skip_oof": bool(args.skip_oof),
        },
        "row_counts": {
            "homology_test": homology_test_rows,
            "supervised_test": supervised_test_rows,
            "ensemble_test": ensemble_test_rows,
        },
        "cv_scores": scores.to_dict("records"),
    }
    write_json(report, args.report)
    return report


def main() -> int:
    args = parse_args()
    report = blend_predictions(args)

    print("CAFA 6 ensemble predictions built")
    print(f"- oof: {report['outputs']['ensemble_oof']}")
    print(f"- test: {report['outputs']['ensemble_test']}")
    print(f"- scores: {report['outputs']['cv_scores']}")
    for row in report["cv_scores"]:
        print(f"- {row['branch']}: fmax={row['fmax']:.6f}, threshold={row['threshold']:.4f}")
    print(f"- mean_fmax: {report['cv_scores'][0]['mean_fmax']:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
