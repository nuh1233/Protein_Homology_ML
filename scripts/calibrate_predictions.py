"""Calibrate supervised predictions and repair GO ancestor consistency."""

from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cafa6.calibration import apply_branch_calibration, fit_branch_calibration
from cafa6.ensemble import prune_top_k_by_group, repair_go_hierarchy
from cafa6.io import PROCESSED_DIR, REPORTS_DIR, write_json, write_parquet
from cafa6.metrics import score_branch_fmaxes


PREDICTIONS_DIR = PROJECT_ROOT / "artifacts" / "predictions"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oof-predictions", type=Path, default=PREDICTIONS_DIR / "esm2_oof.parquet")
    parser.add_argument("--test-predictions", type=Path, default=PREDICTIONS_DIR / "esm2_test.parquet")
    parser.add_argument("--truth", type=Path, default=PROCESSED_DIR / "train_terms_closure.parquet")
    parser.add_argument("--go-terms", type=Path, default=PROCESSED_DIR / "go_terms.parquet")
    parser.add_argument("--go-ancestors", type=Path, default=PROCESSED_DIR / "go_ancestors.parquet")
    parser.add_argument("--method", choices=["none", "max"], default="max")
    parser.add_argument("--top-k-per-branch", type=int, default=100)
    parser.add_argument(
        "--score-repaired-oof",
        action="store_true",
        help="Compute Fmax on repaired OOF predictions. This is expensive for full top-k artifacts.",
    )
    parser.add_argument(
        "--entry-chunk-size",
        type=int,
        default=5000,
        help="Number of proteins to GO-repair at once per branch. Lower this if RAM is tight.",
    )
    parser.add_argument("--oof-output", type=Path, default=PREDICTIONS_DIR / "esm2_oof_repaired.parquet")
    parser.add_argument("--test-output", type=Path, default=PREDICTIONS_DIR / "esm2_test_repaired.parquet")
    parser.add_argument("--report", type=Path, default=REPORTS_DIR / "calibration_report.json")
    return parser.parse_args()


def _check_inputs(paths: list[Path]) -> None:
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"Missing required input: {path}")


def _repair_by_branch(
    predictions: pd.DataFrame,
    ancestors: pd.DataFrame,
    terms: pd.DataFrame,
    top_k_per_branch: int,
    entry_chunk_size: int = 5000,
    label: str = "predictions",
) -> pd.DataFrame:
    if entry_chunk_size <= 0:
        raise ValueError("entry_chunk_size must be positive.")

    frames: list[pd.DataFrame] = []
    for branch in sorted(predictions["branch"].dropna().unique()):
        branch_predictions = predictions.loc[predictions["branch"] == branch].copy()
        if branch_predictions.empty:
            continue

        branch_terms = terms.loc[terms["branch"] == branch, ["term", "branch"]].copy()
        branch_ancestors = ancestors.loc[
            (ancestors["term_branch"] == branch) & (ancestors["ancestor_branch"] == branch)
        ].copy()

        entry_codes, _ = pd.factorize(branch_predictions["entry_id"], sort=False)
        chunk_ids = entry_codes // entry_chunk_size
        n_chunks = int(chunk_ids.max() + 1) if len(chunk_ids) else 0
        print(
            f"Repairing {label} branch {branch}: {branch_predictions['entry_id'].nunique()} entries, "
            f"{len(branch_predictions)} rows, {n_chunks} chunks",
            flush=True,
        )
        for _, chunk_predictions in branch_predictions.groupby(chunk_ids, sort=False):
            repaired = repair_go_hierarchy(chunk_predictions, branch_ancestors, terms=branch_terms)
            repaired = prune_top_k_by_group(repaired, top_k=top_k_per_branch, group_columns=("entry_id", "branch"))
            frames.append(repaired)

    if not frames:
        return pd.DataFrame(columns=["entry_id", "term", "branch", "score"])
    output = pd.concat(frames, ignore_index=True)
    return output.sort_values(
        ["entry_id", "branch", "score", "term"],
        ascending=[True, True, False, True],
        kind="mergesort",
    ).reset_index(drop=True)


def calibrate_predictions(args: argparse.Namespace) -> dict[str, object]:
    _check_inputs([args.oof_predictions, args.test_predictions, args.truth, args.go_terms, args.go_ancestors])

    truth = pd.read_parquet(args.truth)
    terms = pd.read_parquet(args.go_terms)
    ancestors = pd.read_parquet(args.go_ancestors)

    oof_predictions = pd.read_parquet(args.oof_predictions)
    oof_input_rows = int(len(oof_predictions))
    calibration = fit_branch_calibration(oof_predictions, truth=truth, method=args.method, average="protein_macro")
    raw_score_columns = [
        "branch",
        "fmax",
        "threshold",
        "precision",
        "recall",
        "n_truth_entries",
        "n_truth_terms",
        "n_prediction_rows",
        "mean_fmax",
        "average",
    ]
    raw_scores = calibration.loc[:, raw_score_columns].copy()
    calibrated_oof = apply_branch_calibration(oof_predictions, calibration)
    repaired_oof = _repair_by_branch(
        calibrated_oof,
        ancestors,
        terms,
        top_k_per_branch=args.top_k_per_branch,
        entry_chunk_size=args.entry_chunk_size,
        label="oof",
    )
    oof_repaired_rows = int(len(repaired_oof))
    write_parquet(repaired_oof, args.oof_output)
    if args.score_repaired_oof:
        repaired_scores = score_branch_fmaxes(truth=truth, predictions=repaired_oof, average="protein_macro")
    else:
        repaired_scores = pd.DataFrame()

    del oof_predictions, calibrated_oof, repaired_oof
    gc.collect()

    test_predictions = pd.read_parquet(args.test_predictions)
    test_input_rows = int(len(test_predictions))
    calibrated_test = apply_branch_calibration(test_predictions, calibration)
    repaired_test = _repair_by_branch(
        calibrated_test,
        ancestors,
        terms,
        top_k_per_branch=args.top_k_per_branch,
        entry_chunk_size=args.entry_chunk_size,
        label="test",
    )
    test_repaired_rows = int(len(repaired_test))
    write_parquet(repaired_test, args.test_output)

    report = {
        "inputs": {
            "oof_predictions": str(args.oof_predictions),
            "test_predictions": str(args.test_predictions),
            "truth": str(args.truth),
            "go_terms": str(args.go_terms),
            "go_ancestors": str(args.go_ancestors),
        },
        "outputs": {
            "oof_repaired": str(args.oof_output),
            "test_repaired": str(args.test_output),
            "report": str(args.report),
        },
        "parameters": {
            "method": args.method,
            "top_k_per_branch": int(args.top_k_per_branch),
            "entry_chunk_size": int(args.entry_chunk_size),
            "score_repaired_oof": bool(args.score_repaired_oof),
        },
        "raw_scores": raw_scores.to_dict("records"),
        "calibration": calibration.to_dict("records"),
        "repaired_scores": repaired_scores.to_dict("records"),
        "row_counts": {
            "oof_input": oof_input_rows,
            "test_input": test_input_rows,
            "oof_repaired": oof_repaired_rows,
            "test_repaired": test_repaired_rows,
        },
    }
    write_json(report, args.report)
    return report


def main() -> int:
    args = parse_args()
    report = calibrate_predictions(args)

    print("CAFA 6 predictions calibrated and GO-repaired")
    print(f"- oof_repaired: {report['outputs']['oof_repaired']}")
    print(f"- test_repaired: {report['outputs']['test_repaired']}")
    print(f"- report: {report['outputs']['report']}")
    for row in report["repaired_scores"]:
        print(f"- {row['branch']}: fmax={row['fmax']:.6f}, threshold={row['threshold']:.4f}")
    if report["repaired_scores"]:
        print(f"- mean_fmax: {report['repaired_scores'][0]['mean_fmax']:.6f}")
    else:
        print("- repaired OOF scoring skipped; rerun with --score-repaired-oof if needed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
