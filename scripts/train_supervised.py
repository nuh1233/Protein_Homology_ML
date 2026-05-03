"""Train branch-specific supervised heads on cached ESM-2 embeddings."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cafa6.features import (
    align_rows,
    build_label_matrix,
    filter_trainable_labels,
    load_embedding_matrix,
    select_branch_terms,
)
from cafa6.io import PROCESSED_DIR, REPORTS_DIR, write_json, write_parquet
from cafa6.metrics import score_branch_fmaxes
from cafa6.models import predict_branch_topk, train_branch_model


EMBEDDINGS_DIR = PROJECT_ROOT / "artifacts" / "embeddings"
MODELS_DIR = PROJECT_ROOT / "artifacts" / "models"
PREDICTIONS_DIR = PROJECT_ROOT / "artifacts" / "predictions"

DEFAULT_MAX_LABELS = {
    "MF": 1500,
    "BP": 2500,
    "CC": 1000,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-manifest", type=Path, default=EMBEDDINGS_DIR / "esm2_train" / "manifest.csv")
    parser.add_argument("--test-manifest", type=Path, default=EMBEDDINGS_DIR / "esm2_test" / "manifest.csv")
    parser.add_argument("--train-terms", type=Path, default=PROCESSED_DIR / "train_terms_closure.parquet")
    parser.add_argument("--folds", type=Path, default=PROCESSED_DIR / "folds_clustered.parquet")
    parser.add_argument("--branches", nargs="+", choices=["MF", "BP", "CC"], default=["MF", "BP", "CC"])
    parser.add_argument("--min-label-count", type=int, default=50)
    parser.add_argument("--mf-max-labels", type=int, default=DEFAULT_MAX_LABELS["MF"])
    parser.add_argument("--bp-max-labels", type=int, default=DEFAULT_MAX_LABELS["BP"])
    parser.add_argument("--cc-max-labels", type=int, default=DEFAULT_MAX_LABELS["CC"])
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--min-score", type=float, default=0.0)
    parser.add_argument("--predict-batch-size", type=int, default=2048)
    parser.add_argument("--alpha", type=float, default=1e-4)
    parser.add_argument("--max-iter", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--oof-output", type=Path, default=PREDICTIONS_DIR / "esm2_oof.parquet")
    parser.add_argument("--test-output", type=Path, default=PREDICTIONS_DIR / "esm2_test.parquet")
    parser.add_argument("--report", type=Path, default=REPORTS_DIR / "supervised_report.json")
    parser.add_argument("--score-output", type=Path, default=REPORTS_DIR / "cv_scores.csv")
    return parser.parse_args()


def _max_labels_by_branch(args: argparse.Namespace) -> dict[str, int]:
    return {
        "MF": args.mf_max_labels,
        "BP": args.bp_max_labels,
        "CC": args.cc_max_labels,
    }


def _save_branch_model(model, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_path)


def _concat_prediction_frames(frames: list[pd.DataFrame]) -> pd.DataFrame:
    if not frames:
        return pd.DataFrame(columns=["entry_id", "term", "branch", "score"])
    output = pd.concat(frames, ignore_index=True)
    output["score"] = output["score"].clip(0.0, 1.0)
    output = (
        output.groupby(["entry_id", "term", "branch"], as_index=False)
        .agg(score=("score", "max"))
        .sort_values(["entry_id", "branch", "score", "term"], ascending=[True, True, False, True], kind="mergesort")
        .reset_index(drop=True)
    )
    return output


def train_supervised(args: argparse.Namespace) -> dict[str, object]:
    for path in [args.train_manifest, args.test_manifest, args.train_terms, args.folds]:
        if not path.is_file():
            raise FileNotFoundError(f"Missing required input: {path}")

    train_embeddings = load_embedding_matrix(args.train_manifest)
    test_embeddings = load_embedding_matrix(args.test_manifest)
    train_terms = pd.read_parquet(args.train_terms)
    folds = pd.read_parquet(args.folds).loc[:, ["entry_id", "fold"]].copy()
    folds["entry_id"] = folds["entry_id"].astype(str)
    folds = folds.sort_values("entry_id", kind="mergesort").reset_index(drop=True)

    train_order = folds["entry_id"].tolist()
    train_indices = align_rows(train_embeddings.entry_ids, train_order)
    x_train = train_embeddings.matrix[train_indices]
    x_test = test_embeddings.matrix
    test_entry_ids = list(map(str, test_embeddings.entry_ids))

    max_labels_by_branch = _max_labels_by_branch(args)
    branch_reports: list[dict[str, object]] = []
    oof_frames: list[pd.DataFrame] = []
    test_frames: list[pd.DataFrame] = []

    for branch in args.branches:
        selected_terms = select_branch_terms(
            train_terms,
            branch=branch,
            min_count=args.min_label_count,
            max_labels=max_labels_by_branch[branch],
        )
        label_terms = selected_terms["term"].tolist()
        if not label_terms:
            raise ValueError(f"No labels selected for branch {branch}. Lower --min-label-count.")

        y_all = build_label_matrix(train_order, train_terms, label_terms, branch=branch)
        fold_reports: list[dict[str, object]] = []

        for fold in sorted(folds["fold"].unique()):
            valid_mask = folds["fold"].to_numpy() == fold
            train_mask = ~valid_mask
            y_fold, fold_label_terms, kept_indices = filter_trainable_labels(
                y_all[train_mask],
                label_terms,
                min_positive=1,
            )
            fold_model = train_branch_model(
                x_train[train_mask],
                y_fold,
                fold_label_terms,
                branch=branch,
                alpha=args.alpha,
                max_iter=args.max_iter,
                random_state=args.seed + int(fold),
                n_jobs=args.n_jobs,
            )
            fold_predictions = predict_branch_topk(
                fold_model,
                x_train[valid_mask],
                [entry_id for entry_id, is_valid in zip(train_order, valid_mask, strict=True) if is_valid],
                top_k=args.top_k,
                min_score=args.min_score,
                batch_size=args.predict_batch_size,
            )
            oof_frames.append(fold_predictions)
            fold_reports.append(
                {
                    "fold": int(fold),
                    "n_train": int(train_mask.sum()),
                    "n_valid": int(valid_mask.sum()),
                    "n_labels": int(len(fold_label_terms)),
                    "dropped_fold_labels": int(len(label_terms) - len(fold_label_terms)),
                    "oof_prediction_rows": int(len(fold_predictions)),
                }
            )

        y_final, final_label_terms, _ = filter_trainable_labels(y_all, label_terms, min_positive=1)
        final_model = train_branch_model(
            x_train,
            y_final,
            final_label_terms,
            branch=branch,
            alpha=args.alpha,
            max_iter=args.max_iter,
            random_state=args.seed,
            n_jobs=args.n_jobs,
        )
        model_path = MODELS_DIR / f"{branch.lower()}_model.joblib"
        _save_branch_model(final_model, model_path)

        branch_test = predict_branch_topk(
            final_model,
            x_test,
            test_entry_ids,
            top_k=args.top_k,
            min_score=args.min_score,
            batch_size=args.predict_batch_size,
        )
        test_frames.append(branch_test)

        branch_reports.append(
            {
                "branch": branch,
                "selected_labels": int(len(label_terms)),
                "final_trainable_labels": int(len(final_label_terms)),
                "model_path": str(model_path),
                "test_prediction_rows": int(len(branch_test)),
                "folds": fold_reports,
            }
        )

    oof_predictions = _concat_prediction_frames(oof_frames)
    test_predictions = _concat_prediction_frames(test_frames)
    write_parquet(oof_predictions, args.oof_output)
    write_parquet(test_predictions, args.test_output)

    scores = score_branch_fmaxes(
        truth=train_terms,
        predictions=oof_predictions,
        average="protein_macro",
    )
    args.score_output.parent.mkdir(parents=True, exist_ok=True)
    scores.to_csv(args.score_output, index=False)

    report = {
        "inputs": {
            "train_manifest": str(args.train_manifest),
            "test_manifest": str(args.test_manifest),
            "train_terms": str(args.train_terms),
            "folds": str(args.folds),
        },
        "outputs": {
            "esm2_oof": str(args.oof_output),
            "esm2_test": str(args.test_output),
            "cv_scores": str(args.score_output),
            "report": str(args.report),
        },
        "parameters": {
            "min_label_count": int(args.min_label_count),
            "branches": list(args.branches),
            "max_labels_by_branch": max_labels_by_branch,
            "top_k": int(args.top_k),
            "min_score": float(args.min_score),
            "alpha": float(args.alpha),
            "max_iter": int(args.max_iter),
            "seed": int(args.seed),
        },
        "embedding_shapes": {
            "train": [int(value) for value in x_train.shape],
            "test": [int(value) for value in x_test.shape],
        },
        "branches": branch_reports,
        "cv_scores": scores.to_dict("records"),
    }
    write_json(report, args.report)
    return report


def main() -> int:
    args = parse_args()
    report = train_supervised(args)

    print("CAFA 6 supervised ESM-2 heads trained")
    print(f"- oof: {report['outputs']['esm2_oof']}")
    print(f"- test: {report['outputs']['esm2_test']}")
    print(f"- scores: {report['outputs']['cv_scores']}")
    for row in report["cv_scores"]:
        print(f"- {row['branch']}: fmax={row['fmax']:.6f}, threshold={row['threshold']:.4f}")
    print(f"- mean_fmax: {report['cv_scores'][0]['mean_fmax']:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
