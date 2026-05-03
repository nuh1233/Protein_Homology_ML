"""Prepare deterministic ESM-2 embedding batches for Colab extraction."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cafa6.embeddings import DEFAULT_BATCH_SIZE, DEFAULT_MODEL_NAME, manifest_summary, prepare_embedding_batches
from cafa6.io import PROCESSED_DIR


EMBEDDINGS_DIR = PROJECT_ROOT / "artifacts" / "embeddings"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train-sequences",
        type=Path,
        default=PROCESSED_DIR / "train_sequences.parquet",
        help="Canonical train sequence table.",
    )
    parser.add_argument(
        "--test-sequences",
        type=Path,
        default=PROCESSED_DIR / "test_sequences.parquet",
        help="Canonical test sequence table.",
    )
    parser.add_argument("--output-root", type=Path, default=EMBEDDINGS_DIR, help="Embedding artifact root.")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Sequences per extraction shard.")
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME, help="fair-esm model loader name.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.train_sequences.is_file():
        raise FileNotFoundError(f"Missing train sequence table: {args.train_sequences}")
    if not args.test_sequences.is_file():
        raise FileNotFoundError(f"Missing test sequence table: {args.test_sequences}")

    train_manifest, train_batches = prepare_embedding_batches(
        sequence_path=args.train_sequences,
        output_dir=args.output_root / "esm2_train",
        split="train",
        batch_size=args.batch_size,
        model_name=args.model_name,
    )
    test_manifest, test_batches = prepare_embedding_batches(
        sequence_path=args.test_sequences,
        output_dir=args.output_root / "esm2_test",
        split="test",
        batch_size=args.batch_size,
        model_name=args.model_name,
    )

    print("CAFA 6 ESM-2 embedding batches prepared")
    print(f"- train_manifest: {args.output_root / 'esm2_train' / 'manifest.csv'}")
    print(f"- test_manifest: {args.output_root / 'esm2_test' / 'manifest.csv'}")
    print(f"- train_batches: {len(train_batches)}")
    print(f"- test_batches: {len(test_batches)}")
    print(f"- train_summary: {manifest_summary(train_manifest)}")
    print(f"- test_summary: {manifest_summary(test_manifest)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
