from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cafa6.embeddings import (
    EmbeddingBatchConfig,
    create_embedding_manifest,
    make_batch_ids,
    manifest_summary,
    mark_batch_complete,
    normalize_sequence_table,
    pending_batch_ids,
    prepare_embedding_batches,
    read_manifest,
    update_manifest_from_existing_shards,
    write_embedding_batches,
    write_manifest,
)


def _sequences() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"entry_id": "P2", "sequence": "MMMM"},
            {"entry_id": "P1", "sequence": "MA"},
            {"entry_id": "P1", "sequence": "DUPLICATE"},
        ]
    )


def test_normalize_sequence_table_sorts_and_deduplicates() -> None:
    normalized = normalize_sequence_table(_sequences())

    assert normalized["entry_id"].tolist() == ["P1", "P2"]
    assert normalized["sequence"].tolist() == ["MA", "MMMM"]
    assert normalized["sequence_length"].tolist() == [2, 4]
    assert normalized["row_index"].tolist() == [0, 1]


def test_normalize_sequence_table_requires_columns() -> None:
    with pytest.raises(ValueError, match="sequence"):
        normalize_sequence_table(pd.DataFrame([{"entry_id": "P1"}]))


def test_make_batch_ids_validates_size() -> None:
    assert make_batch_ids(5, batch_size=2) == ["batch_00000", "batch_00001", "batch_00002"]

    with pytest.raises(ValueError, match="positive"):
        make_batch_ids(5, batch_size=0)


def test_create_manifest_and_write_batches(tmp_path: Path) -> None:
    config = EmbeddingBatchConfig(split="train", output_dir=tmp_path / "esm2_train", batch_size=1)
    manifest = create_embedding_manifest(_sequences(), config)
    written = write_embedding_batches(_sequences(), manifest)

    assert manifest["entry_id"].tolist() == ["P1", "P2"]
    assert manifest["batch_id"].tolist() == ["batch_00000", "batch_00001"]
    assert len(written) == 2
    assert all(path.is_file() for path in written)

    first_batch = pd.read_parquet(written[0])
    assert first_batch.loc[0, "entry_id"] == "P1"
    assert first_batch.loc[0, "row_index_in_batch"] == 0


def test_manifest_read_write_pending_and_completion(tmp_path: Path) -> None:
    config = EmbeddingBatchConfig(split="test", output_dir=tmp_path / "esm2_test", batch_size=2)
    manifest = create_embedding_manifest(_sequences(), config)
    manifest_path = write_manifest(manifest, config.output_dir / "manifest.csv")

    loaded = read_manifest(manifest_path)
    assert pending_batch_ids(loaded) == ["batch_00000"]

    shard_path = Path(loaded["shard_path"].iloc[0])
    shard_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(shard_path, np.zeros((2, 3), dtype="float32"))
    refreshed = update_manifest_from_existing_shards(loaded)

    assert refreshed["status"].unique().tolist() == ["complete"]

    completed = mark_batch_complete(manifest_path, "batch_00000", completed_at="2026-04-26T00:00:00Z")
    assert completed["completed_at"].unique().tolist() == ["2026-04-26T00:00:00Z"]


def test_mark_batch_complete_rejects_unknown_batch(tmp_path: Path) -> None:
    config = EmbeddingBatchConfig(split="test", output_dir=tmp_path / "esm2_test", batch_size=2)
    manifest = create_embedding_manifest(_sequences(), config)
    manifest_path = write_manifest(manifest, config.output_dir / "manifest.csv")

    with pytest.raises(ValueError, match="Unknown batch_id"):
        mark_batch_complete(manifest_path, "bad_batch", completed_at="")


def test_prepare_embedding_batches_and_summary(tmp_path: Path) -> None:
    sequence_path = tmp_path / "sequences.parquet"
    _sequences().to_parquet(sequence_path, index=False)

    manifest, written = prepare_embedding_batches(
        sequence_path=sequence_path,
        output_dir=tmp_path / "esm2_train",
        split="train",
        batch_size=1,
        model_name="test_model",
    )
    summary = manifest_summary(manifest)

    assert (tmp_path / "esm2_train" / "manifest.csv").is_file()
    assert len(written) == 2
    assert summary["n_sequences"] == 2
    assert summary["n_batches"] == 2
    assert summary["models"] == ["test_model"]
