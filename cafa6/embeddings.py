"""Colab-compatible ESM-2 embedding extraction utilities."""

from __future__ import annotations

from datetime import datetime, timezone
from dataclasses import asdict, dataclass
from pathlib import Path, PureWindowsPath
from typing import Iterable

import numpy as np
import pandas as pd


DEFAULT_MODEL_NAME = "esm2_t33_650M_UR50D"
DEFAULT_BATCH_SIZE = 512
DEFAULT_INFERENCE_BATCH_SIZE = 8
DEFAULT_MAX_SEQUENCE_LENGTH = 1022
MANIFEST_COLUMNS: tuple[str, ...] = (
    "entry_id",
    "split",
    "batch_id",
    "row_index",
    "row_index_in_batch",
    "sequence_length",
    "batch_path",
    "shard_path",
    "model_name",
    "status",
    "completed_at",
)


@dataclass(frozen=True)
class EmbeddingBatchConfig:
    """Configuration for deterministic embedding batch preparation."""

    split: str
    output_dir: Path
    batch_size: int = DEFAULT_BATCH_SIZE
    model_name: str = DEFAULT_MODEL_NAME


def normalize_sequence_table(sequences: pd.DataFrame) -> pd.DataFrame:
    """Normalize sequence tables to entry_id, sequence, and length columns."""

    required = {"entry_id", "sequence"}
    missing = required.difference(sequences.columns)
    if missing:
        raise ValueError(f"sequences are missing required columns: {', '.join(sorted(missing))}")

    output = sequences.loc[:, ["entry_id", "sequence"]].copy()
    output["entry_id"] = output["entry_id"].astype(str)
    output["sequence"] = output["sequence"].astype(str)
    output = output.drop_duplicates("entry_id", keep="first")
    output["sequence_length"] = output["sequence"].str.len().astype(int)
    output = output.sort_values("entry_id", kind="mergesort").reset_index(drop=True)
    output["row_index"] = np.arange(len(output), dtype=int)
    return output


def make_batch_ids(n_rows: int, batch_size: int) -> list[str]:
    """Create deterministic batch IDs for a row count."""

    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")

    n_batches = int(np.ceil(n_rows / batch_size)) if n_rows else 0
    return [f"batch_{index:05d}" for index in range(n_batches)]


def create_embedding_manifest(
    sequences: pd.DataFrame,
    config: EmbeddingBatchConfig,
) -> pd.DataFrame:
    """Create a row-per-sequence embedding manifest."""

    normalized = normalize_sequence_table(sequences)
    batch_ids = make_batch_ids(len(normalized), config.batch_size)

    rows: list[dict[str, object]] = []
    for batch_number, batch_id in enumerate(batch_ids):
        start = batch_number * config.batch_size
        stop = min(start + config.batch_size, len(normalized))
        batch_path = config.output_dir / "batches" / f"{batch_id}.parquet"
        shard_path = config.output_dir / f"shard_{batch_number:05d}.npy"

        batch_rows = normalized.iloc[start:stop]
        for row_index_in_batch, row in enumerate(batch_rows.itertuples(index=False)):
            rows.append(
                {
                    "entry_id": row.entry_id,
                    "split": config.split,
                    "batch_id": batch_id,
                    "row_index": int(row.row_index),
                    "row_index_in_batch": int(row_index_in_batch),
                    "sequence_length": int(row.sequence_length),
                    "batch_path": str(batch_path),
                    "shard_path": str(shard_path),
                    "model_name": config.model_name,
                    "status": "pending",
                    "completed_at": "",
                }
            )

    return pd.DataFrame.from_records(rows, columns=MANIFEST_COLUMNS)


def write_embedding_batches(
    sequences: pd.DataFrame,
    manifest: pd.DataFrame,
) -> list[Path]:
    """Write deterministic batch parquet files referenced by a manifest."""

    normalized = normalize_sequence_table(sequences)
    written: list[Path] = []

    for batch_id, batch_manifest in manifest.groupby("batch_id", sort=True):
        batch_path = Path(batch_manifest["batch_path"].iloc[0])
        batch_path.parent.mkdir(parents=True, exist_ok=True)

        row_indices = batch_manifest["row_index"].astype(int).tolist()
        batch = normalized.loc[normalized["row_index"].isin(row_indices), ["entry_id", "sequence", "sequence_length", "row_index"]]
        batch = batch.sort_values("row_index", kind="mergesort").reset_index(drop=True)
        batch["batch_id"] = str(batch_id)
        batch["row_index_in_batch"] = np.arange(len(batch), dtype=int)
        batch.to_parquet(batch_path, index=False)
        written.append(batch_path)

    return written


def update_manifest_from_existing_shards(manifest: pd.DataFrame) -> pd.DataFrame:
    """Mark manifest rows complete when their shard files already exist."""

    output = manifest.copy()
    shard_exists = output["shard_path"].map(lambda value: Path(value).is_file())
    output.loc[shard_exists, "status"] = "complete"
    return output


def write_manifest(manifest: pd.DataFrame, output_path: str | Path) -> Path:
    """Write a manifest CSV with deterministic column order."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest.loc[:, MANIFEST_COLUMNS].to_csv(path, index=False)
    return path


def _rebase_manifest_artifact_path(value: str, manifest_path: Path) -> str:
    """Rebase stale absolute artifact paths to the current manifest directory."""

    if not value:
        return value

    current_path = Path(value)
    if current_path.exists():
        return value

    normalized = value.replace("\\", "/")
    split_marker = "/artifacts/embeddings/"
    if split_marker in normalized:
        relative_to_embeddings = normalized.split(split_marker, 1)[1]
        embeddings_root = manifest_path.resolve().parents[1]
        return str(embeddings_root / relative_to_embeddings)

    if ":" in value[:3]:
        windows_path = PureWindowsPath(value)
        parts = list(windows_path.parts)
        if "artifacts" in parts and "embeddings" in parts:
            marker_index = parts.index("embeddings")
            relative_to_embeddings = Path(*parts[marker_index + 1 :])
            embeddings_root = manifest_path.resolve().parents[1]
            return str(embeddings_root / relative_to_embeddings)

    return value


def read_manifest(path: str | Path) -> pd.DataFrame:
    """Read an embedding manifest CSV."""

    manifest_path = Path(path)
    manifest = pd.read_csv(manifest_path, dtype=str).fillna("")
    for column in ["row_index", "row_index_in_batch", "sequence_length"]:
        if column in manifest.columns:
            manifest[column] = manifest[column].astype(int)
    missing = set(MANIFEST_COLUMNS).difference(manifest.columns)
    if missing:
        raise ValueError(f"manifest is missing required columns: {', '.join(sorted(missing))}")
    for path_column in ["batch_path", "shard_path"]:
        manifest[path_column] = manifest[path_column].map(
            lambda value: _rebase_manifest_artifact_path(value, manifest_path)
        )
    return manifest.loc[:, MANIFEST_COLUMNS]


def prepare_embedding_batches(
    sequence_path: str | Path,
    output_dir: str | Path,
    split: str,
    batch_size: int = DEFAULT_BATCH_SIZE,
    model_name: str = DEFAULT_MODEL_NAME,
) -> tuple[pd.DataFrame, list[Path]]:
    """Read a sequence table and write resumable embedding batches plus manifest."""

    sequence_table = pd.read_parquet(sequence_path)
    config = EmbeddingBatchConfig(
        split=split,
        output_dir=Path(output_dir),
        batch_size=batch_size,
        model_name=model_name,
    )
    manifest = create_embedding_manifest(sequence_table, config)
    written_batches = write_embedding_batches(sequence_table, manifest)
    manifest = update_manifest_from_existing_shards(manifest)
    write_manifest(manifest, config.output_dir / "manifest.csv")
    return manifest, written_batches


def pending_batch_ids(manifest: pd.DataFrame) -> list[str]:
    """Return batch IDs whose shard file is not complete."""

    refreshed = update_manifest_from_existing_shards(manifest)
    pending = refreshed.loc[refreshed["status"] != "complete", "batch_id"].drop_duplicates()
    return pending.sort_values(kind="mergesort").tolist()


def mark_batch_complete(manifest_path: str | Path, batch_id: str, completed_at: str) -> pd.DataFrame:
    """Mark one batch complete in an on-disk manifest."""

    manifest = read_manifest(manifest_path)
    mask = manifest["batch_id"] == batch_id
    if not mask.any():
        raise ValueError(f"Unknown batch_id in manifest: {batch_id}")
    manifest.loc[mask, "status"] = "complete"
    manifest.loc[mask, "completed_at"] = completed_at
    write_manifest(manifest, manifest_path)
    return manifest


def load_esm2_model(model_name: str = DEFAULT_MODEL_NAME, device: str | None = None):
    """Load a fair-esm ESM-2 model and alphabet lazily."""

    try:
        import torch
        import esm
    except ImportError as exc:
        raise RuntimeError("ESM-2 extraction requires torch and fair-esm. Install them in Colab first.") from exc

    if not hasattr(esm.pretrained, model_name):
        raise ValueError(f"fair-esm does not expose model loader: {model_name}")

    loader = getattr(esm.pretrained, model_name)
    model, alphabet = loader()
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()
    return model, alphabet, device


def mean_pool_esm_representations(token_representations, batch_tokens, alphabet) -> np.ndarray:
    """Mean-pool ESM token representations over amino-acid tokens."""

    lengths = (batch_tokens != alphabet.padding_idx).sum(dim=1)
    pooled = []
    for index, tokens_len in enumerate(lengths.tolist()):
        pooled.append(token_representations[index, 1 : tokens_len - 1].mean(dim=0).detach().cpu().numpy())
    return np.vstack(pooled).astype("float32")


def extract_batch_embeddings(
    batch: pd.DataFrame,
    model,
    alphabet,
    device: str,
    repr_layer: int | None = None,
    inference_batch_size: int = DEFAULT_INFERENCE_BATCH_SIZE,
    max_sequence_length: int | None = DEFAULT_MAX_SEQUENCE_LENGTH,
) -> np.ndarray:
    """Extract mean-pooled ESM-2 embeddings for one sequence batch."""

    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("ESM-2 extraction requires torch.") from exc

    if repr_layer is None:
        repr_layer = int(getattr(model, "num_layers", 33))

    converter = alphabet.get_batch_converter()
    embeddings: list[np.ndarray] = []
    records = list(zip(batch["entry_id"].astype(str), batch["sequence"].astype(str), strict=True))
    if max_sequence_length is not None:
        records = [(entry_id, sequence[:max_sequence_length]) for entry_id, sequence in records]

    with torch.no_grad():
        for start in range(0, len(records), inference_batch_size):
            chunk = records[start : start + inference_batch_size]
            _, _, batch_tokens = converter(chunk)
            batch_tokens = batch_tokens.to(device)
            result = model(batch_tokens, repr_layers=[repr_layer], return_contacts=False)
            token_representations = result["representations"][repr_layer]
            embeddings.append(mean_pool_esm_representations(token_representations, batch_tokens, alphabet))

    if not embeddings:
        return np.empty((0, 0), dtype="float32")
    return np.vstack(embeddings).astype("float32")


def extract_manifest_batch(
    manifest_path: str | Path,
    batch_id: str,
    model_name: str = DEFAULT_MODEL_NAME,
    device: str | None = None,
    repr_layer: int | None = None,
    inference_batch_size: int = DEFAULT_INFERENCE_BATCH_SIZE,
    max_sequence_length: int | None = DEFAULT_MAX_SEQUENCE_LENGTH,
    overwrite: bool = False,
    completed_at: str = "",
) -> Path:
    """Extract one manifest batch to its shard path and update the manifest."""

    manifest = read_manifest(manifest_path)
    batch_rows = manifest.loc[manifest["batch_id"] == batch_id]
    if batch_rows.empty:
        raise ValueError(f"Unknown batch_id in manifest: {batch_id}")

    batch_path = Path(batch_rows["batch_path"].iloc[0])
    shard_path = Path(batch_rows["shard_path"].iloc[0])
    if shard_path.is_file() and not overwrite:
        mark_batch_complete(manifest_path, batch_id, completed_at=completed_at)
        return shard_path

    batch = pd.read_parquet(batch_path)
    model, alphabet, resolved_device = load_esm2_model(model_name=model_name, device=device)
    embeddings = extract_batch_embeddings(
        batch=batch,
        model=model,
        alphabet=alphabet,
        device=resolved_device,
        repr_layer=repr_layer,
        inference_batch_size=inference_batch_size,
        max_sequence_length=max_sequence_length,
    )

    shard_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(shard_path, embeddings)
    mark_batch_complete(manifest_path, batch_id, completed_at=completed_at)
    return shard_path


def extract_manifest_batches(
    manifest_path: str | Path,
    batch_ids: Iterable[str] | None = None,
    model_name: str = DEFAULT_MODEL_NAME,
    device: str | None = None,
    repr_layer: int | None = None,
    inference_batch_size: int = DEFAULT_INFERENCE_BATCH_SIZE,
    max_sequence_length: int | None = DEFAULT_MAX_SEQUENCE_LENGTH,
    overwrite: bool = False,
    max_batches: int | None = None,
) -> list[Path]:
    """Extract multiple manifest batches while loading the ESM-2 model once."""

    manifest_path = Path(manifest_path)
    manifest = read_manifest(manifest_path)
    selected_batch_ids = list(batch_ids) if batch_ids is not None else pending_batch_ids(manifest)
    if max_batches is not None:
        selected_batch_ids = selected_batch_ids[:max_batches]
    if not selected_batch_ids:
        return []

    model, alphabet, resolved_device = load_esm2_model(model_name=model_name, device=device)
    written: list[Path] = []

    for batch_id in selected_batch_ids:
        manifest = read_manifest(manifest_path)
        batch_rows = manifest.loc[manifest["batch_id"] == batch_id]
        if batch_rows.empty:
            raise ValueError(f"Unknown batch_id in manifest: {batch_id}")

        batch_path = Path(batch_rows["batch_path"].iloc[0])
        shard_path = Path(batch_rows["shard_path"].iloc[0])
        completed_at = datetime.now(timezone.utc).isoformat()
        if shard_path.is_file() and not overwrite:
            mark_batch_complete(manifest_path, batch_id, completed_at=completed_at)
            written.append(shard_path)
            continue

        batch = pd.read_parquet(batch_path)
        embeddings = extract_batch_embeddings(
            batch=batch,
            model=model,
            alphabet=alphabet,
            device=resolved_device,
            repr_layer=repr_layer,
            inference_batch_size=inference_batch_size,
            max_sequence_length=max_sequence_length,
        )
        shard_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(shard_path, embeddings)
        mark_batch_complete(manifest_path, batch_id, completed_at=completed_at)
        written.append(shard_path)

    return written


def manifest_summary(manifest: pd.DataFrame) -> dict[str, object]:
    """Summarize manifest coverage and completion state."""

    manifest = update_manifest_from_existing_shards(manifest)
    status_counts = {
        status: int(count)
        for status, count in manifest.groupby("status")["entry_id"].count().sort_index().items()
    }
    return {
        "n_sequences": int(len(manifest)),
        "n_batches": int(manifest["batch_id"].nunique()) if not manifest.empty else 0,
        "splits": sorted(manifest["split"].drop_duplicates().tolist()),
        "models": sorted(manifest["model_name"].drop_duplicates().tolist()),
        "status_counts": status_counts,
        "pending_batches": pending_batch_ids(manifest),
    }


def config_to_dict(config: EmbeddingBatchConfig) -> dict[str, object]:
    """Return a JSON/YAML-friendly config dictionary."""

    data = asdict(config)
    data["output_dir"] = str(config.output_dir)
    return data
