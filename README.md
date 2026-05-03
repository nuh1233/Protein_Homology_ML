# Protein Homology ML — CAFA 6 Function Prediction

A full end-to-end machine-learning pipeline for the [CAFA 6 Kaggle competition](https://www.kaggle.com/competitions/cafa-6-protein-function-prediction): predicting Gene Ontology (GO) function terms for proteins using ESM-2 language-model embeddings and sequence-homology transfer.

---

## Table of Contents

1. [Overview](#overview)
2. [Repository Structure](#repository-structure)
3. [Pipeline Architecture](#pipeline-architecture)
4. [Installation](#installation)
5. [Data Download](#data-download)
6. [Step-by-Step Pipeline](#step-by-step-pipeline)
   - [1. Verify raw files](#1-verify-raw-files)
   - [2. Build processed tables](#2-build-processed-tables)
   - [3. Build GO ontology tables](#3-build-go-ontology-tables)
   - [4. Create cross-validation folds](#4-create-cross-validation-folds)
   - [5. Prepare ESM-2 embedding batches](#5-prepare-esm-2-embedding-batches)
   - [6. Extract ESM-2 embeddings (Colab/GPU)](#6-extract-esm-2-embeddings-colab--gpu)
   - [7. Run homology transfer](#7-run-homology-transfer)
   - [8. Train supervised heads](#8-train-supervised-heads)
   - [9. Calibrate predictions](#9-calibrate-predictions)
   - [10. Blend predictions](#10-blend-predictions)
   - [11. Score predictions (CV)](#11-score-predictions-cv)
   - [12. Create submission file](#12-create-submission-file)
7. [Configuration Files](#configuration-files)
8. [Module Reference](#module-reference)
9. [Evaluation Metric](#evaluation-metric)
10. [Running Tests](#running-tests)

---

## Overview

This project tackles the [Critical Assessment of Functional Annotation (CAFA 6)](https://www.kaggle.com/competitions/cafa-6-protein-function-prediction) challenge: given protein sequences, predict which Gene Ontology terms they are annotated with across three ontology branches:

| Branch | Abbreviation | Description |
|--------|-------------|-------------|
| Molecular Function | MF | What a protein does at the molecular level |
| Biological Process | BP | Broader biological processes the protein is involved in |
| Cellular Component | CC | Where the protein is found in the cell |

The pipeline combines two complementary prediction strategies:

- **Homology transfer** — sequence similarity search (MMseqs2/DIAMOND) propagates GO annotations from labelled training proteins to query proteins via a noisy-OR aggregation formula weighted by alignment quality.
- **Supervised learning** — branch-specific one-vs-rest logistic classifiers trained on ESM-2 protein language model embeddings (650M parameter `esm2_t33_650M_UR50D`).

Both streams are combined in a final weighted ensemble.

---

## Repository Structure

```
Protein_Homology_ML/
├── cafa6/                      # Core Python library
│   ├── __init__.py
│   ├── calibration.py          # Branch-specific score calibration
│   ├── embeddings.py           # Embedding shard manifest utilities
│   ├── ensemble.py             # Prediction blending & GO hierarchy repair
│   ├── features.py             # Embedding matrix loading & label matrix building
│   ├── folds.py                # Cluster-aware cross-validation fold creation
│   ├── homology.py             # Hit table parsing, normalization & noisy-OR aggregation
│   ├── io.py                   # FASTA/TSV/Parquet I/O, file-check utilities
│   ├── metrics.py              # Fmax scoring (protein-macro & micro)
│   ├── models.py               # SGD one-vs-rest classifier training & inference
│   ├── ontology.py             # GO OBO parsing & transitive ancestor closure
│   └── submission.py           # Submission frame preparation & validation
│
├── scripts/                    # Runnable pipeline scripts
│   ├── download_data.py        # Download CAFA 6 files from Kaggle
│   ├── verify_raw_files.py     # Confirm all required raw files are present
│   ├── build_tables.py         # Parse raw files → Parquet sequence/label tables
│   ├── build_ontology.py       # Parse go-basic.obo → ancestor/edge tables
│   ├── make_folds.py           # Assign cluster-aware CV folds
│   ├── prepare_embedding_batches.py  # Shard sequences for Colab ESM-2 extraction
│   ├── run_homology.py         # Aggregate MMseqs2/DIAMOND hit tables
│   ├── train_supervised.py     # Train branch SGD classifiers on ESM-2 embeddings
│   ├── calibrate_predictions.py     # Fit & apply branch calibration
│   ├── blend_predictions.py    # Blend homology + supervised predictions
│   ├── score_predictions.py    # Compute Fmax CV scores
│   └── make_submission.py      # Write final Kaggle TSV submission
│
├── notebooks/
│   ├── 00_colab_setup.ipynb    # Install dependencies on Colab
│   └── 01_extract_esm2_embeddings.ipynb  # GPU embedding extraction (run on Colab)
│
├── configs/
│   ├── baseline_esm2.yaml      # ESM-2 model & batch configuration
│   └── homology_transfer.yaml  # Homology aggregation configuration
│
├── tests/                      # pytest unit tests (one file per module)
├── requirements.txt
└── README.md
```

### Data & Artifact Layout (created at runtime)

```
data/
├── raw/                        # Downloaded CAFA 6 competition files (git-ignored)
└── processed/                  # Parquet tables derived from raw data (git-ignored)

artifacts/
├── embeddings/
│   ├── esm2_train/             # ESM-2 .npy shards + manifest.csv for train set
│   └── esm2_test/              # ESM-2 .npy shards + manifest.csv for test set
├── retrieval/                  # Homology hit tables & aggregated predictions
├── models/                     # Serialized joblib branch models
├── predictions/                # OOF and test prediction Parquet files
└── reports/                    # JSON/CSV diagnostic reports
```

---

## Pipeline Architecture

```
Raw FASTA + TSV                  go-basic.obo
      │                               │
 build_tables.py              build_ontology.py
      │                               │
 data/processed/           ancestor closure tables
 (sequences, labels)               │
      │────────────────────────────┘
      │
 make_folds.py ────► folds_clustered.parquet
      │
      ├──► prepare_embedding_batches.py
      │         │
      │    [Colab GPU]
      │    01_extract_esm2_embeddings.ipynb
      │         │
      │    artifacts/embeddings/
      │         │
      │    train_supervised.py ────► esm2_oof.parquet
      │                              esm2_test.parquet
      │
      └──► [External: MMseqs2 / DIAMOND]
                │
           run_homology.py ────► homology_oof.parquet
                                  homology_test.parquet
                │
          calibrate_predictions.py
                │
           blend_predictions.py ────► ensemble_oof.parquet
                                       ensemble_test.parquet
                │
           make_submission.py ────► submission.tsv  (Kaggle upload)
```

---

## Installation

**Python 3.11+ recommended.**

```bash
pip install -r requirements.txt
```

Core dependencies:

| Package | Version | Purpose |
|---------|---------|---------|
| numpy | ≥1.26 | Numerical arrays |
| pandas | ≥2.2 | DataFrame operations |
| pyarrow | ≥15.0 | Parquet I/O |
| scipy | ≥1.12 | Sparse matrix operations |
| scikit-learn | ≥1.4 | SGD classifiers |
| joblib | ≥1.3 | Model serialization & parallelism |
| kagglehub | ≥0.3 | Kaggle competition data download |
| pytest | ≥8.0 | Test runner |

**ESM-2 embeddings** additionally require `fair-esm` and `torch`. These are heavy GPU dependencies and are only needed during the embedding extraction step; install them inside Google Colab using `notebooks/00_colab_setup.ipynb`.

**Homology search** requires an external binary — either [MMseqs2](https://github.com/soedinglab/MMseqs2) or [DIAMOND](https://github.com/bbuchfink/diamond) — installed separately and accessible on your `PATH`.

---

## Data Download

Authenticate with Kaggle by placing a valid `kaggle.json` in `~/.kaggle/` (standard Kaggle API key), then run:

```bash
python scripts/download_data.py
```

This downloads the CAFA 6 competition files into `data/raw/`. Required files:

| File | Description |
|------|-------------|
| `train_sequences.fasta` | Training protein sequences |
| `train_terms.tsv` | GO term annotations for training proteins |
| `train_taxonomy.tsv` | NCBI taxon IDs for training proteins |
| `testsuperset.fasta` | Test protein sequences |
| `testsuperset-taxon-list.tsv` | Taxon IDs for test proteins |
| `go-basic.obo` | GO ontology definition file |
| `IA.tsv` | Information Accretion weights |
| `sample_submission.tsv` | Submission format reference |

To verify files without re-downloading:

```bash
python scripts/verify_raw_files.py
```

---

## Step-by-Step Pipeline

### 1. Verify raw files

```bash
python scripts/verify_raw_files.py
```

Prints a summary of which required files are present/missing in `data/raw/`.

---

### 2. Build processed tables

```bash
python scripts/build_tables.py
```

Parses the raw FASTA and TSV files into normalized Parquet tables under `data/processed/`:

- `train_sequences.parquet` — sequence metadata (entry_id, length, taxon_id, …)
- `test_sequences.parquet`
- `train_terms.parquet` — direct GO annotations
- `train_terms_closure.parquet` — annotations expanded with transitive ontology ancestors
- `label_frequency.csv` — per-branch GO term frequency report

---

### 3. Build GO ontology tables

```bash
python scripts/build_ontology.py
```

Parses `go-basic.obo` and produces:

- `data/processed/go_terms.parquet` — active GO terms with branch labels
- `data/processed/go_edges.parquet` — `is_a` and `part_of` edges
- `data/processed/go_ancestors.parquet` — transitive ancestor closure
- `artifacts/reports/ontology_report.json` — parsing statistics

---

### 4. Create cross-validation folds

```bash
python scripts/make_folds.py
```

Assigns proteins to 5 stratified folds, respecting cluster boundaries so that sequence-similar proteins always land in the same fold (preventing data leakage):

- `data/processed/folds_clustered.parquet` — per-protein fold assignments

Optionally supply a pre-computed cluster file with `--clusters <path>`. Without one, each protein is treated as its own singleton cluster.

---

### 5. Prepare ESM-2 embedding batches

```bash
python scripts/prepare_embedding_batches.py
```

Splits train and test sequences into fixed-size batches and writes a `manifest.csv` describing shard paths and row indices. No GPU required for this step.

Output paths are configured in `configs/baseline_esm2.yaml`:
- `artifacts/embeddings/esm2_train/manifest.csv`
- `artifacts/embeddings/esm2_test/manifest.csv`

---

### 6. Extract ESM-2 embeddings (Colab / GPU)

Open `notebooks/01_extract_esm2_embeddings.ipynb` in Google Colab (or any GPU machine) and run all cells. The notebook:

1. Installs `fair-esm` and `torch`.
2. Loads `esm2_t33_650M_UR50D` (33-layer, 650M parameter model).
3. Iterates over manifest batches, extracts layer-33 mean representations.
4. Saves one `.npy` shard per batch.

Each shard is a `float32` array of shape `(batch_size, 1280)`. Progress is tracked in the manifest so interrupted runs resume automatically.

---

### 7. Run homology transfer

**First, generate hit tables with MMseqs2 (outside Python):**

```bash
# Train-vs-train (for OOF predictions)
mmseqs easy-search data/raw/train_sequences.fasta \
       data/raw/train_sequences.fasta \
       artifacts/retrieval/mmseqs_hits_valid.tsv \
       tmp_mmseqs \
       --format-output query,target,pident,evalue,bits,qcov,tcov

# Test-vs-train
mmseqs easy-search data/raw/testsuperset.fasta \
       data/raw/train_sequences.fasta \
       artifacts/retrieval/mmseqs_hits_test.tsv \
       tmp_mmseqs \
       --format-output query,target,pident,evalue,bits,qcov,tcov
```

Or print suggested commands without running them:

```bash
python scripts/run_homology.py --mode prepare
```

**Then aggregate hit labels:**

```bash
python scripts/run_homology.py --mode aggregate \
  --valid-hits artifacts/retrieval/mmseqs_hits_valid.tsv \
  --test-hits  artifacts/retrieval/mmseqs_hits_test.tsv
```

Hit table formats supported: Parquet, CSV, tab-separated (with or without header). Any column naming convention from MMseqs2, DIAMOND, or BLAST tabular output is auto-detected.

Outputs:
- `artifacts/retrieval/homology_oof.parquet` — OOF predictions (fold-filtered)
- `artifacts/retrieval/homology_test.parquet` — Test predictions
- `artifacts/reports/homology_report.json` — Aggregation statistics

The scoring formula is **noisy-OR** over hit weights:

```
score(query, term) = 1 - ∏(1 - weight_i)  for all hits i carrying that term
```

Hit weights are derived from bitscore (relative to per-query max), percent identity, or e-value — whichever is available — multiplied by a coverage factor `√(qcov × tcov)`.

---

### 8. Train supervised heads

```bash
python scripts/train_supervised.py
```

Trains one branch-specific one-vs-rest SGD logistic classifier per GO branch (MF, BP, CC) on ESM-2 embeddings with 5-fold cross-validation:

- Loads ESM-2 embedding matrices from shard manifests.
- Selects the top-frequency GO labels per branch (up to 1500 MF / 2500 BP / 1000 CC).
- Runs 5-fold CV for OOF predictions and fits a final model on all training data.
- Saves models to `artifacts/models/{mf,bp,cc}_model.joblib`.

Outputs:
- `artifacts/predictions/esm2_oof.parquet`
- `artifacts/predictions/esm2_test.parquet`
- `artifacts/reports/cv_scores.csv`

Key flags:

| Flag | Default | Description |
|------|---------|-------------|
| `--min-label-count` | 50 | Minimum training proteins per GO label |
| `--alpha` | 1e-4 | L2 regularization strength |
| `--max-iter` | 50 | Max SGD epochs |
| `--n-jobs` | -1 | Parallel workers (-1 = all cores) |

---

### 9. Calibrate predictions

```bash
python scripts/calibrate_predictions.py
```

Fits a deterministic per-branch max-normalization calibration from OOF predictions, then applies it to both OOF and test prediction files. This step is optional but improves score comparability across branches.

Outputs:
- `artifacts/predictions/esm2_oof_repaired.parquet`
- `artifacts/predictions/esm2_test_repaired.parquet`
- `artifacts/reports/calibration_report.json`

The script also calls `repair_go_hierarchy` to propagate prediction scores to GO ancestor terms, ensuring the hierarchy constraint is satisfied.

---

### 10. Blend predictions

```bash
python scripts/blend_predictions.py
```

Combines homology and supervised predictions with a weighted max-score blend:

```
ensemble_score(entry, term) = max(homology_score × w_h,  supervised_score × w_s)
```

Default weights: `w_h = 1.0`, `w_s = 0.05`. The supervised stream is down-weighted because it is sparser (top-25 predictions per branch per protein).

Outputs:
- `artifacts/predictions/ensemble_oof.parquet`
- `artifacts/predictions/ensemble_test.parquet`
- `artifacts/reports/ensemble_cv_scores.csv`

---

### 11. Score predictions (CV)

```bash
python scripts/score_predictions.py \
  --predictions artifacts/predictions/ensemble_oof.parquet \
  --truth       data/processed/train_terms_closure.parquet
```

Computes per-branch **Fmax** (maximum F-measure over all decision thresholds) and prints results. Useful for evaluating any intermediate prediction file.

---

### 12. Create submission file

```bash
python scripts/make_submission.py
```

Reads the ensemble test predictions, filters to valid test entry IDs and GO terms, optionally prunes to top-K predictions per protein, and writes a headerless three-column TSV:

```
<entry_id>\t<GO_term>\t<score>
```

Output: `artifacts/predictions/submission.tsv` — ready to upload to Kaggle.

---

## Configuration Files

### `configs/baseline_esm2.yaml`

Controls ESM-2 model selection and batch sizes for embedding extraction:

```yaml
model:
  name: esm2_t33_650M_UR50D   # Hugging Face / fair-esm model name
  repr_layer: 33               # Which transformer layer to extract
  max_sequence_length: 1022    # Sequences truncated beyond this length

batches:
  sequence_batch_size: 512     # Sequences per shard
```

### `configs/homology_transfer.yaml`

Controls homology aggregation behaviour:

```yaml
aggregation:
  max_hits_per_query: 100      # Top hits kept per query protein
  min_hit_weight: 0.0          # Discard hits below this weight
  closure_relations: [is_a, part_of]
  score_formula: noisy_or_over_hit_weights
  exclude_oof_self_hits: true
  exclude_oof_same_fold_hits: true
  exclude_test_self_hits: true
```

---

## Module Reference

### `cafa6.io`

Low-level I/O primitives: FASTA record parsing (`iter_fasta_records`, `read_fasta`), training term loading (`read_train_terms`), taxonomy reading, raw-file presence checks, and helpers for writing JSON and Parquet.

### `cafa6.ontology`

Parses `go-basic.obo` files into `GoTerm` / `GoEdge` dataclasses, filters active non-obsolete terms, and computes a transitive ancestor closure table via BFS. Entry point: `build_go_tables(obo_path)`.

### `cafa6.homology`

Reads and normalizes hit tables from any MMseqs2/DIAMOND/BLAST tabular format, computes `hit_weight` values from alignment statistics, and runs **noisy-OR** label aggregation. Key functions:

- `read_hit_table(path)` — auto-detects format
- `add_hit_weights(hits)` — derives weights from bitscore/pident/evalue + coverage
- `aggregate_hit_labels(hits, train_terms)` — noisy-OR aggregation
- `make_oof_homology_predictions(...)` — fold-safe OOF predictions
- `make_test_homology_predictions(...)` — test predictions

### `cafa6.folds`

Cluster-aware cross-validation split that assigns whole sequence-similarity clusters to the same fold, preventing leakage from homology-based models. Entry point: `make_fold_assignments(sequences, clusters, n_folds)`.

### `cafa6.features`

Loads ESM-2 embedding shards described by a manifest CSV into an aligned `EmbeddingMatrix`, builds sparse multilabel indicator matrices, and provides `make_prediction_frame` to convert dense score arrays to long-form DataFrames. Entry points: `load_embedding_matrix(manifest_path)`, `build_label_matrix(...)`.

### `cafa6.models`

Branch-specific one-vs-rest logistic SGD classifier. Entry points:

- `train_branch_model(x_train, y_train, label_terms, branch)` — returns a `BranchSupervisedModel`
- `predict_branch_topk(model, x, entry_ids, top_k)` — batched top-K inference

### `cafa6.metrics`

Computes the official CAFA **protein-macro Fmax** metric across all decision thresholds. Also supports micro-averaged Fmax. Entry point: `score_branch_fmaxes(truth, predictions)`.

### `cafa6.calibration`

Per-branch max-normalization: scales each branch's scores so the maximum equals 1.0, making scores comparable across branches. Also provides `fit_branch_calibration` and `apply_branch_calibration` for calibration metadata round-trips.

### `cafa6.ensemble`

- `blend_prediction_frames(frames, weights)` — weighted mean blend
- `repair_go_hierarchy(predictions, ancestors)` — propagates child scores to GO ancestors, satisfying the monotonicity constraint
- `prune_top_k_by_group(predictions, top_k)` — keeps highest-scoring K rows per (entry, branch) group

### `cafa6.submission`

- `prepare_submission_frame(predictions, ...)` — filters to valid entries/terms, deduplicates, prunes to top-K
- `validate_submission_predictions(...)` — returns a detailed validity report
- `write_submission(frame, path)` — writes headerless TSV

---

## Evaluation Metric

The competition uses **protein-centric macro-averaged Fmax**:

1. For each decision threshold τ ∈ [0, 1]:
   - For each protein, compute precision and recall against the ground-truth GO term set.
   - Average precision and recall across all proteins with at least one prediction above τ (precision) or with at least one ground-truth annotation (recall).
   - Compute F1 from the means.
2. **Fmax** = maximum F1 over all τ.
3. The final competition score is the mean Fmax across MF, BP, and CC branches.

This is implemented in `cafa6.metrics.score_branch_fmaxes`.

---

## Running Tests

```bash
pytest tests/
```

The `tests/` directory contains one test file per module (e.g. `test_homology.py`, `test_metrics.py`, …). Tests use synthetic in-memory DataFrames and do not require the competition data files.

To run a specific test module:

```bash
pytest tests/test_homology.py -v
```