# Feature Spec: Phase 1 — Corpus Engineering & Micro-Clustering

## Objective

Produce a set of neuroscience micro-clusters from the DAPT corpus, assign every document to a cluster, and create a deterministic three-way data split (dev / val / sealed) per cluster. The cluster assignments and split indices are the primary inputs for Phase 2 teacher election and trace generation.

---

## Scope

This feature implements **Phase 1 (Steps 1.1 and 1.2)** of the pipeline. It sits after RAD Prep (`s3_rad_prep`) and produces the cluster structure consumed by Phase 2 (`s5_teacher_election`, future).

Two outputs:
1. Cluster assignment for every document in the DAPT corpus
2. Three-way per-cluster split indices with imbalance reweighting caps for Phase 2

---

## Module Layout

```
lib/
└── s4_clustering/
    ├── __init__.py              # run_clustering_pipeline(cfg) entry point
    ├── embedder.py              # Batch-embed documents via sentence-transformers; cache to disk
    ├── clusterer.py             # HDBSCAN clustering + noise-point resolution
    ├── splitter.py              # Per-cluster 70/20/10 split with imbalance reweighting caps
    └── cluster_reporter.py     # Cluster size stats, imbalance report, phase manifest
```

---

## Data Flow

```
DAPT Corpus (JSONL)
        ↓
   embedder.py        — SentenceTransformer(all-mpnet-base-v2) → embeddings.npy + doc_ids.json
        ↓
   clusterer.py       — HDBSCAN(cosine) → ClusterAssignment records; noise → nearest centroid
        ↓
cluster_reporter.py   — cluster count, size stats, imbalance report
        ↓
   splitter.py        — per-cluster 70/20/10 split → splits.json with reweight_cap per cluster
        ↓
Output:
  data/clustering/embeddings.npy                  — float32 (n_docs, dim), gitignored
  data/clustering/doc_ids.json                    — ordered doc_id list matching embeddings rows
  data/clustering/cluster_assignments.jsonl       — one record per doc
  data/clustering/splits.json                     — per-cluster dev/val/sealed doc_id lists
  data/clustering/cluster_manifest.json           — stats + pass/fail
  logs/clustering/cluster_report.json             — detailed per-cluster stats
```

---

## Component Specifications

### 1. `embedder.py`

Reads the DAPT corpus JSONL. Embeds each document's `text` field using
`sentence_transformers.SentenceTransformer(cfg.clustering.embedding_model)`.

**Cache behavior:** If `cfg.clustering.embeddings_cache_path` and the paired
`doc_ids.json` both exist and the cached doc count matches the current corpus
size, skip re-embedding and load from disk. Log a warning if cache is reused
with a different model name than what is recorded in the manifest.

Returns `(embeddings: np.ndarray, doc_ids: List[str])`. Shape: `(n_docs, 768)`
for `all-mpnet-base-v2`.

Batch size: `cfg.clustering.embed_batch_size` (default 64). Progress via tqdm.

### 2. `clusterer.py`

Applies HDBSCAN to the cached embeddings.

```python
@dataclass
class ClusterAssignment:
    doc_id: str
    cluster_id: int       # HDBSCAN label (noise originally -1, reassigned)
    cluster_label: str    # "cluster_007" (zero-padded to 3 digits)
    is_noise: bool        # True if HDBSCAN originally labeled as noise
    assigned_by: str      # "hdbscan" | "nearest_centroid" | "dropped"
```

**Noise handling** (controlled by `cfg.clustering.noise_assignment`):
- `nearest` (default): compute cosine distance from noise point to each cluster
  centroid (mean of cluster member embeddings); assign to nearest cluster.
  Set `is_noise=True`, `assigned_by="nearest_centroid"`.
- `drop`: exclude noise points from all downstream processing.
  Set `assigned_by="dropped"`. Log count of dropped documents.

HDBSCAN parameters exposed in config: `min_cluster_size`, `min_samples`, `metric`.
Default: `min_cluster_size=10`, `min_samples=5`, `metric="cosine"`.

Returns `List[ClusterAssignment]`.

### 3. `splitter.py`

For each cluster, splits its assigned `doc_id` list into dev / val / sealed at
the configured ratios (default 0.70 / 0.20 / 0.10). Split is random with
`cfg.misc.seed` for reproducibility. For clusters with fewer than 3 documents,
assign all to dev and mark `val_doc_ids=[]`, `sealed_doc_ids=[]`.

**Imbalance reweighting caps:**

Computes the total document count across all clusters. For each cluster:
- `raw_fraction = cluster_doc_count / total_docs`
- If `raw_fraction < cfg.clustering.cluster_min_fraction`:
  `reweight_cap = ceil(total_docs * cluster_min_fraction)`
- If `raw_fraction > cfg.clustering.cluster_max_fraction`:
  `reweight_cap = floor(total_docs * cluster_max_fraction)`
- Otherwise: `reweight_cap = None` (no cap)

`reweight_cap` is a recommended **trace count ceiling** for Phase 2 — it is
stored in `splits.json` so Phase 2 can enforce it without re-reading clustering
data.

```python
@dataclass
class ClusterSplit:
    cluster_id: int
    cluster_label: str
    dev_doc_ids: List[str]
    val_doc_ids: List[str]
    sealed_doc_ids: List[str]
    total_docs: int
    raw_fraction: float
    reweight_cap: Optional[int]
```

`splits.json` format:
```json
{
  "total_docs": 12500,
  "total_clusters": 87,
  "seed": 42,
  "clusters": {
    "cluster_007": {
      "cluster_id": 7,
      "total_docs": 144,
      "dev_doc_ids": [...],
      "val_doc_ids": [...],
      "sealed_doc_ids": [...],
      "raw_fraction": 0.0115,
      "reweight_cap": 250
    },
    ...
  }
}
```

### 4. `cluster_reporter.py`

Produces `data/clustering/cluster_manifest.json`:

```json
{
  "status": "complete",
  "embedding_model": "all-mpnet-base-v2",
  "total_docs": 12500,
  "noise_docs": 87,
  "noise_fraction": 0.007,
  "total_clusters": 87,
  "cluster_sizes": {"min": 12, "max": 1820, "mean": 143.7, "median": 98, "std": 201.4},
  "capped_min_count": 12,
  "capped_max_count": 3,
  "warnings": ["High noise fraction: 0.35 > 0.30 threshold"]
}
```

Also writes `logs/clustering/cluster_report.json` with per-cluster doc count,
fraction, and reweight_cap for every cluster.

---

## Config Extensions

Add `ClusteringConfig` to [lib/utils/config.py](lib/utils/config.py) and include
it as `clustering: ClusteringConfig` in `PipelineConfig`.

```python
@dataclass
class ClusteringConfig:
    # Input
    corpus_path: Path              # CLUSTERING_CORPUS_PATH, default: data/dapt/domain_dapt_corpus.jsonl

    # Embedding
    embedding_model: str           # CLUSTERING_EMBEDDING_MODEL, default: all-mpnet-base-v2
    embed_batch_size: int          # CLUSTERING_EMBED_BATCH_SIZE, default: 64
    embeddings_cache_path: Path    # CLUSTERING_EMBEDDINGS_CACHE, default: data/clustering/embeddings.npy
    doc_ids_cache_path: Path       # CLUSTERING_DOC_IDS_CACHE, default: data/clustering/doc_ids.json

    # HDBSCAN
    hdbscan_min_cluster_size: int  # HDBSCAN_MIN_CLUSTER_SIZE, default: 10
    hdbscan_min_samples: int       # HDBSCAN_MIN_SAMPLES, default: 5
    hdbscan_metric: str            # HDBSCAN_METRIC, default: cosine

    # Noise handling
    noise_assignment: str          # CLUSTERING_NOISE_ASSIGNMENT, default: nearest (nearest|drop)

    # Imbalance reweighting
    cluster_min_fraction: float    # CLUSTER_MIN_FRACTION, default: 0.02
    cluster_max_fraction: float    # CLUSTER_MAX_FRACTION, default: 0.15

    # Split ratios
    split_dev_ratio: float         # SPLIT_DEV_RATIO, default: 0.70
    split_val_ratio: float         # SPLIT_VAL_RATIO, default: 0.20
    split_sealed_ratio: float      # SPLIT_SEALED_RATIO, default: 0.10

    # Output paths
    output_dir: Path               # CLUSTERING_OUTPUT_DIR, default: data/clustering
    assignments_path: Path         # CLUSTERING_ASSIGNMENTS_PATH, default: data/clustering/cluster_assignments.jsonl
    splits_path: Path              # CLUSTERING_SPLITS_PATH, default: data/clustering/splits.json
    cluster_manifest_path: Path    # CLUSTERING_MANIFEST_PATH, default: data/clustering/cluster_manifest.json
```

---

## Pipeline Integration

Add step `s4` to `pipeline.py` argparse choices:

```
--step choices: ["s1", "s1.5", "s2", "s3", "s4", "all"]
```

Add `s4()` function that calls `run_clustering_pipeline(cfg)`. The `all` path
runs: s1 → s1.5 → s2 → s3 → s4.

No sub-modes for s4; it always runs embedding + clustering + splitting in sequence.

---

## Output Artifacts

| Artifact | Path | Phase Log Entry |
|---|---|---|
| Embeddings | `data/clustering/embeddings.npy` | model, n_docs, dim (gitignored) |
| Doc ID index | `data/clustering/doc_ids.json` | ordered list matching embeddings rows |
| Cluster assignments | `data/clustering/cluster_assignments.jsonl` | n_docs records |
| Splits | `data/clustering/splits.json` | per-cluster dev/val/sealed lists + caps |
| Cluster manifest | `data/clustering/cluster_manifest.json` | stats, warnings, pass/fail |
| Cluster report | `logs/clustering/cluster_report.json` | per-cluster detail |

---

## Validation Gate

After clustering and splitting, the manifest records `status: complete` if all
of the following hold (logged as warnings, not hard failures, unless noted):

| Check | Threshold | Action if failed |
|---|---|---|
| Cluster count | ≥ 10 | **Hard fail** — HDBSCAN params need tuning |
| Cluster count | ≥ 50 | Warning — fewer micro-clusters than expected |
| Noise fraction | ≤ 0.30 | Warning — consider lowering `min_cluster_size` |
| Largest cluster fraction | ≤ 0.40 | Warning — severe imbalance before reweighting |
| All clusters with ≥ 3 docs have non-empty val split | — | **Hard fail** |

`status: complete` requires zero hard failures. Warnings are recorded in
`cluster_manifest.json["warnings"]` and do not block the pipeline.

---

## Dependencies

New packages to add to `pyproject.toml`:
- `hdbscan>=0.8` — HDBSCAN clustering algorithm

`sentence-transformers>=3.0` was already added in the `s3_rad_prep` spec.
`scikit-learn` is a transitive dependency of `hdbscan`.

---

## Test Plan

File: `tests/test_clustering.py`

| Test | What it checks |
|---|---|
| `test_embedder_output_shape` | output shape is `(n_docs, 768)`, doc_ids length matches |
| `test_embedder_cache_reuse` | second call with same corpus skips embedding, loads from disk |
| `test_embedder_cache_miss_on_count_change` | cache is ignored when corpus size changes |
| `test_clusterer_basic` | HDBSCAN assigns labels, returns `ClusterAssignment` objects |
| `test_clusterer_noise_nearest` | noise points get `assigned_by="nearest_centroid"` |
| `test_clusterer_noise_drop` | noise points get `assigned_by="dropped"`, excluded from assignments |
| `test_clusterer_labels_zero_padded` | `cluster_label` is zero-padded to 3 digits |
| `test_splitter_ratios` | dev/val/sealed sizes match 70/20/10 within ±1 doc rounding |
| `test_splitter_small_cluster` | cluster < 3 docs → val and sealed lists are empty, no crash |
| `test_splitter_reweight_min` | cluster below min_fraction gets non-None `reweight_cap` |
| `test_splitter_reweight_max` | cluster above max_fraction gets non-None `reweight_cap` |
| `test_splitter_within_range` | cluster within bounds has `reweight_cap=None` |
| `test_cluster_reporter_manifest` | manifest has correct fields, status="complete" for valid input |
| `test_cluster_reporter_warnings` | manifest includes warning when noise fraction > 0.30 |
| `test_pipeline_end_to_end` | runs full s4 with mock corpus; checks all output artifacts exist |

---

## Key Design Decisions

1. **Embeddings cache.** HDBSCAN is sensitive to `min_cluster_size` and `min_samples`;
   users will iterate on these parameters. Caching the embeddings (the expensive step)
   avoids re-running inference on every tuning pass.

2. **Cosine metric for HDBSCAN.** `all-mpnet-base-v2` produces L2-normalized vectors;
   cosine distance is the natural metric and matches the metric used in the RAD index
   (FAISS `IndexFlatIP`), keeping the two stages consistent.

3. **Nearest-centroid noise assignment (default).** Dropping noise silently reduces
   corpus coverage and can leave topic areas unrepresented. Nearest-centroid assignment
   keeps every document in the pipeline while flagging its origin. High noise fractions
   surface as a manifest warning, prompting parameter tuning.

4. **Imbalance caps are trace-count recommendations, not document-count filters.**
   The spec says caps apply "during trace generation in Phase 2." The splitter stores
   `reweight_cap` per cluster so Phase 2 can enforce it without re-reading clustering
   data — the split file is the single source of truth for per-cluster budgets.

5. **Sealed split is index-only.** Sealed doc_ids are stored deterministically (seeded)
   but must not be opened or used until Step 7.1. The pipeline only writes the index;
   it does not read or process sealed documents.

6. **`sentence-transformers` over raw AutoModel.** `SentenceTransformer` handles
   mean pooling and L2 normalization correctly for `all-mpnet-base-v2` out of the box.
   Manual pooling via AutoModel is error-prone and produces inconsistent embeddings.
