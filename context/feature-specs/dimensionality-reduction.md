# Feature Spec: Refactoring Dimensionality Reduction in Step 5 (`s5_clustering`)

## Objective

Decouple and refactor dimensionality reduction in **Step 5 (Corpus Engineering & Micro-Clustering)** to support multiple reduction algorithms (**UMAP**, **PCA**, and **Pass-through / None**). 

PCA (Principal Component Analysis) is a linear reduction technique that can discard non-linear manifold relationships in high-dimensional transformer embeddings (`all-mpnet-base-v2`). **UMAP + HDBSCAN** preserves local neighborhood topology significantly better, yielding tighter, more semantically cohesive micro-clusters for domain knowledge distillation.

---

## Scope

This feature refactors the embedding transformation pipeline within `lib/s5_clustering/`:
1. Creates a modular dimensionality reducer component (`lib/s5_clustering/dim_reducer.py`).
2. Configures UMAP as the default reduction strategy while preserving PCA and Pass-through options.
3. Extends `ClusteringConfig` in `lib/utils/config.py` with UMAP hyperparameters and backwards-compatible configuration flags.
4. Updates `clusterer.py` and `cluster_reporter.py` to log reduction metadata in manifests and execution reports.
5. Adds dependency `umap-learn>=0.5.5` to `pyproject.toml`.
6. Provides unit test coverage for UMAP, PCA, Pass-through, and fallback behaviors in `tests/test_clustering.py`.

---

## Module Architecture

```
lib/
└── s5_clustering/
    ├── __init__.py              # Pipeline entry point
    ├── embedder.py              # Dense sentence embeddings generator & disk cache
    ├── dim_reducer.py           # [NEW] Modular UMAP / PCA / Pass-through reduction engine
    ├── clusterer.py             # HDBSCAN micro-clustering & nearest-centroid noise resolution
    ├── splitter.py              # 70/20/10 dev/val/sealed split & imbalance capping
    └── cluster_reporter.py      # Quality gating & manifest reporting
```

---

## Data Flow

```
Raw Embeddings (n_docs, 768)
        ↓
   dim_reducer.py  — Select strategy via cfg.clustering.dim_reduction_method
                     ├─ "umap"        : UMAP(n_components=15, n_neighbors=15, min_dist=0.0, metric="cosine") -> L2-normalize
                     ├─ "pca"         : L2-normalize -> PCA(n_components=50) -> L2-normalize
                     └─ "passthrough" : Return original embeddings unchanged
        ↓
Reduced Embeddings (n_docs, target_dim)
        ↓
   clusterer.py    — HDBSCAN(metric="euclidean" or "cosine") → ClusterAssignment records
```

---

## Component Specifications

### 1. `lib/s5_clustering/dim_reducer.py` [NEW]

Provides `apply_dimensionality_reduction(embeddings: np.ndarray, cfg: PipelineConfig) -> Tuple[np.ndarray, Dict[str, Any]]`.

**Reduction Modes**:
- `umap` (Default):
  - Uses `umap.UMAP(n_components=cfg.clustering.umap_n_components, n_neighbors=cfg.clustering.umap_n_neighbors, min_dist=cfg.clustering.umap_min_dist, metric=cfg.clustering.umap_metric, random_state=cfg.misc.seed)`.
  - Recommended `min_dist=0.0` packages points tightly for HDBSCAN density clustering.
  - L2-normalizes the reduced vectors to ensure consistent distance computations.
  - Fallback logic: If `n_docs <= n_neighbors` or `n_docs <= n_components` or `umap-learn` import fails, gracefully fall back to PCA or Pass-through with warning logs.
- `pca`:
  - L2-normalizes input embeddings, fits `sklearn.decomposition.PCA(n_components=cfg.clustering.pca_components, random_state=cfg.misc.seed)`, and re-normalizes output.
- `none` / `passthrough`:
  - Passes original embeddings without reduction.

Returns `(reduced_embeddings, reduction_metadata)` where `reduction_metadata` contains:
```python
{
    "method": "umap",               # "umap" | "pca" | "passthrough"
    "original_dim": 768,
    "reduced_dim": 15,
    "fallback_triggered": False,
    "fallback_reason": None
}
```

### 2. `lib/s5_clustering/clusterer.py` [MODIFY]

- Refactors `run_clustering` to call `apply_dimensionality_reduction(embeddings, cfg)`.
- Removes hardcoded inline `PCA` code.
- Uses `clustering_metric = "euclidean"` for L2-normalized reduced spaces (or `"cosine"` for passthrough cosine metrics).

### 3. `lib/s5_clustering/cluster_reporter.py` [MODIFY]

- Includes dimensionality reduction metadata (`dim_reduction_method`, `original_dim`, `reduced_dim`) into `cluster_manifest.json` and `logs/clustering/cluster_report.json`.

---

## Configuration Extensions

Added to `ClusteringConfig` in `lib/utils/config.py`:

| Parameter | Type | Default | Env Var | Description |
|---|---|---|---|---|
| `dim_reduction_method` | `str` | `"umap"` | `CLUSTERING_DIM_REDUCTION_METHOD` | Reduction method (`umap`, `pca`, `none`, `passthrough`). |
| `umap_n_components` | `int` | `15` | `UMAP_N_COMPONENTS` | Target UMAP component dimensions. |
| `umap_n_neighbors` | `int` | `15` | `UMAP_N_NEIGHBORS` | Size of local neighborhood for UMAP manifold estimation. |
| `umap_min_dist` | `float` | `0.0` | `UMAP_MIN_DIST` | Minimum distance between embedded points (0.0 optimal for HDBSCAN). |
| `umap_metric` | `str` | `"cosine"` | `UMAP_METRIC` | Distance metric for UMAP input space manifold construction. |
| `use_pca` | `bool` | `True` | `CLUSTERING_USE_PCA` | Backwards-compatibility flag. If set to `False` and `dim_reduction_method` is default, maps to `none`. |
| `pca_components` | `int` | `50` | `CLUSTERING_PCA_COMPONENTS` | Target components when `dim_reduction_method="pca"`. |

---

## Dependencies

Update `pyproject.toml`:
- Add `umap-learn>=0.5.5` to `[project.dependencies]`.

---

## Test Plan

Updated / New tests in `tests/test_clustering.py`:

| Test Case | Objective |
|---|---|
| `test_dim_reducer_umap` | Verifies UMAP reduces 768-dim embeddings to `umap_n_components` (15) with output L2 normalization. |
| `test_dim_reducer_pca` | Verifies PCA reduction to `pca_components` (50) works accurately. |
| `test_dim_reducer_passthrough` | Verifies `dim_reduction_method="none"` returns raw 768-dim embeddings unchanged. |
| `test_dim_reducer_umap_fallback_small_corpus` | Verifies fallback when `n_docs <= n_neighbors` without crashing. |
| `test_clusterer_with_umap` | End-to-end HDBSCAN clustering execution using UMAP reduced vectors. |
| `test_manifest_dim_reduction_metadata` | Asserts `cluster_manifest.json` records correct reduction metadata. |

---

## Key Design Decisions

1. **UMAP min_dist=0.0 for Density Clustering**: Standard UMAP visualization uses `min_dist=0.1` to prevent points from clustering tightly. For HDBSCAN density clustering, `min_dist=0.0` is recommended by UMAP authors as it allows dense clusters to form tightly without artificial spacing.
2. **L2 Normalization Post-Reduction**: Normalizing reduced vectors ensures cosine distance is equivalent to Euclidean distance dot products, keeping HDBSCAN metrics fast and numerically stable.
3. **Graceful Fallbacks for Small Corpora**: In small unit test environments (`n_docs < 15`), UMAP cannot compute 15 neighbors. The reducer automatically adjusts `n_neighbors = min(n_neighbors, n_docs - 1)` or falls back cleanly to PCA / passthrough.
