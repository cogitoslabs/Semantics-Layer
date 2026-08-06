# Feature Spec: Hybrid Retrieval Refactoring with BGE-Large & Cross-Encoder Reranking

## Objective

Upgrade Step 4 (`s4_rad_prep`) retrieval architecture from single-stage Reciprocal Rank Fusion / linear score combination to a state-of-the-art **Two-Stage Hybrid Retrieval & Cross-Encoder Reranking** pipeline:
1. **Stage 1 (Candidate Retrieval)**: Dense Retrieval using **BGE-Large** (`BAAI/bge-large-en-v1.5`) combined with Sparse Retrieval (**BM25Okapi**).
2. **Stage 2 (Reranking)**: Heavyweight **Cross-Encoder Reranker** (`BAAI/bge-reranker-large`) scoring candidate query-chunk pairs jointly.
3. **Stage 3 (Relevance Gating & Routing)**: Dynamic score thresholding to filter low-relevance chunks and route low-context samples to the no-retrieval track.

---

## Background & Rationale

- **Dense Embedding Shift**: BioLinkBERT / PubMedBERT base models provide initial dense retrieval, but modern embedding benchmarks demonstrate that **BGE-Large** (`BAAI/bge-large-en-v1.5`) significantly outperforms previous domain embeddings on complex technical & scientific query retrieval.
- **Reranking vs Simple Fusion**: Reciprocal Rank Fusion (RRF) and linear score averaging ($0.5 \times \text{Dense} + 0.5 \times \text{Sparse}$) treat query and document representations independently (bi-encoder). A **Cross-Encoder Reranker** attends across all query tokens and chunk tokens simultaneously, computing deeper semantic interaction logits.

---

## Module Layout Changes

```
lib/
└── s4_rad_prep/
    ├── __init__.py               # Exports run_rad_prep_pipeline
    ├── chunker.py                # Document chunking
    ├── indexer.py                # FAISS indexing; updated with 'bge-large' model key support
    ├── reranker.py               # [NEW] CrossEncoderReranker class using HuggingFace / CrossEncoder
    ├── retriever.py              # Updated Retriever incorporating candidate pool fetch + Cross-Encoder reranking
    ├── trace_generator.py        # Prompt generation for grounded / no-retrieval traces
    └── no_retrieval_router.py    # Gating & no-retrieval routing
```

---

## Detailed Technical Specifications

### 1. `indexer.py` & `DenseEmbedder`
- Add support for `"bge-large"` mapping to `"BAAI/bge-large-en-v1.5"`.
- Retain full backward compatibility with existing dense embedding models: `"biolinkbert"` (`michiyasunaga/BioLinkBERT-large`), `"pubmedbert"` (`microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract`), or any arbitrary HuggingFace model ID / local path passed via configuration.
- Handle query embedding instruction prefixing for BGE models (e.g. prefixing query strings with `"Represent this sentence for searching relevant passages: "` when generating query embeddings).

### 2. `reranker.py` [NEW]
- Define `CrossEncoderReranker`:
  ```python
  class CrossEncoderReranker:
      def __init__(self, model_name: str, device: Optional[str] = None):
          ...
      def rerank(self, query: str, chunks: List[Chunk], top_k: int) -> tuple[List[Chunk], List[float]]:
          ...
  ```
- Uses `AutoModelForSequenceClassification` and `AutoTokenizer` (or `sentence_transformers.CrossEncoder`) on PyTorch with explicit device allocation (`cuda` / `cpu`).
- Evaluates `(query, chunk.text)` text pairs in batches (`reranker_batch_size`), computes relevance logits, applies sigmoid normalization to map scores to $[0, 1]$, and ranks candidates descending.

### 3. `retriever.py` Integration
- **Hybrid Retrieval Workflow**:
  1. Retrieve `rerank_candidate_k` (default: 50) candidates using RRF of Dense (e.g. `bge-large`, `biolinkbert`, `pubmedbert`, etc.) and Sparse (BM25).
  2. If `cfg.rad.use_reranker` is True:
     - Pass candidate chunks to `CrossEncoderReranker.rerank(query, candidate_chunks, top_k=cfg.rad.top_k)`.
     - Output top-$K$ chunks and their normalized reranker scores.
  3. Relevance Threshold Gate:
     - Filter chunks with score $< \text{cfg.rad.relevance\_threshold}$.
     - Return `RetrievalResult`.

### 4. Configuration Extensions (`RADPrepConfig` in `lib/utils/config.py` & `.env.common`)
Add fields to `RADPrepConfig` and `.env.common`:
- `RAD_EMBEDDING_MODEL`: `bge-large` | `biolinkbert` | `pubmedbert` | `<hf_model_id>` (default: `bge-large`)
- `RAD_USE_RERANKER`: `True` | `False` (default: `True`)
- `RAD_RERANKER_MODEL`: `BAAI/bge-reranker-large` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | `<hf_model_id>` (default: `BAAI/bge-reranker-large`)
- `RAD_RERANK_CANDIDATE_K`: `50` (number of top candidates passed from Stage 1 to Stage 2)
- `RAD_RERANKER_BATCH_SIZE`: `32`

Validation in `PipelineConfig.validate()`:
- `rerank_candidate_k >= top_k`
- `reranker_batch_size >= 1`


---

## Test Plan

Existing test suite `tests/test_rad_prep.py` will be updated and expanded:
1. `test_bge_large_dense_embedder`: Verify `DenseEmbedder` correctly initializes and embeds using `bge-large`.
2. `test_cross_encoder_reranker`: Mock `AutoModelForSequenceClassification` to verify pair tokenization, scoring, sigmoid scaling, and ranking order.
3. `test_hybrid_retrieval_with_reranker`: Test end-to-end first-stage candidate retrieval followed by second-stage cross-encoder reranking and threshold gating.
4. `test_reranker_disabled_fallback`: Verify fallback behavior when `use_reranker = False`.

---

## Key Design Decisions

1. **Two-Stage Architecture**: Decouples fast candidate retrieval (bi-encoder dense + BM25 sparse over thousands of chunks) from high-precision reranking (cross-encoder transformer over top-50 candidates).
2. **Sigmoid Logit Normalization**: Normalizes Cross-Encoder output logits via $\sigma(x)$ so that threshold gating (`relevance_threshold`, default 0.65) operates on a stable $[0, 1]$ scale.
3. **Graceful Fallback**: If `use_reranker` is set to `False` or if reranker model loading is bypassed, the system falls back seamlessly to Stage 1 RRF hybrid retrieval.
