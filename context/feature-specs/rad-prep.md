# Feature Spec: Step 0.3 — Retrieval-Augmented Distillation Preparation (RAD Prep)

## Objective

Build an indexed biomedical retrieval corpus and a grounded teacher trace generation pipeline. This is the primary architectural defense against hallucination: instead of training the student on parametric-only teacher outputs, traces are anchored to retrieved evidence from a domain-matched corpus.

---

## Scope

This feature implements **Step 0.3** of the pipeline, which sits between DAPT (`s2_dapt`) and Corpus Engineering (`s4_clustering`, future). It produces two outputs consumed by Phase 2 trace generation:

1. A searchable index over the biomedical retrieval corpus
2. A JSONL corpus of grounded teacher traces (Question + Context → Reasoning + Answer)

---

## Module Layout

```
lib/
└── s3_rad_prep/
    ├── __init__.py               # run_rad_prep_pipeline(cfg) entry point
    ├── chunker.py                # Document chunking with configurable size/overlap
    ├── indexer.py                # Build/save/load FAISS index from chunked corpus
    ├── retriever.py              # Dense, sparse, and hybrid retrieval + relevance gating
    ├── trace_generator.py        # Teacher trace generation with structured prompt format
    └── no_retrieval_router.py    # Routes samples below threshold to no-retrieval track
```

---

## Data Flow

```
Retrieval Corpus (JSONL)
        ↓
   chunker.py          — splits by doc_type: 512/64 tokens (long-form), 256/32 (abstracts)
        ↓
   indexer.py          — embeds chunks via biomedical dense model → FAISS index on disk
        ↓
   retriever.py        — for each Q: Top-K retrieval → cosine threshold → gated chunks
        ↓
no_retrieval_router.py — if <2 chunks pass → no-retrieval flag; log rate per cluster
        ↓
trace_generator.py     — format [CONTEXT]/[QUESTION]/[GROUND TRUTH] → call teacher → trace
        ↓
   Output: data/rad_prep/traces/grounded_traces.jsonl
           data/rad_prep/traces/no_retrieval_traces.jsonl
           logs/rad_prep/no_retrieval_rates.jsonl
```

---

## Component Specifications

### 1. `chunker.py`

Reads `data/rad_prep/retrieval_corpus.jsonl`. Each record must have `text` and `doc_type` fields (`long_form` | `abstract`). Yields `Chunk` objects.

```python
@dataclass
class Chunk:
    chunk_id: str    # "{doc_id}_{chunk_index}"
    doc_id: str
    doc_type: str    # "long_form" | "abstract"
    text: str
    token_count: int
```

Chunking parameters per `doc_type`:

| doc_type | chunk_tokens | overlap_tokens |
|---|---|---|
| long_form | 512 | 64 |
| abstract | 256 | 32 |

Tokenization uses the same tokenizer as the DAPT model (`cfg.model.base_model_name`) to ensure consistent token counts. Write chunks to `data/rad_prep/chunks.jsonl` before indexing.

### 2. `indexer.py`

Embeds all chunks and builds a FAISS `IndexFlatIP` (inner product, for cosine sim with normalized vectors).

**Supported embedding models** (selected via `RAD_EMBEDDING_MODEL` env var):

| Key | HF model ID | Strength |
|---|---|---|
| `biolinkbert` | `michiyasunaga/BioLinkBERT-large` | entity linking |
| `pubmedbert` | `microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract` | abstract retrieval |
| `hybrid` | BM25 + `biolinkbert` | best recall |

Index is saved to `cfg.rad.index_dir` (default: `data/rad_prep/index/`). Alongside the FAISS binary, save:
- `chunks_metadata.jsonl` — parallel list of chunk metadata (no text, just IDs and doc_type)
- `index_manifest.json` — model used, chunk count, token counts, build timestamp

Batched embedding with `cfg.rad.embed_batch_size` (default: 64).

### 3. `retriever.py`

Implements three retrieval modes:

**Dense retrieval** (BioLinkBERT or PubMedBERT):
- Embed query → cosine similarity against FAISS index → top-K results with scores

**Sparse retrieval** (BM25 via `rank_bm25`):
- BM25Okapi over tokenized chunk texts → top-K results with normalized scores

**Hybrid retrieval** (reciprocal rank fusion):
- Combine dense and sparse rankings: `score = 1/(k + rank_dense) + 1/(k + rank_sparse)` where `k=60`
- Re-rank by fused score, return top-K

Relevance threshold gate: after retrieval, filter chunks where `cosine_sim < cfg.rad.relevance_threshold` (default: 0.65). Return `RetrievalResult`:

```python
@dataclass
class RetrievalResult:
    chunks: List[Chunk]
    scores: List[float]
    passed_threshold: int   # chunks that met the threshold
    retrieval_mode: str
```

### 4. `no_retrieval_router.py`

If `result.passed_threshold < 2`, return `no_retrieval=True`. Route this sample to the no-retrieval track. Log the event per `cluster_id` (if provided):

```python
@dataclass
class RoutingDecision:
    sample_id: str
    cluster_id: Optional[str]
    no_retrieval: bool
    passed_chunks: int
    reason: str
```

Log rates to `logs/rad_prep/no_retrieval_rates.jsonl` after each batch. Flush aggregate stats at pipeline end.

### 5. `trace_generator.py`

**Input:** JSONL with fields `question`, `answer`, `cluster_id` (optional), `sample_id`. Source: existing Q&A probe data or a separately prepared sample set (path: `cfg.rad.qa_samples_path`).

**Teacher call interface:** Supports two backends via `RAD_TEACHER_BACKEND` env var:
- `hf_local` — loads teacher from `cfg.rad.teacher_model_name` via transformers
- `api` — calls an OpenAI-compatible endpoint at `cfg.rad.teacher_api_url` with `cfg.rad.teacher_api_key`

**Prompt format (retrieval-grounded):**
```
[SYSTEM]: You are a neuroscientist. Reason step-by-step using the provided context. Cite specific passages where applicable. Wrap your final answer inside \boxed{}.

[CONTEXT]: {retrieved_chunks_text}
[QUESTION]: {question}
[GROUND TRUTH]: {answer}
```

**Prompt format (no-retrieval):**
```
[SYSTEM]: You are a neuroscientist. Reason step-by-step using only your knowledge. Wrap your final answer inside \boxed{}.

[NO CONTEXT AVAILABLE]
[QUESTION]: {question}
[GROUND TRUTH]: {answer}
```

**Output record (JSONL):**
```json
{
  "sample_id": "...",
  "cluster_id": "...",
  "question": "...",
  "answer": "...",
  "retrieved_context": "...",
  "no_retrieval": false,
  "teacher_trace": "...",
  "token_count": 412,
  "teacher_model": "...",
  "embedding_model": "..."
}
```

Trace token count is validated at write time: discard (do not truncate) any trace with `token_count < 200 or token_count > 2500`. Log discarded traces to `logs/rad_prep/discarded_traces.jsonl` with reason.

---

## Config Extensions

Add `RADPrepConfig` to `lib/utils/config.py` and add it as `rad: RADPrepConfig` to `PipelineConfig`.

```python
@dataclass
class RADPrepConfig:
    # Corpus paths
    retrieval_corpus_path: Path    # RAD_CORPUS_PATH, default: data/rad_prep/retrieval_corpus.jsonl
    chunks_path: Path              # RAD_CHUNKS_PATH, default: data/rad_prep/chunks.jsonl
    index_dir: Path                # RAD_INDEX_DIR, default: data/rad_prep/index
    traces_dir: Path               # RAD_TRACES_DIR, default: data/rad_prep/traces
    qa_samples_path: Path          # RAD_QA_SAMPLES_PATH, default: evals/dapt/probe_qa.jsonl

    # Retrieval settings
    embedding_model: str           # RAD_EMBEDDING_MODEL, default: biolinkbert
    retrieval_mode: str            # RAD_RETRIEVAL_MODE, default: hybrid (dense|sparse|hybrid)
    top_k: int                     # RAD_TOP_K, default: 7
    relevance_threshold: float     # RAD_RELEVANCE_THRESHOLD, default: 0.65
    embed_batch_size: int          # RAD_EMBED_BATCH_SIZE, default: 64

    # Chunking
    long_form_chunk_tokens: int    # RAD_LONG_FORM_CHUNK_TOKENS, default: 512
    long_form_overlap_tokens: int  # RAD_LONG_FORM_OVERLAP_TOKENS, default: 64
    abstract_chunk_tokens: int     # RAD_ABSTRACT_CHUNK_TOKENS, default: 256
    abstract_overlap_tokens: int   # RAD_ABSTRACT_OVERLAP_TOKENS, default: 32

    # Teacher
    teacher_backend: str           # RAD_TEACHER_BACKEND, default: hf_local
    teacher_model_name: str        # RAD_TEACHER_MODEL_NAME, default: Qwen/Qwen3-1.7B
    teacher_api_url: Optional[str] # RAD_TEACHER_API_URL
    teacher_api_key: Optional[str] # RAD_TEACHER_API_KEY
    teacher_max_new_tokens: int    # RAD_TEACHER_MAX_NEW_TOKENS, default: 1024
    teacher_batch_size: int        # RAD_TEACHER_BATCH_SIZE, default: 4

    # Trace filtering
    trace_min_tokens: int          # RAD_TRACE_MIN_TOKENS, default: 200
    trace_max_tokens: int          # RAD_TRACE_MAX_TOKENS, default: 2500
```

---

## Pipeline Integration (`pipeline.py`)

Add step `s3` to the argparse choices:

```
--step choices: ["s1", "s1.5", "s2", "s3", "all"]
```

Add `s3()` function that calls `run_rad_prep_pipeline(cfg)`. The `all` path runs steps in order: s1 → s1.5 → s2 → s3.

The pipeline can be run in two sub-modes via `--rad-mode`:
- `index` — only build/rebuild the FAISS index (chunking + embedding)
- `traces` — only generate traces (requires existing index)
- `full` (default) — index then traces

---

## Output Artifacts

| Artifact | Path | Phase Log Entry |
|---|---|---|
| Chunks JSONL | `data/rad_prep/chunks.jsonl` | chunk count, token stats |
| FAISS index | `data/rad_prep/index/index.faiss` | embedding model, chunk count |
| Index manifest | `data/rad_prep/index/index_manifest.json` | model, timestamp, chunk count |
| Grounded traces | `data/rad_prep/traces/grounded_traces.jsonl` | trace count, token stats |
| No-retrieval traces | `data/rad_prep/traces/no_retrieval_traces.jsonl` | count, % of total |
| No-retrieval rates | `logs/rad_prep/no_retrieval_rates.jsonl` | per-cluster rates |
| Discarded traces | `logs/rad_prep/discarded_traces.jsonl` | count, reasons |
| Phase manifest | `logs/rad_prep/phase_manifest.json` | all above stats + pass/fail |

---

## Validation Gate

After trace generation, the pipeline logs a summary. No hard gate is defined at this step (gating is in Phase 2 trace harmonization), but the following must pass for the phase manifest to record `status: complete`:

- Grounded trace count ≥ `min(cfg.rad.min_traces, int(0.95 * total_attempted))` (default: 1000 or 95% of attempted evaluation samples, preventing false incomplete status on smaller sample sets)
- No-retrieval rate per cluster ≤ 30% (warn if exceeded; does not block — indicates indexing gap)
- Discarded trace rate ≤ 20% (warn if exceeded)

---

## Dependencies

New packages to add to `pyproject.toml`:
- `sentence-transformers>=3.0` — dense embeddings
- `faiss-cpu>=1.7` (or `faiss-gpu` if CUDA available at install time)
- `rank-bm25>=0.2` — BM25 sparse retrieval

No new runtime dependencies beyond what's in the existing DAPT stack (`transformers`, `torch`, `numpy`, `tqdm`).

---

## Test Plan

File: `tests/test_rad_prep.py`

| Test | What it checks |
|---|---|
| `test_chunker_long_form` | 512-token chunks with 64 overlap, correct `doc_type` |
| `test_chunker_abstract` | 256-token chunks with 32 overlap |
| `test_chunker_short_doc` | doc shorter than chunk size → single chunk, no error |
| `test_indexer_build_and_load` | builds index, saves manifest, reloads and retrieves |
| `test_dense_retrieval` | returns ≤ top_k results, scores in [0,1] |
| `test_relevance_threshold_gate` | samples below 0.65 filtered, passed_threshold count correct |
| `test_no_retrieval_router` | triggers when < 2 chunks pass |
| `test_hybrid_fusion` | hybrid scores differ from pure dense/sparse |
| `test_trace_generator_grounded` | [CONTEXT] field present, \boxed{} in output |
| `test_trace_generator_no_retrieval` | [NO CONTEXT AVAILABLE] field present |
| `test_trace_token_filter` | traces < 200 or > 2500 tokens are discarded, not truncated |
| `test_pipeline_end_to_end` | runs full s3 step with minimal mock corpus, checks artifacts exist |

---

## Key Design Decisions

1. **Hybrid retrieval by default.** BM25 handles exact neuroscience nomenclature queries (receptor subtypes, gene names); dense handles semantic similarity. Reciprocal rank fusion avoids needing to tune score normalization across the two modalities.

2. **FAISS `IndexFlatIP` with L2-normalized vectors** gives exact cosine similarity without approximation error. For the expected corpus size (textbooks + PMC = millions of chunks), exact search is feasible on CPU. Switch to `IndexIVFFlat` only if index exceeds 10M vectors.

3. **Discard, don't truncate, over-length traces.** A trace cut at token 2500 mid-reasoning teaches broken chain-of-thought. This follows the project-overview spec exactly.

4. **`[NO CONTEXT AVAILABLE]` as explicit token, not field omission.** Prevents model confusion at inference time between missing context and successful empty retrieval.

5. **Chunking uses DAPT tokenizer, not embedding model tokenizer.** Chunk sizes are defined in terms of the student model's token budget. This ensures chunks fit in the inference-time context window.

6. **No re-ranker in v1.** The spec lists re-ranking as "optional but recommended." First build validates retrieval quality with threshold gating alone. Re-ranker (e.g., `cross-encoder/ms-marco-MiniLM-L-6-v2`) is a follow-up item.

7. **Teacher backend abstraction.** The `hf_local` / `api` split lets the same pipeline run against a local Qwen model during development and an API-hosted teacher (DeepSeek, Nvidia) in production without code changes.
