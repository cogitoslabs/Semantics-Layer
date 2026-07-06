# Feature Spec: Phase 2, Step 2.1 — Teacher Benchmarking

## Objective

Benchmark every candidate teacher model on every micro-cluster across four dimensions: answer accuracy, reasoning quality, citation accuracy, and hallucination rate. Produce a per-teacher per-cluster score matrix that is consumed by Step 2.2 (multi-metric teacher election).

---

## Scope

This feature implements **Step 2.1** of Phase 2 (Teacher Election & Trace Harmonization). It sits after Micro-Clustering (`s4_clustering`) and before teacher election (`s5b_teacher_election`, future). It produces the score matrix that drives teacher election.

Included in this step:
1. Evaluation sample preparation — draw from cluster validation splits + retrieve context
2. Multi-teacher trace generation — generate fresh traces from each candidate teacher on the eval set
3. Scoring — measure each of the four benchmark dimensions per teacher per cluster
4. LLM-as-judge calibration protocol — calibrate reasoning judge against optional human spot-check labels
5. Score reporting — write per-teacher per-cluster manifest

Not included (separate steps):
- Teacher election (Step 2.2, `s5b_teacher_election`)
- Full trace generation for distillation (Step 2.3)
- Trace harmonization (Step 2.4)

---

## Module Layout

```
lib/
└── s5_teacher_benchmarking/
    ├── __init__.py                    # run_teacher_benchmarking(cfg) entry point
    ├── eval_sampler.py                # Draw eval samples from cluster validation splits + re-retrieve context
    ├── benchmark_runner.py            # Orchestrate: for each teacher × cluster → generate + score
    ├── answer_accuracy.py             # Extract \boxed{} answer; exact match / F1 token overlap
    ├── reasoning_judge.py             # LLM-as-judge: step validity, logical coherence, no circular reasoning
    ├── citation_accuracy.py           # Citation precision/recall vs. retrieved context
    ├── hallucination_detector.py      # NLI entailment + invented terminology detection
    └── benchmark_reporter.py          # Aggregate scores, calibration log, manifest
```

---

## Data Flow

```
data/clustering/splits.json
        ↓
eval_sampler.py    — draw val-split doc_ids per cluster → reload Q+GT from traces →
                     re-retrieve context via existing RAD index
        ↓
benchmark_runner.py — for each (candidate_teacher × cluster):
    TeacherBackend (reuses s3_rad_prep.trace_generator backends)
        → generate teacher traces on eval samples
        ↓
    answer_accuracy.py        → accuracy ∈ [0, 1]
    reasoning_judge.py        → reasoning_quality ∈ [0, 1]  (LLM-as-judge)
    citation_accuracy.py      → citation_accuracy ∈ [0, 1]  (grounded only)
    hallucination_detector.py → hallucination_rate ∈ [0, 1]
        ↓
benchmark_reporter.py  → per-teacher × cluster scores + aggregate manifest
        ↓
Output:
  data/benchmarking/scores.jsonl                        — one record per teacher×cluster
  data/benchmarking/benchmark_manifest.json             — stats + validation gate
  logs/benchmarking/judge_calibration.jsonl             — LLM judge outputs (calibration traces)
  logs/benchmarking/inter_rater_agreement.json          — Cohen's κ (if human labels provided)
```

---

## Component Specifications

### 1. `eval_sampler.py`

Reads `data/clustering/splits.json` to get per-cluster `val_doc_ids`. For each cluster, samples up to `cfg.benchmarking.eval_sample_size` (default 200) document IDs from the validation split.

Maps each sampled doc_id back to its corresponding trace record in `data/rad_prep/traces/grounded_traces.jsonl` and `no_retrieval_traces.jsonl` (joined on `sample_id == doc_id`). Extracts `question`, `answer`, `retrieved_context`, and `no_retrieval` flag — these are the shared evaluation samples used across all candidate teachers.

If a doc_id has no matching trace record, skip it and log the miss count. If total samples for a cluster fall below `cfg.benchmarking.min_eval_samples` (default 10) after skipping, log a warning but do not hard-fail.

Returns `Dict[cluster_label, List[EvalSample]]`:

```python
@dataclass
class EvalSample:
    sample_id: str
    cluster_id: str
    cluster_label: str
    question: str
    ground_truth: str
    retrieved_context: str       # empty string if no_retrieval
    no_retrieval: bool
```

### 2. `benchmark_runner.py`

Main orchestration loop. For each `teacher_name` in `cfg.benchmarking.candidate_teachers`:

1. Instantiate a teacher backend (reuse `LocalHFBackend`, `APIBackend`, or `BedrockBackend` from `lib.s3_rad_prep.trace_generator`). Teacher backend selection follows the same `RAD_TEACHER_BACKEND` pattern but is overridden by `BENCHMARK_TEACHER_BACKEND`.

2. For each cluster, call `backend.generate_batch(prompts)` on the eval samples (batched at `cfg.benchmarking.teacher_batch_size`). Prompts are built with `format_prompt()` from `s3_rad_prep.trace_generator` — same format as RAD prep to ensure consistency.

3. Pass each `(eval_sample, teacher_trace)` pair to the four scorers. Collect a `BenchmarkRecord` per sample.

4. Aggregate per-cluster scores (mean over samples).

```python
@dataclass
class BenchmarkRecord:
    teacher_model: str
    cluster_label: str
    cluster_id: int
    sample_id: str
    answer_accuracy: float
    reasoning_quality: float           # None if judge call failed
    citation_precision: Optional[float]
    citation_recall: Optional[float]
    citation_accuracy: Optional[float] # None for no-retrieval samples
    hallucination_rate: float
    no_retrieval: bool
    teacher_trace: str
    token_count: int

@dataclass
class ClusterScore:
    teacher_model: str
    cluster_label: str
    cluster_id: int
    eval_sample_size: int
    answer_accuracy: float
    reasoning_quality: float
    citation_accuracy: Optional[float]   # None if all samples were no-retrieval
    hallucination_rate: float
    no_retrieval_fraction: float
```

`ClusterScore` records are written to `data/benchmarking/scores.jsonl`, one per teacher×cluster pair.

### 3. `answer_accuracy.py`

Extracts the final answer from a teacher trace by locating the last `\boxed{...}` span using regex. If no `\boxed{}` is found, use the last sentence of the trace as the answer (fallback).

**Multiple-choice questions** (sample has `choices` list + `answer_idx`): normalize both extracted answer and ground truth (strip whitespace, lowercase, remove punctuation). Accuracy = 1.0 if match else 0.0.

**Free-form questions**: compute F1 token overlap (standard SQuAD-style):

```
precision = |overlap_tokens| / |predicted_tokens|
recall    = |overlap_tokens| / |reference_tokens|
accuracy  = F1(precision, recall)
```

Tokenization uses `.split()` (whitespace split), both strings lowercased and punctuation-stripped before tokenization.

Returns `accuracy ∈ [0, 1]`.

### 4. `reasoning_judge.py`

Uses a frontier judge model (separate from the teachers being evaluated) to score reasoning quality on three dimensions.

**Judge prompt format:**

```
[SYSTEM]: You are an expert neuroscience reviewer evaluating reasoning quality.
Rate the following reasoning trace on three dimensions, each from 1 to 5.

[QUESTION]: {question}
[CONTEXT]: {retrieved_context}
[TRACE]: {teacher_trace}

Scoring rubric:
- step_validity (1-5): Are individual reasoning steps factually and logically sound?
- logical_coherence (1-5): Do steps follow from each other without gaps or leaps?
- absence_of_circular_reasoning (1-5): Is circular reasoning absent? (5=fully absent, 1=pervasive)

Respond ONLY with valid JSON: {"step_validity": N, "logical_coherence": N, "absence_of_circular_reasoning": N}
```

Parse the JSON response. Average the three scores and normalize to [0, 1] by dividing by 5.

```
reasoning_quality = mean(step_validity, logical_coherence, absence_of_circular_reasoning) / 5
```

**On judge failures:** If the judge response cannot be parsed as valid JSON with the three required keys, retry once. If the second attempt also fails, set `reasoning_quality = None` and log the failure. Do not crash the batch.

**Calibration mode** (`cfg.benchmarking.enable_calibration = True`): Save the first `cfg.benchmarking.human_calibration_size` traces per cluster (with judge scores) to `logs/benchmarking/judge_calibration.jsonl` for human annotation. When `cfg.benchmarking.human_labels_path` is set and the file exists, load human labels and compute Cohen's κ between LLM and human scores (per dimension, rounded to integer bins 1-5). Write results to `logs/benchmarking/inter_rater_agreement.json`.

**Judge backends:** The judge uses the same `APIBackend` or `BedrockBackend` infrastructure as the teacher, but is configured independently via `BENCHMARK_JUDGE_*` env vars. The judge must not be the same model as any of the candidate teachers.

### 5. `citation_accuracy.py`

Only evaluated for retrieval-grounded traces (where `no_retrieval = False`).

**Citation extraction:** Find cited text in the teacher trace using two heuristics:
1. Quoted spans: `"..."` or `'...'` longer than 10 characters
2. Reference phrases: patterns like `"according to ..."`, `"as described in ..."`, `"the context states ..."`, followed by 1-3 sentences

**Scoring:** For each cited span, check overlap with retrieved context chunks using character-level Jaccard similarity ≥ `cfg.benchmarking.citation_min_overlap` (default: 0.30). A citation is "supported" if any context chunk meets the overlap threshold.

```
citation_precision = supported_citations / total_citations_made
citation_recall    = unique_context_chunks_cited / total_context_chunks
citation_accuracy  = F1(citation_precision, citation_recall)
```

If no citations are extracted from the trace, set `citation_precision = 0`, `citation_recall = 0`, `citation_accuracy = 0.0`.

If the sample has `no_retrieval = True`, set all citation fields to `None`.

Returns `(citation_precision, citation_recall, citation_accuracy)`.

### 6. `hallucination_detector.py`

Two-pass detection:

**Pass 1 — NLI entailment check:**

Split the teacher trace into sentences using `.split(".")` (simple heuristic; adequate for structured scientific traces). For each sentence:
1. Build a premise from: retrieved context (if available) + the ground truth answer
2. Use a small NLI cross-encoder (`cross-encoder/nli-deberta-v3-small`) to compute entailment probability for the claim (sentence)
3. Flag the sentence as hallucinated if `entailment_score < cfg.benchmarking.hallucination_nli_threshold` (default: 0.5)

**Pass 2 — Invented terminology check:**

Apply regex to detect tokens matching the pattern `[A-Z][a-z]+-[A-Z][a-z]+\s+(receptor|protein|cell|neuron|pathway|channel)` that are NOT present in the curated vocabulary at `evals/dapt/vocab_cloze_set.json`. Flag any matched terms not in the vocabulary as potential invented terminology.

```
hallucination_rate = (nli_flagged_sentences + invented_term_sentences) / total_sentences
hallucination_rate = min(hallucination_rate, 1.0)
```

Deduplicate — a sentence flagged by both passes counts as one.

Returns `hallucination_rate ∈ [0, 1]`.

### 7. `benchmark_reporter.py`

Reads all `ClusterScore` records and writes two outputs:

**`data/benchmarking/scores.jsonl`** — one JSON line per teacher×cluster pair:
```json
{
  "teacher_model": "Qwen/Qwen3-1.7B",
  "cluster_label": "cluster_007",
  "cluster_id": 7,
  "eval_sample_size": 200,
  "answer_accuracy": 0.72,
  "reasoning_quality": 0.81,
  "citation_precision": 0.68,
  "citation_recall": 0.54,
  "citation_accuracy": 0.60,
  "hallucination_rate": 0.08,
  "no_retrieval_fraction": 0.15
}
```

**`data/benchmarking/benchmark_manifest.json`:**
```json
{
  "status": "complete",
  "candidate_teachers": ["Qwen/Qwen3-1.7B", "deepseek-ai/DeepSeek-V3"],
  "total_clusters": 87,
  "eval_sample_size": 200,
  "judge_model": "...",
  "judge_calibrated": false,
  "per_teacher_aggregate": {
    "Qwen/Qwen3-1.7B": {
      "mean_answer_accuracy": 0.71,
      "mean_reasoning_quality": 0.79,
      "mean_citation_accuracy": 0.58,
      "mean_hallucination_rate": 0.10
    }
  },
  "warnings": ["reasoning_quality null rate > 5% for teacher X"]
}
```

---

## Config Extensions

Add `TeacherBenchmarkingConfig` to [lib/utils/config.py](lib/utils/config.py) and include as `benchmarking: TeacherBenchmarkingConfig` in `PipelineConfig`.

```python
@dataclass
class TeacherBenchmarkingConfig:
    # Candidate teachers (comma-separated model names)
    candidate_teachers: List[str]       # BENCHMARK_TEACHERS, default: ["Qwen/Qwen3-1.7B"]

    # Judge model (must differ from all candidate teachers)
    judge_backend: str                  # BENCHMARK_JUDGE_BACKEND, default: api (api|bedrock|hf_local)
    judge_model_name: str               # BENCHMARK_JUDGE_MODEL, default: ""
    judge_api_url: Optional[str]        # BENCHMARK_JUDGE_API_URL
    judge_api_key: Optional[str]        # BENCHMARK_JUDGE_API_KEY
    judge_max_new_tokens: int           # BENCHMARK_JUDGE_MAX_NEW_TOKENS, default: 256

    # Teacher generation (reuses RAD backends)
    teacher_backend: str                # BENCHMARK_TEACHER_BACKEND, default: inherits RAD_TEACHER_BACKEND
    teacher_batch_size: int             # BENCHMARK_TEACHER_BATCH_SIZE, default: 4

    # Evaluation sampling
    eval_sample_size: int               # BENCHMARK_EVAL_SAMPLE_SIZE, default: 200
    min_eval_samples: int               # BENCHMARK_MIN_EVAL_SAMPLES, default: 10

    # Calibration
    enable_calibration: bool            # BENCHMARK_ENABLE_CALIBRATION, default: False
    human_calibration_size: int         # BENCHMARK_CALIBRATION_SIZE, default: 200
    human_labels_path: Optional[Path]   # BENCHMARK_HUMAN_LABELS_PATH, default: None

    # Scoring thresholds
    hallucination_nli_threshold: float  # BENCHMARK_HALLUCINATION_NLI_THRESHOLD, default: 0.5
    citation_min_overlap: float         # BENCHMARK_CITATION_MIN_OVERLAP, default: 0.30
    nli_model: str                      # BENCHMARK_NLI_MODEL, default: cross-encoder/nli-deberta-v3-small

    # Output paths
    output_dir: Path                    # BENCHMARKING_OUTPUT_DIR, default: data/benchmarking
    scores_path: Path                   # BENCHMARKING_SCORES_PATH, default: data/benchmarking/scores.jsonl
    manifest_path: Path                 # BENCHMARKING_MANIFEST_PATH, default: data/benchmarking/benchmark_manifest.json
    calibration_log_path: Path          # BENCHMARKING_CALIBRATION_LOG, default: logs/benchmarking/judge_calibration.jsonl
    inter_rater_log_path: Path          # BENCHMARKING_INTER_RATER_LOG, default: logs/benchmarking/inter_rater_agreement.json
```

`candidate_teachers` is parsed from a comma-separated env var:
```python
field(default_factory=lambda: [t.strip() for t in _get("BENCHMARK_TEACHERS", "Qwen/Qwen3-1.7B").split(",")])
```

---

## Pipeline Integration

Add step `s5` to `pipeline.py` argparse choices:

```
--step choices: ["s1", "s1.5", "s2", "s3", "s4", "s5", "all"]
```

Add `s5()` function:

```python
def s5(cfg: PipelineConfig) -> None:
    print("Initializing Teacher Benchmarking (Phase 2, Step 2.1)")
    try:
        run_teacher_benchmarking(cfg)
    except Exception as e:
        print(f"Error executing teacher benchmarking: {e}", file=sys.stderr)
        sys.exit(1)
```

The `all` path runs: s1 → s1.5 → s2 → s3 → s4 → s5.

No sub-modes for s5; it always runs eval sampling → multi-teacher generation → scoring in sequence.

---

## Output Artifacts

| Artifact | Path | Phase Log Entry |
|---|---|---|
| Score records | `data/benchmarking/scores.jsonl` | n_records = teachers × clusters |
| Benchmark manifest | `data/benchmarking/benchmark_manifest.json` | per-teacher aggregates, warnings |
| Calibration traces | `logs/benchmarking/judge_calibration.jsonl` | if calibration enabled |
| Inter-rater agreement | `logs/benchmarking/inter_rater_agreement.json` | if human labels provided |

---

## Validation Gate

After benchmarking, the manifest records `status: complete` if:

| Check | Threshold | Action if failed |
|---|---|---|
| Score records written | = teachers × clusters | **Hard fail** |
| reasoning_quality null rate per teacher | ≤ 5% of samples | Warning — judge failures above threshold |
| Mean hallucination_rate per teacher | ≤ 0.50 | Warning only — election weight handles penalization |
| Eval sample coverage per cluster | ≥ min_eval_samples | Warning — low-coverage clusters noted in manifest |

`status: complete` requires zero hard failures. Warnings are recorded in `benchmark_manifest.json["warnings"]`.

---

## Dependencies

New packages to add to `pyproject.toml`:
- `sentence-transformers>=3.0` — already added in s3_rad_prep
- `scikit-learn>=1.3` — Cohen's κ via `sklearn.metrics.cohen_kappa_score`

New:
- `nltk>=3.8` — sentence tokenization fallback (punkt)

The NLI model (`cross-encoder/nli-deberta-v3-small`) is loaded via `sentence-transformers.CrossEncoder`. No additional package needed.

---

## Test Plan

File: `tests/test_teacher_benchmarking.py`

| Test | What it checks |
|---|---|
| `test_eval_sampler_draws_from_val_split` | Returns only val-split doc_ids, capped at eval_sample_size |
| `test_eval_sampler_skips_missing_doc_ids` | Missing trace records are skipped; miss count is logged |
| `test_eval_sampler_min_samples_warning` | Cluster with < min_eval_samples triggers warning, not hard fail |
| `test_answer_accuracy_multiple_choice` | Exact match on boxed answer vs. correct option |
| `test_answer_accuracy_free_form_f1` | F1 token overlap computed correctly for partial overlap |
| `test_answer_accuracy_no_boxed_fallback` | Trace without \boxed{} falls back to last sentence |
| `test_reasoning_judge_parses_valid_json` | Valid 3-key JSON → normalized score in [0, 1] |
| `test_reasoning_judge_retries_on_parse_failure` | Single parse failure → retry; second failure → None |
| `test_reasoning_judge_calibration_saves_traces` | Calibration mode writes to judge_calibration.jsonl |
| `test_citation_accuracy_supported` | Citation matching real context chunk → precision = 1.0 |
| `test_citation_accuracy_unsupported` | Citation not in context → precision = 0.0 |
| `test_citation_accuracy_no_retrieval` | no_retrieval=True sample → all citation fields None |
| `test_hallucination_nli_flags_unsupported_claim` | Claim with low entailment → flagged |
| `test_hallucination_invented_term_regex` | Token matching pattern but not in vocab → flagged |
| `test_hallucination_rate_clamped_to_one` | Rate cannot exceed 1.0 even with double-flagging |
| `test_benchmark_runner_produces_cluster_scores` | Full loop with mock backend; ClusterScore records written |
| `test_benchmark_reporter_manifest_complete` | Manifest has all required fields; status=complete on valid input |
| `test_benchmark_reporter_warning_on_null_rate` | >5% null reasoning_quality → warning in manifest |
| `test_pipeline_end_to_end` | Full s5 with minimal mock corpus; scores.jsonl and manifest exist |

---

## Key Design Decisions

1. **Fresh generation per candidate teacher.** The benchmark evaluates each teacher on the same shared evaluation set — it does not score existing RAD prep traces, which were generated by a single teacher and would bias evaluation toward that model. Each candidate teacher generates fresh outputs on identical prompts.

2. **Shared eval samples across all teachers.** All teachers see the same `(question, retrieved_context)` pairs, drawn from the cluster validation split. This eliminates question-level variance as a confound when comparing teachers.

3. **LLM-as-judge with explicit rubric.** At trace volumes of 300K–2M, human evaluation is not scalable. The three-dimension rubric (step validity, logical coherence, absence of circular reasoning) operationalizes "reasoning quality" into scorable criteria. JSON-format response enforcement reduces free-text parsing fragility.

4. **Calibration is optional but logged.** The judge produces scores immediately without requiring human labels. When `BENCHMARK_ENABLE_CALIBRATION=True`, it writes a calibration corpus for offline human spot-check. Cohen's κ is computed post-hoc when human labels are available. The gate never blocks on calibration status — it records it.

5. **NLI model size tradeoff.** `cross-encoder/nli-deberta-v3-small` is chosen over larger models (e.g., DeBERTa-v3-large) for runtime feasibility at scale. If hallucination detection quality is insufficient, swap to the large variant via `BENCHMARK_NLI_MODEL`.

6. **Citation accuracy is grounded-only.** No-retrieval traces have no context to cite. Setting citation fields to `None` rather than 0 avoids artificially penalizing no-retrieval performance in teacher election composites (Step 2.2 must handle `None` gracefully).

7. **Reuse of teacher backend infrastructure.** `LocalHFBackend`, `APIBackend`, and `BedrockBackend` from `lib.s3_rad_prep.trace_generator` are reused verbatim for candidate teacher generation. No duplication — benchmark runner imports and instantiates them directly with `cfg.benchmarking` overrides where needed.

8. **Eval samples from val split, not dev.** The dev split is reserved for training in Phase 3 (SFT). Using val-split samples for benchmarking keeps the splits clean and avoids any data contamination between teacher election and subsequent student training.
