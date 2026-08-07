# Feature Specification: Improving Teacher Evaluation with LLM-as-a-Judge

## Overview

This feature refactors the **Step 6 (Teacher Benchmarking)** evaluation system to rely on an **LLM-as-a-Judge** framework for evaluating candidate Teacher LLM responses. Rather than using hybrid deterministic Python heuristic scripts (regex string extraction for answer accuracy, character Jaccard overlap for citations, and DeBERTa-v3 cross-encoder NLI models for hallucination detection), all four evaluation dimensions are assessed using a unified, generalized LLM Judge rubric prompt. 

Crucially, the LLM Judge is required to output both **quantitative numerical scores** and **qualitative explanations** for every metric.

---

## Objectives

1. **Generalized Evaluation Rubric**: Design a comprehensive evaluation prompt and structured JSON schema for the LLM Judge.
2. **Four-Dimension Coverage**:
   - **Answer Accuracy**: Evaluate correctness against Ground Truth and options, providing a score $[0, 1]$ and explanation.
   - **Reasoning Quality**: Evaluate `step_validity` (1–5), `logical_coherence` (1–5), and `absence_of_circular_reasoning` (1–5), normalized to $[0, 1]$, with explanation.
   - **Citation Accuracy**: Evaluate precision, recall, and overall citation accuracy against context passages, with explanation.
   - **Hallucination Detection**: Evaluate trace claims against context/domain knowledge, providing a hallucination rate $[0, 1]$ and explanation listing any detected ungrounded claims.
3. **Single-Pass Judge Inference**: Consolidate multi-metric evaluation into a single LLM Judge API call per sample trace to minimize evaluation latency and API cost.
4. **Rich Failure & Calibration Logging**: Log explanations alongside numerical scores in failure logs (`logs/benchmarking/failed_traces.jsonl`) and judge logs (`logs/benchmarking/judge_calibration.jsonl`).

---

## Detailed Design & Architecture

### 1. LLM Judge Prompt & Schema (`metric_eval_judge.py`)

The LLM Judge prompt provides clear instructions and rubric definitions, requesting a raw JSON object response without markdown wrappers or thinking tags.

#### Prompt Template:
```text
[SYSTEM]: You are an expert scientific and neuroscience reviewer evaluating an AI model response trace.
Evaluate the model response on four core evaluation dimensions based on the provided Question, Ground Truth, and Context Passages (if available).

[QUESTION]: {question}
[GROUND TRUTH]: {ground_truth}
[CONTEXT PASSAGES]: {context_str}
[MODEL RESPONSE TRACE]: {trace}

EVALUATION RUBRIC:
1. answer_accuracy:
   - score (float 0.0 to 1.0): 1.0 if the final answer is completely correct and matches ground truth; 0.0 if completely incorrect; partial credit (e.g. 0.5) for partially correct responses.
   - explanation (string): Concise explanation of answer correctness.

2. reasoning_quality:
   - step_validity (int 1 to 5): Factual and logical soundness of individual reasoning steps.
   - logical_coherence (int 1 to 5): Smooth progression without gaps or leaps in logic.
   - absence_of_circular_reasoning (int 1 to 5): Freedom from circular logic (5=fully absent, 1=pervasive).
   - explanation (string): Concise explanation of reasoning quality.

3. citation_accuracy:
   - precision (float 0.0 to 1.0): Proportion of citations/references in trace that accurately reflect context.
   - recall (float 0.0 to 1.0): Proportion of key context facts used that are properly cited.
   - accuracy (float 0.0 to 1.0): Overall citation accuracy (F1 or harmonic mean of precision & recall).
   - explanation (string): Concise explanation of citation accuracy. (For NO CONTEXT items, set scores to 1.0 and explain 'No context provided').

4. hallucination:
   - rate (float 0.0 to 1.0): Proportion/severity of hallucinated or ungrounded claims (0.0=no hallucinations, 1.0=completely hallucinated).
   - explanation (string): Concise explanation of detected hallucinations or confirmation of factual accuracy.

CRITICAL: Do NOT output thinking, reasoning tags, or markdown formatting. Output ONLY raw JSON:
{
  "answer_accuracy": {"score": float, "explanation": "string"},
  "reasoning_quality": {"step_validity": int, "logical_coherence": int, "absence_of_circular_reasoning": int, "explanation": "string"},
  "citation_accuracy": {"precision": float, "recall": float, "accuracy": float, "explanation": "string"},
  "hallucination": {"rate": float, "explanation": "string"}
}
```

---

### 2. Output Schema & Score Parsing

The `_parse_response` method extracts and parses the JSON response:

```json
{
  "answer_accuracy": {
    "score": 1.0,
    "explanation": "The response correctly identifies GABA as the chief inhibitory neurotransmitter."
  },
  "reasoning_quality": {
    "step_validity": 5,
    "logical_coherence": 5,
    "absence_of_circular_reasoning": 5,
    "explanation": "Reasoning follows logical physiological pathways without circular arguments."
  },
  "citation_accuracy": {
    "precision": 1.0,
    "recall": 1.0,
    "accuracy": 1.0,
    "explanation": "All cited passages [Passage 1] match the provided text accurately."
  },
  "hallucination": {
    "rate": 0.0,
    "explanation": "No hallucinated or ungrounded assertions detected."
  }
}
```

Scores are derived as follows:
- **Answer Accuracy**: `data["answer_accuracy"]["score"]`
- **Reasoning Quality**: `mean(step_validity, logical_coherence, absence_of_circular_reasoning) / 5.0`
- **Citation Precision/Recall/Accuracy**: `data["citation_accuracy"]["precision"]`, `recall`, `accuracy` (or `None` for `no_retrieval`)
- **Hallucination Rate**: `data["hallucination"]["rate"]`

---

### 3. Pipeline Integration (`benchmark_runner.py` & `failure_logger.py`)

- `benchmark_runner.py`: Calls `ReasoningJudge.evaluate_trace(...)` which executes the generalized prompt and returns all 4 dimension scores and explanations.
- `failure_logger.py`: Updated to record judge explanations in failure logs when any metric falls below warning thresholds.
- `benchmark_reporter.py`: Aggregates the derived float scores into `data/benchmarking/scores.jsonl` and enforces validation gates.

---

## File Changes Overview

| File Path | Status | Summary of Changes |
| :--- | :---: | :--- |
| [metric_eval_judge.py](file:///e:/Projects/cnd/Semantics/lib/s6_teacher_benchmarking/metric_eval_judge.py) | **NEW / RENAME** | Implement generalized multi-metric judge prompt (`MetricEvalJudge`), JSON parser with explanation fields, and score derivation logic. |
| [benchmark_runner.py](file:///e:/Projects/cnd/Semantics/lib/s6_teacher_benchmarking/benchmark_runner.py) | **MODIFY** | Update trace scoring loop to use generalized LLM judge evaluations and pass metric explanations. |
| [failure_logger.py](file:///e:/Projects/cnd/Semantics/lib/s6_teacher_benchmarking/failure_logger.py) | **MODIFY** | Incorporate judge explanations into failure records for rich diagnostic logs. |
| [test_teacher_benchmarking.py](file:///e:/Projects/cnd/Semantics/tests/test_teacher_benchmarking.py) | **MODIFY** | Update unit tests to mock and verify generalized judge output and explanations. |
| [S6_TEACHER_BENCHMARKING.md](file:///e:/Projects/cnd/Semantics/docs/S6_TEACHER_BENCHMARKING.md) | **MODIFY** | Update Step 6 documentation reflecting LLM-as-a-Judge for all metrics. |

---

## Verification Plan

### Automated Tests
- Run full pytest test suite:
  ```bash
  pytest tests/test_teacher_benchmarking.py
  pytest
  ```
- Verify 100% test pass rate across unit tests.

### Manual / Integration Verification
- Execute dry-run benchmark test on sample data to confirm judge prompt formatting, JSON parsing, explanation logging, and score generation.
