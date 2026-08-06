# Feature Spec: Model Tracing Decorator (`model_trace`)

## Overview
Adds a `@model_trace` decorator in `lib/utils` and a configuration toggle `MODEL_TRACING` to log complete model inputs, generated outputs, and generation hyperparameters to a CSV trace file (`dapt_model_traces.csv`).

## Key Requirements
1. Add environment variable `MODEL_TRACING=False` and `MODEL_TRACE_FILE=logs/dapt_model_traces.csv` to `.env.common` and `.env.example`.
2. Add `model_tracing: bool` (default `False`) and `model_trace_file: Path` (default `logs/dapt_model_traces.csv`) to `LoggingConfig` in `lib/utils/config.py`.
3. Create `lib/utils/model_tracer.py` containing `@model_trace` decorator and export it from `lib/utils/__init__.py`.
4. Apply `@model_trace` to `generate_response` and `generate_responses_batch` in `concept_probe.py`, and `generate_topk_completions` and `generate_topk_completions_batch` in `cloze_probe.py`.
5. CSV Schema for Model Trace File:
   - `Timestamp`: ISO 8601 UTC timestamp
   - `Function`: Name of the decorated generation function
   - `Prompt`: Complete input prompt text or JSON list of prompts
   - `Output`: Complete generated response text or JSON list of completions
   - `Parameters`: JSON string of generation parameters (`max_new_tokens`, `k`, `temperature`, `repetition_penalty`, `device`, `batch_size`, `max_length`)

## Behavior
- If `MODEL_TRACING` is `False` (default), the decorator executes the underlying model generation function with zero file I/O or performance overhead.
- If `MODEL_TRACING` is `True`, the decorator logs each call to `model_trace_file`.
