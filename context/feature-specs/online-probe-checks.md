# Feature Specification: Online Probe Checks (Cloze & Concept)

## Overview
Interactive and CLI utility scripts to test and probe model generation behavior in real time for Cloze and Concept probes using the exact model, tokenizer, and generation settings configured in `.env.common` and `lib/utils/config.py`.

## Requirements
1. **Config & Model Loading**:
   - Load `.env.common` and associated environment variables via `PipelineConfig`.
   - Instantiate model and tokenizer using `lib.s3_dapt.model_utils.load_model_and_tokenizer`.
   - Automatic GPU/CPU device placement (`cuda` if available else `cpu`).

2. **Cloze Probe Check (`scripts/online_cloze_check.py`)**:
   - Input: Terminal prompt (interactive loop or `--prompt` argument).
   - Format: Converts prompt using `format_cloze_prompt` to append few-shot examples if applicable.
   - Generation: Uses `generate_topk_completions` with exact cloze probe parameters:
     - `k`: `cfg.probes.cloze_top_k` (default 5)
     - `max_new_tokens`: `cfg.probes.cloze_max_new_tokens` (default 3)
     - `max_length`: `cfg.probes.cloze_max_seq_len` (default 256)
     - `do_sample`: `False`
     - `repetition_penalty`: `1.1`
     - `eos_token_id`: stop token IDs (EOS + newline)

3. **Concept Probe Check (`scripts/online_concept_check.py`)**:
   - Input: Terminal prompt (interactive loop or `--prompt` argument).
   - Generation: Uses `generate_response` with exact concept probe parameters:
     - `max_new_tokens`: `cfg.probes.concept_max_new_tokens` (default 100)
     - `max_length`: `cfg.probes.concept_max_seq_len` (default 256)
     - `do_sample`: `False`
     - `temperature`: `1.0`
     - `repetition_penalty`: `1.1`
     - `eos_token_id`: stop token IDs (EOS)

4. **CLI Interface**:
   - Accepts optional `--prompt` / `-p` flag for single prompt generation.
   - Defaults to interactive terminal prompt (`quit` or `exit` to stop).
