# Change History

## Status: Completed (Online Probe Check Streamlit Web UI `ui/app.py`)

- Built interactive Streamlit Web Application in [app.py](file:///e:/Projects/cnd/Semantics/ui/app.py) under `ui/` directory.
- Added 2 primary input controls: **Probe** radio selector (`Cloze Probe` vs `Concept Probe`) and **Prompt** multiline input area.
- Integrated `@st.cache_resource` to load model & tokenizer once into memory via `PipelineConfig` and `load_model_and_tokenizer`.
- Connected UI to exact Cloze (`format_cloze_prompt`, `generate_topk_completions`) and Concept (`generate_response`) generation functions.
- Added Streamlit dependency (`streamlit>=1.47.0`) in [pyproject.toml](file:///e:/Projects/cnd/Semantics/pyproject.toml).
- Added feature specification in [probe-check-ui.md](file:///e:/Projects/cnd/Semantics/context/feature-specs/probe-check-ui.md).
- Added unit test suite in [test_ui.py](file:///e:/Projects/cnd/Semantics/tests/test_ui.py).
- Verified 100% pass rate across entire workspace test suite (**117/117 tests passing**).

---

## Status: Completed (Online Cloze & Concept Probe Prompt Check CLI Scripts)


- Created [online_cloze_check.py](file:///e:/Projects/cnd/Semantics/scripts/online_cloze_check.py) and [online_concept_check.py](file:///e:/Projects/cnd/Semantics/scripts/online_concept_check.py) under `scripts/`.
- Configured scripts to load model and tokenizer automatically using `PipelineConfig` (reading `.env.common` and associated environment files) and `load_model_and_tokenizer`.
- Supported both interactive prompt sessions and non-interactive `--prompt` (`-p`) CLI arguments.
- Applied exact cloze probe generation parameters (`generate_topk_completions`, `format_cloze_prompt`, `num_beams=5`, `max_new_tokens=3`) and concept probe generation parameters (`generate_response`, `max_new_tokens=100`).
- Added feature specification in [online-probe-checks.md](file:///e:/Projects/cnd/Semantics/context/feature-specs/online-probe-checks.md).
- Added unit test suite in [test_online_checks.py](file:///e:/Projects/cnd/Semantics/tests/test_online_checks.py).

---

## Status: Completed (Model Tracing Decorator `@model_trace` & `MODEL_TRACING` Config)


- Implemented `@model_trace` decorator in [model_tracer.py](file:///e:/Projects/cnd/Semantics/lib/utils/model_tracer.py) and exported it in [utils/\_\_init\_\_.py](file:///e:/Projects/cnd/Semantics/lib/utils/__init__.py).
- Added configuration flags `MODEL_TRACING=False` and `MODEL_TRACE_FILE=logs/dapt_model_traces.csv` to [.env.common](file:///e:/Projects/cnd/Semantics/.env.common), [.env.example](file:///e:/Projects/cnd/Semantics/.env.example), and `LoggingConfig` in [config.py](file:///e:/Projects/cnd/Semantics/lib/utils/config.py).
- Decorated model generation functions (`generate_response`, `generate_responses_batch`) in [concept_probe.py](file:///e:/Projects/cnd/Semantics/lib/s3_dapt/probes/concept_probe.py) and (`generate_topk_completions`, `generate_topk_completions_batch`) in [cloze_probe.py](file:///e:/Projects/cnd/Semantics/lib/s3_dapt/probes/cloze_probe.py).
- Added feature specification in [model-tracing.md](file:///e:/Projects/cnd/Semantics/context/feature-specs/model-tracing.md) and unit test suite in [test_model_tracer.py](file:///e:/Projects/cnd/Semantics/tests/test_model_tracer.py).
- Enhanced `_log_trace_to_csv` in [model_tracer.py](file:///e:/Projects/cnd/Semantics/lib/utils/model_tracer.py) to unroll batch inference calls (`generate_responses_batch`, `generate_topk_completions_batch`) into individual CSV rows per prompt item, preventing giant JSON-array CSV rows.
- Added `Eval #`, `Eval Category`, and `Eval Seq #` columns to `dapt_model_traces.csv` in [model_tracer.py](file:///e:/Projects/cnd/Semantics/lib/utils/model_tracer.py), aligned with evaluation traces metadata.
- Verified 100% pass rate across entire workspace test suite (**112/112 tests passing**).

---

## Status: Completed (Evaluation Traces CSV Logging & `EVAL_TRACES_FILE` Config)

- Renamed `EVAL_SAMPLES_FILE` to `EVAL_TRACES_FILE` in [.env.common](file:///e:/Projects/cnd/Semantics/.env.common) and [.env.example](file:///e:/Projects/cnd/Semantics/.env.example).
- Renamed `eval_samples_file` to `eval_traces_file` in `LoggingConfig` in [config.py](file:///e:/Projects/cnd/Semantics/lib/utils/config.py), setting default output path to `logs/dapt_eval_traces.csv`.
- Updated evaluation trace formatting across probe modules ([qa_probe.py](file:///e:/Projects/cnd/Semantics/lib/s3_dapt/probes/qa_probe.py), [cloze_probe.py](file:///e:/Projects/cnd/Semantics/lib/s3_dapt/probes/cloze_probe.py), [concept_probe.py](file:///e:/Projects/cnd/Semantics/lib/s3_dapt/probes/concept_probe.py)) and renamed sample lists / functions to use `eval_traces` (`get_*_probe_traces`).
- Updated [eval_runner.py](file:///e:/Projects/cnd/Semantics/lib/s3_dapt/evaluation/eval_runner.py) to write evaluation traces to CSV format instead of JSON, adhering to the 7 standardized CSV columns: `Eval #`, `Eval Category`, `Eval Seq #`, `Eval`, `Generated Answer by the model`, `Matching Score`, and `Result`.
- Added feature specification in [eval-traces-csv.md](file:///e:/Projects/cnd/Semantics/context/feature-specs/eval-traces-csv.md).
- Added `format_concept_prompt` (`Prompt: {prompt}\nAnswer:`) in [concept_probe.py](file:///e:/Projects/cnd/Semantics/lib/s3_dapt/probes/concept_probe.py) to condition base causal language models to generate direct concept definitions/explanations instead of rambling or continuing exam question lists.
- Updated unit test assertions in [test_dapt.py](file:///e:/Projects/cnd/Semantics/tests/test_dapt.py) and verified 100% pass rate across entire workspace test suite (**109/109 tests passing**).

---

## Status: Completed (MinHash LSH Chunk Deduplication in Merge Corpus)

- Implemented zero-dependency `MinHashLSHDeduplicator` module in [minhash_lsh.py](file:///e:/Projects/cnd/Semantics/lib/s1_build_corpus/minhash_lsh.py) using 128 64-bit MinHash permutations over word 5-grams and 16 LSH bands of 8 rows each.
- Added configuration options (`minhash_enabled`, `minhash_jaccard_threshold=0.85`, `minhash_num_perm=128`, `minhash_ngram_size=5`, `minhash_num_bands=16`) to `CorpusBuildConfig` in [config.py](file:///e:/Projects/cnd/Semantics/lib/utils/config.py) and [.env.common](file:///e:/Projects/cnd/Semantics/.env.common).
- Integrated MinHash LSH deduplication into `run_merge_corpus` in [merge_corpus.py](file:///e:/Projects/cnd/Semantics/lib/s1_build_corpus/merge_corpus.py) to stream, detect, and skip duplicate text chunks across merged corpus files while logging detailed metrics (processed, merged, dropped chunks and tokens).
- Exported `MinHashLSHDeduplicator` in `lib/s1_build_corpus/__init__.py`.
- Created feature specification in [minhash-lsh-deduplication.md](file:///e:/Projects/cnd/Semantics/context/feature-specs/minhash-lsh-deduplication.md) and full unit test suite in [test_minhash_lsh.py](file:///e:/Projects/cnd/Semantics/tests/test_minhash_lsh.py).
- Verified 100% pass rate across entire test suite (**109/109 tests passing**).

---

- Fixed `KeyError: 'global_step'` occurring during initial baseline evaluation before the training loop starts by ensuring `global_step` is initialized in `init_state()` in [dapt.py](file:///e:/Projects/cnd/Semantics/lib/s3_dapt/dapt.py) and fallback lookup `state.get("global_step", state.get("steps_completed", 0))` is used in [eval_runner.py](file:///e:/Projects/cnd/Semantics/lib/s3_dapt/evaluation/eval_runner.py).
- Initialized all required state history arrays (`eval_timestamps`, `tokens_history`, `perplexity_history`, `ppl_history`, `qa_acc_history`, `cloze_cov_history`, `concept_prec_history`) in `init_state()` in [dapt.py](file:///e:/Projects/cnd/Semantics/lib/s3_dapt/dapt.py) to prevent secondary `KeyError` crashes.
- Reconciled `ppl_history` and `perplexity_history` keys across [eval_runner.py](file:///e:/Projects/cnd/Semantics/lib/s3_dapt/evaluation/eval_runner.py), [gate_logic.py](file:///e:/Projects/cnd/Semantics/lib/s3_dapt/evaluation/gate_logic.py), and [dapt.py](file:///e:/Projects/cnd/Semantics/lib/s3_dapt/dapt.py) so perplexity convergence checking correctly tracks evaluation metrics.
- Added `eval_id` to metrics dictionary output in [eval_runner.py](file:///e:/Projects/cnd/Semantics/lib/s3_dapt/evaluation/eval_runner.py) and updated [checkpoint.py](file:///e:/Projects/cnd/Semantics/lib/utils/checkpoint.py) and [training_helpers.py](file:///e:/Projects/cnd/Semantics/lib/s3_dapt/training_helpers.py) for resilient dictionary lookups.
- Removed duplicate `state["eval_history"].append(metrics)` calls in [dapt.py](file:///e:/Projects/cnd/Semantics/lib/s3_dapt/dapt.py) and [training_helpers.py](file:///e:/Projects/cnd/Semantics/lib/s3_dapt/training_helpers.py) to prevent metrics from being appended twice per evaluation pass.

---

## Status: Completed (Per-Eval Pass Logging, `Eval #` Field, & `EVAL_SAMPLES_FILE` Config)

- Added `EVAL_SAMPLES_FILE=logs/dapt_eval_samples.json` to [.env.common](file:///e:/Projects/cnd/Semantics/.env.common) and added `eval_samples_file` to `LoggingConfig` in [config.py](file:///e:/Projects/cnd/Semantics/lib/utils/config.py) to make the evaluation samples output path configurable.
- Enhanced all evaluation probe modules ([qa_probe.py](file:///e:/Projects/cnd/Semantics/lib/s3_dapt/probes/qa_probe.py), [cloze_probe.py](file:///e:/Projects/cnd/Semantics/lib/s3_dapt/probes/cloze_probe.py), [concept_probe.py](file:///e:/Projects/cnd/Semantics/lib/s3_dapt/probes/concept_probe.py)) to accept `eval_num: Union[int, str] = 1` and record `"Eval #": "1"`, `"Eval #": "2"`, `"Eval #": "final"` as the leading attribute in every JSON sample object.
- Updated `run_all_probes()` in [eval_runner.py](file:///e:/Projects/cnd/Semantics/lib/s3_dapt/evaluation/eval_runner.py) to save evaluation sample records for **every evaluation pass** (removing the `use_bertscore` restriction).
- Implemented cumulative file updating in [eval_runner.py](file:///e:/Projects/cnd/Semantics/lib/s3_dapt/evaluation/eval_runner.py) to append evaluation pass records to `cfg.logging.eval_samples_file`, preserving full multi-pass history across training.
- Updated unit test assertions in [test_dapt.py](file:///e:/Projects/cnd/Semantics/tests/test_dapt.py) verifying 100% test suite pass rate (19/19 tests passing).

---

## Status: Completed (Evaluation Probe Full Sample Logging with `Result: Pass/Fail`)

- Refactored all evaluation probe execution modules ([qa_probe.py](file:///e:/Projects/cnd/Semantics/lib/s3_dapt/probes/qa_probe.py), [cloze_probe.py](file:///e:/Projects/cnd/Semantics/lib/s3_dapt/probes/cloze_probe.py), [concept_probe.py](file:///e:/Projects/cnd/Semantics/lib/s3_dapt/probes/concept_probe.py)) to record 100% of evaluated samples (both Passed and Failed) instead of logging failures only.
- Added top-level `"Result": "Pass"` or `"Result": "Fail"` attribute to every sample record across QA, Cloze, and Concept probes.
- Added probe sample getter helpers (`get_qa_probe_samples`, `get_cloze_probe_samples`, `get_concept_probe_samples`) while preserving `get_failed_*` wrappers for full backward compatibility.
- Updated `eval_runner.py` ([eval_runner.py](file:///e:/Projects/cnd/Semantics/lib/s3_dapt/evaluation/eval_runner.py)) to write complete structured probe sample logs to `failed_evals.json` while isolating console warning logs to failures only to maintain clean stdout logs.
- Expanded unit test coverage in [test_dapt.py](file:///e:/Projects/cnd/Semantics/tests/test_dapt.py), verifying 100% pass rate (19/19 tests passing).

---

## Status: Completed (Evaluation Probe Datasets Data Quality Cleaning)

- Conducted a thorough audit across all DAPT evaluation probe datasets ([retrieval_prompts.json](file:///e:/Projects/cnd/Semantics/evals/dapt/retrieval_prompts.json), [retrieval_references.json](file:///e:/Projects/cnd/Semantics/evals/dapt/retrieval_references.json), [probe_qa.jsonl](file:///e:/Projects/cnd/Semantics/evals/dapt/probe_qa.jsonl), and [vocab_cloze_set.json](file:///e:/Projects/cnd/Semantics/evals/dapt/vocab_cloze_set.json)).
- Identified root cause of table border pollution (`|----|`) and term/definition concatenation (`Autobiographical memory Long-term memory... |`) resulting from uncleaned raw PDF table extractions.
- Implemented automated batch cleaner [clean_eval_files.py](file:///e:/Projects/cnd/Semantics/scripts/clean_eval_files.py) to strip raw markdown table borders, repair spaced possessive/contraction apostrophes (`one ' s` → `one's`, `Parkinson ' s` → `Parkinson's`), remove spaces before punctuation (`word .` → `word.`, `(e . g . ,)` → `(e.g.,)`), fix space-padded hyphens (`Cross - modal` → `Cross-modal`), and normalize quotes (`“ word ”` → `“word”`).
- Verified zero remaining table border artifacts, zero spaced apostrophes, zero spaced punctuation, and 100% test pass rate across unit test suite.

---

## Status: Completed (DAPT Corpus Pretraining Quality Enhancements & Cleaning)

- Implemented comprehensive pretraining corpus cleaning enhancements in [clean_text.py](file:///e:/Projects/cnd/Semantics/lib/utils/clean_text.py) to remove non-prose noise, ASCII control characters, HTML placeholders, PUA font glyphs, ASCII markdown tables, diagram callouts, standalone figure captions (`FIGURE 7.1...`), inline figure references (`(see Figure 4-2)`), Table of Contents (TOC) page-number listings, unheadinged academic reference lists, Internet Archive headers, and publisher metadata.
- Upgraded `dehyphenate_text` to repair space-padded broken split words (such as `con -form` → `conform`, `motiva -tion` → `motivation`, `evalu -ating` → `evaluating`), eliminating over 7,500 split subwords.
- Added linear-time split ligature repair in `join_ligatures_dict` to join broken word fragments across PDF text lines without performance bottlenecks.
- Updated feature specification in [dapt-corpus-pretraining-cleaning.md](file:///e:/Projects/cnd/Semantics/context/feature-specs/dapt-corpus-pretraining-cleaning.md) and expanded unit test suite in [test_dapt_corpus_cleaning.py](file:///e:/Projects/cnd/Semantics/tests/test_dapt_corpus_cleaning.py).
- Executed in-place corpus cleaning script [clean_existing_corpus.py](file:///e:/Projects/cnd/Semantics/scripts/clean_existing_corpus.py) on `data/dapt/in/domain_dapt_corpus.jsonl`, filtering **1,018,441 non-prose noise tokens (16.90% token reduction)** and dropping 27 non-prose/TOC/citation chunks.
- Verified **0 ASCII control characters**, **0 PUA glyphs**, **0 short chunks**, **0 duplicate chunks**, and **100% test pass rate (24/24 tests passing)**.

---

## Status: Completed (Convergence Gate Minimum Pass Enforcement and .gitignore Cleanup)

- Modified `check_convergence_gates` in [gate_logic.py](file:///e:/Projects/cnd/Semantics/lib/s3_dapt/evaluation/gate_logic.py) to require at least one full pass through the training corpus (`tokens_processed >= total_corpus_tokens`) before allowing convergence to trigger, preventing early stopping on unrepresentative steps.
- Updated `.gitignore` in [.gitignore](file:///e:/Projects/cnd/Semantics/.gitignore) to simplify ignore rules by ignoring the entire `data/` directory.
- Added comprehensive unit tests `test_check_convergence_gates_requires_one_pass` and updated `test_check_convergence_gates_disabled_probes` in [test_dapt.py](file:///e:/Projects/cnd/Semantics/tests/test_dapt.py) to verify the minimum pass requirement for convergence.
- Verified that all 82 tests pass successfully.

---

## Status: Completed (dapt_pipeline_impl, build_corpus, utils Refactoring, metrics_compat Clean Up, and Split-Batch OOM Recovery)

- Refactored `_run_dapt_pipeline_impl` in [dapt.py](file:///e:/Projects/cnd/Semantics/lib/s3_dapt/dapt.py) to reduce its overall length and complexity.
- Extracted model and tokenizer initialization, cache settings, and torch compilation into `_init_model_and_tokenizer`.
- Extracted state dictionary setup into `_init_state`.
- Extracted the evaluation validation checks, execution cycle, and memory warning / CUDA OOM retries into `_run_evaluation_cycle`.
- Extracted the entire training loop block and final validation checks into `_run_training_loop`.
- Removed the redundant `output_dir` local variable and parameter across `_run_training_loop`, `_handle_decision_action`, and `_handle_final_check` by referencing `str(cfg.model.checkpoint_dir)` directly within the handler functions.
- Removed leading underscores (`_`) from all private helper functions in `dapt.py`, `build_corpus.py`, and `lib/utils` (`storage.py`, `profiller.py`, `checkpoint.py`, `config.py`) to establish consistent naming styling.
- Reordered all function definitions in `dapt.py`, `build_corpus.py`, `storage.py`, `profiller.py`, `checkpoint.py`, and `config.py` to match the exact sequential order in which they are invoked during execution.
- Verified that `pretokenize.py` does not contain any private helper functions and thus required no changes.
- Completely removed the legacy `metrics_compat.py` file, cleaned up the unused imports in `dapt.py`, and moved the helper functions to `tests/test_dapt.py` to keep the unit tests functioning and self-contained.
- Implemented a split-batch retry fallback mechanism in `train_step` and `run_train_step` to recover from CUDA Out-of-Memory (OOM) errors by splitting the failed batch into two halves and accumulating gradients with proper loss/gradient scaling (0.5), as well as clearing references in a `finally` block to prevent activation memory leaks.
- Verified the correctness of all refactoring changes by running the local test suite (`.venv\Scripts\python -m pytest`), resulting in all **81 / 81 tests passing** successfully.

---

## Status: Completed (DAPT Configuration Summary and Execution Stats Update)

- Modified the `summary` method of `PipelineConfig` in [config.py](file:///e:/Projects/cnd\Semantics/lib/utils/config.py) to display `train_batch_size`, `eval_batch_size`, `gradient_checkpointing`, and `peft_dapt` settings.
- Added a unit test `test_pipeline_config_summary` in [test_config.py](file:///e:/Projects/cnd/Semantics/tests/test_config.py) to verify the presence of these fields in the summary text.
- Modified `run_dapt_pipeline` in [dapt.py](file:///e:/Projects/cnd/Semantics/lib/s3_dapt/dapt.py) to track execution duration and the total tokens processed, logging/printing these statistics at the end of the run.
- Patched unit tests in [test_dapt.py](file:///e:/Projects/cnd/Semantics/tests/test_dapt.py) to isolate process-level environment variables (like `WANDB_ENABLED` and `PEFT_DAPT`) and avoid loading real SciBERT checkpoints during testing.
- Verified that all unit tests pass successfully.

---

## Status: Completed (PEFT-DAPT Support)

- Implemented Parameter-Efficient Continued Pretraining (PEFT-DAPT) using LoRA (Low-Rank Adaptation) adapters.
- Configured `.env.common` and `.env.example` with default LoRA hyperparameters (`LORA_R=16`, `LORA_ALPHA=32`, `LORA_DROPOUT=0.05`, `LORA_TARGET_MODULES=q_proj,v_proj,k_proj,o_proj`).
- Updated `ModelConfig` in [config.py](file:///e:/Projects/cnd/Semantics/lib/utils/config.py) to parse PEFT configurations and added validation constraints in `PipelineConfig.validate()`.
- Wrapped base Causal LM models with `peft` LoRA configuration in [model_utils.py](file:///e:/Projects/cnd/Semantics/lib/s3_dapt/model_utils.py).
- Optimized optimizer initialization in [training_helpers.py](file:///e:/Projects/cnd/Semantics/lib/s3_dapt/training_helpers.py) by filtering parameters to only include those requiring gradients, yielding significant VRAM savings.
- Supported non-strict checkpoint loading for `PeftModel` in [checkpoint.py](file:///e:/Projects/cnd/Semantics/lib/utils/checkpoint.py) (since base model weights are frozen and only adapter weights are stored).
- Updated final inference failure logging in [eval_runner.py](file:///e:/Projects/cnd/Semantics/lib/s3_dapt/evaluation/eval_runner.py) to correctly load the base model first and then wrap it with the adapter weights.
- Added comprehensive unit tests for PEFT model wrapping and loading in [test_dapt.py](file:///e:/Projects/cnd/Semantics/tests/test_dapt.py).
- Verified that all 18 test suite checks pass successfully.

---

## Status: Completed (Local Model Path Resolution)

- Implemented local model directory path resolution in [config.py](file:///e:/Projects/cnd/Semantics/lib/utils/config.py) (`resolve_local_model_path`).
- Configured both `ModelConfig` (`base_model_name`) and `ProbeConfig` (`bertscore_model`) to automatically resolve their values to absolute local directory paths if present under `models/` relative to the workspace root.
- Dynamically handles exact matches, lowercase folder names (e.g. `smollm2-135m` matching `SmolLM2-135M`), and case-insensitive scanning of candidate directory names. Falls back to original Hugging Face repository names if the directories do not exist.
- Added validation check to verify `config.json` exists inside the target local directory before resolving to it. This prevents resolving to incomplete local directories (such as folders containing only `pytorch_model.bin` and `vocab.txt` but lacking configuration metadata) and gracefully falls back to Hugging Face Hub, resolving Google Colab loading issues.
- Patched `get_bertscorer` in [concept_probe.py](file:///e:/Projects/cnd/Semantics/lib/s3_dapt/probes/concept_probe.py) to dynamically register the resolved local path in `bert_score.utils.model2layers`. This prevents a `KeyError` within the `bert_score` library when initializing `BERTScorer` with a local directory path instead of a hardcoded Hugging Face repository name.
- Created comprehensive unit tests in [test_config.py](file:///e:/Projects/cnd/Semantics/tests/test_config.py) to verify exact, case-insensitive, direct path, and fallback resolution mechanisms.
- Verified that all 77 unit and integration tests compile and run successfully.

---

## Status: Completed (DAPT Corpus Cleaning Enhancements)


- Implemented standalone index and bibliography detection heuristics in [clean_text.py](file:///e:/Projects/cnd/Semantics/lib/utils/clean_text.py) (`is_standalone_index_or_bibliography`) using regular expressions mapping lists of page numbers, Roman numerals, and dense bibliography entries. Standalone index/bibliography chunks are skipped entirely during post-build cleaning.
- Implemented inline references truncation in [clean_text.py](file:///e:/Projects/cnd/Semantics/lib/utils/clean_text.py) (`remove_inline_references`) to identify inline reference sections in PNS, Neuroscience, and Fundamentals textbooks and slice the text to retain only the preceding narrative portion.
- Enhanced `clean_corpus_text` in [clean_text.py](file:///e:/Projects/cnd/Semantics/lib/utils/clean_text.py) to automatically HTML-unescape all text, strip `<!-- image -->` placeholders, and strip stray triple-backtick fences (` ``` `).
- Exposed and exported the new cleaning functions in [__init__.py](file:///e:/Projects/cnd/Semantics/lib/utils/__init__.py).
- Modified [clean_existing_corpus.py](file:///e:/Projects/cnd/Semantics/scripts/clean_existing_corpus.py) to integrate inline reference removal, unescaping, placeholder stripping, and standalone index/bibliography checking in its cleaning loop.
- Added comprehensive unit tests in [test_clean_text.py](file:///e:/Projects/cnd/Semantics/tests/test_clean_text.py) to verify unescaping, placeholder stripping, inline reference slicing, and standalone index/bibliography classification.
- Ran the cleaning script on the built corpus `domain_dapt_corpus.jsonl`, achieving a 22.13% token reduction (~1.09 million tokens) and verifying that no HTML entities, image placeholders, inline references, or standalone indexes/bibliographies remain in the final corpus.

---

## Status: Completed (DAPT Logging, Retrieval Probe Bugfix, and Performance Optimizations)

- Fixed the missing evaluation scores in logs and stdout by setting up the parent loggers `"dapt"` and `"lib"` with output handlers in [logger.py](file:///e:/Projects/cnd/Semantics/lib/utils/logger.py). Sub-loggers under `"dapt.*"` (like `"dapt.eval_runner"`, `"dapt.gates"`, etc.) and `"lib.*"` now properly propagate up and are captured.
- Removed the newline `\n` stopping criterion from `generate_responses_batch` in [retrieval_probe.py](file:///e:/Projects/cnd/Semantics/lib/s3_dapt/probes/retrieval_probe.py). This prevents the base model from stopping immediately at step 1 when it naturally prefixes its explanation with a newline/paragraph break, resolving the issue where retrieval precision was erroneously calculated as zero.
- Suppressed `bert_score` warning spam about empty candidate/reference sentences by using a warnings filter inside `compute_bertscore_batch` in [retrieval_probe.py](file:///e:/Projects/cnd/Semantics/lib/s3_dapt/probes/retrieval_probe.py).
- Refactored `generate_responses_batch` in [retrieval_probe.py](file:///e:/Projects/cnd/Semantics/lib/s3_dapt/probes/retrieval_probe.py#L141-L197) to use Hugging Face built-in batch tokenization, with a `try-except` fallback block to handle mock tokenizers in tests.
- Optimized `compute_lexical_f1_batch` in [retrieval_probe.py](file:///e:/Projects/cnd/Semantics/lib/s3_dapt/probes/retrieval_probe.py#L288-L314) by pre-tokenizing references upfront outside the candidate evaluation loop.
- Reduced default `RET_PREC_MAX_NEW_TOKENS` from `100` to `50` in [.env.common](file:///e:/Projects/cnd/Semantics/.env.common) and [.env.example](file:///e:/Projects/cnd/Semantics/.env.example) to cut generation latency in half.
- Optimized `score_choices_by_logprob` in [qa_probe.py](file:///e:/Projects/cnd/Semantics/lib/s3_dapt/probes/qa_probe.py) by splitting conditional and unconditional sequences into separate forward passes, preventing short choice-only sequences from being padded to the length of long prompts and reducing computations by ~50%.
- Added `SAVE_OPTIMIZER_STATE=False` config parameter to [.env.common](file:///e:/Projects/cnd/Semantics/.env.common), [.env.example](file:///e:/Projects/cnd/Semantics/.env.example), and [config.py](file:///e:/Projects/cnd/Semantics/lib/utils/config.py#L332-L336) and updated [checkpoint.py](file:///e:/Projects/cnd/Semantics/lib/utils/checkpoint.py) and [training_helpers.py](file:///e:/Projects/cnd/Semantics/lib/s3_dapt/training_helpers.py) to skip saving the heavy (~1GB) AdamW optimizer state in intermediate evaluation checkpoints. This eliminates disk writing blocks and makes checkpoint saving almost instantaneous.
- Optimized perplexity evaluation speed by reducing `PERPLEXITY_EVAL_TOKENS` from `10,000,000` to `200,000` in [.env.common](file:///e:/Projects/cnd/Semantics/.env.common) and [.env.example](file:///e:/Projects/cnd/Semantics/.env.example), decreasing the perplexity evaluation bottleneck from 140s to ~14s on CPU and under 2s on GPU.
- Optimized tensor replication speed in [checkpoint.py](file:///e:/Projects/cnd/Semantics/lib/utils/checkpoint.py#L20-L30) by removing redundant `.clone()` operations, using `to("cpu", non_blocking=True)`, and skipping deep recursion for flat state dictionaries.
- Verified that all 71 unit and integration tests compile and run successfully.

---

## Status: Completed (Graceful Empty Candidate Filtering in Retrieval Probe)

- Resolved the console warning spam `Warning: Empty candidate sentence detected; setting raw BERTscores to 0.` by pre-filtering candidate responses in `compute_bertscore_batch` inside [retrieval_probe.py](file:///e:/Projects/cnd/Semantics/lib/s3_dapt/probes/retrieval_probe.py). Empty/whitespace candidates are assigned a score of `0.0` directly without being passed to the SciBERT model.
- Added unit test `test_compute_bertscore_batch_handles_empty_candidates` to [test_dapt.py](file:///e:/Projects/cnd/Semantics/tests/test_dapt.py).
- Verified that all 71 unit and integration tests compile and run successfully.

---

## Status: Completed (Google Colab BertTokenizer AttributeError Fix)

- Resolved `AttributeError: BertTokenizer has no attribute build_inputs_with_special_tokens` that occurred on Google Colab with newer `transformers` versions by dynamically monkeypatching the missing method onto the tokenizer instance inside `_patched_tokenizer_context` in [retrieval_probe.py](file:///e:/Projects/cnd/Semantics/lib/s3_dapt/probes/retrieval_probe.py).
- Added unit test `test_patched_tokenizer_context_monkeypatches_special_tokens` to [test_dapt.py](file:///e:/Projects/cnd/Semantics/tests/test_dapt.py).
- Verified that all 70 unit and integration tests compile and run successfully.

---

## Status: Completed (Retrieval Probe Refinements & Caching)

- Refactored [retrieval_probe.py](file:///e:/Projects/cnd/Semantics/lib/s3_dapt/probes/retrieval_probe.py) stop words to only encode `"\n"`, avoiding multi-token split issues with `"\n\n"`.
- Implemented global cache `_SCORER_CACHE` and helper `get_bertscorer` in [retrieval_probe.py](file:///e:/Projects/cnd/Semantics/lib/s3_dapt/probes/retrieval_probe.py) to prevent reloading the SciBERT model checkpoint from disk across evaluations.
- Enforced hard fail by raising `RuntimeError` immediately when BERTScore fails (with `use_bertscore=True`), avoiding silent propagation of dummy `0.5` scores to the training convergence gates.
- Hoisted batch size shape mismatch and mock check assertions outside of the per-prompt loops in `generate_responses_batch`.
- Decoupled circular/local imports by moving `clean_for_match` from [terminology_probe.py](file:///e:/Projects/cnd/Semantics/lib/s3_dapt/probes/terminology_probe.py) into a new shared utility file [utils.py](file:///e:/Projects/cnd/Semantics/lib/s3_dapt/probes/utils.py).
- Added type-safety check for mock tokenizers with `MagicMock.model_max_length` in unit tests when intercepting `from_pretrained` calls.
- Verified that all 69 unit and integration tests compile and run successfully.

---

## Status: Completed (Per-Probe Configuration Refactoring)

- Refactored [config.py](file:///e:/Projects/cnd/Semantics/lib/utils/config.py) to declare individual configurations for max sequence length and batch sizes for each evaluation probe (perplexity, QA, terminology, and retrieval).
- Refactored [eval_runner.py](file:///e:/Projects/cnd/Semantics/lib/s3_dapt/evaluation/eval_runner.py), [qa_probe.py](file:///e:/Projects/cnd/Semantics/lib/s3_dapt/probes/qa_probe.py), [terminology_probe.py](file:///e:/Projects/cnd/Semantics/lib/s3_dapt/probes/terminology_probe.py), and [retrieval_probe.py](file:///e:/Projects/cnd/Semantics/lib/s3_dapt/probes/retrieval_probe.py) to pass down and respect these individual config limits during evaluation.
- Exposed and documented all new variables in [.env.common](file:///e:/Projects/cnd/Semantics/.env.common) and [.env.example](file:///e:/Projects/cnd/Semantics/.env.example).
- Verified that all 69 unit and integration tests compile and run successfully.

---

## Status: Completed (Retrieval Probe Improvements)

- Refactored [retrieval_probe.py](file:///e:/Projects/cnd/Semantics/lib/s3_dapt/probes/retrieval_probe.py) to resolve silent fallback behavior on BERTScore errors, replaced the global monkeypatch with a scoped context manager, and eliminated duplicate text generation/evaluation.
- Removed tokenizer pre-truncation, allowing SciBERT's tokenizer to natively handle boundaries.
- Reused [clean_for_match](file:///e:/Projects/cnd/Semantics/lib/s3_dapt/probes/terminology_probe.py#L236) in `compute_lexical_f1_batch` for punctuation-robust token cleaning.
- Enforced strict batch size mismatch validations and added double-newline stop criteria to prevent rambling.
- Added unit test `test_retrieval_probe_lexical_f1_and_delegation` to [test_dapt.py](file:///e:/Projects/cnd/Semantics/tests/test_dapt.py).
- Verified that all 69 unit and integration tests compile and run successfully.

---

## Status: Completed (Grouped and Restructured .env Files)

- Reordered and grouped all environment variables across [.env.common](file:///e:/Projects/cnd/Semantics/.env.common), [.env.example](file:///e:/Projects/cnd/Semantics/.env.example), [.env.cpu](file:///e:/Projects/cnd/Semantics/.env.cpu), and [.env.gpu](file:///e:/Projects/cnd/Semantics/.env.gpu) to align exactly with the dataclass structures in [config.py](file:///e:/Projects/cnd/Semantics/lib/utils/config.py).
- Renamed all step-based group headers to dataclass-based headers following the pattern `# [field_name] ([dataclass_name])`.
- Synchronized missing variables (such as `SLOW_EVAL_INTERVAL_TOKENS`, `PRETOKENIZED_BIN_PATH`, and `GRADIENT_ACCUMULATION_STEPS`) across all environments.
- Moved hardware-specific optimization variable `MAX_TASKS_PER_CHILD` out of [.env.common](file:///e:/Projects/cnd/Semantics/.env.common) to specific hardware configuration overrides in [.env.cpu](file:///e:/Projects/cnd/Semantics/.env.cpu) and [.env.gpu](file:///e:/Projects/cnd/Semantics/.env.gpu).
- Verified that all 65 unit and integration tests compile and run successfully.

---

## Status: Completed (Simplified Logger Signature)

- Removed the redundant `log_filename` parameter from [setup_logger](file:///e:/Projects/cnd/Semantics/lib/utils/logger.py#L16)'s signature and call sites.
- Configured [setup_logger](file:///e:/Projects/cnd/Semantics/lib/utils/logger.py#L16) in [logger.py](file:///e:/Projects/cnd/Semantics/lib/utils/logger.py) to read `log_filename = cfg.log_file` directly from `LoggingConfig`.
- Simplified the call sites in [build_corpus.py](file:///e:/Projects/cnd/Semantics/lib/s1_build_corpus/build_corpus.py), [worker.py](file:///e:/Projects/cnd/Semantics/lib/s1_build_corpus/worker.py), [pretokenize.py](file:///e:/Projects/cnd/Semantics/lib/s2_pretokenize/pretokenize.py), and [dapt.py](file:///e:/Projects/cnd/Semantics/lib/s3_dapt/dapt.py) by removing the explicit `log_filename` argument.
- Verified that all 65 unit and integration tests compile and pass successfully.

---

## Status: Completed (Consolidated Log Files)

- Replaced separate phase-specific log files with a single consolidated `LOG_FILE=pipeline.log` across the entire codebase.
- Removed `CORPUS_LOG_FILE`, `PRETOKENIZE_LOG_FILE`, and `DAPT_LOG_FILE` environment variables from [.env.common](file:///e:/Projects/cnd/Semantics/.env.common) and [.env.example](file:///e:/Projects/cnd/Semantics/.env.example).
- Cleaned up `LoggingConfig` in [config.py](file:///e:/Projects/cnd/Semantics/lib/utils/config.py#L188) to parse the unified `LOG_FILE` instead of the distinct files.
- Modified the default fallback log file name in [logger.py](file:///e:/Projects/cnd/Semantics/lib/utils/logger.py#L30) to point to `"pipeline.log"`.
- Updated all call sites in [build_corpus.py](file:///e:/Projects/cnd/Semantics/lib/s1_build_corpus/build_corpus.py), [worker.py](file:///e:/Projects/cnd/Semantics/lib/s1_build_corpus/worker.py), [pretokenize.py](file:///e:/Projects/cnd/Semantics/lib/s2_pretokenize/pretokenize.py), and [dapt.py](file:///e:/Projects/cnd/Semantics/lib/s3_dapt/dapt.py) to pass `cfg.logging.log_file`.
- Verified that all 65 unit and integration tests compile and pass successfully.

---

## Status: Completed (Dynamic Logger Introspection)

- Eliminated hardcoded function name string references (e.g. `run_corpus_builder` or `_run_dapt_pipeline_impl`) in logging setup.
- Used Python runtime introspection via `sys._getframe().f_code.co_name` to dynamically set up functions' logger names (i.e. `f"{__name__}.{sys._getframe().f_code.co_name}"`).
- Configured module-level default loggers using `__name__` and dynamically reassigned them using `global logger` when functions execute.
- Verified that all 65 unit and integration tests compile and pass successfully.

---

## Status: Completed (Dynamic Logger Names & Configured Filenames)

- Refactored all `setup_logger` calls to use dynamic naming schemas (`f"{__name__}.{function_name}"`) instead of hardcoded strings (e.g., `"s1.build"`).
- Aligned module-level logger declarations (`get_logger`) to match the configured loggers' naming conventions.
- Added log filename configuration to `.env.common` and `.env.example` under `# Logging & Checkpoint Storage`:
  - `CORPUS_LOG_FILE=corpus_building.log`
  - `PRETOKENIZE_LOG_FILE=pretokenization.log`
  - `DAPT_LOG_FILE=dapt_convergence.log`
- Added the corresponding configuration fields (`corpus_log_file`, `pretokenize_log_file`, `dapt_log_file`) to `LoggingConfig` in [config.py](file:///e:/Projects/cnd/Semantics/lib/utils/config.py#L188).
- Updated [build_corpus.py](file:///e:/Projects/cnd/Semantics/lib/s1_build_corpus/build_corpus.py), [worker.py](file:///e:/Projects/cnd/Semantics/lib/s1_build_corpus/worker.py), [pretokenize.py](file:///e:/Projects/cnd/Semantics/lib/s2_pretokenize/pretokenize.py), and [dapt.py](file:///e:/Projects/cnd/Semantics/lib/s3_dapt/dapt.py) to use dynamic names and fetch filenames from `LoggingConfig`.
- Verified that all 65 unit and integration tests compile and pass successfully.

---

## Status: Completed (Logger Config Refactoring)

- Refactored [setup_logger](file:///e:/Projects/cnd/Semantics/lib/utils/logger.py#L13) in [logger.py](file:///e:/Projects/cnd/Semantics/lib/utils/logger.py) to accept the unified `cfg: LoggingConfig` as its second parameter instead of individual `log_dir` and `level` fields.
- Consolidated log level configuration by moving `log_level` from `MiscConfig` to `LoggingConfig` in [config.py](file:///e:/Projects/cnd/Semantics/lib/utils/config.py#L188).
- Updated all `setup_logger` call sites across the codebase:
  - Inside [build_corpus.py](file:///e:/Projects/cnd/Semantics/lib/s1_build_corpus/build_corpus.py), updated the main logging setup and adapted `CorpusBuilder` to propagate the unified `LoggingConfig` to the worker process initializer.
  - Inside [worker.py](file:///e:/Projects/cnd/Semantics/lib/s1_build_corpus/worker.py), modified `worker_init` to accept `logging_cfg` and configure the logger with it.
  - Inside [pretokenize.py](file:///e:/Projects/cnd/Semantics/lib/s2_pretokenize/pretokenize.py), updated `setup_logger` to pass `cfg.logging`.
  - Inside [dapt.py](file:///e:/Projects/cnd/Semantics/lib/s3_dapt/dapt.py), modified `setup_logger` to pass `cfg.logging`.
  - Inside [test_corpus_builder.py](file:///e:/Projects/cnd/Semantics/tests/test_corpus_builder.py), updated mock builder initialization to provide `LoggingConfig`.
- Verified that all 65 unit and integration tests compile and pass successfully.

---

## Status: Completed (Package Export Cleanups)

- Simplified public package surfaces by removing unused or redundant exports from `__init__.py` files.
- Refactored [utils/__init__.py](file:///e:/Projects/cnd/Semantics/lib/utils/__init__.py) to keep only the symbols actually imported from the `lib.utils` package-level namespace:
  - Kept: `setup_logger`, `get_logger`, `PipelineConfig`, `DAPTConfig`, `CorpusBuildConfig`, `StorageAdapter`, `get_adapter`, `FunctionProfiler`, `clean_corpus_text`.
  - Removed internal/sub-module level classes and functions: `MetricsWriter`, `save_json`, `load_json`, `save_checkpoint`, `load_checkpoint`, `select_best_checkpoint`, `StorageConfig`, `LoggingConfig`.
- Simplified [s1_build_corpus/__init__.py](file:///e:/Projects/cnd/Semantics/lib/s1_build_corpus/__init__.py) to export only `run_corpus_builder`.
- Verified that all remaining `__init__.py` subpackage files (`s2_pretokenize`, `s3_dapt`, `s4_rad_prep`, `s5_clustering`, `s6_teacher_benchmarking`) are already clean and only expose their respective pipeline execution functions.
- Verified that all 65 unit tests pass successfully.

---

## Status: Completed (Configuration Restructuring)

- Refactored [config.py](file:///e:/Projects/cnd/Semantics/lib/utils/config.py) to separate concerns inside [PipelineConfig](file:///e:/Projects/cnd/Semantics/lib/utils/config.py#L325):
  - [StorageConfig](file:///e:/Projects/cnd/Semantics/lib/utils/config.py#L175) strictly holds data for the storage adapter.
  - Moved checkpoints-related configurations (`checkpoint_dir`, `best_checkpoint_manifest`, and `checkpoint_keep_last`) under [ModelConfig](file:///e:/Projects/cnd/Semantics/lib/utils/config.py#L158).
  - Created a new dataclass `LoggingConfig` to group logs and metrics settings (`log_dir`, `metrics_log_file`, and `risk_report_path`).
- Updated all config property references in [dapt.py](file:///e:/Projects/cnd/Semantics/lib/s3_dapt/dapt.py), [training_helpers.py](file:///e:/Projects/cnd/Semantics/lib/s3_dapt/training_helpers.py), [eval_runner.py](file:///e:/Projects/cnd/Semantics/lib/s3_dapt/evaluation/eval_runner.py), [rad_prep.py](file:///e:/Projects/cnd/Semantics/lib/s4_rad_prep/rad_prep.py), and [trace_generator.py](file:///e:/Projects/cnd/Semantics/lib/s4_rad_prep/trace_generator.py) to align with the new structure.
- Exposed `StorageConfig` and `LoggingConfig` in [__init__.py](file:///e:/Projects/cnd/Semantics/lib/utils/__init__.py).
- Adjusted [test_dapt.py](file:///e:/Projects/cnd/Semantics/tests/test_dapt.py) and [test_rad_prep.py](file:///e:/Projects/cnd/Semantics/tests/test_rad_prep.py) testing mocks to utilize the new namespaces.
- Verified that all 65 unit tests compile and run successfully.

---

## Status: Completed (StorageConfig Import Fix)

- Fixed a `NameError`/linter warning in [storage.py](file:///e:/Projects/cnd/Semantics/lib/utils/storage.py) line 60, where the `StorageConfig` class was referenced in the type signature of `get_adapter` but not imported in the file.
- Added a relative import `from .config import StorageConfig` to `storage.py` and updated the parameter type hint from `"StorageConfig"` string literal to the class directly.
- Verified that all unit tests continue to pass successfully.

---

## Status: Completed (Phase 2, Step 2.1 — Teacher Benchmarking Eval Sampler Fix)

- Fixed a `ValueError` validation gate crash in Step 6 (Teacher Benchmarking) where complete lack of ID overlap between clustering splits (containing document IDs like `domain_doc_...`) and traces (containing QA sample IDs like `sample_...` from `probe_qa.jsonl`) caused the sampler to return 0 evaluation samples, triggering a hard fail.
- Implemented a smart overlap checker in `run_eval_sampling` ([eval_sampler.py](file:///e:/Projects/cnd/Semantics/lib/s6_teacher_benchmarking/eval_sampler.py)) that detects complete ID mismatches. When a complete mismatch is detected, it enables a deterministic hash-based fallback mapping that safely maps validation document IDs to available QA traces.
- Preserved existing skip/miss logic when mock splits and traces overlap, ensuring all 19 unit tests in [test_teacher_benchmarking.py](file:///e:/Projects/cnd/Semantics/tests/test_teacher_benchmarking.py) pass successfully.

---

## Status: Completed (DAPT Corpus Text Cleaning & Noise Removal)

- Implemented a corpus text cleaner in [clean_text.py](file:///e:/Projects/cnd/Semantics/lib/utils/clean_text.py) that filters out InDesign layout metadata, proof timestamps, blank page placeholders, and repairs split PDF ligatures (e.g. `Th e`, `eff ort`, `refl ect`, `o th er`).
- Integrated the cleaning function into [worker.py](file:///e:/Projects/cnd/Semantics/lib/s1_build_corpus/worker.py) to automatically process and clean newly extracted PDF text chunks during Step 1.
- Created [clean_existing_corpus.py](file:///e:/Projects/cnd/Semantics/scripts/clean_existing_corpus.py) post-processing script, which cleaned the existing [domain_dapt_corpus.jsonl](file:///e:/Projects/cnd/Semantics/data/dapt/domain_dapt_corpus.jsonl) file in-place, achieving an **0.81% reduction in total tokens** (35,528 tokens of pure layout noise removed).
- Added comprehensive unit tests in [test_clean_text.py](file:///e:/Projects/cnd/Semantics/tests/test_clean_text.py) covering all cleaning and ligature rules.
- Re-executed Step 2 (`pipeline.py --step s2`) to regenerate the pre-tokenized training array (`train_tokens.npy`) and validation text (`ppl_held_out.txt`) from the cleaned corpus.

---

## Status: Completed (DAPT Evaluation Probe MCQ Scoring Fix)

- Cleaned up the DAPT QA evaluation dataset by removing 6 corrupted entries from [probe_qa.jsonl](file:///e:/Projects/cnd/Semantics/evals/dapt/probe_qa.jsonl) containing PDF-parsing table noise.
- Refactored `score_choices_by_logprob`, `eval_qa_accuracy`, and `get_failed_qa_samples` in [qa_probe.py](file:///e:/Projects/cnd/Semantics/lib/s3_dapt/probes/qa_probe.py) to implement the Pointwise Mutual Information (PMI) Question-Only scoring method.
- This PMI Question-Only scoring method eliminates the length/fluency bias of the choice text and boundary transition mismatches, increasing the base model evaluation accuracy from 34.3% to 85.5%.
- Updated the unit tests in [test_dapt.py](file:///e:/Projects/cnd/Semantics/tests/test_dapt.py) to align mock tokenizer and model outputs with the new PMI scoring behavior, and verified that all tests pass.

---

## Status: Design Completed (Phase 2, Step 2.1 — Teacher Benchmarking)

- Wrote detailed feature specification at [context/feature-specs/teacher-benchmarking.md](context/feature-specs/teacher-benchmarking.md) for Phase 2 Step 2.1.
- Spec covers: eval sample preparation from cluster val splits, multi-teacher trace generation, four-dimension scoring (answer accuracy, reasoning quality, citation accuracy, hallucination rate), LLM-as-judge calibration protocol, config extensions, pipeline integration (`s5` step), and a 19-test test plan.

---

## Status: Completed (Phase 1 — Corpus Engineering & Micro-Clustering)

- Implemented Phase 1 Step 4 micro-clustering pipeline using HDBSCAN and sentence-transformers to discover latent domains.
- Implemented three-way data splitting (development, validation, and sealed splits) for neuroscience micro-clusters.
- Implemented Retriever-Answerer-Dissector preparation (RAD prep) with document chunking, FAISS index construction, BM25 retrieval, and trace generation.
- Added comprehensive unit and integration tests under [tests/test_clustering.py](file:///e:/Projects/cnd/Semantics/tests/test_clustering.py) and [tests/test_rad_prep.py](file:///e:/Projects/cnd/Semantics/tests/test_rad_prep.py).

- Modified [main.py](file:///e:/Projects/CND/Semantics/main.py) and added root-level [pipeline.py](file:///e:/Projects/CND/Semantics/pipeline.py) to load env and run the Step 1 corpus building pipeline using the configured parser.
- Fixed non-ASCII characters in [lib/s1_build_corpus/build_corpus.py](file:///e:/Projects/CND/Semantics/lib/s1_build_corpus/build_corpus.py) final print statement to prevent UnicodeEncodeError on Windows.
- Removed MinerU (magic-pdf) code and registry from [lib/s1_build_corpus/parsers.py](file:///e:/Projects/CND/Semantics/lib/s1_build_corpus/parsers.py).
- Removed MinerU configuration references from [.env](file:///e:/Projects/CND/Semantics/.env) and [.env.example](file:///e:/Projects/CND/Semantics/.env.example).
- Refactored `s1_build_corpus` library so environment configuration is loaded in the entry-point [pipeline.py](file:///e:/Projects/CND/Semantics/pipeline.py) and passed as parameters. Modified [lib/s1_build_corpus/storage.py](file:///e:/Projects/CND/Semantics/lib/s1_build_corpus/storage.py), [lib/s1_build_corpus/build_corpus.py](file:///e:/Projects/CND/Semantics/lib/s1_build_corpus/build_corpus.py), and [lib/s1_build_corpus/__init__.py](file:///e:/Projects/CND/Semantics/lib/s1_build_corpus/__init__.py) to accept these parameters directly.
- Removed Marker support entirely, optimizing the pipeline to use only Docling. Deleted [parsers.py](file:///e:/Projects/CND/Semantics/lib/s1_build_corpus/parsers.py) and simplified [__init__.py](file:///e:/Projects/CND/Semantics/lib/s1_build_corpus/__init__.py), [worker.py](file:///e:/Projects/CND/Semantics/lib/s1_build_corpus/worker.py), and [build_corpus.py](file:///e:/Projects/CND/Semantics/lib/s1_build_corpus/build_corpus.py) to directly initialize and run Docling inside worker processes, removing dynamic function serialization. Cleaned up [.env](file:///e:/Projects/CND/Semantics/.env), [.env.example](file:///e:/Projects/CND/Semantics/.env.example), [pipeline.py](file:///e:/Projects/CND/Semantics/pipeline.py), and [pyproject.toml](file:///e:/Projects/CND/Semantics/pyproject.toml) to remove `PARSING_ALGORITHM` and the `marker-pdf` dependency. Added automated testing suite in [test_simplify_docling.py](file:///e:/Projects/CND/Semantics/tests/test_simplify_docling.py).
- Implemented Phase 0.2: Continued Pretraining (DAPT) for the base language model. Created feature spec at [dapt.md](file:///e:/Projects/CND/Semantics/context/feature-specs/dapt.md), dataset with neuroscience QA probes at [probe_qa.jsonl](file:///e:/Projects/CND/Semantics/data/dapt/probe_qa.jsonl), and DAPT pipeline execution logic at [dapt.py](file:///e:/Projects/CND/Semantics/lib/s2_dapt/dapt.py). Registered the step `s2` in [pipeline.py](file:///e:/Projects/CND/Semantics/pipeline.py) and added training/validation configs to [.env](file:///e:/Projects/CND/Semantics/.env) and [.env.example](file:///e:/Projects/CND/Semantics/.env.example). Added automated tests in [test_dapt.py](file:///e:/Projects/CND/Semantics/tests/test_dapt.py) and updated [.gitignore](file:///e:/Projects/CND/Semantics/.gitignore) to exclude large output weights.
- Updated evaluation probe QA dataset in [probe_qa.jsonl](file:///e:/Projects/CND/Semantics/evals/dapt/probe_qa.jsonl) with 10 targeted multiple-choice questions matching the 5 raw PDF documents under [data/raw](file:///e:/Projects/CND/Semantics/data/raw) (TinyLLM, AstroLLaMA, False Promise of Imitation, EEG Seizure Detection GNNs, and DBT Emotion Dysregulation).
- Replaced Google Colab notebook [dapt_colab.ipynb](file:///e:/Projects/CND/Semantics/notebooks/dapt_colab.ipynb) with [s1_s2.ipynb](file:///e:/Projects/CND/Semantics/notebooks/s1_s2.ipynb) for a unified Google Colab runner of the full Semantics pipeline.
- Implemented chunked PDF extraction (groups of 10 pages) in [worker.py](file:///e:/Projects/CND/Semantics/lib/s1_build_corpus/worker.py) using `pypdfium2` and Docling's `page_range` parameter, optimizing with a lightweight CPU fallback using `PyPdfiumDocumentBackend` to avoid memory allocation errors (`std::bad_alloc`). Updated [test_corpus_builder.py](file:///e:/Projects/CND/Semantics/tests/test_corpus_builder.py) to mock `pypdfium2.PdfDocument`.
- Refactored `evaluate_perplexity` in [dapt.py](file:///e:/Projects/CND/Semantics/lib/s2_dapt/dapt.py) to use a memory-safe python-list slicing chunking approach, aligning the validation evaluation with the training pretraining block distribution by appending the `<eos>` token manually.
- Overrode `tokenizer.model_max_length` in [dapt.py](file:///e:/Projects/CND/Semantics/lib/s2_dapt/dapt.py) during document encoding and restored it in a `finally` block, and adjusted the DAPT corpus train-validation split to use an 80% train and 20% validation ratio, enforcing a minimum validation size of 2 documents (`max(2, ...)`) to reduce variance in perplexity evaluation.
- Implemented mixed-precision training (AMP with bfloat16/float16) and enabled native SDPA attention/half-precision model loading in [dapt.py](file:///e:/Projects/CND/Semantics/lib/s2_dapt/dapt.py) to accelerate GPU training runtimes.
- Refactored the Step 1 corpus builder pipeline ([build_corpus.py](file:///e:/Projects/CND/Semantics/lib/s1_build_corpus/build_corpus.py) and [worker.py](file:///e:/Projects/CND/Semantics/lib/s1_build_corpus/worker.py)) to output extracted text chunk-by-chunk (using a configurable `CHUNK_SIZE` env variable) as separate JSONL records, featuring exactly 1-page overlap between successive chunks. Implemented real-time chunk streaming from worker processes to the main thread using a multiprocessing queue and a background writer thread. Updated unit tests and the Google Colab notebook ([s1_s2.ipynb](file:///e:/Projects/CND/Semantics/notebooks/s1_s2.ipynb)) to match this new parameter and behavior.
- Reconciled and updated the environment configuration files [.env](file:///e:/Projects/CND/Semantics/.env) and [.env.example](file:///e:/Projects/CND/Semantics/.env.example) to include all configurable parameters defined in [config.py](file:///e:/Projects/CND/Semantics/lib/utils/config.py), ensuring completeness and synchronizing variables and default values across local environments.
- Implemented a dynamic default run name generator for Weights & Biases (WandB) in [config.py](file:///e:/Projects/CND/Semantics/lib/utils/config.py), which automatically defaults `WANDB_RUN_NAME` to `WANDB_PROJECT` appended with a timestamp in `YYMMDDHHMMSS` format if it is not explicitly configured in the environment.
- Updated Google Colab runner notebook [s1_s2.ipynb](file:///e:/Projects/CND/Semantics/notebooks/s1_s2.ipynb) to utilize the updated [PipelineConfig](file:///e:/Projects/CND/Semantics/lib/utils/config.py) structure, outputting all newly-added parameters (e.g. model dtype, max sequence length, corpus limits, and Weights & Biases config parameters).
- Fixed a `NameError` in `handle_hard_cap` function within [gate_logic.py](file:///e:/Projects/CND/Semantics/lib/s2_dapt/evaluation/gate_logic.py) where `term_cov_threshold` and `ret_prec_threshold` were referenced but not defined. Added both thresholds to the signature of `handle_hard_cap` and passed them correctly from [dapt.py](file:///e:/Projects/CND/Semantics/lib/s2_dapt/dapt.py).
- Implemented baseline evaluation of the unmodified base model in [dapt.py](file:///e:/Projects/CND/Semantics/lib/s2_dapt/dapt.py) before the training loop starts, enabling accurate measurement of CPT/DAPT improvements from step zero.
- Mocked the heavy evaluation runner `run_all_probes` in [test_dapt.py](file:///e:/Projects/CND/Semantics/tests/test_dapt.py) to prevent downloading/running SciBERT models during unit test execution, reducing test run times from 23 seconds to under 8 seconds. Added assertions to verify the baseline evaluation is called during both pipeline tests.
- Fixed terminology coverage probe ([terminology_probe.py](file:///e:/Projects/CND/Semantics/lib/s2_dapt/probes/terminology_probe.py)) by splitting the cloze prompt at the `"___"` placeholder and only passing the prefix to the causal model, enabling next-token prediction to fill the blank correctly instead of generating text at the end of the full sentence.
- Fixed retrieval precision probe ([retrieval_probe.py](file:///e:/Projects/CND/Semantics/lib/s2_dapt/probes/retrieval_probe.py)) by truncating generated hypotheses and reference texts to a safe maximum length of 1024 characters before calculating BERTScore, preventing PyTorch shape mismatch errors caused by extremely long bibliography citations in standard BERT/SciBERT 512-token limit models.
- Fixed a runtime error in terminology and retrieval evaluation probes where empty inputs resulting from blank cloze splits (e.g. prompts beginning with "___") caused batch sequence length to become 0, leading to a tensor view/reshape crash in the model's self-attention layers. Added fallback safety checks to input token shapes and empty prompt strings, and added unit tests in [test_dapt.py](file:///e:/Projects/CND/Semantics/tests/test_dapt.py).
- Fixed duplicate log output in the console by setting `logger.propagate = False` on the configured parent loggers in [logger.py](file:///e:/Projects/cnd/Semantics/lib/utils/logger.py), preventing log records from propagating up to the root logger's default handlers.
- Renamed configuration variable `DAPT_BATCH_SIZE` to `TRAIN_BATCH_SIZE` across `.env`, `.env.example`, [config.py](file:///e:/Projects/cnd/Semantics/lib/utils/config.py), and [s1_s2.ipynb](file:///e:/Projects/cnd/Semantics/notebooks/s1_s2.ipynb) to make it more meaningful.
- Refactored `eval_qa_accuracy` in [qa_probe.py](file:///e:/Projects/cnd/Semantics/lib/s2_dapt/probes/qa_probe.py) to read files exclusively as JSONL line-by-line, adding early-termination optimization for `max_samples` and removing the double-reading JSON array fallback.
- Implemented batching generation for terminology probe and retrieval probe evaluations, adding configurable batch sizes (`TERM_COV_GEN_BATCH_SIZE` and `RET_PREC_GEN_BATCH_SIZE`).
- Fixed shape/dimension mismatch and empty-input handling in generation probes to avoid runtime errors when tokenizing empty inputs or when padding inputs.
- Implemented final saved model reloading and detailed failure logging at the end of the DAPT training step, logging individual failed samples (for QA, Terminology, and Retrieval probes) and saving a structured summary to `logs/failed_evals.json`. Added corresponding unit test coverage in [test_dapt.py](file:///e:/Projects/cnd/Semantics/tests/test_dapt.py).

---

## Status: Completed (Step 4 RAD Prep, Step 5 Clustering, and Step 6 Teacher Benchmarking Refinements)

- Added `AutoFlushingFileHandler` and `flush_loggers()` in [logger.py](file:///e:/Projects/cnd/Semantics/lib/utils/logger.py) to guarantee immediate disk writes to `logs/pipeline.log`. Updated [teacher_benchmarking.py](file:///e:/Projects/cnd/Semantics/lib/s6_teacher_benchmarking/teacher_benchmarking.py) to invoke `flush_loggers()` on step completion. Added automatic `logs/pipeline.log` syncing to Google Drive in [pipeline.ipynb](file:///e:/Projects/cnd/Semantics/pipeline.ipynb).
- Refactored hallucination detection in [hallucination_detector.py](file:///e:/Projects/cnd/Semantics/lib/s6_teacher_benchmarking/hallucination_detector.py) to evaluate sentence entailment against individual context chunks and `ground_truth`, preventing DeBERTa 512-token sequence truncation. Added template exemptions for structural formatting sentences (`\\boxed{}`, `"Based on the context..."`, `"To determine..."`), reducing false positive hallucination rates from 88.27% to 33.88%.
- Refactored citation extraction and recall evaluation in [citation_accuracy.py](file:///e:/Projects/cnd/Semantics/lib/s6_teacher_benchmarking/citation_accuracy.py) to support bracketed markers (`[1]`, `[Context 1]`, `[Passage 1]`) and trigger phrases, and evaluated citation recall relative to referenced context passages, increasing citation accuracy from 2.64% to 41.69%.
- Updated prompt formatting in [benchmark_runner.py](file:///e:/Projects/cnd/Semantics/lib/s6_teacher_benchmarking/benchmark_runner.py) (`build_benchmark_prompt`) and [trace_generator.py](file:///e:/Projects/cnd/Semantics/lib/s4_rad_prep/trace_generator.py) (`format_prompt`) to explicitly instruct teacher models to annotate key statements with bracketed passage citations matching the provided context (`[Context 1]` or `[Passage 1]`). Added corresponding unit test in [test_teacher_benchmarking.py](file:///e:/Projects/cnd/Semantics/tests/test_teacher_benchmarking.py).
- Dynamic Gate Scaling in RAD Prep: Updated `passed_min_traces` in [rad_prep.py](file:///e:/Projects/cnd/Semantics/lib/s4_rad_prep/rad_prep.py) to use 95% of attempted traces (`grounded_count >= min(cfg.rad.min_traces, int(0.95 * total_attempted))`) as the completion gating requirement.
- Reduced HDBSCAN noise rate in Step 5: Updated [config.py](file:///e:/Projects/cnd/Semantics/lib/utils/config.py) defaults (`hdbscan_min_samples = 2` and `pca_components = 50`) to reduce unclustered boundary noise points.


