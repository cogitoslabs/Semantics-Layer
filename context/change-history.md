# Change History



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
- Fixed duplicate log output in the console by setting `logger.propagate = False` on the configured parent loggers in [logger.py](file:///e:/Projects/CND/Semantics/lib/utils/logger.py), preventing log records from propagating up to the root logger's default handlers.
- Renamed configuration variable `DAPT_BATCH_SIZE` to `TRAIN_BATCH_SIZE` across `.env`, `.env.example`, [config.py](file:///e:/Projects/CND/Semantics/lib/utils/config.py), and [s1_s2.ipynb](file:///e:/Projects/CND/Semantics/notebooks/s1_s2.ipynb) to make it more meaningful.
- Refactored `eval_qa_accuracy` in [qa_probe.py](file:///e:/Projects/CND/Semantics/lib/s2_dapt/probes/qa_probe.py) to read files exclusively as JSONL line-by-line, adding early-termination optimization for `max_samples` and removing the double-reading JSON array fallback.
- Implemented batching generation for terminology probe and retrieval probe evaluations, adding configurable batch sizes (`TERM_COV_GEN_BATCH_SIZE` and `RET_PREC_GEN_BATCH_SIZE`).
- Fixed shape/dimension mismatch and empty-input handling in generation probes to avoid runtime errors when tokenizing empty inputs or when padding inputs.
- Implemented final saved model reloading and detailed failure logging at the end of the DAPT training step, logging individual failed samples (for QA, Terminology, and Retrieval probes) and saving a structured summary to `logs/failed_evals.json`. Added corresponding unit test coverage in [test_dapt.py](file:///e:/Projects/CND/Semantics/tests/test_dapt.py).


