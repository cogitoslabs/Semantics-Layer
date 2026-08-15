# Feature Spec: Pre-tokenization Metadata Validation Check

## Objective
Prevent catastrophic training degradation and huge baseline perplexity (PPL) caused by mismatch between offline pre-tokenized token ID arrays (`train_tokens.npy`, `ppl_validation_tokens.npy`) and the active base model (`cfg.model.base_model_name`).

## Background
- Pre-tokenization converts raw text to integer token IDs saved as flat `.npy` binary arrays.
- Different models (e.g. `SmolLM2-135M` vs `Qwen3-0.6B`) use distinct vocabularies and tokenizers.
- If `BASE_MODEL_NAME` is changed without re-running pre-tokenization, DAPT loads token IDs for the wrong tokenizer, causing:
  - Huge initial PPL (~17,000+).
  - Training gradients based on scrambled target tokens, rapidly destroying model weights and collapsing QA/Cloze/Concept scores.

## Proposed Design
1. **`DataConfig` Enhancement**:
   - Add `pretokenized_meta_path: Path` defaulting to `data/dapt/pretokenized_metadata.json` (configurable via `PRETOKENIZED_META_PATH`).
2. **`pretokenize.py` Enhancement**:
   - When saving `train_tokens.npy` and `ppl_validation_tokens.npy`, write a companion metadata JSON file containing:
     - `base_model_name`: normalized base model name or path
     - `tokenizer_name_or_path`: tokenizer path
     - `tokenizer_class`: tokenizer class name
     - `vocab_size`: `len(tokenizer)`
     - `eos_token_id`: EOS token ID
     - `train_tokens_count`: integer count of training tokens
     - `val_tokens_count`: integer count of validation tokens
     - `train_doc_count`: number of processed training documents
     - `val_doc_count`: number of processed validation documents
     - `created_at`: ISO timestamp string
3. **`training_helpers.py` Validation Helper**:
   - Add `verify_pretokenized_metadata(cfg: DAPTConfig, tokenizer: Optional[Any] = None) -> None`:
     - If metadata file exists:
       - Validate `meta["base_model_name"]` against `cfg.model.base_model_name` (handling local paths, case-insensitivity, and repo names).
       - Validate `meta["vocab_size"]` against `len(tokenizer)` if tokenizer is provided.
       - Raise a descriptive `ValueError` on mismatch prompting the user to run `python pipeline.py --step s2`.
     - If metadata file does not exist:
       - Log a clear warning.
4. **`dapt.py` & `perplexity_probe.py` Integration**:
   - Call `verify_pretokenized_metadata` during DAPT startup before training and baseline evaluation.
   - In `perplexity_probe.py`, guard against token IDs exceeding `len(tokenizer)` / `model.config.vocab_size`.
