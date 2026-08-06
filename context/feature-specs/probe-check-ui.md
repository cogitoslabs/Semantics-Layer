# Feature Specification: Online Probe Check Web UI

## Overview
A Streamlit web application located at `ui/online_probes.py` providing an interactive UI for evaluating models (Base Model or DAPT Checkpoints) against Cloze and Concept probes.

## Requirements
1. **Model Selection**:
   - **Model Source**: Choice between `Base Model` and `Checkpoint`.
   - **Checkpoint Selection**: When `Checkpoint` is selected, dynamically scan `cfg.model.checkpoint_dir` for available checkpoints (`dapt_eval_*.pt`).
   - **Checkpoint Loading**: Load base model & tokenizer via `load_model_and_tokenizer(cfg, device)` and restore weights using `load_checkpoint(checkpoint_path, model)`.
   - **Caching**: Use `@st.cache_resource` parameterized by model source and checkpoint selection to prevent redundant reloading.

2. **Probe Logic & Inputs**:
   - **Probe Selection**: `Cloze Probe` or `Concept Probe`.
   - **Prompt**: Multiline text input for the test prompt.

3. **Execution & Generation**:
   - **Cloze Probe**: Calls `format_cloze_prompt` and `generate_topk_completions`.
   - **Concept Probe**: Calls `generate_response`.

4. **Output Rendering**:
   - Displays formatted outputs, active model information, device, and probe results cleanly.
