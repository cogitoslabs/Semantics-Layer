# Feature Specification: Online Probe Check Web UI

## Overview
A Streamlit web application located at `ui/online_probes.py` providing an interactive UI for real-time comparison between Base Model and fine-tuned DAPT Checkpoint outputs across Cloze, QA, and Concept probes.

## Requirements & Modifications
1. **Header Layout**:
   - Clean interface without top title or caption banners to maximize screen real estate.
   - Set page layout to `wide` for side-by-side comparison.

2. **Sidebar Controls**:
   - **Probe Selection**: Dropdown (`st.sidebar.selectbox`) allowing selection between `Cloze Probe`, `QA Probe`, and `Concept Probe`.
   - **Checkpoint Selection**: Dropdown (`st.sidebar.selectbox`) scanning and listing available DAPT checkpoints (`dapt_eval_*.pt`).
   - **Model Information**: Displays base model name, selected checkpoint, and compute device.

3. **Prompt & Action Layout**:
   - "Question / Prompt" input area placed side-by-side with the "Generate Response" action button using horizontal layout columns.
   - For QA Probe, answer choices (A, B, C, D) are configured cleanly under the prompt area.

4. **Dual-Column Model Comparison**:
   - Automatically executes inference on both the **Base Model** and the **Selected Checkpoint**.
   - Output displayed in two side-by-side columns:
     - **Column 1**: Output from Base Model.
     - **Column 2**: Output from Selected Checkpoint.
   - Uses unified output rendering for Cloze completions, QA choice predictions, and Concept generation.

5. **Resource Caching**:
   - Retains `@st.cache_resource` for efficient model loading and switching across base and checkpoint states.
