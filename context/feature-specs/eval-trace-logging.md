# Feature Specification: Evaluation Trace Logging and UI Viewer

## 1. Overview
This feature introduces structured, per-probe evaluation trace logging during training and evaluation runs, coupled with an interactive Streamlit UI in `ui/trace_log.py` to inspect, compare, and analyze evaluation prompt completions between the Base Model and fine-tuned checkpoints over time.

---

## 2. Requirements & Objectives

### A. Trace Logging
1. **Per-Probe Directory Partitioning**:
   - Whenever an evaluation cycle runs (in `eval_runner.py` or standalone probe checkers), traces must be logged to:
     ```
     logs/traces/
     ├── cloze/
     │   └── YYYYMMDD_HHMMSS_eval_NNNN.csv
     ├── qa/
     │   └── YYYYMMDD_HHMMSS_eval_NNNN.csv
     └── concept/
         └── YYYYMMDD_HHMMSS_eval_NNNN.csv
     ```
2. **Standardized CSV Trace Schema**:
   Each CSV file must contain standardized columns:
   - `eval_num`: Evaluation cycle number (e.g., `0` for base, `1`, `2`, ...).
   - `seq_num`: Item sequence ID within the evaluation dataset for 1:1 cross-checkpoint alignment.
   - `prompt`: The raw prompt/question presented to the model.
   - `target_answer`: The reference/ground-truth target (cloze target term, expected QA choice, or concept reference definition).
   - `model_output`: The exact generated text from the model.
   - `matching_score`: Numerical score (`1.0`/`0.0` for Cloze/QA, BERTScore F1 for Concept).
   - `result`: Binary outcome (`Pass` / `Fail`).
   - `category`: Sub-category or cluster (e.g. neurotransmitters, cognitive domain).
   - `checkpoint_name`: Identifier for the model source (e.g. `base_model` or `dapt_eval_0007.pt`).
   - `timestamp`: ISO timestamp of evaluation.

3. **Non-Breaking Pipeline Integration**:
   - Integrated into `eval_runner.py` via a modular helper `lib/utils/trace_logger.py`.
   - Existing cumulative `dapt_eval_traces.csv` logging remains intact for backward compatibility.

---

### B. Trace Log UI Viewer (`ui/trace_log.py`)
1. **Sidebar Navigation & Controls**:
   - **Evaluation Category Dropdown**: Select between `Cloze`, `QA`, and `Concept`.
   - **Timestamp / Checkpoint Dropdown**: Select available trace logs for the chosen category, ordered from newest to oldest (with baseline/base model clearly tagged).
   - **Base Model Reference Selector**: Select the baseline trace file (defaults automatically to the earliest trace or `eval_0` baseline).
   - **Filter Checkbox**: `"Show only traces that changed from base model output"` to quickly pinpoint learning shifts and regressions.

2. **Interactive Trace Browser**:
   - Navigation controls: Previous / Next buttons, direct slider, or index jump.
   - Progress and summary stats: Total traces, count of changed traces, pass/fail counts for baseline vs checkpoint.

3. **Detailed Comparison View**:
   - **Top Card**: Prompt & Expected Target Answer.
   - **Two-Column Side-by-Side Output**:
     - **Left Column (Base Model)**: Generated output, score, and Pass/Fail badge.
     - **Right Column (Selected Checkpoint)**: Generated output, score, and Pass/Fail badge.
   - **Delta Summary**: Visual indicator showing if the prediction changed (`Identical` vs `Changed`, `Improved` 🟢, `Regressed` 🔴, or `Unchanged` ⚪).

---

## 3. Architecture & File Structure

```
Semantics/
├── lib/
│   └── utils/
│       └── trace_logger.py         # Modular utility for saving & loading per-probe CSV trace logs
├── logs/
│   └── traces/
│       ├── cloze/                 # Partitioned Cloze evaluation trace CSVs
│       ├── qa/                    # Partitioned QA evaluation trace CSVs
│       └── concept/               # Partitioned Concept evaluation trace CSVs
├── ui/
│   ├── online_probes.py           # Streamlit probe interactive testing application
│   └── trace_log.py               # Streamlit trace log browser & comparator application
└── tests/
    └── test_trace_logger.py       # Unit tests for trace logger and trace loader logic
```

---

## 4. Key Design Decisions

1. **Deterministic Alignment via `seq_num`**:
   - By embedding `seq_num` across all probe datasets and trace outputs, the UI can reliably pair any checkpoint trace with the corresponding base model trace without relying on string matching of prompts.

2. **Decoupled Trace Logger**:
   - Creating `lib/utils/trace_logger.py` keeps `eval_runner.py` clean and allows both training runs and interactive UI scripts to read/write trace files consistently.

3. **Resilient UI Loading**:
   - The UI automatically handles cases where no baseline trace is recorded yet (fallback to displaying single-checkpoint results), or when directories are empty.

---

## 5. Verification Plan
- **Unit Tests**:
   - `tests/test_trace_logger.py`: Test saving traces to partitioned folders, reading available trace files, and computing trace diffs.
- **Integration**:
   - Run evaluation cycle in `eval_runner.py` and verify created CSVs under `logs/traces/`.
   - Launch `streamlit run ui/trace_log.py` and verify all filters, dropdowns, and comparison cards render correctly.
