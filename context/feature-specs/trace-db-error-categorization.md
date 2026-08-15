# Feature Spec: Evaluation Trace SQLite Database & Error Categorization UI

## Objective
Provide an SQLite database and interactive UI workflow for evaluation trace logs without modifying the main training pipeline:
1. Main pipeline continues to output trace CSV files in `logs/traces/{probe}/`.
2. A standalone loader (`lib/utils/trace_db.py` and CLI script `scripts/load_traces_to_db.py`) parses the trace CSV logs and ingests them into an SQLite database (`logs/traces.db`), with one table for each probe (`cloze_traces`, `qa_traces`, `concept_traces`).
3. Composite primary key: `(run_id, checkpoint, seq_num)` where:
   - `run_id`: Run timestamp (e.g. `20260814_081654`).
   - `checkpoint`: Integer (`0` for Base Model baseline `eval_0000`, `1` for `eval_0001`, etc.).
   - `seq_num`: Item sequence index (1, 2, 3, ...).
4. The Streamlit UI (`ui/trace_log.py`):
   - Reads probe traces and run/checkpoint options directly from SQLite (with automatic sync from trace CSVs if new files exist).
   - Provides an interactive **Error Category** widget for each item.
   - The dropdown dynamically lists unique categories existing in the database for that probe (plus option to add new custom categories).
   - Default category is `Pass` (if result is Pass) or `Fail` (if result is Fail), or whatever custom category was previously saved.
   - Provides sidebar filtering by Error Category.

## SQLite Database Schema (`logs/traces.db`)

### Tables: `cloze_traces`, `qa_traces`, `concept_traces`
```sql
CREATE TABLE IF NOT EXISTS {probe}_traces (
    run_id TEXT NOT NULL,           -- Run timestamp identifier (e.g. '20260814_081654')
    checkpoint INTEGER NOT NULL,     -- 0 for Base Model baseline, 1, 2, ... for checkpoints
    checkpoint_name TEXT NOT NULL,   -- e.g. 'eval_0', 'eval_0001', 'dapt_eval_0001.pt'
    seq_num INTEGER NOT NULL,        -- 1-based index of prompt in probe
    category TEXT,                   -- Dataset domain topic / cluster
    prompt TEXT,                     -- Evaluation prompt/question
    target_answer TEXT,              -- Expected reference/target answer
    model_output TEXT,               -- Model generation
    matching_score REAL,             -- Score (0.0 to 1.0)
    result TEXT,                     -- 'Pass' or 'Fail'
    error_category TEXT,             -- User annotation (e.g. 'Pass', 'Fail', 'Hallucination')
    notes TEXT,                      -- User notes
    timestamp TEXT,                  -- ISO timestamp
    updated_at TEXT,                 -- ISO timestamp of last update
    PRIMARY KEY (run_id, checkpoint, seq_num)
);

CREATE INDEX IF NOT EXISTS idx_{probe}_run_ckpt ON {probe}_traces (run_id, checkpoint);
CREATE INDEX IF NOT EXISTS idx_{probe}_err_cat ON {probe}_traces (error_category);
```

## Modules
1. **`lib/utils/trace_db.py`**:
   - `init_trace_db(db_path)`
   - `ingest_csv_to_db(csv_path, db_path)`
   - `sync_all_traces_to_db(traces_dir, db_path)`
   - `list_runs_and_checkpoints(probe, db_path)`
   - `load_probe_traces(probe, run_id, checkpoint, db_path)`
   - `update_error_category(probe, run_id, checkpoint, seq_num, error_category, notes, db_path)`
   - `get_unique_error_categories(probe, db_path)`
2. **`scripts/load_traces_to_db.py`**:
   - Standalone CLI script to sync trace CSVs to SQLite database.
3. **`ui/trace_log.py`**:
   - UI viewer integrated with SQLite database, dynamic error category dropdowns, and instant persistence.
