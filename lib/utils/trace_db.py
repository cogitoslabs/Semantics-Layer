"""
lib/utils/trace_db.py — SQLite Database Layer for Evaluation Traces & Interactive Error Categorization

Provides durable, relational storage for evaluation probe traces across Cloze, QA, and Concept probes,
indexed by (run_id, checkpoint, seq_num). Enables interactive error categorization and filtering in the UI.
"""

import csv
import os
from datetime import datetime, timezone
from pathlib import Path
import sqlite3
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd

from lib.utils.logger import get_logger

logger = get_logger("lib.utils.trace_db")

PROBE_TABLE_MAP = {
    "cloze": "cloze_traces",
    "qa": "qa_traces",
    "concept": "concept_traces",
}

DEFAULT_ERROR_CATEGORIES = [
    "Pass",
    "Fail",
    "Hallucination",
    "Repetition Degeneration",
    "Off-target / Irrelevant",
    "Incomplete Answer",
    "Syntax / Format Error",
]


def normalize_probe_name(probe: str) -> str:
    """Normalize probe string to standard key ('cloze', 'qa', 'concept')."""
    norm = str(probe).strip().lower()
    if "cloze" in norm:
        return "cloze"
    elif "qa" in norm:
        return "qa"
    elif "concept" in norm or "retrieval" in norm:
        return "concept"
    return norm


def get_table_name(probe: str) -> str:
    """Get the SQLite table name for a given probe."""
    norm = normalize_probe_name(probe)
    return PROBE_TABLE_MAP.get(norm, f"{norm}_traces")


from contextlib import contextmanager


@contextmanager
def get_db_connection(db_path: Union[str, Path]):
    """Create a thread-safe SQLite connection with parent directory auto-creation and automatic closing."""
    p = Path(db_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(p), timeout=30.0)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def init_trace_db(db_path: Union[str, Path] = "logs/traces.db") -> None:
    """Initialize the SQLite database schema with dedicated tables for each probe."""
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        for probe_key, table in PROBE_TABLE_MAP.items():
            cursor.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {table} (
                    run_id TEXT NOT NULL,
                    checkpoint INTEGER NOT NULL,
                    checkpoint_name TEXT NOT NULL,
                    seq_num INTEGER NOT NULL,
                    category TEXT,
                    prompt TEXT,
                    target_answer TEXT,
                    model_output TEXT,
                    matching_score REAL,
                    result TEXT,
                    error_category TEXT,
                    notes TEXT,
                    timestamp TEXT,
                    updated_at TEXT,
                    PRIMARY KEY (run_id, checkpoint, seq_num)
                );
                """
            )
            cursor.execute(
                f"CREATE INDEX IF NOT EXISTS idx_{probe_key}_run_ckpt ON {table} (run_id, checkpoint);"
            )
            cursor.execute(
                f"CREATE INDEX IF NOT EXISTS idx_{probe_key}_err_cat ON {table} (error_category);"
            )
        conn.commit()


def parse_trace_file_metadata(file_path: Union[str, Path]) -> Tuple[str, str, int, str]:
    """
    Extract (probe, run_id, checkpoint, checkpoint_name) from CSV file path.
    
    Supports formats:
      - New: logs/traces/<run_id>/<probe>/eval_XXXX.csv (e.g. logs/traces/20260814_081654/qa/eval_0001.csv)
      - Flat: logs/traces/<run_id>/<probe>_eval_XXXX.csv
      - Legacy: logs/traces/<probe>/<run_id>_eval_XXXX.csv
    """
    p = Path(file_path).resolve()
    stem = p.stem  # e.g. 'eval_0001' or 'qa_eval_0001' or '20260814_081654_eval_0001'
    parent_name = p.parent.name.lower()
    grandparent_name = p.parent.parent.name.lower()

    probe = "qa"
    run_id = "default_run"
    checkpoint = 0
    checkpoint_name = stem

    # Case 1: .../traces/<run_id>/<probe>/eval_XXXX.csv
    if parent_name in PROBE_TABLE_MAP:
        probe = normalize_probe_name(parent_name)
        if grandparent_name not in ["traces", "logs", "."]:
            run_id = p.parent.parent.name
    elif any(k in stem.lower() for k in PROBE_TABLE_MAP):
        for k in PROBE_TABLE_MAP:
            if k in stem.lower():
                probe = k
                break
        if parent_name not in ["traces", "logs", "."]:
            run_id = p.parent.name

    # Checkpoint number extraction
    if "eval_" in stem:
        eval_part = stem.split("eval_")[-1]
        checkpoint_name = f"eval_{eval_part}"
        try:
            checkpoint = int(eval_part)
        except ValueError:
            checkpoint = 0

        # Legacy fallback if run_id not found from path hierarchy:
        # e.g. stem was '20260814_081654_eval_0001'
        if run_id == "default_run" and "_" in stem:
            prefix = stem.split("eval_")[0].rstrip("_")
            if prefix:
                run_id = prefix
    else:
        checkpoint = 0

    return probe, run_id, checkpoint, checkpoint_name


def ingest_trace_csv(
    csv_path: Union[str, Path],
    db_path: Union[str, Path] = "logs/traces.db",
) -> int:
    """
    Ingest a single evaluation trace CSV into the SQLite database.
    Preserves existing user annotations (error_category, notes) if previously modified.
    """
    p = Path(csv_path)
    if not p.exists() or p.stat().st_size == 0:
        return 0

    init_trace_db(db_path)
    probe, run_id, checkpoint, checkpoint_name = parse_trace_file_metadata(p)
    table = get_table_name(probe)

    df = pd.read_csv(p, encoding="utf-8", dtype=str)
    if df.empty:
        return 0

    now_iso = datetime.now(timezone.utc).isoformat()
    rows_inserted = 0

    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        for idx, row in df.iterrows():
            # Extract sequence number
            seq_val = row.get("seq_num")
            try:
                seq_num = int(seq_val)
            except (ValueError, TypeError):
                seq_num = idx + 1

            prompt = str(row.get("prompt", "") or "")
            target_answer = str(row.get("target_answer", "") or "")
            model_output = str(row.get("model_output", "") or "")
            
            # Score
            try:
                score = round(float(row.get("matching_score", 0.0)), 4)
            except (ValueError, TypeError):
                score = 0.0

            result = str(row.get("result", "Fail")).strip().capitalize()
            category = str(row.get("category", "") or "")
            ts = str(row.get("timestamp", "") or now_iso)

            # Default error_category to Pass or Fail
            default_err = "Pass" if result == "Pass" else "Fail"
            error_cat = str(row.get("error_category", "") or "").strip() or default_err
            notes = str(row.get("notes", "") or "")

            # Upsert into SQLite
            cursor.execute(
                f"""
                INSERT INTO {table} (
                    run_id, checkpoint, checkpoint_name, seq_num, category, prompt,
                    target_answer, model_output, matching_score, result, error_category,
                    notes, timestamp, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(run_id, checkpoint, seq_num) DO UPDATE SET
                    category = excluded.category,
                    prompt = excluded.prompt,
                    target_answer = excluded.target_answer,
                    model_output = excluded.model_output,
                    matching_score = excluded.matching_score,
                    result = excluded.result,
                    error_category = CASE 
                        WHEN {table}.error_category IS NOT NULL AND {table}.error_category NOT IN ('Pass', 'Fail', '')
                        THEN {table}.error_category
                        ELSE excluded.error_category
                    END,
                    notes = CASE
                        WHEN {table}.notes IS NOT NULL AND {table}.notes != ''
                        THEN {table}.notes
                        ELSE excluded.notes
                    END,
                    updated_at = excluded.updated_at;
                """,
                (
                    run_id,
                    checkpoint,
                    checkpoint_name,
                    seq_num,
                    category,
                    prompt,
                    target_answer,
                    model_output,
                    score,
                    result,
                    error_cat,
                    notes,
                    ts,
                    now_iso,
                ),
            )
            rows_inserted += 1
        conn.commit()

    logger.info(f"Ingested {rows_inserted} trace rows from {p.name} into SQLite table '{table}' (run_id={run_id}, ckpt={checkpoint})")
    return rows_inserted


def sync_all_traces_to_db(
    traces_dir: Union[str, Path] = "logs/traces",
    db_path: Union[str, Path] = "logs/traces.db",
    latest_only: bool = False,
) -> Dict[str, int]:
    """
    Scan trace directory recursively for CSV files across cloze, qa, and concept, and ingest into SQLite.
    """
    base_path = Path(traces_dir)
    if not base_path.exists():
        return {k: 0 for k in PROBE_TABLE_MAP}

    init_trace_db(db_path)
    counts = {k: 0 for k in PROBE_TABLE_MAP}

    csv_files = sorted(
        [f for f in base_path.rglob("*.csv") if "dapt_eval_traces" not in f.name],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )

    if latest_only and csv_files:
        csv_files = csv_files[:1]

    for f in csv_files:
        probe, _, _, _ = parse_trace_file_metadata(f)
        cnt = ingest_trace_csv(f, db_path=db_path)
        if probe in counts:
            counts[probe] += cnt

    return counts


def list_db_runs(
    probe: str,
    db_path: Union[str, Path] = "logs/traces.db",
) -> List[Dict[str, Any]]:
    """
    List distinct evaluation run IDs stored in SQLite for a probe.
    """
    table = get_table_name(probe)
    init_trace_db(db_path)

    query = f"""
        SELECT 
            run_id, 
            MIN(timestamp) as start_time,
            MAX(timestamp) as end_time,
            COUNT(DISTINCT checkpoint) as checkpoint_count,
            COUNT(*) as item_count
        FROM {table}
        GROUP BY run_id
        ORDER BY run_id DESC;
    """

    runs = []
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        for row in cursor.execute(query).fetchall():
            run_id = row["run_id"]
            st_time = row["start_time"] or run_id
            ckpt_count = row["checkpoint_count"]
            item_count = row["item_count"]

            if len(run_id) >= 15 and "_" in run_id:
                parts = run_id.split("_")
                d_str = parts[0]
                t_str = parts[1]
                formatted = f"{d_str[:4]}-{d_str[4:6]}-{d_str[6:8]} {t_str[:2]}:{t_str[2:4]}:{t_str[4:6]}"
                label = f"{formatted} (Run: {run_id}) — {ckpt_count} checkpoint(s)"
            elif len(run_id) == 8 and run_id.isdigit():
                formatted_date = f"{run_id[:4]}-{run_id[4:6]}-{run_id[6:8]}"
                label = f"{formatted_date} (Run: {run_id}) — {ckpt_count} checkpoint(s)"
            else:
                label = f"Run: {run_id} — {ckpt_count} checkpoint(s)"

            runs.append({
                "run_id": run_id,
                "label": label,
                "checkpoint_count": ckpt_count,
                "item_count": item_count,
            })

    return runs


def list_db_checkpoints_for_run(
    probe: str,
    run_id: str,
    db_path: Union[str, Path] = "logs/traces.db",
) -> List[Dict[str, Any]]:
    """
    List all checkpoints available under a specific run_id for a probe.
    """
    table = get_table_name(probe)
    init_trace_db(db_path)

    query = f"""
        SELECT 
            checkpoint, 
            checkpoint_name, 
            MIN(timestamp) as timestamp,
            COUNT(*) as item_count
        FROM {table}
        WHERE run_id = ?
        GROUP BY checkpoint, checkpoint_name
        ORDER BY checkpoint ASC;
    """

    checkpoints = []
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        for row in cursor.execute(query, (run_id,)).fetchall():
            ckpt = row["checkpoint"]
            ckpt_name = row["checkpoint_name"]
            item_count = row["item_count"]
            is_base = (ckpt == 0)

            if is_base:
                label = f"Checkpoint 0 — Base Model Baseline ({ckpt_name})"
            else:
                label = f"Checkpoint {ckpt} ({ckpt_name})"

            checkpoints.append({
                "checkpoint": ckpt,
                "checkpoint_name": ckpt_name,
                "label": label,
                "is_base": is_base,
                "item_count": item_count,
            })

    return checkpoints


def list_db_runs_and_checkpoints(
    probe: str,
    db_path: Union[str, Path] = "logs/traces.db",
) -> List[Dict[str, Any]]:
    """
    List distinct evaluation runs and checkpoints stored in the SQLite database for a probe.
    Returns list of dicts with formatted display labels, ordered newest first.
    """
    table = get_table_name(probe)
    init_trace_db(db_path)

    query = f"""
        SELECT 
            run_id, 
            checkpoint, 
            checkpoint_name, 
            MIN(timestamp) as start_time,
            COUNT(*) as item_count
        FROM {table}
        GROUP BY run_id, checkpoint, checkpoint_name
        ORDER BY run_id DESC, checkpoint DESC;
    """

    results = []
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        for row in cursor.execute(query).fetchall():
            run_id = row["run_id"]
            ckpt = row["checkpoint"]
            ckpt_name = row["checkpoint_name"]
            st_time = row["start_time"] or run_id
            item_cnt = row["item_count"]

            # Human readable label
            is_base = (ckpt == 0)
            base_tag = "Base Model Baseline" if is_base else f"Checkpoint {ckpt}"
            
            # Format timestamp string if in YYYYMMDD_HHMMSS format
            if len(run_id) >= 15 and "_" in run_id:
                parts = run_id.split("_")
                d_str = parts[0]
                t_str = parts[1]
                formatted_dt = f"{d_str[:4]}-{d_str[4:6]}-{d_str[6:8]} {t_str[:2]}:{t_str[2:4]}:{t_str[4:6]}"
            else:
                formatted_dt = st_time[:19].replace("T", " ")

            label = f"{formatted_dt} — {base_tag} ({ckpt_name})"

            results.append({
                "run_id": run_id,
                "checkpoint": ckpt,
                "checkpoint_name": ckpt_name,
                "label": label,
                "is_base": is_base,
                "item_count": item_cnt,
            })

    return results


def load_probe_traces_from_db(
    probe: str,
    run_id: Optional[str] = None,
    checkpoint: Optional[int] = None,
    db_path: Union[str, Path] = "logs/traces.db",
) -> pd.DataFrame:
    """
    Load trace records from the database for a specific probe, run_id, and checkpoint.
    """
    table = get_table_name(probe)
    init_trace_db(db_path)

    query = f"SELECT * FROM {table}"
    params = []

    conditions = []
    if run_id is not None:
        conditions.append("run_id = ?")
        params.append(run_id)
    if checkpoint is not None:
        conditions.append("checkpoint = ?")
        params.append(checkpoint)

    if conditions:
        query += " WHERE " + " AND ".join(conditions)

    query += " ORDER BY seq_num ASC;"

    with get_db_connection(db_path) as conn:
        df = pd.read_sql_query(query, conn, params=params)

    if df.empty:
        return pd.DataFrame(columns=[
            "run_id", "checkpoint", "checkpoint_name", "seq_num", "category",
            "prompt", "target_answer", "model_output", "matching_score",
            "result", "error_category", "notes", "timestamp", "updated_at"
        ])

    df["seq_num"] = pd.to_numeric(df["seq_num"], errors="coerce").fillna(0).astype(int)
    df["matching_score"] = pd.to_numeric(df["matching_score"], errors="coerce").fillna(0.0).astype(float)
    return df


def update_trace_annotation(
    probe: str,
    run_id: str,
    checkpoint: int,
    seq_num: int,
    error_category: str,
    notes: str = "",
    db_path: Union[str, Path] = "logs/traces.db",
) -> bool:
    """
    Update error category and notes for a specific trace item.
    """
    table = get_table_name(probe)
    init_trace_db(db_path)
    now_iso = datetime.now(timezone.utc).isoformat()

    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            f"""
            UPDATE {table}
            SET error_category = ?, notes = ?, updated_at = ?
            WHERE run_id = ? AND checkpoint = ? AND seq_num = ?;
            """,
            (error_category, notes, now_iso, run_id, checkpoint, seq_num),
        )
        conn.commit()
        updated = cursor.rowcount > 0

    if updated:
        logger.info(f"Updated annotation for {probe} item (run: {run_id}, ckpt: {checkpoint}, seq: {seq_num}) -> '{error_category}'")
    return updated


def get_distinct_error_categories(
    probe: str,
    db_path: Union[str, Path] = "logs/traces.db",
) -> List[str]:
    """
    Query distinct error category values existing in the database for a probe,
    merged with default choices, returned in clean order.
    """
    table = get_table_name(probe)
    init_trace_db(db_path)

    existing_set = set(DEFAULT_ERROR_CATEGORIES)
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        rows = cursor.execute(
            f"SELECT DISTINCT error_category FROM {table} WHERE error_category IS NOT NULL AND error_category != '';"
        ).fetchall()
        for r in rows:
            val = str(r["error_category"]).strip()
            if val:
                existing_set.add(val)

    # Return with Pass and Fail first, then standard defaults, then custom alphabetical
    ordered = ["Pass", "Fail"]
    for cat in DEFAULT_ERROR_CATEGORIES:
        if cat not in ordered and cat in existing_set:
            ordered.append(cat)
    for cat in sorted(list(existing_set)):
        if cat not in ordered:
            ordered.append(cat)

    return ordered
