"""
tests/test_trace_db.py — Comprehensive Unit Tests for SQLite Trace Database Layer
"""

import os
import sqlite3
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from lib.utils.trace_db import (
    get_db_connection,
    get_distinct_error_categories,
    get_table_name,
    ingest_trace_csv,
    init_trace_db,
    list_db_checkpoints_for_run,
    list_db_runs,
    list_db_runs_and_checkpoints,
    load_probe_traces_from_db,
    normalize_probe_name,
    parse_trace_file_metadata,
    sync_all_traces_to_db,
    update_trace_annotation,
)


def test_normalize_probe_name_and_table_mapping():
    assert normalize_probe_name("QA Probe") == "qa"
    assert normalize_probe_name("Cloze") == "cloze"
    assert normalize_probe_name("concept") == "concept"
    assert normalize_probe_name("Retrieval") == "concept"

    assert get_table_name("qa") == "qa_traces"
    assert get_table_name("cloze") == "cloze_traces"
    assert get_table_name("concept") == "concept_traces"


def test_init_trace_db_creates_tables():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_traces.db"
        init_trace_db(db_path)

        assert db_path.exists()
        with get_db_connection(db_path) as conn:
            cursor = conn.cursor()
            tables = [
                r[0]
                for r in cursor.execute(
                    "SELECT name FROM sqlite_master WHERE type='table';"
                ).fetchall()
            ]
            assert "qa_traces" in tables
            assert "cloze_traces" in tables
            assert "concept_traces" in tables


def test_parse_trace_file_metadata():
    # Nested run_id format: logs/traces/20260814_081654/qa/eval_0001.csv
    f1 = Path("logs/traces/20260814_081654/qa/eval_0001.csv")
    probe, run_id, ckpt, ckpt_name = parse_trace_file_metadata(f1)
    assert probe == "qa"
    assert run_id == "20260814_081654"
    assert ckpt == 1
    assert ckpt_name == "eval_0001"

    # Baseline format: logs/traces/20260814_081654/cloze/eval_0000.csv
    f2 = Path("logs/traces/20260814_081654/cloze/eval_0000.csv")
    probe, run_id, ckpt, ckpt_name = parse_trace_file_metadata(f2)
    assert probe == "cloze"
    assert run_id == "20260814_081654"
    assert ckpt == 0
    assert ckpt_name == "eval_0000"

    # Legacy flat format: logs/traces/qa/20260814_081654_eval_0001.csv
    f3 = Path("logs/traces/qa/20260814_081654_eval_0001.csv")
    probe, run_id, ckpt, ckpt_name = parse_trace_file_metadata(f3)
    assert probe == "qa"
    assert run_id == "20260814_081654"
    assert ckpt == 1
    assert ckpt_name == "eval_0001"


def test_ingest_trace_csv_and_defaults():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        db_path = tmp_path / "traces.db"
        csv_dir = tmp_path / "20260814_081654" / "qa"
        csv_dir.mkdir(parents=True)
        csv_file = csv_dir / "eval_0002.csv"

        df_data = pd.DataFrame([
            {
                "seq_num": 1,
                "prompt": "What is episodic memory?",
                "target_answer": "Memory for personal events.",
                "model_output": "Memory for personal events.",
                "matching_score": 1.0,
                "result": "Pass",
                "category": "cognitive_psychology",
                "timestamp": "2026-08-14T10:00:00Z",
            },
            {
                "seq_num": 2,
                "prompt": "What is semantic memory?",
                "target_answer": "Memory for facts.",
                "model_output": "I don't know.",
                "matching_score": 0.0,
                "result": "Fail",
                "category": "cognitive_psychology",
                "timestamp": "2026-08-14T10:00:00Z",
            }
        ])
        df_data.to_csv(csv_file, index=False)

        count = ingest_trace_csv(csv_file, db_path=db_path)
        assert count == 2

        # Check DB records
        loaded_df = load_probe_traces_from_db("qa", run_id="20260814_081654", checkpoint=2, db_path=db_path)
        assert len(loaded_df) == 2
        assert loaded_df.iloc[0]["error_category"] == "Pass"
        assert loaded_df.iloc[1]["error_category"] == "Fail"


def test_update_trace_annotation_and_dynamic_categories():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        db_path = tmp_path / "traces.db"
        csv_dir = tmp_path / "20260814_081654" / "qa"
        csv_dir.mkdir(parents=True)
        csv_file = csv_dir / "eval_0001.csv"

        df_data = pd.DataFrame([
            {
                "seq_num": 1,
                "prompt": "Test Q",
                "target_answer": "Test A",
                "model_output": "Hallucinated output",
                "matching_score": 0.0,
                "result": "Fail",
            }
        ])
        df_data.to_csv(csv_file, index=False)
        ingest_trace_csv(csv_file, db_path=db_path)

        # Update annotation to custom error category
        success = update_trace_annotation(
            probe="qa",
            run_id="20260814_081654",
            checkpoint=1,
            seq_num=1,
            error_category="Repetition Hallucination",
            notes="Model repeated tokens endlessly.",
            db_path=db_path,
        )
        assert success is True

        # Verify update persisted
        updated_df = load_probe_traces_from_db("qa", run_id="20260814_081654", checkpoint=1, db_path=db_path)
        assert updated_df.iloc[0]["error_category"] == "Repetition Hallucination"
        assert updated_df.iloc[0]["notes"] == "Model repeated tokens endlessly."

        # Verify distinct categories dynamically includes newly added category
        distinct_cats = get_distinct_error_categories("qa", db_path=db_path)
        assert "Repetition Hallucination" in distinct_cats
        assert "Pass" in distinct_cats
        assert "Fail" in distinct_cats


def test_sync_all_traces_and_list_runs():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        db_path = tmp_path / "traces.db"
        traces_dir = tmp_path / "traces"
        run_id = "20260814_081654"

        for probe in ["cloze", "qa", "concept"]:
            p_dir = traces_dir / run_id / probe
            p_dir.mkdir(parents=True)
            f_base = p_dir / "eval_0000.csv"
            f_ckpt = p_dir / "eval_0001.csv"
            
            sample_df = pd.DataFrame([{"seq_num": 1, "prompt": "p", "target_answer": "a", "result": "Pass"}])
            sample_df.to_csv(f_base, index=False)
            sample_df.to_csv(f_ckpt, index=False)

        counts = sync_all_traces_to_db(traces_dir, db_path=db_path)
        assert counts["qa"] == 2
        assert counts["cloze"] == 2
        assert counts["concept"] == 2

        # Check list runs
        runs = list_db_runs("qa", db_path=db_path)
        assert len(runs) == 1
        assert runs[0]["run_id"] == "20260814_081654"
        assert runs[0]["checkpoint_count"] == 2

        # Check list checkpoints for run
        ckpts = list_db_checkpoints_for_run("qa", "20260814_081654", db_path=db_path)
        assert len(ckpts) == 2
        assert ckpts[0]["checkpoint"] == 0 and ckpts[0]["is_base"]
        assert ckpts[1]["checkpoint"] == 1 and not ckpts[1]["is_base"]


def test_standalone_script_execution():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        db_path = tmp_path / "test_standalone.db"
        traces_dir = tmp_path / "traces"
        p_dir = traces_dir / "qa"
        p_dir.mkdir(parents=True)
        csv_file = p_dir / "20260814_110000_eval_0001.csv"
        pd.DataFrame([{"seq_num": 1, "prompt": "q", "target_answer": "a", "result": "Pass"}]).to_csv(csv_file, index=False)

        cmd = [
            sys.executable,
            "scripts/load_traces_to_db.py",
            "--traces-dir",
            str(traces_dir),
            "--db-path",
            str(db_path),
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        assert res.returncode == 0
        assert "Total records processed: 1" in res.stdout
        assert db_path.exists()
