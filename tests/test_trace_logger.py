"""
tests/test_trace_logger.py — Unit and integration tests for per-probe trace logging and comparator logic
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from lib.utils.trace_logger import (
    STANDARD_TRACE_COLUMNS,
    compute_trace_diff,
    list_trace_categories,
    list_trace_files,
    load_trace_file,
    normalize_category_name,
    normalize_trace_record,
    save_probe_traces_csv,
)


def test_normalize_category_name():
    assert normalize_category_name("Cloze") == "cloze"
    assert normalize_category_name("Cloze Probe") == "cloze"
    assert normalize_category_name("QA") == "qa"
    assert normalize_category_name("qa probe") == "qa"
    assert normalize_category_name("Concept") == "concept"
    assert normalize_category_name("concept probe") == "concept"
    assert normalize_category_name("custom probe") == "custom_probe"


def test_normalize_trace_record_cloze():
    cloze_sample = {
        "Eval #": "1",
        "Eval Category": "Cloze",
        "Eval Seq #": 42,
        "Eval": json.dumps({"prompt": "The brain consists of [MASK].", "target": "neurons"}),
        "Generated Answer by the model": "neurons",
        "Matching Score": 1.0,
        "Result": "Pass",
        "prompt": "The brain consists of [MASK].",
        "target_term": "neurons",
        "generated_completions": ["neurons", "cells"],
        "category": "neuroscience",
    }
    rec = normalize_trace_record(cloze_sample, eval_num=1, category="cloze", seq_idx=42)
    assert rec["eval_num"] == "1"
    assert rec["seq_num"] == 42
    assert rec["prompt"] == "The brain consists of [MASK]."
    assert rec["target_answer"] == "neurons"
    assert rec["model_output"] == "neurons"
    assert rec["matching_score"] == 1.0
    assert rec["result"] == "Pass"
    assert rec["category"] == "neuroscience"


def test_normalize_trace_record_qa():
    qa_sample = {
        "Eval #": "2",
        "Eval Category": "QA",
        "Eval Seq #": 5,
        "Eval": json.dumps({"question": "What is dopamine?", "choices": ["A", "B"]}),
        "Generated Answer by the model": "A neurotransmitter",
        "Matching Score": 1.0,
        "Result": "Pass",
        "question": "What is dopamine?",
        "choices": ["A neurotransmitter", "A hormone"],
        "expected_idx": 0,
        "expected_text": "A neurotransmitter",
        "predicted_idx": 0,
        "predicted_text": "A neurotransmitter",
        "cluster": "neurotransmitters",
    }
    rec = normalize_trace_record(qa_sample, eval_num=2, category="qa", seq_idx=5)
    assert rec["eval_num"] == "2"
    assert rec["seq_num"] == 5
    assert rec["prompt"] == "What is dopamine?"
    assert rec["target_answer"] == "A neurotransmitter"
    assert rec["model_output"] == "A neurotransmitter"
    assert rec["matching_score"] == 1.0
    assert rec["result"] == "Pass"
    assert rec["category"] == "neurotransmitters"


def test_normalize_trace_record_concept():
    concept_sample = {
        "Eval #": "0",
        "Eval Category": "Concept",
        "Eval Seq #": 12,
        "Eval": json.dumps({"prompt": "Explain LTP", "reference": "Long-term potentiation"}),
        "Generated Answer by the model": "LTP is synaptic strengthening.",
        "Matching Score": 0.8234,
        "Result": "Pass",
        "prompt": "Explain LTP",
        "reference": "Long-term potentiation",
        "generated": "LTP is synaptic strengthening.",
        "score": 0.8234,
    }
    rec = normalize_trace_record(concept_sample, eval_num=0, category="concept", seq_idx=12)
    assert rec["eval_num"] == "0"
    assert rec["seq_num"] == 12
    assert rec["prompt"] == "Explain LTP"
    assert rec["target_answer"] == "Long-term potentiation"
    assert rec["model_output"] == "LTP is synaptic strengthening."
    assert rec["matching_score"] == 0.8234
    assert rec["result"] == "Pass"


def test_save_and_list_probe_traces_csv():
    import shutil
    test_dir = Path(".pytest_tmp") / "test_traces_save"
    if test_dir.exists():
        shutil.rmtree(test_dir, ignore_errors=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    samples = [
        {
            "seq_num": 1,
            "prompt": "Prompt 1",
            "target_answer": "Target 1",
            "model_output": "Output 1",
            "matching_score": 1.0,
            "result": "Pass",
            "category": "neuro",
        },
        {
            "seq_num": 2,
            "prompt": "Prompt 2",
            "target_answer": "Target 2",
            "model_output": "Output 2 wrong",
            "matching_score": 0.0,
            "result": "Fail",
            "category": "neuro",
        },
    ]

    saved_path = save_probe_traces_csv(
        category="cloze",
        eval_num=1,
        traces=samples,
        checkpoint_name="dapt_eval_0001.pt",
        base_dir=test_dir,
    )

    assert saved_path.exists()
    assert saved_path.parent.name == "cloze"
    assert "eval_0001.csv" in saved_path.name

    # List categories
    categories = list_trace_categories(base_dir=test_dir)
    assert "cloze" in categories

    # List files
    files = list_trace_files(category="cloze", base_dir=test_dir)
    assert len(files) == 1
    assert files[0] == saved_path


def test_load_trace_file():
    import shutil
    test_dir = Path(".pytest_tmp") / "test_traces_load"
    if test_dir.exists():
        shutil.rmtree(test_dir, ignore_errors=True)
    test_dir.mkdir(parents=True, exist_ok=True)


    samples = [
        {
            "seq_num": 1,
            "prompt": "P1",
            "target_answer": "T1",
            "model_output": "M1",
            "matching_score": 0.95,
            "result": "Pass",
            "category": "cat1",
        }
    ]
    csv_path = save_probe_traces_csv(
        category="qa",
        eval_num=0,
        traces=samples,
        base_dir=test_dir,
    )

    df = load_trace_file(csv_path)
    assert not df.empty
    assert len(df) == 1
    for col in STANDARD_TRACE_COLUMNS:
        assert col in df.columns
    assert df["seq_num"].iloc[0] == 1
    assert df["matching_score"].iloc[0] == 0.95


def test_load_empty_or_nonexistent_trace_file():
    non_existent = Path(".pytest_tmp") / "non_existent.csv"
    df = load_trace_file(non_existent)
    assert df.empty
    assert list(df.columns) == STANDARD_TRACE_COLUMNS


def test_compute_trace_diff():
    base_data = [
        {
            "eval_num": "0",
            "seq_num": 1,
            "prompt": "P1",
            "target_answer": "T1",
            "model_output": "M1_base_wrong",
            "matching_score": 0.0,
            "result": "Fail",
            "category": "cat1",
            "checkpoint_name": "base",
            "timestamp": "2026-08-14T00:00:00",
        },
        {
            "eval_num": "0",
            "seq_num": 2,
            "prompt": "P2",
            "target_answer": "T2",
            "model_output": "M2_correct",
            "matching_score": 1.0,
            "result": "Pass",
            "category": "cat2",
            "checkpoint_name": "base",
            "timestamp": "2026-08-14T00:00:00",
        },
        {
            "eval_num": "0",
            "seq_num": 3,
            "prompt": "P3",
            "target_answer": "T3",
            "model_output": "M3_correct",
            "matching_score": 1.0,
            "result": "Pass",
            "category": "cat3",
            "checkpoint_name": "base",
            "timestamp": "2026-08-14T00:00:00",
        },
    ]

    target_data = [
        {
            "eval_num": "1",
            "seq_num": 1,
            "prompt": "P1",
            "target_answer": "T1",
            "model_output": "T1",  # Fixed/Improved
            "matching_score": 1.0,
            "result": "Pass",
            "category": "cat1",
            "checkpoint_name": "ckpt_1",
            "timestamp": "2026-08-14T01:00:00",
        },
        {
            "eval_num": "1",
            "seq_num": 2,
            "prompt": "P2",
            "target_answer": "T2",
            "model_output": "M2_wrong_now",  # Regressed
            "matching_score": 0.0,
            "result": "Fail",
            "category": "cat2",
            "checkpoint_name": "ckpt_1",
            "timestamp": "2026-08-14T01:00:00",
        },
        {
            "eval_num": "1",
            "seq_num": 3,
            "prompt": "P3",
            "target_answer": "T3",
            "model_output": "M3_correct",  # Unchanged Pass
            "matching_score": 1.0,
            "result": "Pass",
            "category": "cat3",
            "checkpoint_name": "ckpt_1",
            "timestamp": "2026-08-14T01:00:00",
        },
    ]

    base_df = pd.DataFrame(base_data)
    target_df = pd.DataFrame(target_data)

    diff_df = compute_trace_diff(base_df, target_df)
    assert len(diff_df) == 3

    # Row 1: Improved
    row1 = diff_df[diff_df["seq_num"] == 1].iloc[0]
    assert row1["delta_status"] == "Improved"
    assert bool(row1["is_output_changed"]) is True
    assert bool(row1["is_result_changed"]) is True

    # Row 2: Regressed
    row2 = diff_df[diff_df["seq_num"] == 2].iloc[0]
    assert row2["delta_status"] == "Regressed"
    assert bool(row2["is_output_changed"]) is True
    assert bool(row2["is_result_changed"]) is True

    # Row 3: Unchanged Pass
    row3 = diff_df[diff_df["seq_num"] == 3].iloc[0]
    assert row3["delta_status"] == "Unchanged Pass"
    assert bool(row3["is_output_changed"]) is False
    assert bool(row3["is_result_changed"]) is False


def test_compute_trace_diff_empty_or_single():
    # Empty
    assert compute_trace_diff(pd.DataFrame(), pd.DataFrame()).empty

    # Single target without base
    target_df = pd.DataFrame([
        {"seq_num": 1, "prompt": "P1", "target_answer": "T1", "model_output": "M1", "matching_score": 1.0, "result": "Pass", "category": "c"}
    ])
    single_diff = compute_trace_diff(pd.DataFrame(), target_df)
    assert len(single_diff) == 1
    assert single_diff["delta_status"].iloc[0] == "Single Checkpoint"


def test_eval_runner_saves_partitioned_traces():
    import shutil
    from lib.s3_dapt.evaluation.eval_runner import run_all_probes
    from lib.utils.config import PipelineConfig

    cfg = PipelineConfig()
    cfg.logging.log_dir = Path(".pytest_tmp") / "test_eval_runner_traces"
    if cfg.logging.log_dir.exists():
        shutil.rmtree(cfg.logging.log_dir, ignore_errors=True)
    cfg.logging.log_dir.mkdir(parents=True, exist_ok=True)
    cfg.logging.eval_traces_file = cfg.logging.log_dir / "dapt_eval_traces.csv"
    cfg.corpus.total_corpus_tokens = 1000


    cfg.probes.run_perplexity = False
    cfg.probes.run_qa = True
    cfg.probes.run_cloze = True
    cfg.probes.run_concept = True

    state = {
        "eval_count": 1,
        "tokens_processed": 500,
        "global_step": 10,
        "eval_history": [],
    }

    mock_metrics_writer = MagicMock()

    mock_qa_result = {
        "accuracy": 1.0,
        "correct": 1,
        "total": 1,
        "per_cluster_accuracy": {},
        "eval_traces": [
            {
                "Eval #": "1",
                "Eval Category": "QA",
                "Eval Seq #": 1,
                "Eval": json.dumps({"question": "Q1"}),
                "Generated Answer by the model": "Ans1",
                "Matching Score": 1.0,
                "Result": "Pass",
                "question": "Q1",
                "choices": ["Ans1"],
                "expected_idx": 0,
                "expected_text": "Ans1",
                "predicted_idx": 0,
                "predicted_text": "Ans1",
                "cluster": "c1",
            }
        ],
    }

    mock_cloze_result = {
        "coverage": 1.0,
        "covered": 1,
        "total": 1,
        "per_category": {},
        "missed_terms": [],
        "eval_traces": [
            {
                "Eval #": "1",
                "Eval Category": "Cloze",
                "Eval Seq #": 1,
                "Eval": json.dumps({"prompt": "P1"}),
                "Generated Answer by the model": "Term1",
                "Matching Score": 1.0,
                "Result": "Pass",
                "prompt": "P1",
                "target_term": "Term1",
                "generated_completions": ["Term1"],
                "category": "cat1",
            }
        ],
    }

    mock_concept_result = {
        "precision": 1.0,
        "mean_bertscore_f1": 1.0,
        "eval_traces": [
            {
                "Eval #": "1",
                "Eval Category": "Concept",
                "Eval Seq #": 1,
                "Eval": json.dumps({"prompt": "Prompt1"}),
                "Generated Answer by the model": "ConceptDef1",
                "Matching Score": 1.0,
                "Result": "Pass",
                "prompt": "Prompt1",
                "reference": "ConceptDef1",
                "generated": "ConceptDef1",
                "score": 1.0,
                "threshold": 0.5,
            }
        ],
    }

    with patch("lib.s3_dapt.evaluation.eval_runner.eval_qa_accuracy", return_value=mock_qa_result), \
         patch("lib.s3_dapt.evaluation.eval_runner.eval_cloze_coverage", return_value=mock_cloze_result), \
         patch("lib.s3_dapt.evaluation.eval_runner.eval_concept_precision", return_value=mock_concept_result):

        metrics = run_all_probes(
            model=MagicMock(),
            tokenizer=MagicMock(),
            cfg=cfg,
            state=state,
            metrics_writer=mock_metrics_writer,
            device="cpu",
            run_slow_probes=True,
            use_bertscore=False,
        )

        assert metrics["eval_count"] == 1

        # Check partitioned trace folders
        traces_dir = cfg.logging.log_dir / "traces"
        qa_files = list_trace_files("qa", base_dir=traces_dir)
        cloze_files = list_trace_files("cloze", base_dir=traces_dir)
        concept_files = list_trace_files("concept", base_dir=traces_dir)

        assert len(qa_files) == 1
        assert len(cloze_files) == 1
        assert len(concept_files) == 1

        # Verify load_trace_file on generated files
        df_qa = load_trace_file(qa_files[0])
        assert len(df_qa) == 1
        assert df_qa["result"].iloc[0] == "Pass"

