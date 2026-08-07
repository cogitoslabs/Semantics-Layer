import importlib.metadata
_orig_meta_ver = importlib.metadata.version
def _safe_meta_version(name):
    try:
        v = _orig_meta_ver(name)
        if v is not None:
            return v
    except Exception:
        pass
    return "25.0.0"
importlib.metadata.version = _safe_meta_version

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest
import numpy as np

from lib.utils import PipelineConfig
from lib.s6_teacher_benchmarking.eval_sampler import run_eval_sampling, EvalSample
from lib.s6_teacher_benchmarking.metric_eval_judge import MetricEvalJudge, run_cohen_kappa_evaluation
from lib.s6_teacher_benchmarking.benchmark_reporter import run_benchmark_reporting
from lib.s6_teacher_benchmarking.benchmark_runner import run_benchmark_generation_and_scoring
from lib.s6_teacher_benchmarking import run_teacher_benchmarking


class SimpleMockTokenizer:
    def __init__(self):
        self.pad_token = "<pad>"
        self.eos_token = "</s>"
        self.eos_token_id = 2
        self.pad_token_id = 1

    def encode(self, text, add_special_tokens=False):
        return [ord(c) for c in text]

    def decode(self, tokens, skip_special_tokens=True):
        return "".join(chr(t) for t in tokens)


class MockBackend:
    def __init__(self, responses=None):
        self.responses = responses or ["{}"]
        self.call_count = 0
        self.prompts = []

    def generate_batch(self, prompts):
        self.prompts.extend(prompts)
        res = []
        for _ in prompts:
            resp = self.responses[self.call_count % len(self.responses)]
            res.append(resp)
            self.call_count += 1
        return res


@pytest.fixture
def test_cfg():
    cfg = PipelineConfig()
    cfg.benchmarking.candidate_teachers = ["Qwen/Qwen3-1.7B"]
    cfg.benchmarking.eval_sample_size = 5
    cfg.benchmarking.min_eval_samples = 2
    cfg.benchmarking.teacher_batch_size = 2
    cfg.benchmarking.enable_calibration = False
    return cfg


@pytest.fixture
def mock_tokenizer():
    with patch("lib.s6_teacher_benchmarking.benchmark_runner.AutoTokenizer.from_pretrained") as mock_auto:
        mock_tok = SimpleMockTokenizer()
        mock_auto.return_value = mock_tok
        yield mock_tok


@pytest.fixture
def temp_dir_setup(test_cfg):
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        test_cfg.clustering.splits_path = tmp / "splits.json"
        test_cfg.rad.traces_dir = tmp / "traces"
        test_cfg.rad.traces_dir.mkdir()
        
        test_cfg.benchmarking.output_dir = tmp / "benchmarking"
        test_cfg.benchmarking.scores_path = tmp / "benchmarking" / "scores.jsonl"
        test_cfg.benchmarking.manifest_path = tmp / "benchmarking" / "benchmark_manifest.json"
        test_cfg.benchmarking.calibration_log_path = tmp / "benchmarking" / "judge_calibration.jsonl"
        test_cfg.benchmarking.inter_rater_log_path = tmp / "benchmarking" / "inter_rater_agreement.json"
        
        yield test_cfg


def write_mock_splits(path: Path):
    splits = {
        "total_docs": 10,
        "total_clusters": 2,
        "seed": 42,
        "clusters": {
            "cluster_000": {
                "cluster_id": 0,
                "total_docs": 5,
                "dev_doc_ids": ["doc_0"],
                "val_doc_ids": ["doc_1", "doc_2"],
                "sealed_doc_ids": []
            },
            "cluster_001": {
                "cluster_id": 1,
                "total_docs": 5,
                "dev_doc_ids": ["doc_3"],
                "val_doc_ids": ["doc_4", "doc_5"],
                "sealed_doc_ids": []
            }
        }
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(splits, f)


def write_mock_traces(traces_dir: Path):
    grounded = [
        {"sample_id": "doc_1", "cluster_id": "0", "question": "Q1?", "answer": "Ans1", "retrieved_context": "Context 1", "no_retrieval": False, "teacher_trace": "T1", "token_count": 5},
        {"sample_id": "doc_4", "cluster_id": "1", "question": "Q4?", "answer": "Ans4", "retrieved_context": "Context 4", "no_retrieval": False, "teacher_trace": "T4", "token_count": 5}
    ]
    no_ret = [
        {"sample_id": "doc_2", "cluster_id": "0", "question": "Q2?", "answer": "Ans2", "retrieved_context": "", "no_retrieval": True, "teacher_trace": "T2", "token_count": 5},
        {"sample_id": "doc_5", "cluster_id": "1", "question": "Q5?", "answer": "Ans5", "retrieved_context": "", "no_retrieval": True, "teacher_trace": "T5", "token_count": 5}
    ]
    
    with open(traces_dir / "grounded_traces.jsonl", "w", encoding="utf-8") as f:
        for r in grounded:
            f.write(json.dumps(r) + "\n")
            
    with open(traces_dir / "no_retrieval_traces.jsonl", "w", encoding="utf-8") as f:
        for r in no_ret:
            f.write(json.dumps(r) + "\n")


# 1. Sampler Tests
def test_eval_sampler_draws_from_val_split(temp_dir_setup):
    write_mock_splits(temp_dir_setup.clustering.splits_path)
    write_mock_traces(temp_dir_setup.rad.traces_dir)
    
    eval_samples = run_eval_sampling(temp_dir_setup)
    assert len(eval_samples) == 2
    assert "cluster_000" in eval_samples
    assert len(eval_samples["cluster_000"]) == 2


def test_eval_sampler_max_clusters_limit(temp_dir_setup):
    write_mock_splits(temp_dir_setup.clustering.splits_path)
    write_mock_traces(temp_dir_setup.rad.traces_dir)
    temp_dir_setup.benchmarking.max_eval_clusters = 1
    
    eval_samples = run_eval_sampling(temp_dir_setup)
    assert len(eval_samples) == 1
    assert "cluster_000" in eval_samples
    assert "cluster_001" not in eval_samples


def test_eval_sampler_skips_missing_doc_ids(temp_dir_setup):
    write_mock_splits(temp_dir_setup.clustering.splits_path)
    grounded = [{"sample_id": "doc_1", "cluster_id": "0", "question": "Q1?", "answer": "Ans1", "retrieved_context": "Context 1", "no_retrieval": False}]
    with open(temp_dir_setup.rad.traces_dir / "grounded_traces.jsonl", "w", encoding="utf-8") as f:
        f.write(json.dumps(grounded[0]) + "\n")
        
    eval_samples = run_eval_sampling(temp_dir_setup)
    assert len(eval_samples["cluster_000"]) == 1
    assert eval_samples["cluster_000"][0].sample_id == "doc_1"


def test_eval_sampler_min_samples_warning(temp_dir_setup, caplog):
    write_mock_splits(temp_dir_setup.clustering.splits_path)
    write_mock_traces(temp_dir_setup.rad.traces_dir)
    temp_dir_setup.benchmarking.min_eval_samples = 10
    
    import logging
    with caplog.at_level(logging.WARNING):
        run_eval_sampling(temp_dir_setup)
        assert any("below min_eval_samples" in r.message for r in caplog.records)


# 2. MetricEvalJudge Tests
def test_metric_eval_judge_parses_valid_json(test_cfg):
    judge_json = json.dumps({
        "answer_accuracy": {"score": 1.0, "explanation": "Answer is correct."},
        "reasoning_quality": {"step_validity": 4, "logical_coherence": 5, "absence_of_circular_reasoning": 3, "explanation": "Reasoning is solid."},
        "citation_accuracy": {"precision": 1.0, "recall": 1.0, "accuracy": 1.0, "explanation": "All citations match context."},
        "hallucination": {"rate": 0.0, "explanation": "No hallucinations detected."}
    })
    backend = MockBackend([judge_json])
    judge = MetricEvalJudge(test_cfg, backend)
    res = judge.evaluate_trace("doc_1", "cluster_000", "Q?", "GT", "Ctx", "Trace", False)
    
    assert res is not None
    assert res["answer_accuracy"] == 1.0
    assert res["answer_explanation"] == "Answer is correct."
    assert res["reasoning_quality"] == pytest.approx((4 + 5 + 3) / 15.0)
    assert res["citation_accuracy"] == 1.0
    assert res["hallucination_rate"] == 0.0


def test_metric_eval_judge_retries_on_parse_failure(test_cfg):
    judge_json = json.dumps({
        "answer_accuracy": {"score": 0.5, "explanation": "Partially correct."},
        "reasoning_quality": {"step_validity": 3, "logical_coherence": 3, "absence_of_circular_reasoning": 3, "explanation": "Average."},
        "citation_accuracy": {"precision": 0.8, "recall": 0.8, "accuracy": 0.8, "explanation": "Minor gap."},
        "hallucination": {"rate": 0.1, "explanation": "Minor ungrounded claim."}
    })
    backend = MockBackend(["invalid json response", judge_json])
    judge = MetricEvalJudge(test_cfg, backend)
    res = judge.evaluate_trace("doc_1", "cluster_000", "Q?", "GT", "Ctx", "Trace", False)
    
    assert res is not None
    assert res["answer_accuracy"] == 0.5
    assert backend.call_count == 2


def test_metric_eval_judge_calibration_saves_traces(temp_dir_setup):
    temp_dir_setup.benchmarking.enable_calibration = True
    temp_dir_setup.benchmarking.human_calibration_size = 2
    
    judge_json = json.dumps({
        "answer_accuracy": {"score": 1.0, "explanation": "OK"},
        "reasoning_quality": {"step_validity": 4, "logical_coherence": 4, "absence_of_circular_reasoning": 4, "explanation": "OK"},
        "citation_accuracy": {"precision": 1.0, "recall": 1.0, "accuracy": 1.0, "explanation": "OK"},
        "hallucination": {"rate": 0.0, "explanation": "OK"}
    })
    backend = MockBackend([judge_json])
    judge = MetricEvalJudge(temp_dir_setup, backend)
    
    judge.evaluate_trace("doc_1", "cluster_000", "Q1", "GT1", "Ctx1", "Trace1", False)
    judge.evaluate_trace("doc_2", "cluster_000", "Q2", "GT2", "Ctx2", "Trace2", False)
    
    log_path = temp_dir_setup.benchmarking.calibration_log_path
    assert log_path.exists()
    
    with open(log_path, "r") as f:
        lines = f.readlines()
        assert len(lines) == 2
        record = json.loads(lines[0])
        assert record["sample_id"] == "doc_1"


# 3. Runner and Reporter Tests
def test_build_benchmark_prompt():
    from lib.s6_teacher_benchmarking.benchmark_runner import build_benchmark_prompt
    prompt = build_benchmark_prompt("What is X?", "X is Y.", "Context passage text.", no_retrieval=False)
    assert "[Passage 1]" in prompt or "passage citations" in prompt
    assert "\\boxed{answer}" in prompt

    # Test top-5 capping
    many_passages = "\n\n".join([f"Passage content {i}" for i in range(10)])
    prompt_capped = build_benchmark_prompt("What is X?", "X is Y.", many_passages, no_retrieval=False)
    assert "[Passage 5]" in prompt_capped
    assert "[Passage 6]" not in prompt_capped


def test_benchmark_runner_produces_cluster_scores(temp_dir_setup, mock_tokenizer):
    write_mock_splits(temp_dir_setup.clustering.splits_path)
    write_mock_traces(temp_dir_setup.rad.traces_dir)
    
    eval_samples = run_eval_sampling(temp_dir_setup)
    
    t_backend = MockBackend(["The answer is \\boxed{Ans1}."])
    j_backend = MockBackend([json.dumps({
        "answer_accuracy": {"score": 1.0, "explanation": "Correct"},
        "reasoning_quality": {"step_validity": 4, "logical_coherence": 4, "absence_of_circular_reasoning": 4, "explanation": "Good reasoning"},
        "citation_accuracy": {"precision": 1.0, "recall": 1.0, "accuracy": 1.0, "explanation": "Good citations"},
        "hallucination": {"rate": 0.0, "explanation": "No hallucination"}
    })])
    
    records = run_benchmark_generation_and_scoring(
        temp_dir_setup, 
        eval_samples, 
        judge_backend_override=j_backend,
        teacher_backend_overrides={"Qwen/Qwen3-1.7B": t_backend}
    )
    
    assert len(records) == 4
    assert records[0]["teacher_model"] == "Qwen/Qwen3-1.7B"
    assert records[0]["answer_accuracy"] == 1.0
    assert records[0]["hallucination_rate"] == 0.0


def test_benchmark_reporter_manifest_complete(temp_dir_setup):
    records = [
        {
            "teacher_model": "Qwen/Qwen3-1.7B", "cluster_label": "cluster_000", "cluster_id": 0, "sample_id": "doc_1",
            "answer_accuracy": 1.0, "reasoning_quality": 0.8, "citation_precision": 1.0, "citation_recall": 1.0, "citation_accuracy": 1.0,
            "hallucination_rate": 0.0, "no_retrieval": False, "teacher_trace": "trace", "token_count": 5
        },
        {
            "teacher_model": "Qwen/Qwen3-1.7B", "cluster_label": "cluster_001", "cluster_id": 1, "sample_id": "doc_4",
            "answer_accuracy": 1.0, "reasoning_quality": 0.8, "citation_precision": 1.0, "citation_recall": 1.0, "citation_accuracy": 1.0,
            "hallucination_rate": 0.0, "no_retrieval": False, "teacher_trace": "trace", "token_count": 5
        }
    ]
    
    manifest = run_benchmark_reporting(temp_dir_setup, records, expected_teacher_count=1, expected_cluster_count=2)
    assert manifest["status"] == "complete"
    assert "Qwen/Qwen3-1.7B" in manifest["per_teacher_aggregate"]


# 4. End-to-End Pipeline test
def test_pipeline_end_to_end(temp_dir_setup, mock_tokenizer):
    write_mock_splits(temp_dir_setup.clustering.splits_path)
    write_mock_traces(temp_dir_setup.rad.traces_dir)
    
    t_backend = MockBackend(["The answer is \\boxed{Ans1}."])
    j_backend = MockBackend([json.dumps({
        "answer_accuracy": {"score": 1.0, "explanation": "Correct"},
        "reasoning_quality": {"step_validity": 4, "logical_coherence": 4, "absence_of_circular_reasoning": 4, "explanation": "Good"},
        "citation_accuracy": {"precision": 1.0, "recall": 1.0, "accuracy": 1.0, "explanation": "Accurate"},
        "hallucination": {"rate": 0.0, "explanation": "None"}
    })])
    
    manifest = run_teacher_benchmarking(
        temp_dir_setup,
        judge_backend_override=j_backend,
        teacher_backend_overrides={"Qwen/Qwen3-1.7B": t_backend}
    )
    
    assert manifest["status"] == "complete"
    assert temp_dir_setup.benchmarking.scores_path.exists()
    assert temp_dir_setup.benchmarking.manifest_path.exists()


def test_failure_logger_creates_logs(temp_dir_setup):
    from lib.s6_teacher_benchmarking.failure_logger import BenchmarkFailureLogger
    logger = BenchmarkFailureLogger(temp_dir_setup, reset=True)
    
    logger.check_and_log_failures(
        seq_num=1,
        teacher="teacher_model_a",
        sample_id="doc_123",
        cluster_label="cluster_001",
        question="What is X?",
        ground_truth="Y",
        retrieved_context="Ctx",
        prompt="[PROMPT]",
        response="[RESPONSE]",
        answer_accuracy=0.0,
        reasoning_quality=0.4,
        citation_accuracy=0.1,
        hallucination_rate=0.6,
        no_retrieval=False,
        judge_prompt="[J_PROMPT]",
        judge_response="[J_RESP]",
        explanations={
            "answer_explanation": "Answer was wrong.",
            "reasoning_explanation": "Reasoning failed.",
            "citation_explanation": "Missing citations.",
            "hallucination_explanation": "Invented claim."
        }
    )
    
    failures_dir = Path(temp_dir_setup.logging.log_dir) / "benchmarking" / "failures"
    assert (failures_dir / "answer_accuracy.json").exists()
    assert (failures_dir / "reasoning_quality.json").exists()
    assert (failures_dir / "citation_accuracy.json").exists()
    assert (failures_dir / "hallucination_rate.json").exists()
    
    with open(failures_dir / "answer_accuracy.json", "r", encoding="utf-8") as f:
        records = json.load(f)
        assert len(records) == 1
        assert records[0]["explanation"] == "Answer was wrong."
