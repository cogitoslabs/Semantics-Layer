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
from lib.s6_teacher_benchmarking.answer_accuracy import score_answer_accuracy, extract_answer, compute_f1_overlap, compute_mc_accuracy
from lib.s6_teacher_benchmarking.citation_accuracy import score_citation_accuracy, character_jaccard, extract_citations
from lib.s6_teacher_benchmarking.hallucination_detector import HallucinationDetector
from lib.s6_teacher_benchmarking.reasoning_judge import ReasoningJudge, run_cohen_kappa_evaluation
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
    cfg.benchmarking.nli_model = "mock-nli"
    cfg.benchmarking.enable_calibration = False
    return cfg


@pytest.fixture
def mock_tokenizer():
    with patch("lib.s6_teacher_benchmarking.benchmark_runner.AutoTokenizer.from_pretrained") as mock_auto:
        mock_tok = SimpleMockTokenizer()
        mock_auto.return_value = mock_tok
        yield mock_tok


@pytest.fixture
def mock_cross_encoder():
    with patch("sentence_transformers.CrossEncoder") as mock_class:
        mock_instance = MagicMock()
        # logits output: contradiction, neutral, entailment. High index 2 = entailment.
        mock_instance.predict.return_value = np.array([[0.0, 0.0, 3.0]])
        mock_class.return_value = mock_instance
        yield mock_instance


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


def test_eval_sampler_skips_missing_doc_ids(temp_dir_setup):
    # Splits has doc_1, doc_2. Traces has doc_1, but doc_2 is missing.
    write_mock_splits(temp_dir_setup.clustering.splits_path)
    
    # Write traces without doc_2
    grounded = [{"sample_id": "doc_1", "cluster_id": "0", "question": "Q1?", "answer": "Ans1", "retrieved_context": "Context 1", "no_retrieval": False}]
    with open(temp_dir_setup.rad.traces_dir / "grounded_traces.jsonl", "w", encoding="utf-8") as f:
        f.write(json.dumps(grounded[0]) + "\n")
        
    eval_samples = run_eval_sampling(temp_dir_setup)
    assert len(eval_samples["cluster_000"]) == 1
    assert eval_samples["cluster_000"][0].sample_id == "doc_1"


def test_eval_sampler_min_samples_warning(temp_dir_setup, caplog):
    write_mock_splits(temp_dir_setup.clustering.splits_path)
    write_mock_traces(temp_dir_setup.rad.traces_dir)
    
    # Set min_eval_samples higher than validation size
    temp_dir_setup.benchmarking.min_eval_samples = 10
    
    import logging
    with caplog.at_level(logging.WARNING):
        run_eval_sampling(temp_dir_setup)
        assert any("below min_eval_samples" in r.message for r in caplog.records)


# 2. Answer Accuracy Tests
def test_answer_accuracy_multiple_choice():
    choices = ["Option A", "Option B", "Option C"]
    
    # Exact text match
    assert score_answer_accuracy("Answer is \\boxed{Option B}", "Option B", choices) == 1.0
    # Letter match (Option B is index 1 -> letter 'b')
    assert score_answer_accuracy("Answer: \\boxed{b}", "Option B", choices) == 1.0
    # Case insensitive + whitespace normalization
    assert score_answer_accuracy("Answer: \\boxed{option  b}", "Option B", choices) == 1.0
    # Mismatch
    assert score_answer_accuracy("Answer: \\boxed{a}", "Option B", choices) == 0.0


def test_answer_accuracy_free_form_f1():
    assert score_answer_accuracy("Answer is \\boxed{calcium channel}", "calcium channel blocker") == pytest.approx(0.8)
    assert score_answer_accuracy("Answer is \\boxed{hippocampus}", "hippocampal circuitry") == pytest.approx(0.0)


def test_answer_accuracy_no_boxed_fallback():
    # Should fall back to the last sentence
    trace = "We observed significant plasticity. CA1."
    assert score_answer_accuracy(trace, "CA1") == 1.0


# 3. Reasoning Judge Tests
def test_reasoning_judge_parses_valid_json(test_cfg):
    backend = MockBackend(['{"step_validity": 4, "logical_coherence": 5, "absence_of_circular_reasoning": 3}'])
    judge = ReasoningJudge(test_cfg, backend)
    score, breakdown = judge.score_reasoning_quality("doc_1", "cluster_000", "Q?", "Ctx", "Trace")
    
    assert score == pytest.approx(4.0 / 5.0)
    assert breakdown["step_validity"] == 4.0
    assert breakdown["logical_coherence"] == 5.0


def test_reasoning_judge_retries_on_parse_failure(test_cfg):
    backend = MockBackend([
        "invalid json text", 
        '{"step_validity": 3, "logical_coherence": 3, "absence_of_circular_reasoning": 3}'
    ])
    judge = ReasoningJudge(test_cfg, backend)
    score, breakdown = judge.score_reasoning_quality("doc_1", "cluster_000", "Q?", "Ctx", "Trace")
    
    assert score == pytest.approx(3.0 / 5.0)
    assert backend.call_count == 2


def test_reasoning_judge_calibration_saves_traces(temp_dir_setup):
    temp_dir_setup.benchmarking.enable_calibration = True
    temp_dir_setup.benchmarking.human_calibration_size = 2
    
    backend = MockBackend(['{"step_validity": 4, "logical_coherence": 4, "absence_of_circular_reasoning": 4}'])
    judge = ReasoningJudge(temp_dir_setup, backend)
    
    # Score two samples to trigger log
    judge.score_reasoning_quality("doc_1", "cluster_000", "Q1", "Ctx1", "Trace1")
    judge.score_reasoning_quality("doc_2", "cluster_000", "Q2", "Ctx2", "Trace2")
    
    log_path = temp_dir_setup.benchmarking.calibration_log_path
    assert log_path.exists()
    
    with open(log_path, "r") as f:
        lines = f.readlines()
        assert len(lines) == 2
        record = json.loads(lines[0])
        assert record["sample_id"] == "doc_1"


# 4. Citation Accuracy Tests
def test_citation_accuracy_supported():
    context = "Pyramidal neurons in CA1 exhibit synaptic plasticity."
    trace = 'The paper states "neurons in CA1 exhibit" during high frequency stimulation.'
    
    prec, rec, acc = score_citation_accuracy(trace, context, no_retrieval=False, min_overlap=0.30)
    assert prec == 1.0
    assert rec == 1.0
    assert acc == 1.0


def test_citation_accuracy_unsupported():
    context = "Pyramidal neurons in CA1 exhibit synaptic plasticity."
    trace = 'The paper states "astrocytes are primary drivers of LTD" in the cortex.'
    
    prec, rec, acc = score_citation_accuracy(trace, context, no_retrieval=False, min_overlap=0.30)
    assert prec == 0.0
    assert rec == 0.0
    assert acc == 0.0


def test_citation_accuracy_no_retrieval():
    prec, rec, acc = score_citation_accuracy("Trace", "Context", no_retrieval=True)
    assert prec is None
    assert rec is None
    assert acc is None


def test_extract_citations_bracketed_and_phrases():
    trace = "As described in [Context 1], CA1 pyramidal neurons exhibit plasticity. See [Passage 2] for details."
    citations = extract_citations(trace)
    assert len(citations) >= 2
    assert any("[Context 1]" in c or "[Passage 2]" in c or "As described in" in c for c in citations)


# 5. Hallucination Scorer Tests
def test_hallucination_nli_flags_unsupported_claim(test_cfg, mock_cross_encoder):
    # Mock NLI return low entailment (index 2 < threshold)
    # Contradiction: high score, Entailment: low score
    mock_cross_encoder.predict.return_value = np.array([[3.0, 0.0, -1.0]])
    
    detector = HallucinationDetector(test_cfg)
    # Trace sentence is flagged as hallucination
    rate = detector.score_hallucination_rate("CA1 astrocytes fire action potentials.", "Context", "GT")
    assert rate == 1.0


def test_hallucination_invented_term_regex(test_cfg, mock_cross_encoder):
    # Matches: Glur-Alpha receptor
    detector = HallucinationDetector(test_cfg)
    assert detector.is_invented_term("We found the Glur-Alpha receptor in synapses.")
    # Exists in vocab (empty vocab here, let's add it)
    detector.vocab.add("glur-alpha receptor")
    assert not detector.is_invented_term("We found the Glur-Alpha receptor in synapses.")


def test_hallucination_rate_clamped_to_one(test_cfg, mock_cross_encoder):
    # Mock NLI return low entailment
    mock_cross_encoder.predict.return_value = np.array([[3.0, 0.0, -1.0]])
    
    detector = HallucinationDetector(test_cfg)
    # Even if both NLI and Regex flag, clamped to 1.0
    rate = detector.score_hallucination_rate("Unrelated-receptor is present.", "Context", "GT")
    assert rate == 1.0


# 6. Runner and Reporter Tests
def test_build_benchmark_prompt_explicit_citation():
    from lib.s6_teacher_benchmarking.benchmark_runner import build_benchmark_prompt
    prompt = build_benchmark_prompt("What is X?", "X is Y.", "Context passage text.", no_retrieval=False)
    assert "[Context 1]" in prompt or "bracketed passage citations" in prompt
    assert "\\boxed{}" in prompt


def test_benchmark_runner_produces_cluster_scores(temp_dir_setup, mock_tokenizer, mock_cross_encoder):
    write_mock_splits(temp_dir_setup.clustering.splits_path)
    write_mock_traces(temp_dir_setup.rad.traces_dir)
    
    eval_samples = run_eval_sampling(temp_dir_setup)
    
    # Mock LLM response for teacher (returns trace) and judge (returns valid scoring JSON)
    t_backend = MockBackend(["The answer is \\boxed{Ans1}."])
    j_backend = MockBackend(['{"step_validity": 4, "logical_coherence": 4, "absence_of_circular_reasoning": 4}'])
    
    records = run_benchmark_generation_and_scoring(
        temp_dir_setup, 
        eval_samples, 
        judge_backend_override=j_backend,
        teacher_backend_overrides={"Qwen/Qwen3-1.7B": t_backend}
    )
    
    assert len(records) == 4  # 2 clusters x 2 samples per cluster = 4
    assert records[0]["teacher_model"] == "Qwen/Qwen3-1.7B"


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


def test_benchmark_reporter_warning_on_null_rate(temp_dir_setup):
    # reasoning_quality is None
    records = [
        {
            "teacher_model": "Qwen/Qwen3-1.7B", "cluster_label": "cluster_000", "cluster_id": 0, "sample_id": "doc_1",
            "answer_accuracy": 1.0, "reasoning_quality": None, "citation_precision": 1.0, "citation_recall": 1.0, "citation_accuracy": 1.0,
            "hallucination_rate": 0.0, "no_retrieval": False, "teacher_trace": "trace", "token_count": 5
        }
    ]
    
    manifest = run_benchmark_reporting(temp_dir_setup, records, expected_teacher_count=1, expected_cluster_count=1)
    assert len(manifest["warnings"]) > 0
    assert any("null rate" in w for w in manifest["warnings"])


# 7. End-to-End Pipeline test
def test_pipeline_end_to_end(temp_dir_setup, mock_tokenizer, mock_cross_encoder):
    write_mock_splits(temp_dir_setup.clustering.splits_path)
    write_mock_traces(temp_dir_setup.rad.traces_dir)
    
    t_backend = MockBackend(["The answer is \\boxed{Ans1}."])
    j_backend = MockBackend(['{"step_validity": 4, "logical_coherence": 4, "absence_of_circular_reasoning": 4}'])
    
    manifest = run_teacher_benchmarking(
        temp_dir_setup,
        judge_backend_override=j_backend,
        teacher_backend_overrides={"Qwen/Qwen3-1.7B": t_backend}
    )
    
    assert manifest["status"] == "complete"
    assert temp_dir_setup.benchmarking.scores_path.exists()
    assert temp_dir_setup.benchmarking.manifest_path.exists()
