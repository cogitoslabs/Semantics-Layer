import os
import json
import logging
import copy
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from transformers import AutoTokenizer

from lib.utils import PipelineConfig
from lib.s6_teacher_benchmarking.eval_sampler import EvalSample
from lib.s6_teacher_benchmarking.answer_accuracy import score_answer_accuracy
from lib.s6_teacher_benchmarking.citation_accuracy import score_citation_accuracy
from lib.s6_teacher_benchmarking.hallucination_detector import HallucinationDetector
from lib.s6_teacher_benchmarking.reasoning_judge import ReasoningJudge

logger = logging.getLogger(__name__)


def make_teacher_backend(cfg: PipelineConfig, teacher_name: str):
    """Instantiate the teacher backend using the candidate teacher name."""
    from lib.s4_rad_prep.trace_generator import LocalHFBackend, APIBackend, BedrockBackend
    
    # Create copy of config to avoid side effects
    teacher_cfg = copy.deepcopy(cfg)
    teacher_cfg.rad.teacher_model_name = teacher_name
    
    # Override backend type from BENCHMARK_TEACHER_BACKEND if set
    backend_type = cfg.benchmarking.teacher_backend or cfg.rad.teacher_backend
    teacher_cfg.rad.teacher_backend = backend_type
    teacher_cfg.rad.teacher_batch_size = cfg.benchmarking.teacher_batch_size
    
    logger.info(f"Instantiating teacher backend '{backend_type}' for model '{teacher_name}'...")
    
    if backend_type == "hf_local":
        return LocalHFBackend(teacher_cfg)
    elif backend_type == "api":
        return APIBackend(teacher_cfg)
    elif backend_type == "bedrock":
        return BedrockBackend(teacher_cfg)
    else:
        raise ValueError(f"Unknown teacher backend: {backend_type}")


def build_benchmark_prompt(question: str, ground_truth: str, retrieved_context: str, no_retrieval: bool) -> str:
    """Format prompt for the candidate teacher model."""
    if no_retrieval:
        return (
            "[SYSTEM]: You are a neuroscientist. Reason step-by-step using only your knowledge. "
            "Wrap your final answer inside \\boxed{}.\n\n"
            "[NO CONTEXT AVAILABLE]\n"
            f"[QUESTION]: {question}\n"
            f"[GROUND TRUTH]: {ground_truth}"
        )
    else:
        return (
            "[SYSTEM]: You are a neuroscientist. Reason step-by-step using the provided context. "
            "Annotate key statements with bracketed passage citations matching the provided context (e.g. [Context 1] or [Passage 1]). "
            "Wrap your final answer inside \\boxed{}.\n\n"
            f"[CONTEXT]: {retrieved_context}\n"
            f"[QUESTION]: {question}\n"
            f"[GROUND TRUTH]: {ground_truth}"
        )


def load_qa_choices(qa_samples_path: Path) -> Dict[str, List[str]]:
    """Loads and maps question text to choices list for multiple-choice resolution."""
    choices_map = {}
    if qa_samples_path.exists():
        try:
            with open(qa_samples_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line)
                        question = data.get("question")
                        choices = data.get("choices")
                        if question and choices:
                            choices_map[question.strip().lower()] = [str(c) for c in choices]
        except Exception as e:
            logger.error(f"Error loading QA choices from {qa_samples_path}: {e}")
    return choices_map


def run_benchmark_generation_and_scoring(
    cfg: PipelineConfig,
    eval_samples: Dict[str, List[EvalSample]],
    judge_backend_override: Optional[Any] = None,
    teacher_backend_overrides: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    """
    Orchestrates the evaluation loop: For each teacher model and cluster,
    generates responses on evaluation samples, evaluates using the 4 scoring submodules,
    and returns a list of BenchmarkRecord dictionaries.
    """
    logger.info("Initializing scorers...")
    hallucination_detector = HallucinationDetector(cfg)
    reasoning_judge = ReasoningJudge(cfg, backend=judge_backend_override)
    
    # Load choices for MC accuracy check
    qa_samples_path = Path(cfg.rad.qa_samples_path)
    choices_map = load_qa_choices(qa_samples_path)
    
    # Load student tokenizer for token counting
    logger.info(f"Loading student tokenizer '{cfg.model.base_model_name}' for token counting...")
    student_tokenizer = AutoTokenizer.from_pretrained(cfg.model.base_model_name)
    
    records = []
    
    for teacher in cfg.benchmarking.candidate_teachers:
        logger.info(f"Starting benchmarking for teacher model: {teacher}")
        
        # Instantiate teacher backend
        if teacher_backend_overrides and teacher in teacher_backend_overrides:
            backend = teacher_backend_overrides[teacher]
        else:
            backend = make_teacher_backend(cfg, teacher)
            
        for cluster_label, samples in eval_samples.items():
            if not samples:
                logger.info(f"No eval samples for cluster {cluster_label}. Skipping.")
                continue
                
            logger.info(f"Evaluating teacher {teacher} on cluster {cluster_label} with {len(samples)} samples...")
            
            # 1. Prepare prompts
            prompts = []
            for sample in samples:
                prompt = build_benchmark_prompt(
                    question=sample.question,
                    ground_truth=sample.ground_truth,
                    retrieved_context=sample.retrieved_context,
                    no_retrieval=sample.no_retrieval
                )
                prompts.append(prompt)
                
            # 2. Batch generate teacher traces
            traces = []
            batch_size = cfg.benchmarking.teacher_batch_size
            for i in range(0, len(prompts), batch_size):
                batch_prompts = prompts[i:i + batch_size]
                batch_results = backend.generate_batch(batch_prompts)
                traces.extend(batch_results)
                
            # 3. Score traces
            for sample, trace in zip(samples, traces):
                # Retrieve MC choices if question matches
                choices = choices_map.get(sample.question.strip().lower(), None)
                
                # Answer Accuracy
                ans_acc = score_answer_accuracy(trace, sample.ground_truth, choices)
                
                # Reasoning Quality (LLM-as-judge)
                judge_result = reasoning_judge.score_reasoning_quality(
                    sample_id=sample.sample_id,
                    cluster_label=cluster_label,
                    question=sample.question,
                    retrieved_context=sample.retrieved_context,
                    trace=trace
                )
                if judge_result is not None:
                    res_quality, _ = judge_result
                else:
                    res_quality = None
                    
                # Citation Accuracy
                cit_prec, cit_rec, cit_acc = score_citation_accuracy(
                    trace=trace,
                    retrieved_context=sample.retrieved_context,
                    no_retrieval=sample.no_retrieval,
                    min_overlap=cfg.benchmarking.citation_min_overlap
                )
                
                # Hallucination Rate
                hal_rate = hallucination_detector.score_hallucination_rate(
                    trace=trace,
                    retrieved_context=sample.retrieved_context,
                    ground_truth=sample.ground_truth
                )
                
                # Token count of generated trace
                token_count = len(student_tokenizer.encode(trace, add_special_tokens=False))
                
                record = {
                    "teacher_model": teacher,
                    "cluster_label": cluster_label,
                    "cluster_id": int(sample.cluster_id) if sample.cluster_id else -1,
                    "sample_id": sample.sample_id,
                    "answer_accuracy": ans_acc,
                    "reasoning_quality": res_quality,
                    "citation_precision": cit_prec,
                    "citation_recall": cit_rec,
                    "citation_accuracy": cit_acc,
                    "hallucination_rate": hal_rate,
                    "no_retrieval": sample.no_retrieval,
                    "teacher_trace": trace,
                    "token_count": token_count
                }
                records.append(record)
                
    return records
