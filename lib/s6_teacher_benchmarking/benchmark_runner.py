import os
import json
import logging
import copy
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from transformers import AutoTokenizer

from lib.utils import PipelineConfig
from lib.s6_teacher_benchmarking.eval_sampler import EvalSample
from lib.s6_teacher_benchmarking.metric_eval_judge import MetricEvalJudge
from lib.s6_teacher_benchmarking.failure_logger import BenchmarkFailureLogger

logger = logging.getLogger(__name__)


def make_teacher_backend(cfg: PipelineConfig, teacher_name: str):
    """Instantiate the teacher backend using the candidate teacher name."""
    from lib.utils.teacher_backend import LocalHFBackend, APIBackend, BedrockBackend

    
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
            "[SYSTEM]: You are an expert neuroscientist. Answer the question using step-by-step reasoning.\n\n"
            "INSTRUCTIONS:\n"
            "1. Provide a step-by-step scientific explanation.\n"
            "2. At the end of your response, state your final concise answer wrapped inside \\boxed{answer}.\n"
            "   Do NOT include extra introductory text or citations inside the \\boxed{} block.\n\n"
            "[NO CONTEXT AVAILABLE]\n"
            f"[QUESTION]: {question}"
        )
    else:
        # Format context chunks into explicitly numbered passages (cap at top 5 to prevent context over-population)
        passages = [p.strip() for p in retrieved_context.split("\n\n") if p.strip()][:5]
        if len(passages) > 1:
            formatted_context = "\n\n".join(f"[Passage {i+1}]: {p}" for i, p in enumerate(passages))
        else:
            formatted_context = f"[Passage 1]: {retrieved_context.strip()}"

        return (
            "[SYSTEM]: You are an expert neuroscientist. Answer the question using the provided context passages.\n\n"
            "INSTRUCTIONS:\n"
            "1. Reason step-by-step using facts from the context passages.\n"
            "2. Annotate key statements with bracketed passage citations matching the provided passages (e.g. [Passage 1] or [Passage 2]) or direct quotes.\n"
            "3. At the end of your response, state your final concise answer wrapped inside \\boxed{answer}.\n"
            "   Do NOT include citations or extra introductory text inside the \\boxed{} block.\n\n"
            f"[CONTEXT PASSAGES]:\n{formatted_context}\n\n"
            f"[QUESTION]: {question}"
        )


def run_benchmark_generation_and_scoring(
    cfg: PipelineConfig,
    eval_samples: Dict[str, List[EvalSample]],
    judge_backend_override: Optional[Any] = None,
    teacher_backend_overrides: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    """
    Orchestrates the evaluation loop: For each teacher model and cluster,
    generates responses on evaluation samples, evaluates using MetricEvalJudge,
    and returns a list of BenchmarkRecord dictionaries.
    """
    logger.info("Initializing MetricEvalJudge and failure logger...")
    metric_judge = MetricEvalJudge(cfg, backend=judge_backend_override)
    failure_logger = BenchmarkFailureLogger(cfg, reset=True)
    
    # Load student tokenizer for token counting
    logger.info(f"Loading student tokenizer '{cfg.model.base_model_name}' for token counting...")
    student_tokenizer = AutoTokenizer.from_pretrained(cfg.model.base_model_name)
    
    records = []
    seq_num = 0
    
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
                
            # 3. Score traces using MetricEvalJudge
            for sample, prompt, trace in zip(samples, prompts, traces):
                seq_num += 1
                
                eval_res = metric_judge.evaluate_trace(
                    sample_id=sample.sample_id,
                    cluster_label=cluster_label,
                    question=sample.question,
                    ground_truth=sample.ground_truth,
                    retrieved_context=sample.retrieved_context,
                    trace=trace,
                    no_retrieval=sample.no_retrieval
                )
                
                if eval_res is not None:
                    ans_acc = eval_res["answer_accuracy"]
                    res_quality = eval_res["reasoning_quality"]
                    cit_prec = eval_res["citation_precision"]
                    cit_rec = eval_res["citation_recall"]
                    cit_acc = eval_res["citation_accuracy"]
                    hal_rate = eval_res["hallucination_rate"]
                else:
                    ans_acc = 0.0
                    res_quality = None
                    cit_prec = None
                    cit_rec = None
                    cit_acc = None
                    hal_rate = 1.0
                
                # Check and log failures for any metrics that fell below thresholds
                failure_logger.check_and_log_failures(
                    seq_num=seq_num,
                    teacher=teacher,
                    sample_id=sample.sample_id,
                    cluster_label=cluster_label,
                    question=sample.question,
                    ground_truth=sample.ground_truth,
                    retrieved_context=sample.retrieved_context,
                    prompt=prompt,
                    response=trace,
                    answer_accuracy=ans_acc,
                    reasoning_quality=res_quality,
                    citation_accuracy=cit_acc,
                    hallucination_rate=hal_rate,
                    no_retrieval=sample.no_retrieval,
                    judge_prompt=eval_res.get("judge_prompt") if eval_res else getattr(metric_judge, "last_prompt", None),
                    judge_response=eval_res.get("judge_response") if eval_res else getattr(metric_judge, "last_response", None),
                    explanations={
                        "answer_explanation": eval_res.get("answer_explanation", "") if eval_res else "",
                        "reasoning_explanation": eval_res.get("reasoning_explanation", "") if eval_res else "",
                        "citation_explanation": eval_res.get("citation_explanation", "") if eval_res else "",
                        "hallucination_explanation": eval_res.get("hallucination_explanation", "") if eval_res else "",
                    } if eval_res else None
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

