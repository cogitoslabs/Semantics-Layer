import logging
from typing import Dict, List, Any, Optional

from lib.utils import PipelineConfig, setup_logger, flush_loggers
from lib.s6_teacher_benchmarking.eval_sampler import run_eval_sampling
from lib.s6_teacher_benchmarking.benchmark_runner import run_benchmark_generation_and_scoring
from lib.s6_teacher_benchmarking.metric_eval_judge import run_cohen_kappa_evaluation
from lib.s6_teacher_benchmarking.benchmark_reporter import run_benchmark_reporting

logger = logging.getLogger(__name__)


def run_teacher_benchmarking(
    cfg: PipelineConfig,
    judge_backend_override: Optional[Any] = None,
    teacher_backend_overrides: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Main entry point for Phase 2, Step 2.1 — Teacher Benchmarking.
    Orchestrates eval sampling, trace generation, multi-dimension scoring,
    and manifest/score aggregation.
    """
    # Guarantee pipeline.log is initialized and attached
    setup_logger("lib", cfg.logging)
    
    logger.info("Starting Phase 2, Step 2.1 — Teacher Benchmarking...")
    cfg.ensure_dirs()
    
    # 1. Draw validation eval samples
    eval_samples = run_eval_sampling(cfg)
    
    expected_teacher_count = len(cfg.benchmarking.candidate_teachers)
    expected_cluster_count = len(eval_samples)
    
    # 2. Run multi-teacher trace generation and scoring
    records = run_benchmark_generation_and_scoring(
        cfg=cfg,
        eval_samples=eval_samples,
        judge_backend_override=judge_backend_override,
        teacher_backend_overrides=teacher_backend_overrides
    )
    
    # 3. Perform inter-rater agreement check if human labels are provided
    if cfg.benchmarking.enable_calibration:
        logger.info("Inter-rater calibration checking...")
        run_cohen_kappa_evaluation(cfg)
        
    # 4. Generate final score reports and check gates
    manifest = run_benchmark_reporting(
        cfg=cfg,
        records=records,
        expected_teacher_count=expected_teacher_count,
        expected_cluster_count=expected_cluster_count
    )
    
    logger.info("Teacher Benchmarking pipeline step completed successfully.")
    flush_loggers()
    return manifest
