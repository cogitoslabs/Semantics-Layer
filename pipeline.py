import sys
import time
import argparse

from lib.s1_build_corpus import run_corpus_builder, run_replay_corpus, run_merge_corpus
from lib.s2_pretokenize import run_pretokenization
from lib.s3_dapt import run_dapt_pipeline
from lib.s3_dapt.evaluation.eval_runner import run_inference_and_log_failures
from lib.s4_rad_prep import run_rad_prep_pipeline
from lib.s5_clustering import run_clustering_pipeline
from lib.s6_teacher_benchmarking import run_teacher_benchmarking

from lib.utils import PipelineConfig, setup_logger, get_logger


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Semantics Layer Pipeline")
    parser.add_argument(
        "--step",
        choices=["s1", "s2", "s3", "s4", "s5", "s6", "all"],
        default="all",
        help="Pipeline step to execute: s1 (Corpus Construction), s2 (Pre-tokenization), s3 (Domain Adaptive Pretraining), s4 (Retrieval-Augmented Distillation Prep), s5 (Corpus Engineering & Micro-Clustering), s6 (Teacher Benchmarking)"
    )
    parser.add_argument(
        "--rad-mode",
        choices=["index", "prompts", "traces", "full"],
        default="full",
        help="Sub-mode for step s4: index (chunk & index corpus), prompts (prepare grounded QA prompts), full (index and prepare prompts)"
    )

    args = parser.parse_args()
    # Instantiate and validate configuration
    cfg = PipelineConfig()
    cfg.validate()
    cfg.ensure_dirs()
    
    # Initialize pipeline logging
    setup_logger("pipeline", cfg.logging)
    logger = get_logger("pipeline")
    
    logger.info(f"Args step are : {args.step}")
    
    if args.step == "s1" or args.step == "all":
        logger.info("Initializing corpus building pipeline using Docling parser")
        run_corpus_builder(cfg)
    if args.step == "s2" or args.step == "all":
        logger.info("Fetching general web corpus replay data...")
        run_replay_corpus(cfg)
        logger.info("Merging extracted and replay corpora...")
        run_merge_corpus(cfg)
        logger.info("Initializing offline pre-tokenization step")
        run_pretokenization(cfg)
    if args.step == "s3" or args.step == "all":
        logger.info(f"Initializing DAPT Continued Pretraining on model: {cfg.model.base_model_name}")
        run_dapt_pipeline(cfg)
        logger.info("Running final inference on saved model and logging failed evaluations...")
        run_inference_and_log_failures(cfg)
    if args.step == "s4" or args.step == "all":
        rad_mode = getattr(args, "rad_mode", "full")
        logger.info(f"Initializing Retrieval-Augmented Distillation Preparation (RAD Prep) in mode: {rad_mode}")
        run_rad_prep_pipeline(cfg, rad_mode)
    if args.step == "s5" or args.step == "all":
        logger.info("Initializing Corpus Engineering & Micro-Clustering (Step 5)")
        run_clustering_pipeline(cfg)
    if args.step == "s6" or args.step == "all":
        logger.info("Initializing Teacher Benchmarking (Phase 2, Step 2.1)")
        run_teacher_benchmarking(cfg)


