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

from lib.utils import PipelineConfig


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
        choices=["index", "traces", "full"],
        default="full",
        help="Sub-mode for step s4: index (chunk & index corpus), traces (generate traces), full (index and generate)"
    )
    args = parser.parse_args()
    print(f"Args step are : {args.step}")
    
    # Instantiate and validate configuration
    cfg = PipelineConfig()
    cfg.validate()
    cfg.ensure_dirs()
    
    if args.step == "s1" or args.step == "all":
        print("Initializing corpus building pipeline using Docling parser")
        run_corpus_builder(cfg)
        print("Fetching general web corpus replay data...")
        run_replay_corpus(cfg)
        print("Merging extracted and replay corpora...")
        run_merge_corpus(cfg)
    if args.step == "s2" or args.step == "all":
        print("Initializing offline pre-tokenization step")
        run_pretokenization(cfg)
    if args.step == "s3" or args.step == "all":
        print("Initializing DAPT Continued Pretraining on model: {cfg.model.base_model_name}")
        run_dapt_pipeline(cfg)
        print("Running final inference on saved model and logging failed evaluations...")
        run_inference_and_log_failures(cfg)
    if args.step == "s4" or args.step == "all":
        rad_mode = getattr(args, "rad_mode", "full")
        print(f"Initializing Retrieval-Augmented Distillation Preparation (RAD Prep) in mode: {rad_mode}")
        run_rad_prep_pipeline(cfg, rad_mode)
    if args.step == "s5" or args.step == "all":
        print("Initializing Corpus Engineering & Micro-Clustering (Step 5)")
        run_clustering_pipeline(cfg)
    if args.step == "s6" or args.step == "all":
        print("Initializing Teacher Benchmarking (Phase 2, Step 2.1)")
        run_teacher_benchmarking(cfg)


