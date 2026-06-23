import sys
import time
import argparse

from lib.s1_build_corpus import run_corpus_builder
from lib.s1_5_pretokenize import run_pretokenization
from lib.s2_dapt import run_dapt_pipeline
from lib.utils import PipelineConfig


def s1(cfg: PipelineConfig) -> None:
    print("Initializing corpus building pipeline using Docling parser")

    try:
        run_corpus_builder(cfg)
    except Exception as e:
        print(f"Error executing corpus builder: {e}", file=sys.stderr)
        sys.exit(1)


def s1_5(cfg: PipelineConfig) -> None:
    print("Initializing offline pre-tokenization step")

    try:
        run_pretokenization(cfg)
    except Exception as e:
        print(f"Error executing pre-tokenization: {e}", file=sys.stderr)
        sys.exit(1)


def s2(cfg: PipelineConfig) -> None:
    print(f"Initializing DAPT Continued Pretraining on model: {cfg.model.base_model_name}")

    try:
        run_dapt_pipeline(cfg)
    except Exception as e:
        print(f"Error executing DAPT pipeline: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Semantics Layer Pipeline")
    parser.add_argument(
        "--step",
        choices=["s1", "s1.5", "s2", "all"],
        default="all",
        help="Pipeline step to execute: s1 (Corpus Construction), s1.5 (Pre-tokenization), s2 (Domain Adaptive Pretraining)"
    )
    args = parser.parse_args()
    print(f"Args step are : {args.step}")
    
    # Instantiate and validate configuration
    cfg = PipelineConfig()
    cfg.validate()
    cfg.ensure_dirs()
    
    if args.step == "s1" or args.step == "all":
        s1(cfg)
    if args.step == "s1.5" or args.step == "all":
        s1_5(cfg)
    if args.step == "s2" or args.step == "all":
        s2(cfg)
        print("Running final inference on saved model and logging failed evaluations...")
        try:
            from lib.s2_dapt.evaluation.eval_runner import run_inference_and_log_failures
            run_inference_and_log_failures(cfg)
        except Exception as e:
            print(f"Error logging failed evaluations: {e}", file=sys.stderr)
            sys.exit(1)



