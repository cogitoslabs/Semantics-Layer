import sys
import time
import argparse

from lib.s1_build_corpus import run_corpus_builder
from lib.s2_dapt import run_dapt_pipeline
from lib.utils import PipelineConfig


def calculate_time(func):
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        print(f"Function '{func.__name__}' took {end - start:.6f} seconds to run.")
        return result
    return wrapper

@calculate_time
def s1(cfg: PipelineConfig) -> None:
    print("Initializing corpus building pipeline using Docling parser")

    try:
        run_corpus_builder(
            output_path=str(cfg.build.output_path),
            storage_target=cfg.build.storage_target,
            local_directory_path=cfg.build.local_directory_path,
            aws_bucket_name=cfg.build.aws_bucket_name,
            aws_prefix=cfg.build.aws_prefix,
            gdrive_folder_id=cfg.build.gdrive_folder_id,
            available_gpus=cfg.build.available_gpus,
            workers_per_gpu=cfg.build.workers_per_gpu,
            chunk_size=cfg.build.chunk_size,
        )
    except Exception as e:
        print(f"Error executing corpus builder: {e}", file=sys.stderr)
        sys.exit(1)


@calculate_time
def s2(cfg: PipelineConfig) -> None:
    print(f"Initializing DAPT Continued Pretraining on model: {cfg.model.base_model_name}")

    try:
        run_dapt_pipeline(
            model_name=cfg.model.base_model_name,
            corpus_path=str(cfg.build.output_path),
            probe_qa_path=str(cfg.data.qa_probe_path),
            epochs=cfg.corpus.max_corpus_passes,
            lr=cfg.optimizer.learning_rate,
            batch_size=cfg.optimizer.train_batch_size,
            output_dir=str(cfg.storage.checkpoint_dir),
        )
    except Exception as e:
        print(f"Error executing DAPT pipeline: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Semantics Layer Pipeline")
    parser.add_argument(
        "--step",
        choices=["s1", "s2", "all"],
        default="all",
        help="Pipeline step to execute: s1 (Corpus Construction), s2 (Domain Adaptive Pretraining)"
    )
    args = parser.parse_args()
    print(f"Args step are : {args.step}")
    
    # Instantiate and validate configuration
    cfg = PipelineConfig()
    cfg.validate()
    cfg.ensure_dirs()
    
    if args.step == "s1" or args.step == "all":
        s1(cfg)
    if args.step == "s2" or args.step == "all":
        s2(cfg)


