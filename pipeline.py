import os
import sys
from dotenv import load_dotenv
import time
import argparse

from lib.s1_build_corpus import run_corpus_builder
from lib.s2_dapt import run_dapt_pipeline


def calculate_time(func):
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        print(f"Function '{func.__name__}' took {end - start:.6f} seconds to run.")
        return result
    return wrapper

@calculate_time
def s1() -> None:
    # Load environment variables from .env file
    output_path = os.getenv("OUTPUT_PATH", "./domain_dapt_corpus.jsonl")
    storage_target = os.getenv("STORAGE_TARGET", "local")
    local_directory_path = os.getenv("LOCAL_DIRECTORY_PATH", ".")
    aws_bucket_name = os.getenv("AWS_BUCKET_NAME")
    aws_prefix = os.getenv("AWS_PREFIX", "")
    gdrive_folder_id = os.getenv("GDRIVE_FOLDER_ID")
    available_gpus = os.getenv("AVAILABLE_GPUS", "0")
    
    try:
        workers_per_gpu = int(os.getenv("WORKERS_PER_GPU", "1"))
    except ValueError:
        workers_per_gpu = 1

    try:
        chunk_size = int(os.getenv("CHUNK_SIZE", "10"))
    except ValueError:
        chunk_size = 10

    print("Initializing corpus building pipeline using Docling parser")

    try:
        run_corpus_builder(
            output_path=output_path,
            storage_target=storage_target,
            local_directory_path=local_directory_path,
            aws_bucket_name=aws_bucket_name,
            aws_prefix=aws_prefix,
            gdrive_folder_id=gdrive_folder_id,
            available_gpus=available_gpus,
            workers_per_gpu=workers_per_gpu,
            chunk_size=chunk_size,
        )
    except Exception as e:
        print(f"Error executing corpus builder: {e}", file=sys.stderr)
        sys.exit(1)


@calculate_time
def s2() -> None:
    # Load DAPT environment variables from .env file
    model_name = os.getenv("BASE_MODEL_NAME", "Qwen/Qwen2.5-0.5B")
    corpus_path = os.getenv("OUTPUT_PATH", "./data/dapt/domain_dapt_corpus.jsonl")
    probe_qa_path = os.getenv("PROBE_QA_PATH", "./data/dapt/probe_qa.jsonl")
    output_dir = os.getenv("DAPT_OUTPUT_DIR", "./models/dapt_model")
    
    try:
        epochs = int(os.getenv("DAPT_EPOCHS", "3"))
        lr = float(os.getenv("DAPT_LR", "5e-5"))
        batch_size = int(os.getenv("DAPT_BATCH_SIZE", "2"))
    except ValueError:
        epochs = 3
        lr = 5e-5
        batch_size = 2

    print(f"Initializing DAPT Continued Pretraining on model: {model_name}")

    try:
        run_dapt_pipeline(
            model_name=model_name,
            corpus_path=corpus_path,
            probe_qa_path=probe_qa_path,
            epochs=epochs,
            lr=lr,
            batch_size=batch_size,
            output_dir=output_dir,
        )
    except Exception as e:
        print(f"Error executing DAPT pipeline: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    load_dotenv()
    
    parser = argparse.ArgumentParser(description="Semantics Layer Pipeline")
    parser.add_argument(
        "--step",
        choices=["s1", "s2", "all"],
        default="all",
        help="Pipeline step to execute: s1 (Corpus Construction), s2 (Domain Adaptive Pretraining)"
    )
    args = parser.parse_args()
    print(f"Args step are : {args.step}")
    
    if args.step == "s1" or args.step == "all":
        s1()
    if args.step == "s2" or args.step == "all":
        s2()


