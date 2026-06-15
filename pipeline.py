import os
import sys
from dotenv import load_dotenv
import time

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

    print("Initializing corpus building pipeline using Docling parser")

    try:
        from lib.s1_build_corpus import run_corpus_builder
        run_corpus_builder(
            output_path=output_path,
            storage_target=storage_target,
            local_directory_path=local_directory_path,
            aws_bucket_name=aws_bucket_name,
            aws_prefix=aws_prefix,
            gdrive_folder_id=gdrive_folder_id,
            available_gpus=available_gpus,
            workers_per_gpu=workers_per_gpu,
        )
    except Exception as e:
        print(f"Error executing corpus builder: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    load_dotenv()
    s1()


