"""
Parallel corpus builder: distributes PDF extraction across GPU workers,
writes results to a JSONL file, and reports progress.
"""

import json
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, Optional

from .storage import get_adapter, StorageAdapter
from .worker import worker_init, worker_task, ExtractionResult


class CorpusBuilder:
    """
    Streams PDFs from the configured storage backend, dispatches extraction
    jobs across GPU-pinned worker processes, and writes a DAPT-ready JSONL corpus.
    """

    def __init__(
        self,
        storage: StorageAdapter,
        output_path: str,
        available_gpus: str = "0",
        workers_per_gpu: int = 1,
    ):
        self.name = "Docling"
        self.storage = storage
        self.output_path = Path(output_path)

        self.gpu_ids = [int(g.strip()) for g in available_gpus.split(",")]
        self.workers_per_gpu = workers_per_gpu
        self.total_workers = len(self.gpu_ids) * self.workers_per_gpu


    def build(self) -> None:
        print(
            f"[START] {self.name} pipeline | "
            f"{self.total_workers} workers | GPUs {self.gpu_ids}"
        )

        gpu_queue = self._make_gpu_queue()

        total_tokens = 0
        doc_index = 0

        with (
            open(self.output_path, "w", encoding="utf-8") as out,
            ProcessPoolExecutor(
                max_workers=self.total_workers,
                initializer=worker_init,
                initargs=(gpu_queue,),
            ) as pool,
        ):
            futures = {}
            for filename, path, is_temp in self.storage.stream_pdfs():
                future = pool.submit(worker_task, filename, path)
                futures[future] = (path, is_temp)

            for future in as_completed(futures):
                path, is_temp = futures[future]
                result: ExtractionResult = future.result()

                if is_temp:
                    _try_delete(path)

                if result.succeeded:
                    total_tokens += result.token_count
                    out.write(json.dumps(self._record(doc_index, result)) + "\n")
                    print(
                        f"[OK] #{doc_index} {result.filename} | "
                        f"{result.token_count:,} tokens | cumulative {total_tokens:,}"
                    )
                    doc_index += 1
                else:
                    print(f"[SKIP] {result.filename} | {result.status}")

        print(
            f"\n[DONE] {self.name} - {doc_index} documents, "
            f"{total_tokens:,} tokens -> {self.output_path}"
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _make_gpu_queue(self):  # returns multiprocessing.managers.AutoProxy[Queue]
        manager = multiprocessing.Manager()
        q = manager.Queue()
        for gpu_id in self.gpu_ids:
            for _ in range(self.workers_per_gpu):
                q.put(gpu_id)
        return q

    @staticmethod
    def _record(index: int, result: ExtractionResult) -> dict:
        return {
            "id": f"domain_doc_{index:06d}",
            "source_file": result.filename,
            "text": result.text,
            "token_count": result.token_count,
        }


def _try_delete(path: str) -> None:
    try:
        os.unlink(path)
    except OSError:
        pass


def run_corpus_builder(
    output_path: str,
    storage_target: str,
    local_directory_path: Optional[str] = None,
    aws_bucket_name: Optional[str] = None,
    aws_prefix: Optional[str] = None,
    gdrive_folder_id: Optional[str] = None,
    available_gpus: str = "0",
    workers_per_gpu: int = 1,
) -> None:
    """
    Unified entry point for the corpus builder pipeline.
    Instantiates the storage adapter and corpus builder, then executes the pipeline.
    """
    storage = get_adapter(
        target=storage_target,
        local_directory_path=local_directory_path,
        aws_bucket_name=aws_bucket_name,
        aws_prefix=aws_prefix,
        gdrive_folder_id=gdrive_folder_id,
    )

    builder = CorpusBuilder(
        storage=storage,
        output_path=output_path,
        available_gpus=available_gpus,
        workers_per_gpu=workers_per_gpu,
    )
    builder.build()

