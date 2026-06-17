"""
Parallel corpus builder: distributes PDF extraction across GPU workers,
writes results to a JSONL file, and reports progress.
"""

import json
import multiprocessing
import os
import threading
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Optional

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
        chunk_size: int = 10,
    ):
        self.name = "Docling"
        self.storage = storage
        self.output_path = Path(output_path)
        self.chunk_size = chunk_size

        self.gpu_ids = [int(g.strip()) for g in available_gpus.split(",")]
        self.workers_per_gpu = workers_per_gpu
        self.total_workers = len(self.gpu_ids) * self.workers_per_gpu


    def build(self) -> None:
        print(
            f"[START] {self.name} pipeline | "
            f"{self.total_workers} workers | GPUs {self.gpu_ids}",
            flush=True
        )

        gpu_queue = self._make_gpu_queue()

        # Create a manager and shared queue for streaming chunk results in real-time
        manager = multiprocessing.Manager()
        chunk_queue = manager.Queue()

        sent_chunks = set()
        sent_lock = threading.Lock()

        total_tokens = 0
        doc_index = 0

        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(self.output_path, "w", encoding="utf-8") as out:
            # Start a background thread to process and write chunks immediately as they are put on the queue
            def write_loop():
                nonlocal total_tokens, doc_index
                while True:
                    item = chunk_queue.get()
                    if item is None:
                        break
                    filename, chunk = item
                    total_tokens += chunk.token_count
                    out.write(json.dumps(self._record(doc_index, filename, chunk)) + "\n")
                    out.flush()
                    print(
                        f"[OK] #{doc_index} {filename} (chunk {chunk.chunk_index}) | "
                        f"{chunk.token_count:,} tokens | cumulative {total_tokens:,}",
                        flush=True
                    )
                    doc_index += 1

                    with sent_lock:
                        sent_chunks.add((filename, chunk.chunk_index))

            writer_thread = threading.Thread(target=write_loop)
            writer_thread.start()

            try:
                with ProcessPoolExecutor(
                    max_workers=self.total_workers,
                    initializer=worker_init,
                    initargs=(gpu_queue,),
                ) as pool:
                    futures = {}
                    for filename, path, is_temp in self.storage.stream_pdfs():
                        future = pool.submit(worker_task, filename, path, self.chunk_size, chunk_queue)
                        futures[future] = (path, is_temp)

                    for future in as_completed(futures):
                        path, is_temp = futures[future]
                        result: ExtractionResult = future.result()

                        if is_temp:
                            _try_delete(path)

                        if result.succeeded:
                            # Fallback for mock/test runs: if chunks are in result but not in the queue
                            for chunk in result.chunks:
                                chunk_key = (result.filename, chunk.chunk_index)
                                with sent_lock:
                                    already_sent = chunk_key in sent_chunks
                                    if not already_sent:
                                        sent_chunks.add(chunk_key)
                                if not already_sent:
                                    chunk_queue.put((result.filename, chunk))
                        else:
                            print(f"[SKIP] {result.filename} | {result.status}", flush=True)
            finally:
                # Signal the writer thread to stop and wait for it
                chunk_queue.put(None)
                writer_thread.join()

        print(
            f"\n[DONE] {self.name} - {doc_index} chunks from documents, "
            f"{total_tokens:,} tokens -> {self.output_path}",
            flush=True
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
    def _record(index: int, filename: str, chunk: Any) -> dict:
        return {
            "id": f"domain_doc_{index:06d}",
            "source_file": filename,
            "chunk_id": chunk.chunk_index,
            "page_range": list(chunk.page_range),
            "text": chunk.text,
            "token_count": chunk.token_count,
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
    chunk_size: int = 10,
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
        chunk_size=chunk_size,
    )
    builder.build()

