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

from lib.utils import setup_logger, get_logger, get_adapter, StorageAdapter, PipelineConfig
from .worker import worker_init, worker_task, ExtractionResult

logger = get_logger("s1.build")


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
        logger.info(
            f"[START] {self.name} pipeline | "
            f"{self.total_workers} workers | GPUs {self.gpu_ids}"
        )

        gpu_queue = self._make_gpu_queue()

        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        total_tokens = 0
        doc_index = 0

        # We use multiprocessing.Pool with maxtasksperchild to prevent memory accumulation on CPU.
        # Recycling the worker processes regularly releases memory back to the OS.
        pool = multiprocessing.Pool(
            processes=self.total_workers,
            initializer=worker_init,
            initargs=(gpu_queue,),
            maxtasksperchild=10,
        )

        try:
            with open(self.output_path, "w", encoding="utf-8") as out:
                import pypdfium2 as pdfium

                for filename, path, is_temp in self.storage.stream_pdfs():
                    try:
                        doc = pdfium.PdfDocument(path)
                        try:
                            page_count = len(doc)
                        finally:
                            doc.close()
                    except Exception as e:
                        logger.error(f"[ERROR] Could not open PDF {filename}: {e}")
                        if is_temp:
                            _try_delete(path)
                        continue

                    if page_count == 0:
                        logger.warning(f"[SKIP] {filename} has 0 pages")
                        if is_temp:
                            _try_delete(path)
                        continue

                    # Create chunk tasks for this PDF
                    chunk_size = max(2, self.chunk_size)
                    stride = chunk_size - 1
                    
                    pdf_tasks = []
                    start = 1
                    chunk_idx = 0
                    while start <= page_count:
                        end = min(start + chunk_size - 1, page_count)
                        pdf_tasks.append((filename, path, start, end, chunk_idx))
                        if end >= page_count:
                            break
                        start += stride
                        chunk_idx += 1

                    logger.info(f"[PLAN] Processing {filename} ({page_count} pages) in {len(pdf_tasks)} chunks...")

                    # Run tasks for this PDF in parallel using the pool
                    results = pool.starmap(worker_task, pdf_tasks, chunksize=1)

                    # Gather and sort valid chunks
                    valid_chunks = []
                    for result in results:
                        if result.succeeded:
                            valid_chunks.extend(result.chunks)
                        else:
                            logger.warning(f"[SKIP CHUNK] {result.filename} | {result.status}")

                    valid_chunks.sort(key=lambda c: c.chunk_index)

                    for chunk in valid_chunks:
                        total_tokens += chunk.token_count
                        out.write(json.dumps(self._record(doc_index, filename, chunk)) + "\n")
                        out.flush()
                        logger.info(
                            f"[OK] #{doc_index} {filename} (chunk {chunk.chunk_index}) | "
                            f"{chunk.token_count:,} tokens | cumulative {total_tokens:,}"
                        )
                        doc_index += 1

                    if is_temp:
                        _try_delete(path)

        finally:
            pool.close()
            pool.join()

        logger.info(
            f"[DONE] {self.name} - {doc_index} chunks from documents, "
            f"{total_tokens:,} tokens -> {self.output_path}"
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _make_gpu_queue(self):
        q = multiprocessing.Queue()
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


def run_corpus_builder(cfg: PipelineConfig) -> None:
    """
    Unified entry point for the corpus builder pipeline.
    Instantiates the storage adapter and corpus builder, then executes the pipeline.
    """
    setup_logger("s1.build", log_dir=Path("logs"), log_filename="corpus_building.log")
    
    storage = get_adapter(
        target=cfg.build.storage_target,
        local_directory_path=cfg.build.local_directory_path,
        aws_bucket_name=cfg.build.aws_bucket_name,
        aws_prefix=cfg.build.aws_prefix,
        gdrive_folder_id=cfg.build.gdrive_folder_id,
    )

    builder = CorpusBuilder(
        storage=storage,
        output_path=str(cfg.build.output_path),
        available_gpus=cfg.build.available_gpus,
        workers_per_gpu=cfg.build.workers_per_gpu,
        chunk_size=cfg.build.chunk_size,
    )
    builder.build()

