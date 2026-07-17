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

from lib.utils import setup_logger, get_logger, get_adapter, StorageAdapter, PipelineConfig, CorpusBuildConfig
from lib.utils.config import LoggingConfig
from .worker import worker_init, worker_task, ExtractionResult

logger = get_logger(__name__)


def run_corpus_builder(cfg: PipelineConfig) -> None:
    """
    Unified entry point for the corpus builder pipeline.
    Instantiates the storage adapter and corpus builder, then executes the pipeline.
    """
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    import sys
    setup_logger(
        f"{__name__}.{sys._getframe().f_code.co_name}",
        cfg.logging,
    )
    global logger
    logger = get_logger(f"{__name__}.{sys._getframe().f_code.co_name}")

    # Log cached auto-resolution details to files and stream
    for log_msg in cfg.build.resolution_logs:
        logger.info(log_msg)
    
    storage = get_adapter(cfg.storage)
 
    builder = CorpusBuilder(
        storage=storage,
        cfg=cfg.build,
        logging_cfg=cfg.logging,
    )
    builder.build()


def try_delete(path: str) -> None:
    try:
        os.unlink(path)
    except OSError:
        pass


class CorpusBuilder:
    """
    Streams PDFs from the configured storage backend, dispatches extraction
    jobs across GPU-pinned worker processes, and writes a DAPT-ready JSONL corpus.
    """

    def __init__(
        self,
        storage: StorageAdapter,
        cfg: CorpusBuildConfig,
        logging_cfg: LoggingConfig,
    ):
        self.name = "Docling"
        self.storage = storage
        self.output_path = Path(cfg.extracted_output_path)
        self.gpu_ids = cfg.gpu_ids
        self.workers_per_gpu = cfg.workers_per_gpu
        self.total_workers = cfg.total_workers
        self.chunk_size = cfg.chunk_size
        self.maxtasksperchild = cfg.maxtasksperchild

        self.logging_cfg = logging_cfg
        self.docling_options = {
            "do_ocr": cfg.docling_use_ocr,
            "do_table_structure": cfg.docling_use_table_structure,
            "do_code_enrichment": cfg.docling_use_code_enrichment,
            "do_formula_enrichment": cfg.docling_use_formula_enrichment,
            "do_picture_classification": cfg.docling_use_picture_classification,
            "do_picture_description": cfg.docling_use_picture_description,
            "num_threads": cfg.docling_num_threads,
        }


    def build(self) -> None:
        logger.info(
            f"[START] {self.name} pipeline | "
            f"{self.total_workers} workers | GPUs {self.gpu_ids}"
        )

        gpu_queue = self.make_gpu_queue()

        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        total_tokens = 0
        doc_index = 0

        # We use multiprocessing.Pool with maxtasksperchild to prevent memory accumulation on CPU.
        # Recycling the worker processes regularly releases memory back to the OS.
        pool = multiprocessing.Pool(
            processes=self.total_workers,
            initializer=worker_init,
            initargs=(gpu_queue, self.docling_options, self.logging_cfg),
            maxtasksperchild=self.maxtasksperchild,
        )

        try:
            with open(self.output_path, "w", encoding="utf-8") as out:
                # Limit the number of concurrently active documents to prevent downloading
                # too many temp files simultaneously.
                max_active_docs = max(2, 2 * self.total_workers)
                active_docs = []  # list of (filename, path, is_temp, list_of_async_results)

                doc_generator = self.storage.stream_documents()
                more_docs = True

                while more_docs or active_docs:
                    # 1. Fill the queue with tasks
                    while more_docs and len(active_docs) < max_active_docs:
                        try:
                            filename, path, is_temp = next(doc_generator)
                        except StopIteration:
                            more_docs = False
                            break

                        suffix = Path(path).suffix.lower()
                        if suffix == ".pdf":
                            success = self.queue_pdf(pool, filename, path, is_temp, active_docs)
                            if not success:
                                continue
                        else:
                            self.queue_text_doc(pool, filename, path, is_temp, active_docs)

                    # 2. Process and write the oldest document in the queue
                    if active_docs:
                        filename, path, is_temp, async_results = active_docs[0]

                        # Block on the chunks of the current document to preserve output file ordering
                        results = []
                        for task in async_results:
                            results.append(task.get())

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
                            out.write(json.dumps(self.record(doc_index, filename, chunk)) + "\n")
                            out.flush()
                            logger.info(
                                f"[OK] #{doc_index} {filename} (chunk {chunk.chunk_index}) | "
                                f"{chunk.token_count:,} tokens | cumulative {total_tokens:,}"
                            )
                            doc_index += 1

                        if is_temp:
                            try_delete(path)

                        active_docs.pop(0)

        finally:
            pool.close()
            pool.join()

        logger.info(
            f"[DONE] {self.name} - {doc_index} chunks from documents, "
            f"{total_tokens:,} tokens -> {self.output_path}"
        )


    def make_gpu_queue(self):
        q = multiprocessing.Queue()
        for gpu_id in self.gpu_ids:
            for _ in range(self.workers_per_gpu):
                q.put(gpu_id)
        return q


    def queue_pdf(
        self,
        pool: Any,
        filename: str,
        path: str,
        is_temp: bool,
        active_docs: list
    ) -> bool:
        """
        Extracts PDF page count, splits pages into chunks, and queues async tasks.
        Returns True if successful, False if the PDF was skipped or failed to parse.
        Handles temporary file cleanup on failure.
        """
        import pypdfium2 as pdfium
        try:
            doc = pdfium.PdfDocument(path)
            try:
                page_count = len(doc)
            finally:
                doc.close()
        except Exception as e:
            logger.error(f"[ERROR] Could not open PDF {filename}: {e}")
            if is_temp:
                try_delete(path)
            return False

        if page_count == 0:
            logger.warning(f"[SKIP] {filename} has 0 pages")
            if is_temp:
                try_delete(path)
            return False

        chunk_size = max(2, self.chunk_size)
        stride = chunk_size - 1

        async_results = []
        start = 1
        chunk_idx = 0
        while start <= page_count:
            end = min(start + chunk_size - 1, page_count)
            task = pool.apply_async(
                worker_task,
                args=(filename, path, start, end, chunk_idx)
            )
            async_results.append(task)
            if end >= page_count:
                break
            start += stride
            chunk_idx += 1

        active_docs.append((filename, path, is_temp, async_results))
        logger.info(
            f"[PLAN] Queued PDF {filename} ({page_count} pages) in "
            f"{len(async_results)} chunks asynchronously..."
        )
        return True


    def queue_text_doc(
        self,
        pool: Any,
        filename: str,
        path: str,
        is_temp: bool,
        active_docs: list
    ) -> None:
        """Queues an async task for extracting the full text/HTML document."""
        async_results = []
        task = pool.apply_async(
            worker_task,
            args=(filename, path, None, None, 0, self.chunk_size)
        )
        async_results.append(task)
        active_docs.append((filename, path, is_temp, async_results))
        logger.info(
            f"[PLAN] Queued document {filename} asynchronously..."
        )


    @staticmethod
    def record(index: int, filename: str, chunk: Any) -> dict:
        return {
            "id": f"domain_doc_{index:06d}",
            "source_file": filename,
            "chunk_id": chunk.chunk_index,
            "page_range": list(chunk.page_range) if chunk.page_range else None,
            "text": chunk.text,
            "token_count": chunk.token_count,
        }
