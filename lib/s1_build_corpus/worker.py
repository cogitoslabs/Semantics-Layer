"""
Multiprocessing worker logic: GPU assignment, PDF extraction, and token counting.

worker_init  — runs once per process at startup to pin the GPU and load the model.
worker_task  — runs per PDF, calls the extractor, counts tokens, returns a result.

Both are module-level functions (required by ProcessPoolExecutor on all platforms).
Stateful callbacks are stored as process-globals after initialisation.
"""

import os
from dataclasses import dataclass
from typing import Any, Callable, Optional

import tiktoken


# Process-local state set during initialisation
_docling_converter: Any = None

MIN_CONTENT_LENGTH = 300


@dataclass
class ExtractionResult:
    filename: str
    text: str
    token_count: int
    status: str  # "SUCCESS" | "SKIPPED" | "ERROR: <msg>"

    @property
    def succeeded(self) -> bool:
        return self.status == "SUCCESS"


def worker_init(
    gpu_queue: Any,  # multiprocessing.managers.AutoProxy[Queue]
) -> None:
    """
    Called once per worker process.
    Pops a GPU id from the shared queue, sets CUDA_VISIBLE_DEVICES,
    then runs the Docling-specific init (model loading, etc.).
    """
    global _docling_converter

    gpu_id = gpu_queue.get()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import AcceleratorOptions, PdfPipelineOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption

    options = PdfPipelineOptions()
    options.accelerator_options = AcceleratorOptions(num_threads=4, device="cuda")

    _docling_converter = DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=options)}
    )
    print(f"[WORKER INIT] Docling ready on GPU {gpu_id}")


def worker_task(filename: str, pdf_path: str) -> ExtractionResult:
    """
    Extracts markdown from a single PDF and counts tokens.
    Returns an ExtractionResult regardless of success or failure.
    """
    try:
        if _docling_converter is None:
            return ExtractionResult(filename, "", 0, "ERROR: Docling converter not initialized")

        text = _docling_converter.convert(pdf_path).document.export_to_markdown()

        if not text or len(text.strip()) < MIN_CONTENT_LENGTH:
            return ExtractionResult(filename, "", 0, "SKIPPED")

        tokenizer = tiktoken.get_encoding("cl100k_base")
        token_count = len(tokenizer.encode(text))
        return ExtractionResult(filename, text, token_count, "SUCCESS")

    except Exception as exc:
        return ExtractionResult(filename, "", 0, f"ERROR: {exc}")
