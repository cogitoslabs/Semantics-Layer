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
class ChunkResult:
    chunk_index: int
    text: str
    token_count: int
    page_range: tuple[int, int]


@dataclass
class ExtractionResult:
    filename: str
    chunks: list[ChunkResult]
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

    import torch
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import AcceleratorOptions, PdfPipelineOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption

    cuda_available = torch.cuda.is_available()
    device = "cuda" if cuda_available else "cpu"

    options = PdfPipelineOptions()
    options.accelerator_options = AcceleratorOptions(num_threads=4, device=device)

    if not cuda_available:
        options.do_ocr = False
        options.do_table_structure = False
        options.do_code_enrichment = False
        options.do_formula_enrichment = False
        options.do_picture_classification = False
        options.do_picture_description = False
        options.generate_page_images = False
        options.generate_picture_images = False

    if cuda_available:
        pdf_format_option = PdfFormatOption(pipeline_options=options)
    else:
        from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
        pdf_format_option = PdfFormatOption(
            pipeline_options=options,
            backend=PyPdfiumDocumentBackend
        )

    _docling_converter = DocumentConverter(
        format_options={InputFormat.PDF: pdf_format_option}
    )
    print(f"[WORKER INIT] Docling ready on {device} (GPU {gpu_id} requested)")


def worker_task(
    filename: str,
    pdf_path: str,
    chunk_size: int = 10,
    chunk_queue: Optional[Any] = None
) -> ExtractionResult:
    """
    Extracts markdown from a single PDF in page chunks and counts tokens.
    Returns an ExtractionResult regardless of success or failure.
    """
    try:
        if _docling_converter is None:
            return ExtractionResult(filename, [], "ERROR: Docling converter not initialized")

        import pypdfium2 as pdfium
        doc = pdfium.PdfDocument(pdf_path)
        try:
            page_count = len(doc)
        finally:
            doc.close()

        if page_count == 0:
            return ExtractionResult(filename, [], "ERROR: PDF has 0 pages or could not be read")

        chunk_size = max(2, chunk_size)  # Ensure chunk_size is at least 2 to allow overlap
        stride = chunk_size - 1
        chunks = []
        chunk_index = 0
        tokenizer = tiktoken.get_encoding("cl100k_base")

        start = 1
        while start <= page_count:
            end = min(start + chunk_size - 1, page_count)
            conv_result = _docling_converter.convert(pdf_path, page_range=(start, end))
            try:
                chunk_text = conv_result.document.export_to_markdown()
                if chunk_text and len(chunk_text.strip()) >= MIN_CONTENT_LENGTH:
                    token_count = len(tokenizer.encode(chunk_text))
                    chunk = ChunkResult(
                        chunk_index=chunk_index,
                        text=chunk_text,
                        token_count=token_count,
                        page_range=(start, end)
                    )
                    chunks.append(chunk)
                    if chunk_queue is not None:
                        chunk_queue.put((filename, chunk))
                    chunk_index += 1
            finally:
                try:
                    if hasattr(conv_result, "input") and conv_result.input and hasattr(conv_result.input, "_backend") and conv_result.input._backend:
                        conv_result.input._backend.unload()
                except Exception:
                    pass

            if end >= page_count:
                break
            start += stride

        if not chunks:
            return ExtractionResult(filename, [], "SKIPPED")

        return ExtractionResult(filename, chunks, "SUCCESS")

    except Exception as exc:
        return ExtractionResult(filename, [], f"ERROR: {exc}")
