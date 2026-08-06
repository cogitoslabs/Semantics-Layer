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
from lib.utils.logger import get_logger
from lib.utils import clean_corpus_text

logger = get_logger(__name__)


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
    page_range: Optional[tuple[int, int]] = None
    chunk_index: Optional[int] = None

    @property
    def succeeded(self) -> bool:
        return self.status == "SUCCESS"


def worker_init(
    gpu_queue: Any,  # multiprocessing.Queue
    docling_options: Optional[dict] = None,
    logging_cfg: Any = None,
) -> None:
    """
    Called once per worker process.
    Pops a GPU id from the shared queue, sets CUDA_VISIBLE_DEVICES,
    then runs the Docling-specific init (model loading, etc.).
    """
    global _docling_converter

    import sys
    from lib.utils import setup_logger
    from lib.utils.config import LoggingConfig
    if logging_cfg is None:
        logging_cfg = LoggingConfig()
    setup_logger(
        f"{__name__}.{sys._getframe().f_code.co_name}",
        logging_cfg,
    )
    global logger
    logger = get_logger(f"{__name__}.{sys._getframe().f_code.co_name}")

    gpu_id = gpu_queue.get()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    gpu_queue.put(gpu_id)  # Self-replenish queue for future/recycled worker processes

    import torch
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import AcceleratorOptions, PdfPipelineOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption

    cuda_available = torch.cuda.is_available()
    device = "cuda" if cuda_available else "cpu"

    options = PdfPipelineOptions()
    num_threads = docling_options.get("num_threads", 4) if docling_options is not None else 4
    options.accelerator_options = AcceleratorOptions(num_threads=num_threads, device=device)

    if docling_options is not None:
        options.do_ocr = docling_options.get("do_ocr", False)
        options.do_table_structure = docling_options.get("do_table_structure", False)
        options.do_code_enrichment = docling_options.get("do_code_enrichment", False)
        options.do_formula_enrichment = docling_options.get("do_formula_enrichment", False)
        options.do_picture_classification = docling_options.get("do_picture_classification", False)
        options.do_picture_description = docling_options.get("do_picture_description", False)
    else:
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
    logger.info(f"[WORKER INIT] Docling ready on {device} (GPU {gpu_id} requested)")


def worker_task(
    filename: str,
    pdf_path: str,
    start_page: Optional[int] = None,
    end_page: Optional[int] = None,
    chunk_index: Optional[int] = None,
    chunk_size: int = 10,
    chunk_queue: Optional[Any] = None
) -> ExtractionResult:
    """
    Extracts markdown from a PDF page range or loops through the PDF in chunks.
    """
    try:
        if _docling_converter is None:
            return ExtractionResult(filename, [], "ERROR: Docling converter not initialized")

        from pathlib import Path
        suffix = Path(pdf_path).suffix.lower()
        is_pdf = suffix == ".pdf"

        tokenizer = tiktoken.get_encoding("cl100k_base")

        if not is_pdf:
            conv_result = _docling_converter.convert(pdf_path)
            try:
                chunk_text = conv_result.document.export_to_markdown()
                chunk_text = clean_corpus_text(chunk_text, filename)
                if not chunk_text or len(chunk_text.strip()) < MIN_CONTENT_LENGTH:
                    return ExtractionResult(filename, [], "SKIPPED", chunk_index=0)

                tokens = tokenizer.encode(chunk_text)
                target_tokens = chunk_size * 400
                overlap_tokens = target_tokens // 10

                chunks = []
                if len(tokens) <= target_tokens:
                    chunks.append(ChunkResult(
                        chunk_index=0,
                        text=chunk_text,
                        token_count=len(tokens),
                        page_range=None
                    ))
                else:
                    step = target_tokens - overlap_tokens
                    c_idx = 0
                    start_idx = 0
                    while start_idx < len(tokens):
                        end_idx = min(start_idx + target_tokens, len(tokens))
                        chunk_tokens = tokens[start_idx:end_idx]
                        dec_text = tokenizer.decode(chunk_tokens)
                        if len(dec_text.strip()) >= MIN_CONTENT_LENGTH:
                            chunks.append(ChunkResult(
                                chunk_index=c_idx,
                                text=dec_text,
                                token_count=len(chunk_tokens),
                                page_range=None
                            ))
                            c_idx += 1
                        if end_idx == len(tokens):
                            break
                        start_idx += step

                if not chunks:
                    return ExtractionResult(filename, [], "SKIPPED")
                return ExtractionResult(filename, chunks, "SUCCESS")
            finally:
                try:
                    if hasattr(conv_result, "input") and conv_result.input and hasattr(conv_result.input, "_backend") and conv_result.input._backend:
                        conv_result.input._backend.unload()
                except Exception:
                    pass

        import pypdfium2 as pdfium
        doc = pdfium.PdfDocument(pdf_path)
        try:
            page_count = len(doc)
        finally:
            doc.close()

        if page_count == 0:
            return ExtractionResult(filename, [], "ERROR: PDF has 0 pages or could not be read")

        # Single page-range chunk mode (used by refactored builder)
        if start_page is not None and end_page is not None:
            c_idx = chunk_index if chunk_index is not None else 0
            conv_result = _docling_converter.convert(pdf_path, page_range=(start_page, end_page))
            try:
                chunk_text = conv_result.document.export_to_markdown()
                chunk_text = clean_corpus_text(chunk_text, filename)
                if chunk_text and len(chunk_text.strip()) >= MIN_CONTENT_LENGTH:
                    token_count = len(tokenizer.encode(chunk_text))
                    chunk = ChunkResult(
                        chunk_index=c_idx,
                        text=chunk_text,
                        token_count=token_count,
                        page_range=(start_page, end_page)
                    )
                    return ExtractionResult(filename, [chunk], "SUCCESS")
                else:
                    return ExtractionResult(filename, [], "SKIPPED", page_range=(start_page, end_page), chunk_index=c_idx)
            finally:
                try:
                    if hasattr(conv_result, "input") and conv_result.input and hasattr(conv_result.input, "_backend") and conv_result.input._backend:
                        conv_result.input._backend.unload()
                except Exception:
                    pass

        # Full PDF loop mode (fallback / backward compatibility)
        chunk_size = max(2, chunk_size)
        stride = chunk_size - 1
        chunks = []
        c_idx = 0
        start = 1
        while start <= page_count:
            end = min(start + chunk_size - 1, page_count)
            conv_result = _docling_converter.convert(pdf_path, page_range=(start, end))
            try:
                chunk_text = conv_result.document.export_to_markdown()
                chunk_text = clean_corpus_text(chunk_text, filename)
                if chunk_text and len(chunk_text.strip()) >= MIN_CONTENT_LENGTH:
                    token_count = len(tokenizer.encode(chunk_text))
                    chunk = ChunkResult(
                        chunk_index=c_idx,
                        text=chunk_text,
                        token_count=token_count,
                        page_range=(start, end)
                    )
                    chunks.append(chunk)
                    if chunk_queue is not None:
                        chunk_queue.put((filename, chunk))
                    c_idx += 1
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

