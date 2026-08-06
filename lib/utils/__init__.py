"""
utils/ — Shared logging, file utilities, and checkpointing helper functions
"""

from .logger import setup_logger, get_logger, flush_loggers, AutoFlushingFileHandler
from .config import PipelineConfig, DAPTConfig, CorpusBuildConfig
from .storage import StorageAdapter, get_adapter
from .profiller import FunctionProfiler
from .clean_text import clean_corpus_text, remove_inline_references, is_standalone_index_or_bibliography
from .pdf_utils import extract_main_text_from_pdfs
from .model_tracer import model_trace

__all__ = [
    "setup_logger",
    "get_logger",
    "flush_loggers",
    "AutoFlushingFileHandler",
    "PipelineConfig",
    "DAPTConfig",
    "CorpusBuildConfig",
    "StorageAdapter",
    "get_adapter",
    "FunctionProfiler",
    "clean_corpus_text",
    "remove_inline_references",
    "is_standalone_index_or_bibliography",
    "extract_main_text_from_pdfs",
    "model_trace",
]
