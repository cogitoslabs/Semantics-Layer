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
from .teacher_backend import TeacherModelBackend, LocalHFBackend, APIBackend, BedrockBackend
from .trace_logger import (
    save_probe_traces_csv,
    list_trace_categories,
    list_trace_files,
    load_trace_file,
    compute_trace_diff,
)

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
    "TeacherModelBackend",
    "LocalHFBackend",
    "APIBackend",
    "BedrockBackend",
    "save_probe_traces_csv",
    "list_trace_categories",
    "list_trace_files",
    "load_trace_file",
    "compute_trace_diff",
]


