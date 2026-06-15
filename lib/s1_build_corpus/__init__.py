"""
pdf_corpus_pipeline.pipeline
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Public surface for the parallel PDF → JSONL corpus builder.
"""

from .build_corpus import CorpusBuilder, run_corpus_builder
from .storage import get_adapter, StorageAdapter

__all__ = ["CorpusBuilder", "run_corpus_builder", "get_adapter", "StorageAdapter"]

