"""
pdf_corpus_pipeline.pipeline
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Public surface for the parallel PDF → JSONL corpus builder.
"""

from .build_corpus import run_corpus_builder

__all__ = ["run_corpus_builder"]

