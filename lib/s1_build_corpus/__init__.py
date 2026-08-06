"""
pdf_corpus_pipeline.pipeline
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Public surface for the parallel PDF → JSONL corpus builder.
"""

from .build_corpus import run_corpus_builder
from .replay_corpus import run_replay_corpus
from .merge_corpus import run_merge_corpus
from .minhash_lsh import MinHashLSHDeduplicator

__all__ = ["run_corpus_builder", "run_replay_corpus", "run_merge_corpus", "MinHashLSHDeduplicator"]

