"""
pdf_corpus_pipeline.s5_clustering
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Public surface for the Step 4 Corpus Engineering & Micro-Clustering pipeline.
"""

from .clustering import run_clustering_pipeline

__all__ = ["run_clustering_pipeline"]
