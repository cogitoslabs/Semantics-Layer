"""
pdf_corpus_pipeline.s4_rad_prep
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Public surface for the Step 0.3 Retrieval-Augmented Distillation Preparation (RAD Prep) pipeline.
"""

from .rad_prep import run_rad_prep_pipeline
from .reranker import CrossEncoderReranker

__all__ = ["run_rad_prep_pipeline", "CrossEncoderReranker"]
