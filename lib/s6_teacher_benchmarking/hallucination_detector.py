import re
import json
import logging
from typing import List, Set, Optional
from pathlib import Path
import numpy as np

from lib.utils import PipelineConfig

logger = logging.getLogger(__name__)

class HallucinationDetector:
    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg
        self.nli_model_name = cfg.benchmarking.nli_model
        self.nli_threshold = cfg.benchmarking.hallucination_nli_threshold
        self._model = None
        
        # Load vocab cloze set terms
        self.vocab: Set[str] = set()
        vocab_path = Path(cfg.data.vocab_cloze_path)
        if vocab_path.exists():
            try:
                with open(vocab_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    for item in data:
                        term = item.get("target_term")
                        if term:
                            self.vocab.add(term.lower().strip())
                logger.info(f"Loaded {len(self.vocab)} terms for invented terminology vocabulary check.")
            except Exception as e:
                logger.error(f"Error loading vocabulary from {vocab_path}: {e}")
        else:
            logger.warning(f"Vocabulary cloze file not found at {vocab_path}.")

    @property
    def model(self):
        """Lazily initialize the sentence-transformers CrossEncoder for NLI."""
        if self._model is None:
            logger.info(f"Loading NLI cross-encoder model: {self.nli_model_name}")
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(self.nli_model_name)
        return self._model

    def detect_sentences(self, trace: str) -> List[str]:
        """Split trace into sentences using NLTK or regex fallback."""
        if not trace:
            return []
        import nltk
        from nltk.tokenize import sent_tokenize
        try:
            return sent_tokenize(trace)
        except Exception:
            # Fallback regex split
            return [s.strip() for s in re.split(r"[.?!]+", trace) if s.strip()]

    def is_invented_term(self, text: str) -> bool:
        r"""
        Check if text contains invented terminology not present in the curated vocabulary.
        Pattern: [A-Z][a-z]+-[A-Z][a-z]+\s+(receptor|protein|cell|neuron|pathway|channel)
        """
        pattern = r"\b[A-Z][a-z]+-[A-Z][a-z]+\s+(?:receptor|protein|cell|neuron|pathway|channel|Receptor|Protein|Cell|Neuron|Pathway|Channel)\b"
        matches = re.findall(pattern, text)
        for match in matches:
            norm_match = match.lower().strip()
            # If the term is not in the vocabulary, it's flagged as invented
            if norm_match not in self.vocab:
                return True
        return False

    def score_hallucination_rate(
        self, 
        trace: str, 
        retrieved_context: str, 
        ground_truth: str
    ) -> float:
        """
        Computes the hallucination rate for a trace.
        Combines Pass 1 (NLI entailment) and Pass 2 (invented terminology detection).
        """
        sentences = self.detect_sentences(trace)
        if not sentences:
            return 0.0

        # Build premise
        if retrieved_context.strip():
            premise = f"{retrieved_context}\n\nGround Truth: {ground_truth}"
        else:
            premise = f"Ground Truth: {ground_truth}"

        # Pass 1: Prepare NLI pairs
        nli_pairs = [(premise, sent) for sent in sentences]
        
        # Load NLI model and predict
        nli_model = self.model
        scores = nli_model.predict(nli_pairs)
        
        # Softmax to get entailment probabilities
        # Check label mapping
        entailment_idx = 2  # default fallback
        if hasattr(nli_model, "model") and hasattr(nli_model.model, "config") and hasattr(nli_model.model.config, "label2id"):
            label2id = nli_model.model.config.label2id
            for k, v in label2id.items():
                if "entail" in k.lower():
                    entailment_idx = v
                    break
        
        # If output shape is 1D (e.g. single sentence)
        if len(scores.shape) == 1:
            scores = np.expand_dims(scores, axis=0)

        exp_scores = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
        probs = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)
        entailment_probs = probs[:, entailment_idx]

        # Pass 2 & Pass 1 combined
        flagged_count = 0
        for i, sent in enumerate(sentences):
            nli_hallucinated = entailment_probs[i] < self.nli_threshold
            invented = self.is_invented_term(sent)
            
            if nli_hallucinated or invented:
                flagged_count += 1
                
        hallucination_rate = flagged_count / len(sentences)
        return min(hallucination_rate, 1.0)
