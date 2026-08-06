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
        vocab_path = Path(cfg.data.cloze_set_path)
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
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Loading NLI cross-encoder model: {self.nli_model_name} on device: {device}")
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(self.nli_model_name, device=device)
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

    def is_structural_template_sentence(self, sentence: str) -> bool:
        """Check if sentence is a structural/formatting template sentence (e.g. boxed answer or intro phrase)."""
        s_clean = sentence.strip()
        if not s_clean:
            return True
        if s_clean.startswith("\\boxed{") or s_clean.startswith("boxed{"):
            return True
        lowered = s_clean.lower()
        structural_triggers = [
            "based on the context provided",
            "to determine the best definition",
            "to determine the definition",
            "from the context provided",
            "given the context",
            "understanding cognitive neuroscience",
            "key elements",
            "matching definitions",
            "standard definition",
            "definition search",
            "anatomical position",
            "contextual usage",
        ]
        if any(trigger in lowered for trigger in structural_triggers):
            return True
        return False

    def score_hallucination_rate(
        self, 
        trace: str, 
        retrieved_context: str, 
        ground_truth: str
    ) -> float:
        """
        Computes the hallucination rate for a trace using chunk-level NLI entailment
        and invented terminology detection.
        """
        sentences = self.detect_sentences(trace)
        if not sentences:
            return 0.0

        # Filter out empty or purely structural sentences for NLI evaluation
        eval_sentences = [s for s in sentences if not self.is_structural_template_sentence(s)]
        if not eval_sentences:
            # If all sentences are structural, evaluate all sentences
            eval_sentences = sentences

        # Build candidate premise list (chunks + ground truth)
        premises = []
        if retrieved_context.strip():
            raw_chunks = [c.strip() for c in retrieved_context.split("\n\n") if c.strip()]
            premises.extend(raw_chunks)
        if ground_truth.strip():
            premises.append(f"Ground Truth: {ground_truth}")
            
        if not premises:
            premises = ["No context available."]

        # Build NLI pairs for each sentence against each premise chunk
        nli_pairs = []
        sentence_premise_indices = []  # keeps track of (sentence_idx, premise_idx)
        
        for s_idx, sent in enumerate(eval_sentences):
            for p_idx, premise in enumerate(premises):
                # Truncate premise to ~350 words to avoid encoder truncation lag
                premise_trunc = " ".join(premise.split()[:350])
                nli_pairs.append((premise_trunc, sent))
                sentence_premise_indices.append((s_idx, p_idx))
        
        # Load NLI model and predict
        nli_model = self.model
        scores = nli_model.predict(nli_pairs)
        
        # Softmax to get entailment probabilities
        entailment_idx = 2  # default fallback for cross-encoder/nli-deberta-v3-small
        if hasattr(nli_model, "model") and hasattr(nli_model.model, "config") and hasattr(nli_model.model.config, "label2id"):
            label2id = nli_model.model.config.label2id
            for k, v in label2id.items():
                if "entail" in k.lower():
                    entailment_idx = v
                    break
        
        if len(scores.shape) == 1:
            scores = np.expand_dims(scores, axis=0)

        exp_scores = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
        probs = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)
        entailment_probs = probs[:, entailment_idx]

        # Aggregate max entailment probability per sentence across premises
        max_entailment_per_sentence = [0.0] * len(eval_sentences)
        for (s_idx, p_idx), prob in zip(sentence_premise_indices, entailment_probs):
            if prob > max_entailment_per_sentence[s_idx]:
                max_entailment_per_sentence[s_idx] = float(prob)

        # Flag hallucinated sentences
        flagged_count = 0
        for i, sent in enumerate(eval_sentences):
            is_template = self.is_structural_template_sentence(sent)
            nli_hallucinated = (max_entailment_per_sentence[i] < self.nli_threshold) and not is_template
            invented = self.is_invented_term(sent) and not is_template
            
            if nli_hallucinated or invented:
                flagged_count += 1
                
        hallucination_rate = flagged_count / len(eval_sentences)
        return min(hallucination_rate, 1.0)

