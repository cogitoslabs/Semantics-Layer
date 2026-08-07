import logging
from typing import List, Optional, Tuple
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from lib.s4_rad_prep.chunker import Chunk

logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    """
    Two-stage Cross-Encoder Reranker that scores query-passage text pairs jointly.
    Normalizes logits via sigmoid to range [0, 1] for relevance threshold gating.
    """

    def __init__(self, model_name: str, device: Optional[str] = None, batch_size: int = 32):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = max(1, batch_size)
        logger.info(f"Initializing CrossEncoderReranker using '{self.model_name}' on device '{self.device}'")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name).to(self.device)
        self.model.eval()

    def rerank(self, query: str, chunks: List[Chunk], top_k: int) -> Tuple[List[Chunk], List[float]]:
        if not chunks:
            return [], []

        pairs = [[query, chunk.text] for chunk in chunks]
        scores_list: List[float] = []

        for i in range(0, len(pairs), self.batch_size):
            batch_pairs = pairs[i : i + self.batch_size]
            inputs = self.tokenizer(
                batch_pairs,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            ).to(self.device)

            use_autocast = self.device.startswith("cuda")
            autocast_ctx = (
                torch.amp.autocast(device_type="cuda", dtype=torch.float16)
                if hasattr(torch, "amp") else torch.cuda.amp.autocast(dtype=torch.float16)
            )
            with torch.no_grad():
                if use_autocast:
                    with autocast_ctx:
                        outputs = self.model(**inputs)
                else:
                    outputs = self.model(**inputs)

                logits = outputs.logits
                if logits.shape[-1] == 1:
                    raw_scores = logits.squeeze(-1)
                else:
                    # Multi-class or binary classification logits (take positive class logit)
                    raw_scores = logits[:, 1]
                probs = torch.sigmoid(raw_scores.float()).cpu().numpy()
                if probs.ndim == 0:
                    scores_list.append(float(probs))
                else:
                    scores_list.extend([float(s) for s in probs])

        # Pair each chunk with its reranked score and sort descending
        ranked = sorted(zip(chunks, scores_list), key=lambda x: x[1], reverse=True)[:top_k]
        reranked_chunks = [item[0] for item in ranked]
        reranked_scores = [float(item[1]) for item in ranked]

        return reranked_chunks, reranked_scores
