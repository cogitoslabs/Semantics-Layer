import json
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import numpy as np
import faiss
from transformers import AutoTokenizer
from rank_bm25 import BM25Okapi
from lib.utils import PipelineConfig
from lib.s3_rad_prep.chunker import Chunk
from lib.s3_rad_prep.indexer import DenseEmbedder

logger = logging.getLogger(__name__)

@dataclass
class RetrievalResult:
    chunks: List[Chunk]
    scores: List[float]
    passed_threshold: int
    retrieval_mode: str


class Retriever:
    def __init__(self, cfg: PipelineConfig, embedder: Optional[DenseEmbedder] = None):
        self.cfg = cfg
        self.embedder = embedder or DenseEmbedder(cfg.rad.embedding_model)

        chunks_path = Path(cfg.rad.chunks_path)
        if not chunks_path.exists():
            raise FileNotFoundError(f"Chunks file not found at {chunks_path}. Run chunking first.")

        logger.info(f"Loading chunks from {chunks_path}")
        self.chunks = []
        with open(chunks_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    self.chunks.append(Chunk(**json.loads(line)))

        index_file = Path(cfg.rad.index_dir) / "index.faiss"
        if not index_file.exists():
            raise FileNotFoundError(f"FAISS index not found at {index_file}. Run indexing first.")

        logger.info(f"Loading FAISS index from {index_file}")
        self.index = faiss.read_index(str(index_file))

        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model.base_model_name)

        self.bm25 = None
        if cfg.rad.retrieval_mode in ("sparse", "hybrid"):
            cache_file = Path(cfg.rad.index_dir) / "bm25_tokenized_corpus.json"
            tokenized_corpus = None

            if cache_file.exists():
                try:
                    logger.info(f"Loading cached BM25 tokenized corpus from {cache_file}")
                    with open(cache_file, "r", encoding="utf-8") as f:
                        tokenized_corpus = json.load(f)
                    
                    if len(tokenized_corpus) != len(self.chunks):
                        logger.warning(
                            f"Cached tokenized corpus length ({len(tokenized_corpus)}) does not match "
                            f"number of chunks ({len(self.chunks)}). Re-tokenizing..."
                        )
                        tokenized_corpus = None
                except Exception as e:
                    logger.error(f"Error loading cached tokenized corpus: {e}. Re-tokenizing...")
                    tokenized_corpus = None

            if tokenized_corpus is None:
                logger.info("Initializing BM25: tokenizing chunks (this may take a few moments for large corpora)")
                tokenized_corpus = [
                    self.tokenizer.tokenize(chunk.text) for chunk in self.chunks
                ]
                try:
                    logger.info(f"Saving tokenized corpus cache to {cache_file}")
                    with open(cache_file, "w", encoding="utf-8") as f:
                        json.dump(tokenized_corpus, f)
                except Exception as e:
                    logger.error(f"Error saving tokenized corpus cache: {e}")

            self.bm25 = BM25Okapi(tokenized_corpus)

    def retrieve(self, query: str) -> RetrievalResult:
        mode = self.cfg.rad.retrieval_mode
        top_k = self.cfg.rad.top_k
        relevance_threshold = self.cfg.rad.relevance_threshold

        retrieved_chunks = []
        if mode == "dense":
            retrieved_chunks, scores = self._retrieve_dense(query, top_k)
        elif mode == "sparse":
            retrieved_chunks, scores = self._retrieve_sparse(query, top_k)
        elif mode == "hybrid":
            retrieved_chunks, scores = self._retrieve_hybrid(query, top_k)
        else:
            raise ValueError(f"Unknown retrieval mode: {mode}")

        # Compute cosine similarities for gating (necessary for sparse/hybrid if scores aren't cosine similarity)
        # For dense, the inner product scores of normalized vectors are already cosine similarities.
        if mode == "dense":
            cosine_similarities = scores
        else:
            cosine_similarities = self._compute_cosine_similarities(query, retrieved_chunks)

        # Filter chunks by relevance threshold
        passed_chunks = []
        passed_scores = []
        for chunk, score in zip(retrieved_chunks, cosine_similarities):
            if score >= relevance_threshold:
                passed_chunks.append(chunk)
                passed_scores.append(score)

        return RetrievalResult(
            chunks=passed_chunks,
            scores=passed_scores,
            passed_threshold=len(passed_chunks),
            retrieval_mode=mode
        )

    def _retrieve_dense(self, query: str, top_k: int) -> tuple[List[Chunk], List[float]]:
        query_vector = self.embedder.embed_batch([query])[0].astype("float32")
        query_vector = np.expand_dims(query_vector, axis=0)

        # FAISS search
        similarities, indices = self.index.search(query_vector, top_k)
        retrieved_chunks = []
        scores = []
        for sim, idx in zip(similarities[0], indices[0]):
            if idx != -1 and idx < len(self.chunks):
                retrieved_chunks.append(self.chunks[idx])
                scores.append(float(sim))
        return retrieved_chunks, scores

    def _retrieve_sparse(self, query: str, top_k: int) -> tuple[List[Chunk], List[float]]:
        if self.bm25 is None:
            raise ValueError("BM25 index is not initialized.")
        tokenized_query = self.tokenizer.tokenize(query)
        bm25_scores = self.bm25.get_scores(tokenized_query)

        # Get top-k indices
        top_indices = np.argsort(bm25_scores)[::-1][:top_k]
        retrieved_chunks = []
        scores = []
        for idx in top_indices:
            score = bm25_scores[idx]
            retrieved_chunks.append(self.chunks[idx])
            scores.append(float(score))
        return retrieved_chunks, scores

    def _retrieve_hybrid(self, query: str, top_k: int) -> tuple[List[Chunk], List[float]]:
        # Candidate generation count
        candidates_count = max(top_k * 5, 100)
        candidates_count = min(candidates_count, len(self.chunks))

        # 1. Dense candidates
        query_vector = self.embedder.embed_batch([query])[0].astype("float32")
        query_vector = np.expand_dims(query_vector, axis=0)
        _, dense_indices = self.index.search(query_vector, candidates_count)
        dense_ranks = {idx: rank + 1 for rank, idx in enumerate(dense_indices[0]) if idx != -1}

        # 2. Sparse candidates
        if self.bm25 is None:
            raise ValueError("BM25 index is not initialized.")
        tokenized_query = self.tokenizer.tokenize(query)
        bm25_scores = self.bm25.get_scores(tokenized_query)
        sparse_indices = np.argsort(bm25_scores)[::-1][:candidates_count]
        sparse_ranks = {idx: rank + 1 for rank, idx in enumerate(sparse_indices)}

        # 3. Reciprocal Rank Fusion
        fused_scores = {}
        union_indices = set(dense_ranks.keys()).union(sparse_ranks.keys())
        k_rrf = 60

        for idx in union_indices:
            score = 0.0
            if idx in dense_ranks:
                score += 1.0 / (k_rrf + dense_ranks[idx])
            if idx in sparse_ranks:
                score += 1.0 / (k_rrf + sparse_ranks[idx])
            fused_scores[idx] = score

        # Sort by fused score descending
        sorted_indices = sorted(fused_scores.keys(), key=lambda idx: fused_scores[idx], reverse=True)[:top_k]

        retrieved_chunks = []
        scores = []
        for idx in sorted_indices:
            retrieved_chunks.append(self.chunks[idx])
            scores.append(float(fused_scores[idx]))

        return retrieved_chunks, scores

    def _compute_cosine_similarities(self, query: str, chunks: List[Chunk]) -> List[float]:
        if not chunks:
            return []
        query_vector = self.embedder.embed_batch([query])[0]
        chunk_texts = [chunk.text for chunk in chunks]
        chunk_vectors = self.embedder.embed_batch(chunk_texts)

        similarities = np.dot(chunk_vectors, query_vector)
        return [float(sim) for sim in similarities]
