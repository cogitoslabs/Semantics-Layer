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
from lib.s4_rad_prep.chunker import Chunk
from lib.s4_rad_prep.indexer import DenseEmbedder
from lib.s4_rad_prep.reranker import CrossEncoderReranker

logger = logging.getLogger(__name__)

@dataclass
class RetrievalResult:
    chunks: List[Chunk]
    scores: List[float]
    passed_threshold: int
    retrieval_mode: str


def _tokenize_bm25(text: str) -> List[str]:
    import re
    return re.findall(r"\w+", text.lower())


class Retriever:
    def __init__(
        self,
        cfg: PipelineConfig,
        embedder: Optional[DenseEmbedder] = None,
        reranker: Optional[CrossEncoderReranker] = None
    ):
        self.cfg = cfg
        self.embedder = embedder or DenseEmbedder(cfg.rad.embedding_model)

        if reranker is not None:
            self.reranker = reranker
        elif cfg.rad.use_reranker:
            self.reranker = CrossEncoderReranker(
                model_name=cfg.rad.reranker_model,
                batch_size=cfg.rad.reranker_batch_size
            )
        else:
            self.reranker = None

        chunks_path = Path(cfg.rad.chunks_path)
        if not chunks_path.exists():
            raise FileNotFoundError(f"Chunks file not found at {chunks_path}. Run chunking first.")

        logger.info(f"Loading chunks from {chunks_path}")
        self.chunks = []
        with open(chunks_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    self.chunks.append(Chunk(**json.loads(line)))

        self.chunk_id_to_idx = {chunk.chunk_id: idx for idx, chunk in enumerate(self.chunks)}

        index_file = Path(cfg.rad.index_dir) / "index.faiss"
        if not index_file.exists():
            raise FileNotFoundError(f"FAISS index not found at {index_file}. Run indexing first.")

        logger.info(f"Loading FAISS index from {index_file}")
        self.index = faiss.read_index(str(index_file))

        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model.base_model_name)

        self.bm25 = None
        if cfg.rad.retrieval_mode in ("sparse", "hybrid"):
            cache_file_json = Path(cfg.rad.index_dir) / "bm25_tokenized_corpus.json"
            cache_file_pkl = Path(cfg.rad.index_dir) / "bm25_tokenized_corpus.pkl"
            tokenized_corpus = None

            if cache_file_pkl.exists():
                try:
                    import pickle
                    logger.info(f"Loading fast cached BM25 tokenized corpus from {cache_file_pkl}")
                    with open(cache_file_pkl, "rb") as f:
                        tokenized_corpus = pickle.load(f)
                    if len(tokenized_corpus) != len(self.chunks):
                        tokenized_corpus = None
                except Exception as e:
                    logger.error(f"Error loading pkl tokenized corpus: {e}")
                    tokenized_corpus = None

            if tokenized_corpus is None and cache_file_json.exists():
                try:
                    logger.info(f"Loading cached BM25 tokenized corpus from {cache_file_json}")
                    with open(cache_file_json, "r", encoding="utf-8") as f:
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
                logger.info("Initializing BM25: fast batch tokenizing chunks across CPU cores...")
                tokenized_corpus = [_tokenize_bm25(chunk.text) for chunk in self.chunks]
                try:
                    import pickle
                    logger.info(f"Saving fast tokenized corpus cache to {cache_file_pkl}")
                    with open(cache_file_pkl, "wb") as f:
                        pickle.dump(tokenized_corpus, f, protocol=pickle.HIGHEST_PROTOCOL)
                except Exception as e:
                    logger.error(f"Error saving tokenized corpus cache: {e}")

            self.bm25 = BM25Okapi(tokenized_corpus)

    def _extract_query_variants(self, query: str) -> List[str]:
        """Extract core entity variants to improve exact-term and conceptual matching."""
        variants = [query]
        if not getattr(self.cfg.rad, "query_expansion", True):
            return variants

        import re
        # Extract quoted entities (e.g. 'Sulcus' -> ["Sulcus definition", "Sulcus"])
        quoted = re.findall(r"['\"]([^'\"]+)['\"]", query)
        for term in quoted:
            term_clean = term.strip()
            if term_clean and term_clean not in variants:
                variants.append(f"{term_clean} definition")
                variants.append(term_clean)

        if len(variants) == 1:
            clean_q = re.sub(r"(?i)^(which|what|describe|definition|best|describes|is)\s+", "", query).strip()
            clean_q = re.sub(r"[?!.]+$", "", clean_q).strip()
            if clean_q and clean_q not in variants and len(clean_q.split()) < len(query.split()):
                variants.append(clean_q)

        return variants

    def retrieve(self, query: str) -> RetrievalResult:
        mode = self.cfg.rad.retrieval_mode
        top_k = self.cfg.rad.top_k
        relevance_threshold = self.cfg.rad.relevance_threshold

        retrieved_chunks = []
        if mode == "dense":
            retrieved_chunks, scores = self._retrieve_dense(query, top_k)
            gating_scores = scores
            effective_threshold = relevance_threshold
        elif mode == "sparse":
            retrieved_chunks, scores = self._retrieve_sparse(query, top_k)
            gating_scores = self._compute_cosine_similarities(query, retrieved_chunks)
            effective_threshold = relevance_threshold
        elif mode == "hybrid":
            retrieved_chunks, scores = self._retrieve_hybrid(query, top_k)
            if self.reranker is not None:
                # Reranker scores are normalized probabilities in [0, 1].
                gating_scores = scores
                effective_threshold = 0.30 if relevance_threshold in (0.65, 0.45) else relevance_threshold
            else:
                gating_scores = self._compute_cosine_similarities(query, retrieved_chunks)
                effective_threshold = relevance_threshold
        else:
            raise ValueError(f"Unknown retrieval mode: {mode}")

        # Filter chunks by relevance threshold
        passed_chunks = []
        passed_scores = []
        for chunk, score in zip(retrieved_chunks, gating_scores):
            if score >= effective_threshold:
                passed_chunks.append(chunk)
                passed_scores.append(score)

        return RetrievalResult(
            chunks=passed_chunks,
            scores=passed_scores,
            passed_threshold=len(passed_chunks),
            retrieval_mode=mode
        )

    def _retrieve_dense(self, query: str, top_k: int) -> tuple[List[Chunk], List[float]]:
        query_vector = self.embedder.embed_batch([query], is_query=True)[0].astype("float32")
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
        tokenized_query = _tokenize_bm25(query)
        bm25_scores = self.bm25.get_scores(tokenized_query)

        k = min(top_k, len(self.chunks))
        if k == 0:
            return [], []

        if k < len(bm25_scores):
            top_partition = np.argpartition(bm25_scores, -k)[-k:]
            top_indices = top_partition[np.argsort(bm25_scores[top_partition])[::-1]]
        else:
            top_indices = np.argsort(bm25_scores)[::-1]

        retrieved_chunks = []
        scores = []
        for idx in top_indices:
            score = bm25_scores[idx]
            retrieved_chunks.append(self.chunks[idx])
            scores.append(float(score))
        return retrieved_chunks, scores

    def _retrieve_hybrid(self, query: str, top_k: int) -> tuple[List[Chunk], List[float]]:
        # Stage 1: Candidate Generation count (N candidates)
        candidate_k = max(self.cfg.rad.rerank_candidate_k, top_k)
        candidates_per_variant = min(candidate_k, len(self.chunks))

        query_variants = self._extract_query_variants(query)
        fused_scores: Dict[int, float] = {}
        k_rrf = 60

        for variant in query_variants:
            # 1. Dense candidates for variant
            query_vector = self.embedder.embed_batch([variant], is_query=True)[0].astype("float32")
            query_vector = np.expand_dims(query_vector, axis=0)
            _, dense_indices = self.index.search(query_vector, candidates_per_variant)
            dense_ranks = {idx: rank + 1 for rank, idx in enumerate(dense_indices[0]) if idx != -1}

            # 2. Sparse candidates for variant
            if self.bm25 is None:
                raise ValueError("BM25 index is not initialized.")
            tokenized_variant = _tokenize_bm25(variant)
            bm25_scores = self.bm25.get_scores(tokenized_variant)
            if candidates_per_variant < len(bm25_scores):
                sparse_top = np.argpartition(bm25_scores, -candidates_per_variant)[-candidates_per_variant:]
                sparse_indices = sparse_top[np.argsort(bm25_scores[sparse_top])[::-1]]
            else:
                sparse_indices = np.argsort(bm25_scores)[::-1]
            sparse_ranks = {idx: rank + 1 for rank, idx in enumerate(sparse_indices)}

            # 3. Reciprocal Rank Fusion aggregation
            union_indices = set(dense_ranks.keys()).union(sparse_ranks.keys())
            for idx in union_indices:
                score = 0.0
                if idx in dense_ranks:
                    score += 1.0 / (k_rrf + dense_ranks[idx])
                if idx in sparse_ranks:
                    score += 1.0 / (k_rrf + sparse_ranks[idx])
                fused_scores[idx] = max(fused_scores.get(idx, 0.0), score)

        # Stage 1 candidate pool sorted by fused score
        sorted_candidate_indices = sorted(fused_scores.keys(), key=lambda idx: fused_scores[idx], reverse=True)[:candidate_k]
        candidate_chunks = [self.chunks[idx] for idx in sorted_candidate_indices]

        # Stage 2: Cross-Encoder Reranking (if enabled)
        if self.reranker is not None:
            reranked_chunks, reranked_scores = self.reranker.rerank(query, candidate_chunks, top_k=top_k)
            return reranked_chunks, reranked_scores

        # Fallback to Stage 1 RRF top-K if reranker is disabled
        top_k_chunks = candidate_chunks[:top_k]
        top_k_scores = [float(fused_scores[self.chunk_id_to_idx[c.chunk_id]]) for c in top_k_chunks]
        return top_k_chunks, top_k_scores

    def _compute_cosine_similarities(self, query: str, chunks: List[Chunk]) -> List[float]:
        if not chunks:
            return []
        query_vector = self.embedder.embed_batch([query], is_query=True)[0].astype("float32")
        try:
            chunk_vectors = []
            for chunk in chunks:
                idx = self.chunk_id_to_idx.get(chunk.chunk_id)
                if idx is not None and idx < self.index.ntotal:
                    chunk_vectors.append(self.index.reconstruct(idx))
                else:
                    raise ValueError(f"Chunk ID {chunk.chunk_id} not found in index")
            chunk_matrix = np.vstack(chunk_vectors).astype("float32")
            similarities = np.dot(chunk_matrix, query_vector)
            return [float(sim) for sim in similarities]
        except Exception as e:
            logger.debug(f"Fast cosine similarity reconstruction failed ({e}), falling back to embedding model")
            chunk_texts = [chunk.text for chunk in chunks]
            chunk_vectors = self.embedder.embed_batch(chunk_texts)
            similarities = np.dot(chunk_vectors, query_vector)
            return [float(sim) for sim in similarities]

