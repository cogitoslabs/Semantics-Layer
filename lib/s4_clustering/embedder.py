"""
s4_clustering/embedder.py - Batch-embed documents using sentence-transformers and cache to disk.
"""

import json
import logging
import numpy as np
from pathlib import Path
from typing import List, Tuple
from sentence_transformers import SentenceTransformer

from lib.utils import PipelineConfig

logger = logging.getLogger(__name__)


def load_corpus(corpus_path: Path) -> List[Tuple[str, str]]:
    """
    Read DAPT corpus JSONL.
    Returns:
        List of (doc_id, text) tuples.
    """
    if not corpus_path.exists():
        raise FileNotFoundError(f"Corpus file not found at {corpus_path}")

    docs = []
    with open(corpus_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            # Expecting 'id' or 'doc_id', fallback to 'id'
            doc_id = data.get("id") or data.get("doc_id")
            text = data.get("text", "")
            if doc_id and text:
                docs.append((doc_id, text))
    return docs


def run_embedding(cfg: PipelineConfig) -> Tuple[np.ndarray, List[str]]:
    """
    Embed documents using sentence-transformers. Checks cache before running.
    """
    corpus_path = Path(cfg.clustering.corpus_path)
    logger.info(f"Loading corpus from {corpus_path}...")
    docs = load_corpus(corpus_path)
    n_docs = len(docs)
    logger.info(f"Loaded {n_docs} documents from corpus.")

    embeddings_cache_path = Path(cfg.clustering.embeddings_cache_path)
    doc_ids_cache_path = Path(cfg.clustering.doc_ids_cache_path)
    manifest_path = Path(cfg.clustering.cluster_manifest_path)

    # Check cache validity
    cache_exists = embeddings_cache_path.exists() and doc_ids_cache_path.exists()
    if cache_exists:
        try:
            with open(doc_ids_cache_path, "r", encoding="utf-8") as f:
                cached_doc_ids = json.load(f)
            
            if len(cached_doc_ids) == n_docs:
                logger.info("Cache hit: Document counts match. Loading embeddings from cache.")
                embeddings = np.load(embeddings_cache_path)
                
                # Check manifest model compatibility
                if manifest_path.exists():
                    try:
                        with open(manifest_path, "r", encoding="utf-8") as f:
                            manifest = json.load(f)
                        cached_model = manifest.get("embedding_model")
                        current_model = cfg.clustering.embedding_model
                        if cached_model and cached_model != current_model:
                            logger.warning(
                                f"Cache reuse warning: Cached embeddings were generated with model "
                                f"'{cached_model}' but current config specifies '{current_model}'."
                            )
                    except Exception as e:
                        logger.debug(f"Could not parse manifest for model name comparison: {e}")

                return embeddings, cached_doc_ids
            else:
                logger.info("Cache invalid: Corpus size changed. Re-embedding.")
        except Exception as e:
            logger.warning(f"Failed to load cached files: {e}. Re-embedding.")

    # Cache miss - compute embeddings
    logger.info(f"Cache miss or invalid. Embedding {n_docs} documents using {cfg.clustering.embedding_model}...")
    
    doc_ids = [d[0] for d in docs]
    texts = [d[1] for d in docs]

    model = SentenceTransformer(cfg.clustering.embedding_model)
    
    # Run batch embedding with tqdm progress bar
    embeddings = model.encode(
        texts,
        batch_size=cfg.clustering.embed_batch_size,
        show_progress_bar=True,
        convert_to_numpy=True
    )

    # Save to cache
    embeddings_cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(embeddings_cache_path, embeddings)
    with open(doc_ids_cache_path, "w", encoding="utf-8") as f:
        json.dump(doc_ids, f, indent=2)

    logger.info(f"Successfully saved {n_docs} embeddings and doc_ids to cache.")
    return embeddings, doc_ids
