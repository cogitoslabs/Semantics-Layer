"""
s5_clustering/dim_reducer.py - Modular dimensionality reduction engine (UMAP, PCA, Pass-through).
"""

import os
import logging
from typing import Dict, Any, Tuple
import numpy as np
from sklearn.decomposition import PCA

from lib.utils import PipelineConfig

logger = logging.getLogger(__name__)


def apply_dimensionality_reduction(
    embeddings: np.ndarray,
    cfg: PipelineConfig
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Apply dimensionality reduction to dense document embeddings.

    Supports strategies:
    - 'umap': Non-linear Uniform Manifold Approximation and Projection (default).
    - 'pca': Linear Principal Component Analysis.
    - 'none' / 'passthrough': Retains high-dimensional embeddings.

    Returns:
        Tuple of (reduced_embeddings, reduction_metadata).
    """
    n_docs, orig_dim = embeddings.shape
    raw_method = cfg.clustering.dim_reduction_method.lower().strip()
    
    # Handle backwards compatibility with use_pca=False when method is pca
    if not cfg.clustering.use_pca and raw_method == "pca":
        method = "none"
        logger.info("use_pca=False specified. Disabling PCA reduction.")
    else:
        method = raw_method

    metadata: Dict[str, Any] = {
        "method": method,
        "original_dim": orig_dim,
        "reduced_dim": orig_dim,
        "fallback_triggered": False,
        "fallback_reason": None,
    }

    if n_docs == 0:
        return embeddings, metadata

    # 1. UMAP Reduction
    if method == "umap":
        n_comps = cfg.clustering.umap_n_components
        requested_neighbors = cfg.clustering.umap_n_neighbors

        if n_docs <= n_comps or n_docs <= 3:
            msg = f"Document count ({n_docs}) is too small for UMAP components ({n_comps}). Falling back to PCA."
            logger.warning(msg)
            metadata["fallback_triggered"] = True
            metadata["fallback_reason"] = msg
            return _apply_pca(embeddings, cfg, metadata)

        try:
            import umap
        except ImportError:
            msg = "umap-learn package is not installed. Falling back to PCA."
            logger.warning(msg)
            metadata["fallback_triggered"] = True
            metadata["fallback_reason"] = msg
            return _apply_pca(embeddings, cfg, metadata)

        effective_neighbors = min(requested_neighbors, max(2, n_docs - 1))
        if effective_neighbors < requested_neighbors:
            logger.info(f"Adjusted UMAP n_neighbors from {requested_neighbors} to {effective_neighbors} for corpus size {n_docs}.")

        logger.info(f"Applying UMAP reduction: components={n_comps}, neighbors={effective_neighbors}, min_dist={cfg.clustering.umap_min_dist}, metric={cfg.clustering.umap_metric}")
        
        reducer = umap.UMAP(
            n_components=n_comps,
            n_neighbors=effective_neighbors,
            min_dist=cfg.clustering.umap_min_dist,
            metric=cfg.clustering.umap_metric,
            random_state=cfg.misc.seed,
        )
        
        reduced_arr = reducer.fit_transform(embeddings)
        
        # L2-normalize reduced space for consistent Euclidean/cosine metrics
        norms = np.linalg.norm(reduced_arr, axis=1, keepdims=True)
        norms[norms == 0] = 1e-12
        normalized_reduced = (reduced_arr / norms).astype(np.float32)

        metadata["reduced_dim"] = n_comps
        metadata["umap_effective_neighbors"] = effective_neighbors
        return normalized_reduced, metadata

    # 2. PCA Reduction
    elif method == "pca":
        return _apply_pca(embeddings, cfg, metadata)

    # 3. Pass-through / None Mode
    elif method in ("none", "passthrough"):
        logger.info("Dimensionality reduction disabled (passthrough mode). Using raw embeddings.")
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1e-12
        normalized_embeddings = (embeddings / norms).astype(np.float32)
        metadata["method"] = "passthrough"
        return normalized_embeddings, metadata

    else:
        raise ValueError(f"Unsupported dim_reduction_method: {method!r}. Choices: 'umap', 'pca', 'none', 'passthrough'.")


def _apply_pca(
    embeddings: np.ndarray,
    cfg: PipelineConfig,
    metadata: Dict[str, Any]
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Helper function to fit and transform PCA reduction."""
    n_docs, orig_dim = embeddings.shape
    pca_comps = cfg.clustering.pca_components

    if n_docs <= 1:
        logger.info("Document count <= 1. Returning original embeddings.")
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1e-12
        return (embeddings / norms).astype(np.float32), metadata

    target_comps = min(pca_comps, n_docs)
    logger.info(f"Applying PCA reduction to {target_comps} components...")

    # L2-normalize input embeddings
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1e-12
    emb_norm = embeddings / norms

    pca = PCA(n_components=target_comps, random_state=cfg.misc.seed)
    reduced = pca.fit_transform(emb_norm)

    # L2-normalize reduced output embeddings
    red_norms = np.linalg.norm(reduced, axis=1, keepdims=True)
    red_norms[red_norms == 0] = 1e-12
    normalized_reduced = (reduced / red_norms).astype(np.float32)

    metadata["method"] = "pca"
    metadata["reduced_dim"] = target_comps
    return normalized_reduced, metadata
