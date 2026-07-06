"""
s5_clustering/clusterer.py - Applies HDBSCAN clustering and resolves noise points.
"""

import logging
from dataclasses import dataclass
import numpy as np
from typing import List
import hdbscan
from sklearn.metrics.pairwise import cosine_distances
from sklearn.decomposition import PCA

from lib.utils import PipelineConfig

logger = logging.getLogger(__name__)


@dataclass
class ClusterAssignment:
    doc_id: str
    cluster_id: int       # HDBSCAN label (noise originally -1, reassigned)
    cluster_label: str    # "cluster_007" (zero-padded to 3 digits)
    is_noise: bool        # True if HDBSCAN originally labeled as noise
    assigned_by: str      # "hdbscan" | "nearest_centroid" | "dropped"


def run_clustering(cfg: PipelineConfig, embeddings: np.ndarray, doc_ids: List[str]) -> List[ClusterAssignment]:
    """
    Perform HDBSCAN clustering on embeddings.
    """
    logger.info("Initializing HDBSCAN...")
    
    # Configure HDBSCAN
    min_cluster_size = cfg.clustering.hdbscan_min_cluster_size
    min_samples = cfg.clustering.hdbscan_min_samples
    metric = cfg.clustering.hdbscan_metric
    use_pca = cfg.clustering.use_pca
    pca_comps = cfg.clustering.pca_components
    
    # Process embeddings
    if use_pca and embeddings.shape[0] > pca_comps:
        logger.info(f"Applying PCA dimensionality reduction to {pca_comps} components...")
        # L2-normalize original embeddings first
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1e-12
        emb_normalized = embeddings / norms
        
        # Fit and apply PCA
        pca = PCA(n_components=pca_comps, random_state=cfg.misc.seed)
        emb_reduced = pca.fit_transform(emb_normalized)
        
        # L2-normalize PCA-reduced embeddings
        reduced_norms = np.linalg.norm(emb_reduced, axis=1, keepdims=True)
        reduced_norms[reduced_norms == 0] = 1e-12
        clustering_embeddings = emb_reduced / reduced_norms
        
        # Use Euclidean metric for normalized reduced space
        clustering_metric = "euclidean"
    else:
        logger.info("PCA is disabled or document count is too small. Using raw embeddings.")
        clustering_embeddings = embeddings
        clustering_metric = metric
        
    logger.info("Fitting HDBSCAN on document embeddings...")
    
    # HDBSCAN does not natively support 'cosine' with space trees, so we use precomputed cosine distances.
    if clustering_metric == "cosine":
        logger.info("Computing pairwise cosine distance matrix...")
        dist_matrix = cosine_distances(clustering_embeddings).astype(np.float64)
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            metric="precomputed"
        )
        labels = clusterer.fit_predict(dist_matrix)
    else:
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            metric=clustering_metric
        )
        labels = clusterer.fit_predict(clustering_embeddings)
    
    # Identify unique clusters (excluding noise -1)
    unique_labels = [int(l) for l in np.unique(labels) if l != -1]
    n_clusters = len(unique_labels)
    logger.info(f"HDBSCAN found {n_clusters} clusters (excluding noise).")

    assignments = []
    
    # Find noise indices
    noise_mask = (labels == -1)
    noise_indices = np.where(noise_mask)[0]
    n_noise = len(noise_indices)
    logger.info(f"HDBSCAN labeled {n_noise} out of {len(doc_ids)} documents as noise.")

    noise_mode = cfg.clustering.noise_assignment
    
    if noise_mode == "nearest" and n_noise > 0 and n_clusters > 0:
        logger.info("Resolving noise points using nearest centroid assignment...")
        
        # Calculate cluster centroids
        centroids = []
        cluster_order = sorted(unique_labels)
        
        for cluster_id in cluster_order:
            cluster_indices = np.where(labels == cluster_id)[0]
            cluster_embeddings = clustering_embeddings[cluster_indices]
            # Centroid is the mean of the embeddings
            centroid = np.mean(cluster_embeddings, axis=0)
            # L2-normalize centroid to simplify cosine similarity calculation
            centroid_norm = np.linalg.norm(centroid)
            if centroid_norm > 0:
                centroid = centroid / centroid_norm
            centroids.append(centroid)
            
        centroids = np.stack(centroids)  # Shape: (n_clusters, embedding_dim)
        
        # Extract noise point embeddings
        noise_embeddings = clustering_embeddings[noise_indices]
        # Calculate cosine similarity: dot product between noise vectors and normalized centroids
        similarities = np.dot(noise_embeddings, centroids.T)
        
        # For each noise point, get index of the highest similarity centroid
        nearest_cluster_indices = np.argmax(similarities, axis=1)
        
        # Map back to the actual cluster label (cluster_id)
        resolved_labels = [cluster_order[idx] for idx in nearest_cluster_indices]
    else:
        resolved_labels = []

    dropped_count = 0
    
    for idx, (doc_id, original_label) in enumerate(zip(doc_ids, labels)):
        is_noise = bool(original_label == -1)
        
        if not is_noise:
            # Point was clustered successfully by HDBSCAN
            cluster_id = int(original_label)
            cluster_label = f"cluster_{cluster_id:03d}"
            assigned_by = "hdbscan"
        else:
            # Point was labeled as noise
            if noise_mode == "nearest" and n_clusters > 0:
                # Find its resolved label from resolved_labels mapping
                noise_pos = np.where(noise_indices == idx)[0][0]
                cluster_id = int(resolved_labels[noise_pos])
                cluster_label = f"cluster_{cluster_id:03d}"
                assigned_by = "nearest_centroid"
            else:
                # Noise mode is 'drop' or no clusters found to assign to
                cluster_id = -1
                cluster_label = "dropped"
                assigned_by = "dropped"
                dropped_count += 1
                
        assignments.append(
            ClusterAssignment(
                doc_id=doc_id,
                cluster_id=cluster_id,
                cluster_label=cluster_label,
                is_noise=is_noise,
                assigned_by=assigned_by
            )
        )
        
    if noise_mode == "drop":
        logger.info(f"Dropped {dropped_count} noise documents from active clusters.")
        
    return assignments
