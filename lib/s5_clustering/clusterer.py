"""
s5_clustering/clusterer.py - Applies HDBSCAN clustering and resolves noise points.
"""

import logging
from dataclasses import dataclass
import numpy as np
from typing import List
import hdbscan
from sklearn.metrics.pairwise import cosine_distances

from lib.utils import PipelineConfig
from lib.s5_clustering.dim_reducer import apply_dimensionality_reduction

logger = logging.getLogger(__name__)


@dataclass
class ClusterAssignment:
    doc_id: str
    cluster_id: int       # HDBSCAN label (noise originally -1, reassigned)
    cluster_label: str    # "cluster_007" (zero-padded to 3 digits)
    is_noise: bool        # True if HDBSCAN originally labeled as noise
    assigned_by: str      # "hdbscan" | "nearest_centroid" | "dropped"


def merge_similar_clusters(
    raw_embeddings: np.ndarray,
    labels: np.ndarray,
    similarity_threshold: float = 0.85
) -> np.ndarray:
    """
    Hierarchically merges clusters whose L2-normalized centroids in 768D raw embedding space
    have cosine similarity >= similarity_threshold.
    """
    unique_labels = sorted([int(l) for l in np.unique(labels) if l != -1])
    if len(unique_labels) <= 1:
        return labels

    # Normalize raw 768D embeddings if not already normalized
    norms = np.linalg.norm(raw_embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1e-12
    norm_raw = raw_embeddings / norms

    # Compute centroids for each active cluster in 768D space
    label_to_centroid = {}
    for cid in unique_labels:
        c_mask = (labels == cid)
        centroid = np.mean(norm_raw[c_mask], axis=0)
        c_norm = np.linalg.norm(centroid)
        if c_norm > 0:
            centroid = centroid / c_norm
        label_to_centroid[cid] = centroid

    # Build union-find / label map for cluster merging
    label_map = {cid: cid for cid in unique_labels}

    for i in range(len(unique_labels)):
        cid1 = unique_labels[i]
        for j in range(i + 1, len(unique_labels)):
            cid2 = unique_labels[j]
            root1 = label_map[cid1]
            root2 = label_map[cid2]
            if root1 == root2:
                continue

            sim = float(np.dot(label_to_centroid[root1], label_to_centroid[root2]))
            if sim >= similarity_threshold:
                # Merge root2 into root1
                for key in label_map:
                    if label_map[key] == root2:
                        label_map[key] = root1

    # Remap merged roots to contiguous cluster IDs (0, 1, 2, ...)
    unique_roots = sorted(list(set(label_map.values())))
    root_to_new_id = {root: new_id for new_id, root in enumerate(unique_roots)}

    new_labels = np.full_like(labels, -1)
    for idx, l in enumerate(labels):
        if l != -1:
            target_root = label_map[l]
            new_labels[idx] = root_to_new_id[target_root]

    merged_count = len(unique_labels) - len(unique_roots)
    if merged_count > 0:
        logger.info(
            f"Cluster merging (768D space): Consolidated {len(unique_labels)} clusters into {len(unique_roots)} clusters "
            f"(merged {merged_count} similar clusters with cosine similarity >= {similarity_threshold})."
        )

    return new_labels


def run_clustering(cfg: PipelineConfig, embeddings: np.ndarray, doc_ids: List[str]) -> List[ClusterAssignment]:
    """
    Perform HDBSCAN clustering on embeddings after applying dimensionality reduction.
    """
    logger.info("Initializing clustering process...")
    
    # Configure HDBSCAN
    min_cluster_size = cfg.clustering.hdbscan_min_cluster_size
    min_samples = cfg.clustering.hdbscan_min_samples
    metric = cfg.clustering.hdbscan_metric
    
    # Apply dimensionality reduction (UMAP / PCA / Pass-through)
    clustering_embeddings, dim_metadata = apply_dimensionality_reduction(embeddings, cfg)
    
    if dim_metadata["method"] in ("umap", "pca"):
        clustering_metric = "euclidean"
    else:
        clustering_metric = metric

    logger.info(f"Fitting HDBSCAN on document embeddings (method={dim_metadata['method']}, dim={clustering_embeddings.shape[1]})...")
    
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

    # Perform cluster consolidation merging in 768D raw embedding space if enabled
    if cfg.clustering.enable_cluster_merging:
        labels = merge_similar_clusters(
            raw_embeddings=embeddings,
            labels=labels,
            similarity_threshold=cfg.clustering.cluster_merge_threshold
        )
    
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
