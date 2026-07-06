"""
s5_clustering/cluster_reporter.py - Generates cluster report logs and manifest validation gates.
"""

import json
import logging
import numpy as np
from pathlib import Path
from typing import List, Dict, Any

from lib.utils import PipelineConfig
from lib.s5_clustering.clusterer import ClusterAssignment

logger = logging.getLogger(__name__)


def run_reporting(
    cfg: PipelineConfig,
    assignments: List[ClusterAssignment],
    splits_data: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Generate the validation manifest and detailed report.
    Checks validation gate thresholds.
    """
    logger.info("Generating cluster reports and running validation gates...")
    
    total_docs = len(assignments)
    
    # Original noise point count
    noise_docs = sum(1 for ass in assignments if ass.is_noise)
    noise_fraction = noise_docs / total_docs if total_docs > 0 else 0.0
    
    # Active clusters info (from the splits_data, which contains only active assignments)
    clusters_info = splits_data["clusters"]
    total_clusters = len(clusters_info)
    
    cluster_sizes = [c["total_docs"] for c in clusters_info.values()]
    total_active_docs = sum(cluster_sizes)
    
    # Stats
    if cluster_sizes:
        min_sz = int(np.min(cluster_sizes))
        max_sz = int(np.max(cluster_sizes))
        mean_sz = float(np.mean(cluster_sizes))
        median_sz = float(np.median(cluster_sizes))
        std_sz = float(np.std(cluster_sizes))
    else:
        min_sz = max_sz = mean_sz = median_sz = std_sz = 0
        
    cluster_sizes_stats = {
        "min": min_sz,
        "max": max_sz,
        "mean": float(f"{mean_sz:.1f}"),
        "median": float(f"{median_sz:.1f}"),
        "std": float(f"{std_sz:.1f}")
    }
    
    # Capped clusters counts
    capped_min_count = 0
    capped_max_count = 0
    
    for c in clusters_info.values():
        cap = c.get("reweight_cap")
        if cap is not None:
            raw_fraction = c["raw_fraction"]
            if raw_fraction < cfg.clustering.cluster_min_fraction:
                capped_min_count += 1
            elif raw_fraction > cfg.clustering.cluster_max_fraction:
                capped_max_count += 1
                
    # Run validation checks
    warnings = []
    hard_failures = []
    
    # 1. Cluster count checks
    if total_clusters < cfg.clustering.min_clusters:
        hard_failures.append(f"Hard fail: Cluster count is {total_clusters} (expected >= {cfg.clustering.min_clusters}). HDBSCAN params need tuning.")
    elif total_clusters < 50:
        warnings.append(f"Fewer micro-clusters than expected: {total_clusters} < 50 threshold.")
        
    # 2. Noise fraction check
    if noise_fraction > 0.30:
        warnings.append(f"High noise fraction: {noise_fraction:.2%} > 30% threshold. Consider lowering min_cluster_size.")
        
    # 3. Largest cluster fraction check
    largest_fraction = max_sz / total_active_docs if total_active_docs > 0 else 0.0
    if largest_fraction > 0.40:
        warnings.append(f"Severe imbalance before reweighting: Largest cluster fraction is {largest_fraction:.2%} > 40% threshold.")
        
    # 4. Validation split non-empty check
    for label, c in clusters_info.items():
        if c["total_docs"] >= 3 and len(c["val_doc_ids"]) == 0:
            hard_failures.append(f"Hard fail: Cluster '{label}' has {c['total_docs']} documents but empty validation split.")
            
    # Write warnings to logger
    for warn in warnings:
        logger.warning(warn)
        
    # Write hard failures to logger
    for fail in hard_failures:
        logger.error(fail)
        
    # Determine status
    status = "complete" if not hard_failures else "failed"
    
    manifest = {
        "status": status,
        "embedding_model": cfg.clustering.embedding_model,
        "total_docs": total_docs,
        "noise_docs": noise_docs,
        "noise_fraction": float(f"{noise_fraction:.4f}"),
        "total_clusters": total_clusters,
        "cluster_sizes": cluster_sizes_stats,
        "capped_min_count": capped_min_count,
        "capped_max_count": capped_max_count,
        "warnings": warnings
    }
    
    # Write manifest file
    manifest_path = Path(cfg.clustering.cluster_manifest_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
        
    # Write report file
    report_data = {}
    for label, c in sorted(clusters_info.items()):
        report_data[label] = {
            "doc_count": c["total_docs"],
            "fraction": c["raw_fraction"],
            "reweight_cap": c["reweight_cap"]
        }
        
    report_path = Path(cfg.clustering.cluster_report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)
        
    logger.info(f"Wrote cluster manifest to {manifest_path}")
    logger.info(f"Wrote cluster report to {report_path}")
    
    if hard_failures:
        raise ValueError("Clustering pipeline failed validation gates:\n" + "\n".join(f"  • {f}" for f in hard_failures))
        
    return manifest
