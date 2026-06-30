"""
s4_clustering/splitter.py - Splits per-cluster documents into dev/val/sealed and computes reweighting caps.
"""

import math
import random
import logging
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

from lib.utils import PipelineConfig
from lib.s4_clustering.clusterer import ClusterAssignment

logger = logging.getLogger(__name__)


@dataclass
class ClusterSplit:
    cluster_id: int
    cluster_label: str
    dev_doc_ids: List[str]
    val_doc_ids: List[str]
    sealed_doc_ids: List[str]
    total_docs: int
    raw_fraction: float
    reweight_cap: Optional[int]


def run_splitting(cfg: PipelineConfig, assignments: List[ClusterAssignment]) -> Dict[str, Any]:
    """
    Perform per-cluster dev/val/sealed splits and calculate imbalance reweighting caps.
    """
    logger.info("Initializing document splitter...")
    
    # 1. Group active documents by cluster
    cluster_docs: Dict[str, List[str]] = {}
    cluster_label_to_id: Dict[str, int] = {}
    
    for ass in assignments:
        if ass.assigned_by == "dropped":
            continue
        cluster_docs.setdefault(ass.cluster_label, []).append(ass.doc_id)
        cluster_label_to_id[ass.cluster_label] = ass.cluster_id
        
    # Calculate total active documents across all clusters
    total_active_docs = sum(len(ids) for ids in cluster_docs.values())
    total_clusters = len(cluster_docs)
    logger.info(f"Splitting {total_active_docs} active documents across {total_clusters} clusters.")
    
    # Setup seed for reproducibility
    seed = cfg.misc.seed
    
    # Retrieve split ratios from config
    dev_ratio = cfg.clustering.split_dev_ratio
    val_ratio = cfg.clustering.split_val_ratio
    sealed_ratio = cfg.clustering.split_sealed_ratio
    
    min_fraction = cfg.clustering.cluster_min_fraction
    max_fraction = cfg.clustering.cluster_max_fraction
    
    splits_dict = {}
    
    for cluster_label, doc_ids in sorted(cluster_docs.items()):
        cluster_id = cluster_label_to_id[cluster_label]
        cluster_size = len(doc_ids)
        
        # Calculate raw fraction
        raw_fraction = cluster_size / total_active_docs if total_active_docs > 0 else 0.0
        
        # Calculate reweight cap recommendation
        reweight_cap = None
        if raw_fraction < min_fraction:
            reweight_cap = int(math.ceil(total_active_docs * min_fraction))
        elif raw_fraction > max_fraction:
            reweight_cap = int(math.floor(total_active_docs * max_fraction))
            
        # Shuffle document IDs deterministically
        shuffled_ids = list(doc_ids)
        rng = random.Random(seed)
        rng.shuffle(shuffled_ids)
        
        if cluster_size < 3:
            # For clusters with < 3 documents, assign all to dev
            dev_ids = shuffled_ids
            val_ids = []
            sealed_ids = []
        else:
            # Deterministic index-based split
            val_count = int(round(cluster_size * val_ratio))
            sealed_count = int(round(cluster_size * sealed_ratio))
            
            # Enforce at least 1 document in validation split if total_docs >= 3
            if val_count == 0:
                val_count = 1
                
            dev_count = cluster_size - val_count - sealed_count
            
            # Adjust if dev count is negative or zero
            if dev_count < 1:
                dev_count = 1
                val_count = cluster_size - dev_count - sealed_count
                if val_count < 1:
                    val_count = 1
                    sealed_count = cluster_size - dev_count - val_count
                    
            dev_ids = shuffled_ids[:dev_count]
            val_ids = shuffled_ids[dev_count:dev_count + val_count]
            sealed_ids = shuffled_ids[dev_count + val_count:]
            
        splits_dict[cluster_label] = ClusterSplit(
            cluster_id=cluster_id,
            cluster_label=cluster_label,
            dev_doc_ids=dev_ids,
            val_doc_ids=val_ids,
            sealed_doc_ids=sealed_ids,
            total_docs=cluster_size,
            raw_fraction=raw_fraction,
            reweight_cap=reweight_cap
        )
        
    # Build the final output JSON dictionary
    output_clusters = {}
    for label, split in sorted(splits_dict.items()):
        output_clusters[label] = {
            "cluster_id": split.cluster_id,
            "total_docs": split.total_docs,
            "dev_doc_ids": split.dev_doc_ids,
            "val_doc_ids": split.val_doc_ids,
            "sealed_doc_ids": split.sealed_doc_ids,
            "raw_fraction": float(f"{split.raw_fraction:.6f}"),
            "reweight_cap": split.reweight_cap
        }
        
    final_output = {
        "total_docs": total_active_docs,
        "total_clusters": total_clusters,
        "seed": seed,
        "clusters": output_clusters
    }
    
    return final_output
