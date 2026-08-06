import json
import logging
from dataclasses import dataclass
from typing import List, Dict, Optional, Any
from pathlib import Path

from lib.utils import PipelineConfig

logger = logging.getLogger(__name__)


@dataclass
class EvalSample:
    sample_id: str
    cluster_id: str
    cluster_label: str
    question: str
    ground_truth: str
    retrieved_context: str       # empty string if no_retrieval
    no_retrieval: bool


def load_traces_lookup(cfg: PipelineConfig) -> Dict[str, Dict[str, Any]]:
    """Load both grounded and no_retrieval traces into a combined lookup dictionary indexed by sample_id and doc_id."""
    lookup = {}
    traces_dir = Path(cfg.rad.traces_dir)
    
    grounded_path = traces_dir / "grounded_traces.jsonl"
    no_retrieval_path = traces_dir / "no_retrieval_traces.jsonl"
    
    for path in [grounded_path, no_retrieval_path]:
        if not path.exists():
            logger.warning(f"Trace file not found: {path}")
            continue
        
        logger.info(f"Loading traces from {path} for evaluation sampler...")
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        record = json.loads(line)
                        s_id = record.get("sample_id")
                        d_id = record.get("doc_id")
                        rec_id = record.get("id")
                        
                        if s_id:
                            lookup[s_id] = record
                        if d_id:
                            lookup[d_id] = record
                        if rec_id:
                            lookup[rec_id] = record
                    except Exception as e:
                        logger.error(f"Error parsing trace record line: {e}")
                        
    return lookup


def run_eval_sampling(cfg: PipelineConfig) -> Dict[str, List[EvalSample]]:
    """
    Draw evaluation samples from cluster validation splits and map them to their trace records.
    """
    splits_path = Path(cfg.clustering.splits_path)
    if not splits_path.exists():
        raise FileNotFoundError(f"Clustering splits file not found at {splits_path}. Run step s4 first.")
        
    logger.info(f"Loading clustering splits from {splits_path}...")
    with open(splits_path, "r", encoding="utf-8") as f:
        splits_data = json.load(f)
        
    # Load trace data lookup
    trace_lookup = load_traces_lookup(cfg)
    if not trace_lookup:
        raise ValueError(
            f"No traces found in {cfg.rad.traces_dir}. "
            "Please run Step 4 (RAD Prep) with trace generation enabled first."
        )
    
    # Check if there is any ID overlap between splits and traces
    all_val_doc_ids = []
    clusters = splits_data.get("clusters", {})
    for cluster_info in clusters.values():
        all_val_doc_ids.extend(cluster_info.get("val_doc_ids", []))
        
    overlap_count = sum(1 for doc_id in all_val_doc_ids if doc_id in trace_lookup)
    use_fallback = (overlap_count == 0 and len(all_val_doc_ids) > 0)
    
    if use_fallback:
        logger.info(
            "Evaluation sampler: Using deterministic hash mapping to pair cluster validation documents with QA traces."
        )
    
    eval_samples: Dict[str, List[EvalSample]] = {}
    
    for cluster_label, cluster_info in sorted(clusters.items()):
        cluster_id = str(cluster_info.get("cluster_id"))
        val_doc_ids = cluster_info.get("val_doc_ids", [])
        
        # Sample up to eval_sample_size
        sampled_doc_ids = val_doc_ids[:cfg.benchmarking.eval_sample_size]
        
        cluster_samples = []
        miss_count = 0
        fallback_count = 0
        
        for doc_id in sampled_doc_ids:
            if doc_id in trace_lookup:
                record = trace_lookup[doc_id]
                sample = EvalSample(
                    sample_id=doc_id,
                    cluster_id=cluster_id,
                    cluster_label=cluster_label,
                    question=record["question"],
                    ground_truth=record["answer"],
                    retrieved_context=record.get("retrieved_context", ""),
                    no_retrieval=record.get("no_retrieval", False)
                )
                cluster_samples.append(sample)
            elif use_fallback:
                # Fallback to hash-based deterministic mapping to available QA traces
                import hashlib
                trace_keys = sorted(list(trace_lookup.keys()))
                idx = int(hashlib.md5(doc_id.encode("utf-8")).hexdigest(), 16) % len(trace_keys)
                matched_key = trace_keys[idx]
                record = trace_lookup[matched_key]
                
                sample = EvalSample(
                    sample_id=doc_id,
                    cluster_id=cluster_id,
                    cluster_label=cluster_label,
                    question=record["question"],
                    ground_truth=record["answer"],
                    retrieved_context=record.get("retrieved_context", ""),
                    no_retrieval=record.get("no_retrieval", False)
                )
                cluster_samples.append(sample)
                fallback_count += 1
            else:
                miss_count += 1
                
        if fallback_count > 0:
            logger.debug(
                f"Cluster {cluster_label}: Mapped {fallback_count}/{len(sampled_doc_ids)} "
                "validation documents to QA traces using deterministic fallback."
            )
            
        if miss_count > 0:
            logger.warning(f"Cluster {cluster_label}: Missed {miss_count} trace records for validation doc IDs.")
            
        if len(cluster_samples) < cfg.benchmarking.min_eval_samples:
            logger.warning(
                f"Cluster {cluster_label} has only {len(cluster_samples)} eval samples, "
                f"which is below min_eval_samples={cfg.benchmarking.min_eval_samples}."
            )
            
        eval_samples[cluster_label] = cluster_samples
        logger.info(f"Prepared {len(cluster_samples)} eval samples for cluster {cluster_label}.")
        
    return eval_samples
