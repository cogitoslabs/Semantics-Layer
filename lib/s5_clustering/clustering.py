import json
import logging
from pathlib import Path
from dataclasses import asdict

from lib.utils import PipelineConfig
from lib.s5_clustering.embedder import run_embedding
from lib.s5_clustering.clusterer import run_clustering
from lib.s5_clustering.splitter import run_splitting
from lib.s5_clustering.cluster_reporter import run_reporting

logger = logging.getLogger(__name__)


def run_clustering_pipeline(cfg: PipelineConfig) -> None:
    """
    Run the Step 4 Corpus Engineering & Micro-Clustering pipeline.
    """
    logger.info("Starting Phase 1 Step 4: Corpus Engineering & Micro-Clustering Pipeline")
    
    # 0. Ensure target output directories exist
    cfg.ensure_dirs()
    Path(cfg.clustering.assignments_path).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg.clustering.splits_path).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg.clustering.cluster_manifest_path).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg.clustering.cluster_report_path).parent.mkdir(parents=True, exist_ok=True)
    
    # 1. Generate or load embeddings
    embeddings, doc_ids = run_embedding(cfg)
    
    # 2. Perform clustering
    assignments = run_clustering(cfg, embeddings, doc_ids)
    
    # 3. Write cluster assignments to JSONL file
    assignments_path = Path(cfg.clustering.assignments_path)
    logger.info(f"Writing cluster assignments to {assignments_path}...")
    with open(assignments_path, "w", encoding="utf-8") as f:
        for ass in assignments:
            f.write(json.dumps(asdict(ass)) + "\n")
            
    # 4. Generate splits and compute reweighting caps
    splits_data = run_splitting(cfg, assignments)
    
    # 5. Write splits to JSON file
    splits_path = Path(cfg.clustering.splits_path)
    logger.info(f"Writing splits to {splits_path}...")
    with open(splits_path, "w", encoding="utf-8") as f:
        json.dump(splits_data, f, indent=2)
        
    # 6. Generate reports, run validation gates, and write manifest
    run_reporting(cfg, assignments, splits_data)
    
    logger.info("Corpus Engineering & Micro-Clustering pipeline execution successfully completed.")
