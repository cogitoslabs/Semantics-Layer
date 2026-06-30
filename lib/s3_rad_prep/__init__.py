import json
import logging
import time
from pathlib import Path
from typing import List, Dict, Any

from lib.utils import PipelineConfig
from lib.s3_rad_prep.chunker import run_chunking, Chunk
from lib.s3_rad_prep.indexer import run_indexing
from lib.s3_rad_prep.retriever import Retriever
from lib.s3_rad_prep.no_retrieval_router import NoRetrievalRouter
from lib.s3_rad_prep.trace_generator import TraceGenerator

logger = logging.getLogger(__name__)

def run_rad_prep_pipeline(cfg: PipelineConfig, rad_mode: str = "full") -> None:
    """Orchestrate the Step 0.3 Retrieval-Augmented Distillation Preparation (RAD Prep) pipeline."""
    logger.info(f"Starting RAD Prep Pipeline in mode: {rad_mode}")
    cfg.ensure_dirs()

    logs_dir = Path(cfg.storage.log_dir) / "rad_prep"
    logs_dir.mkdir(parents=True, exist_ok=True)

    no_retrieval_rates_path = logs_dir / "no_retrieval_rates.jsonl"
    discarded_traces_path = logs_dir / "discarded_traces.jsonl"
    phase_manifest_path = logs_dir / "phase_manifest.json"

    # Reset log files if we are starting a trace generation run
    if rad_mode in ("full", "traces"):
        if no_retrieval_rates_path.exists():
            no_retrieval_rates_path.unlink()
        if discarded_traces_path.exists():
            discarded_traces_path.unlink()

    chunks = None
    if rad_mode in ("full", "index"):
        logger.info("Chunking and indexing retrieval corpus...")
        chunks = run_chunking(cfg)
        run_indexing(cfg, chunks)

    if rad_mode in ("full", "traces"):
        logger.info("Generating grounded teacher traces...")

        # Load samples
        qa_samples_path = Path(cfg.rad.qa_samples_path)
        if not qa_samples_path.exists():
            raise FileNotFoundError(f"QA samples file not found at {qa_samples_path}")

        samples = []
        with open(qa_samples_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    samples.append(json.loads(line))

        if not samples:
            logger.warning("No QA samples found. Exiting trace generation.")
            return

        # Initialize retriever
        retriever = Retriever(cfg)

        # Retrieve context for all samples
        logger.info(f"Retrieving context for {len(samples)} samples using mode {cfg.rad.retrieval_mode}")
        retrieved_results = []
        for sample in samples:
            question = sample["question"]
            res = retriever.retrieve(question)
            retrieved_results.append(res)

        # Initialize router and generator
        router = NoRetrievalRouter(no_retrieval_rates_path)
        generator = TraceGenerator(cfg)

        # Generate traces
        generator.generate_traces(samples, retrieved_results, router)
        router.flush_batch()

        # Calculate statistics for validation gate
        grounded_path = Path(cfg.rad.traces_dir) / "grounded_traces.jsonl"
        no_ret_path = Path(cfg.rad.traces_dir) / "no_retrieval_traces.jsonl"

        grounded_count = 0
        if grounded_path.exists():
            with open(grounded_path, "r", encoding="utf-8") as f:
                grounded_count = sum(1 for line in f if line.strip())

        no_ret_count = 0
        if no_ret_path.exists():
            with open(no_ret_path, "r", encoding="utf-8") as f:
                no_ret_count = sum(1 for line in f if line.strip())

        discarded_count = 0
        if discarded_traces_path.exists():
            with open(discarded_traces_path, "r", encoding="utf-8") as f:
                discarded_count = sum(1 for line in f if line.strip())

        total_attempted = grounded_count + no_ret_count + discarded_count
        discarded_rate = discarded_count / total_attempted if total_attempted > 0 else 0.0

        router_stats = router.get_aggregate_stats()

        # Check validation gates
        passed_min_traces = grounded_count >= cfg.rad.min_traces

        cluster_no_retrieval_rates = router_stats.get("by_cluster", {})
        passed_cluster_no_ret = True
        for cluster, stats in cluster_no_retrieval_rates.items():
            if stats["rate"] > 0.30:
                logger.warning(f"Cluster {cluster} exceeded 30% no-retrieval rate: {stats['rate']:.2%}")
                passed_cluster_no_ret = False

        passed_discarded_rate = discarded_rate <= 0.20
        if not passed_discarded_rate:
            logger.warning(f"Discarded trace rate exceeded 20%: {discarded_rate:.2%}")

        # If any validation warnings occur, it is incomplete according to the spec gate criteria
        status = "complete" if (passed_min_traces and passed_cluster_no_ret and passed_discarded_rate) else "incomplete"

        manifest = {
            "status": status,
            "timestamp": time.time(),
            "metrics": {
                "grounded_trace_count": grounded_count,
                "no_retrieval_trace_count": no_ret_count,
                "discarded_trace_count": discarded_count,
                "total_attempted": total_attempted,
                "discarded_rate": discarded_rate,
                "no_retrieval_stats": router_stats
            },
            "gates": {
                "passed_min_traces": passed_min_traces,
                "passed_cluster_no_ret": passed_cluster_no_ret,
                "passed_discarded_rate": passed_discarded_rate
            }
        }

        logger.info(f"Writing phase manifest to {phase_manifest_path}")
        with open(phase_manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        logger.info(f"RAD Prep pipeline completed trace generation. Status: {status}")
