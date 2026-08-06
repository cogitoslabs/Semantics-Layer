import json
import logging
import time
from pathlib import Path

from lib.utils import PipelineConfig, setup_logger
from lib.s4_rad_prep.chunker import run_chunking
from lib.s4_rad_prep.indexer import run_indexing
from lib.s4_rad_prep.retriever import Retriever
from lib.s4_rad_prep.no_retrieval_router import NoRetrievalRouter
from lib.s4_rad_prep.prompt_generator import PromptGenerator

logger = logging.getLogger(__name__)


def run_rad_prep_pipeline(cfg: PipelineConfig) -> None:
    """Orchestrate Step 4 Retrieval-Augmented Distillation Preparation (RAD Prep) pipeline."""
    # Guarantee pipeline.log is initialized and attached
    setup_logger("lib", cfg.logging)

    logger.info("Starting RAD Prep Pipeline...")
    cfg.ensure_dirs()

    logs_dir = Path(cfg.logging.log_dir) / "rad_prep"
    logs_dir.mkdir(parents=True, exist_ok=True)

    no_retrieval_rates_path = logs_dir / "no_retrieval_rates.jsonl"
    phase_manifest_path = logs_dir / "phase_manifest.json"

    grounded_path = Path(cfg.rad.traces_dir) / "grounded_traces.jsonl"
    no_ret_path = Path(cfg.rad.traces_dir) / "no_retrieval_traces.jsonl"

    # Reset log and output files
    if no_retrieval_rates_path.exists():
        no_retrieval_rates_path.unlink()
    if grounded_path.exists():
        grounded_path.unlink()
    if no_ret_path.exists():
        no_ret_path.unlink()

    logger.info("Chunking and indexing retrieval corpus...")
    chunks = run_chunking(cfg)
    run_indexing(cfg, chunks)

    logger.info("Preparing grounded QA prompts...")

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
        logger.warning("No QA samples found. Exiting prompt preparation.")
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
    generator = PromptGenerator(cfg)

    # Generate prompt records
    trace_counts = generator.generate_prompts(samples, retrieved_results, router)

    grounded_count = trace_counts["grounded_count"]
    no_ret_count = trace_counts["no_retrieval_count"]

    total_attempted = grounded_count + no_ret_count

    router_stats = router.get_aggregate_stats()

    # Check validation gates
    target_min_prompts = int(cfg.rad.min_grounded_pct * total_attempted)
    passed_min_prompts = grounded_count >= target_min_prompts


    cluster_no_retrieval_rates = router_stats.get("by_cluster", {})
    passed_cluster_no_ret = True
    for cluster, stats in cluster_no_retrieval_rates.items():
        if stats["rate"] > 0.30:
            logger.warning(f"Cluster {cluster} exceeded 30% no-retrieval rate: {stats['rate']:.2%}")
            passed_cluster_no_ret = False

    status = "complete" if (passed_min_prompts and passed_cluster_no_ret) else "incomplete"

    manifest = {
        "status": status,
        "timestamp": time.time(),
        "metrics": {
            "grounded_prompt_count": grounded_count,
            "no_retrieval_prompt_count": no_ret_count,
            "grounded_trace_count": grounded_count,
            "no_retrieval_trace_count": no_ret_count,
            "total_attempted": total_attempted,
            "no_retrieval_stats": router_stats,
        },
        "gates": {
            "passed_min_prompts": passed_min_prompts,
            "passed_cluster_no_ret": passed_cluster_no_ret,
        },
    }

    logger.info(f"Writing phase manifest to {phase_manifest_path}")
    with open(phase_manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    # Print formatted summary block to console and pipeline.log
    logger.info(
        f"\n"
        f"================================================================\n"
        f"  Step 4 (RAD Prep) Execution Summary\n"
        f"================================================================\n"
        f"  Overall Status          : {status.upper()}\n"
        f"  Grounded Prompts        : {grounded_count} (Gate: >= {target_min_prompts} -> {'PASS' if passed_min_prompts else 'INCOMPLETE'})\n"
        f"  No-Retrieval Prompts    : {no_ret_count} ({router_stats.get('overall_rate', 0.0):.1%} overall, Gate <= 30% -> {'PASS' if passed_cluster_no_ret else 'WARN'})\n"
        f"  Total Attempted Samples : {total_attempted}\n"
        f"  Manifest Output Path    : {phase_manifest_path}\n"
        f"================================================================\n"
    )




