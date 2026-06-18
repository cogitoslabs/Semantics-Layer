"""
evaluation/eval_runner.py — Orchestrate all four probes in a single evaluation run

This is what the training loop calls at each EVAL_INTERVAL_TOKENS.
It returns a structured metrics dict that is both logged to JSONL and
used by gate_logic.check_convergence_gates().
"""

import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

from lib.utils import DAPTConfig
from lib.s2_dapt.probes.qa_probe         import eval_qa_accuracy
from lib.s2_dapt.probes.perplexity_probe import eval_perplexity
from lib.s2_dapt.probes.terminology_probe import eval_terminology_coverage
from lib.s2_dapt.probes.retrieval_probe  import eval_retrieval_precision
from lib.utils.logger import get_logger, MetricsWriter

logger = get_logger("dapt.eval_runner")


def run_all_probes(
    model,
    tokenizer,
    cfg: DAPTConfig,
    state: Dict[str, Any],
    metrics_writer: MetricsWriter,
    device: str = "cuda",
) -> Dict[str, Any]:
    """
    Run all four probes in sequence, update state, write metrics to JSONL.

    Returns the full metrics dict for this evaluation checkpoint.

    Order: PPL first (cheapest, most stable), then QA, then terminology,
    then retrieval (most expensive due to BERTScore).
    """
    eval_start = time.time()
    logger.info(
        f"\n{'='*62}\n"
        f"  Starting Eval #{state['eval_count']} | "
        f"Tokens: {state['tokens_processed']/1e3:.1f}K | "
        f"Pass: {state['tokens_processed']/cfg.corpus.total_corpus_tokens:.3f}x\n"
        f"{'='*62}"
    )

    # ── Probe B: Perplexity (cheapest — run first) ────────────────────────────
    logger.info("  Running Probe 1: Perplexity...")
    t0 = time.time()
    ppl_result = eval_perplexity(
        model=model,
        tokenizer=tokenizer,
        ppl_corpus_path=cfg.data.ppl_corpus_path,
        max_tokens=cfg.probes.perplexity_eval_tokens,
        seq_len=cfg.model.max_seq_len,
        batch_size=cfg.optimizer.eval_batch_size,
        device=device,
    )
    ppl_elapsed = time.time() - t0

    # ── Probe A: QA Accuracy ──────────────────────────────────────────────────
    logger.info("  Running Probe 2: QA Accuracy...")
    t0 = time.time()
    qa_result = eval_qa_accuracy(
        model=model,
        tokenizer=tokenizer,
        qa_probe_path=cfg.data.qa_probe_path,
        device=device,
    )
    qa_elapsed = time.time() - t0

    # ── Probe C: Terminology Coverage ─────────────────────────────────────────
    logger.info("  Running Probe 3: Terminology Coverage...")
    t0 = time.time()
    term_result = eval_terminology_coverage(
        model=model,
        tokenizer=tokenizer,
        vocab_cloze_path=cfg.data.vocab_cloze_path,
        top_k=cfg.probes.term_cov_top_k,
        max_new_tokens=cfg.probes.term_cov_max_new_tokens,
        device=device,
    )
    term_elapsed = time.time() - t0

    # ── Probe D: Retrieval Precision ──────────────────────────────────────────
    logger.info("  Running Probe 4: Retrieval Precision (BERTScore)...")
    t0 = time.time()
    ret_result = eval_retrieval_precision(
        model=model,
        tokenizer=tokenizer,
        anatomical_prompts_path=cfg.data.anatomical_prompts_path,
        anatomical_references_path=cfg.data.anatomical_references_path,
        bertscore_model=cfg.probes.bertscore_model,
        max_new_tokens=cfg.probes.ret_prec_max_new_tokens,
        device=device,
    )
    ret_elapsed = time.time() - t0

    total_elapsed = time.time() - eval_start

    # Calculate perplexity improvement if possible
    ppl_improvement_pct = None
    if len(state["perplexity_history"]) > 0:
        from lib.s2_dapt.probes.perplexity_probe import compute_ppl_improvement
        ppl_improvement_pct = compute_ppl_improvement(state["perplexity_history"][-1], ppl_result["perplexity"])

    # ── Assemble metrics record ───────────────────────────────────────────────
    metrics = {
        "eval_id"          : state["eval_count"],
        "tokens_processed" : state["tokens_processed"],
        "corpus_pass"      : state["tokens_processed"] / cfg.corpus.total_corpus_tokens,
        "timestamp"        : datetime.now(timezone.utc).isoformat(),
        "ppl_improvement_pct": ppl_improvement_pct,
        "metrics"          : {
            "perplexity"        : ppl_result["perplexity"],
            "avg_nll_nats"      : ppl_result["avg_nll_nats"],
            "qa_accuracy"       : qa_result["accuracy"],
            "qa_correct"        : qa_result["correct"],
            "qa_total"          : qa_result["total"],
            "term_coverage"     : term_result["coverage"],
            "retrieval_precision": ret_result["precision"],
        },
        "per_cluster_qa"   : qa_result.get("per_cluster_accuracy", {}),
        "per_category_term": term_result.get("per_category", {}),
        "probe_elapsed_sec": {
            "perplexity"  : round(ppl_elapsed, 1),
            "qa"          : round(qa_elapsed, 1),
            "terminology" : round(term_elapsed, 1),
            "retrieval"   : round(ret_elapsed, 1),
            "total"       : round(total_elapsed, 1),
        },
    }

    # ── Update rolling state histories ────────────────────────────────────────
    state["perplexity_history"].append(ppl_result["perplexity"])
    state["qa_acc_history"].append(qa_result["accuracy"])
    state["term_cov_history"].append(term_result["coverage"])
    state["ret_prec_history"].append(ret_result["precision"])

    # ── Write to JSONL ────────────────────────────────────────────────────────
    metrics_writer.write(metrics)

    # ── Log to Weights & Biases ───────────────────────────────────────────────
    if cfg.wandb.enabled:
        try:
            import wandb
            wandb.log({
                "eval/perplexity": float(ppl_result["perplexity"]),
                "eval/qa_accuracy": float(qa_result["accuracy"]),
                "eval/terminology_coverage": float(term_result["coverage"]),
                "eval/retrieval_precision": float(ret_result["precision"]),
                "eval/eval_count": int(state["eval_count"]),
                "eval/tokens_processed": int(state["tokens_processed"]),
            })
        except Exception as e:
            logger.warning(f"Failed to log evaluation metrics to wandb: {e}")

    logger.info(
        f"\n  Eval #{state['eval_count']} summary:\n"
        f"    PPL={ppl_result['perplexity']:.3f}  "
        f"QA={qa_result['accuracy']:.3f}  "
        f"Term={term_result['coverage']:.3f}  "
        f"Ret={ret_result['precision']:.3f}  "
        f"(elapsed: {total_elapsed:.0f}s)\n"
    )

    return metrics
