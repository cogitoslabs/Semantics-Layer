"""
evaluation/eval_runner.py — Orchestrate all four probes in a single evaluation run

This is what the training loop calls at each EVAL_INTERVAL_TOKENS.
It returns a structured metrics dict that is both logged to JSONL and
used by gate_logic.check_convergence_gates().
"""

import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from lib.utils import DAPTConfig
from lib.s3_dapt.probes.qa_probe         import eval_qa_accuracy, get_qa_probe_samples
from lib.s3_dapt.probes.perplexity_probe import eval_perplexity
from lib.s3_dapt.probes.cloze_probe import eval_cloze_coverage, get_cloze_probe_samples
from lib.s3_dapt.probes.concept_probe  import eval_concept_precision, clear_scorer_cache, get_concept_probe_samples
from lib.utils.logger import get_logger, MetricsWriter

logger = get_logger("dapt.eval_runner")


def run_all_probes(
    model,
    tokenizer,
    cfg: DAPTConfig,
    state: Dict[str, Any],
    metrics_writer: MetricsWriter,
    device: str = "cuda",
    run_slow_probes: bool = True,
    use_bertscore: bool = True,
) -> Dict[str, Any]:
    """
    Run probes in sequence (all four or just the fast ones), update state, write metrics to JSONL.

    Returns the full metrics dict for this evaluation checkpoint.
    """
    eval_start = time.time()
    logger.info(
        f"\n{'='*62}\n"
        f"  Starting Eval #{state['eval_count']} | "
        f"Tokens: {state['tokens_processed']/1e3:.1f}K | "
        f"Pass: {state['tokens_processed']/cfg.corpus.total_corpus_tokens:.3f}x | "
        f"Slow probes={run_slow_probes} (SciBERT={use_bertscore})\n"
        f"  Active probes: PPL={cfg.probes.run_perplexity}, QA={cfg.probes.run_qa}, "
        f"Cloze={cfg.probes.run_cloze}, Concept={cfg.probes.run_concept}\n"
        f"{'='*62}"
    )

    # ── Probe 1 - Perplexity probe ────────────────────────────────────────────
    if cfg.probes.run_perplexity:
        logger.info("  Running Probe 1 - Perplexity probe...")
        t0 = time.time()
        perplexity_max_seq_len = cfg.probes.perplexity_max_seq_len or cfg.model.max_seq_len
        perplexity_batch_size = cfg.probes.perplexity_batch_size or cfg.optimizer.eval_batch_size
        ppl_result = eval_perplexity(
            model=model,
            tokenizer=tokenizer,
            ppl_corpus_path=cfg.data.ppl_corpus_path,
            max_tokens=cfg.probes.perplexity_eval_tokens,
            seq_len=perplexity_max_seq_len,
            batch_size=perplexity_batch_size,
            device=device,
        )
        ppl_elapsed = time.time() - t0
    else:
        logger.info("  Skipping Probe 1 - Perplexity probe (disabled)...")
        ppl_result = {
            "perplexity": 0.0,
            "avg_nll_nats": 0.0,
        }
        ppl_elapsed = 0.0

    # ── Probe 2 - QA probe ────────────────────────────────────────────────────
    if cfg.probes.run_qa:
        logger.info("  Running Probe 2 - QA probe...")
        t0 = time.time()
        qa_batch_size = cfg.probes.qa_batch_size or cfg.optimizer.eval_batch_size
        qa_result = eval_qa_accuracy(
            model=model,
            tokenizer=tokenizer,
            qa_probe_path=cfg.data.qa_probe_path,
            device=device,
            batch_size=qa_batch_size,
            max_length=cfg.probes.qa_max_seq_len,
            eval_num=state["eval_count"],
        )
        qa_elapsed = time.time() - t0
    else:
        logger.info("  Skipping Probe 2 - QA probe (disabled)...")
        qa_result = {
            "accuracy": 0.0,
            "correct": 0,
            "total": 0,
            "per_cluster_accuracy": {},
        }
        qa_elapsed = 0.0

    # ── Probe 3 - Cloze probe ─────────────────────────────────────────────────
    if cfg.probes.run_cloze:
        if run_slow_probes:
            logger.info("  Running Probe 3 - Cloze probe...")
            t0 = time.time()
            term_result = eval_cloze_coverage(
                model=model,
                tokenizer=tokenizer,
                vocab_cloze_path=cfg.data.cloze_set_path,
                top_k=cfg.probes.cloze_top_k,
                max_new_tokens=cfg.probes.cloze_max_new_tokens,
                device=device,
                generation_batch_size=cfg.probes.cloze_gen_batch_size,
                max_length=cfg.probes.cloze_max_seq_len,
                eval_num=state["eval_count"],
            )
            term_elapsed = time.time() - t0
        else:
            logger.info("  Skipping Probe 3 - Cloze probe (carrying forward last metric)...")
            term_result = {
                "coverage": state["cloze_cov_history"][-1] if state.get("cloze_cov_history") else 0.0,
                "per_category": state["eval_history"][-1].get("per_category_cloze", {}) if state.get("eval_history") else {},
            }
            term_elapsed = 0.0
    else:
        logger.info("  Skipping Probe 3 - Cloze probe (disabled)...")
        term_result = {
            "coverage": 0.0,
            "per_category": {},
        }
        term_elapsed = 0.0

    # ── Probe 4 - Concept Probe ───────────────────────────────────────────────
    if cfg.probes.run_concept:
        if run_slow_probes:
            logger.info("  Running Probe 4 - Concept Probe...")
            t0 = time.time()
            ret_result = eval_concept_precision(
                model=model,
                tokenizer=tokenizer,
                retrieval_prompts_path=cfg.data.concept_prompts_path,
                retrieval_references_path=cfg.data.concept_references_path,
                bertscore_model=cfg.probes.bertscore_model,
                max_new_tokens=cfg.probes.concept_max_new_tokens,
                device=device,
                use_bertscore=use_bertscore,
                generation_batch_size=cfg.probes.concept_gen_batch_size,
                bertscore_batch_size=cfg.probes.concept_bertscore_batch_size,
                failure_threshold=cfg.gates.concept_threshold,
                max_length=cfg.probes.concept_max_seq_len,
                eval_num=state["eval_count"],
            )
            ret_elapsed = time.time() - t0
        else:
            logger.info("  Skipping Probe 4 - Concept Probe (carrying forward last metric)...")
            ret_result = {
                "precision": state["concept_prec_history"][-1] if state.get("concept_prec_history") else 0.0,
            }
            ret_elapsed = 0.0
    else:
        logger.info("  Skipping Probe 4 - Concept Probe (disabled)...")
        ret_result = {
            "precision": 0.0,
        }
        ret_elapsed = 0.0

    # ── Aggregate metrics ──────────────────────────────────────────────────────
    total_elapsed = time.time() - eval_start
    metrics = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "eval_count": state["eval_count"],
        "eval_id": state["eval_count"],
        "step": state.get("global_step", state.get("steps_completed", 0)),
        "tokens_processed": state["tokens_processed"],
        "epoch": state["tokens_processed"] / cfg.corpus.total_corpus_tokens,
        "perplexity": ppl_result["perplexity"],
        "avg_nll_nats": ppl_result["avg_nll_nats"],
        "qa_accuracy": qa_result["accuracy"],
        "qa_correct": qa_result["correct"],
        "qa_total": qa_result["total"],
        "per_cluster_qa": qa_result.get("per_cluster_accuracy", {}),
        "cloze_coverage": term_result["coverage"],
        "per_category_cloze": term_result.get("per_category", {}),
        "concept_precision": ret_result["precision"],
        "concept_mean_bertscore_f1": ret_result.get("mean_bertscore_f1", 0.0),
        "concept_min_bertscore_f1": ret_result.get("min_bertscore_f1", 0.0),
        "concept_max_bertscore_f1": ret_result.get("max_bertscore_f1", 0.0),
        "elapsed_seconds": total_elapsed,
        "ppl_elapsed": ppl_elapsed,
        "qa_elapsed": qa_elapsed,
        "term_elapsed": term_elapsed,
        "ret_elapsed": ret_elapsed,
    }

    # Update state histories
    state.setdefault("eval_history", []).append(metrics)
    state.setdefault("eval_timestamps", []).append(metrics["timestamp"])
    state.setdefault("tokens_history", []).append(state["tokens_processed"])

    if cfg.probes.run_perplexity:
        state.setdefault("perplexity_history", []).append(ppl_result["perplexity"])
        if "ppl_history" in state and isinstance(state["ppl_history"], list):
            if state["ppl_history"] is not state["perplexity_history"]:
                state["ppl_history"].append(ppl_result["perplexity"])
        else:
            state["ppl_history"] = state["perplexity_history"]
    if cfg.probes.run_qa:
        state.setdefault("qa_acc_history", []).append(qa_result["accuracy"])
    if cfg.probes.run_cloze and run_slow_probes:
        state.setdefault("cloze_cov_history", []).append(term_result["coverage"])
    if cfg.probes.run_concept and run_slow_probes:
        state.setdefault("concept_prec_history", []).append(ret_result["precision"])

    metrics_writer.write(metrics)

    logger.info(
        f"\n  Eval #{state['eval_count']} summary:\n"
        f"    PPL={ppl_result['perplexity']:.3f}  "
        f"QA={qa_result['accuracy']:.3f}  "
        f"Cloze={term_result['coverage']:.3f}  "
        f"Concept={ret_result['precision']:.3f}  "
        f"(elapsed: {total_elapsed:.0f}s)\n"
    )

    # ── Compile and save all evaluation probe traces for EVERY evaluation pass ──
    eval_traces_path = (
        getattr(cfg.logging, "eval_traces_file", None)
        or (cfg.logging.log_dir / "dapt_eval_traces.csv")
    )

    current_pass_traces = []
    if cfg.probes.run_qa:
        current_pass_traces.extend(qa_result.get("eval_traces", qa_result.get("samples", [])))
    if cfg.probes.run_cloze:
        current_pass_traces.extend(term_result.get("eval_traces", term_result.get("samples", [])))
    if cfg.probes.run_concept:
        current_pass_traces.extend(ret_result.get("eval_traces", ret_result.get("samples", [])))

    try:
        append_eval_traces_to_csv(eval_traces_path, current_pass_traces)
        logger.info(f"Detailed evaluation probe traces (Pass/Fail, Eval #{state['eval_count']}) saved to {eval_traces_path}")
    except Exception as e:
        logger.error(f"Failed to write evaluation probe traces CSV file: {e}")

    # Clear cached scorers to reclaim VRAM
    clear_scorer_cache()

    return metrics


CSV_FIELDNAMES = [
    "Eval #",
    "Eval Category",
    "Eval Seq #",
    "Eval",
    "Generated Answer by the model",
    "Matching Score",
    "Result",
]


def append_eval_traces_to_csv(eval_traces_path: Path, new_traces: List[Dict[str, Any]]) -> None:
    """
    Append evaluation trace records to a CSV file. If the file does not exist,
    the header row will be written first.
    """
    if not new_traces:
        return
    import csv
    import os
    os.makedirs(eval_traces_path.parent, exist_ok=True)
    file_exists = eval_traces_path.exists() and eval_traces_path.stat().st_size > 0
    with open(eval_traces_path, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES, extrasaction="ignore")
        if not file_exists:
            writer.writeheader()
        for record in new_traces:
            writer.writerow(record)


def run_inference_and_log_failures(cfg: DAPTConfig) -> None:
    """
    Load the final model and tokenizer from cfg.model.checkpoint_dir,
    run inference on active evaluation probes, log failed samples to console,
    and save ALL evaluation probe traces (Pass/Fail) to a CSV file.
    """
    eval_traces_path = (
        getattr(cfg.logging, "eval_traces_file", None)
        or (cfg.logging.log_dir / "dapt_eval_traces.csv")
    )
    import os
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from lib.s3_dapt.probes.qa_probe import get_qa_probe_traces
    from lib.s3_dapt.probes.cloze_probe import get_cloze_probe_traces
    from lib.s3_dapt.probes.concept_probe import get_concept_probe_traces

    model_dir = cfg.model.checkpoint_dir
    if not model_dir.exists():
        logger.error(f"Saved model directory not found at {model_dir}. Cannot run final inference.")
        return

    logger.info(f"Loading final model from {model_dir} for final inference and probe evaluation logging...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        attn_implementation = "sdpa"
    else:
        dtype = torch.float32
        attn_implementation = "eager"

    try:
        if getattr(cfg.model, "peft_dapt", False):
            from peft import PeftModel
            base_model = AutoModelForCausalLM.from_pretrained(
                cfg.model.base_model_name,
                dtype=dtype,
                attn_implementation=attn_implementation
            )
            model = PeftModel.from_pretrained(base_model, str(model_dir))
        else:
            model = AutoModelForCausalLM.from_pretrained(
                str(model_dir),
                dtype=dtype,
                attn_implementation=attn_implementation
            )
        model.to(device)
        tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    except Exception as e:
        logger.error(f"Failed to load final model or tokenizer from {model_dir}: {e}")
        return

    final_traces = []

    # 1. QA Probe Traces
    if cfg.probes.run_qa:
        logger.info("Running QA probe to collect evaluation traces...")
        try:
            qa_batch_size = cfg.probes.qa_batch_size or cfg.optimizer.eval_batch_size
            qa_traces = get_qa_probe_traces(
                model=model,
                tokenizer=tokenizer,
                qa_probe_path=cfg.data.qa_probe_path,
                device=str(device),
                batch_size=qa_batch_size,
                max_length=cfg.probes.qa_max_seq_len,
                eval_num="final",
            )
            final_traces.extend(qa_traces)
            qa_failures = [s for s in qa_traces if s.get("Result") == "Fail"]
            logger.info(f"QA Probe: {len(qa_traces)} total evaluated ({len(qa_failures)} failures found).")
            for idx, failure in enumerate(qa_failures):
                logger.warning(
                    f"QA Failure #{idx+1} [Cluster: {failure.get('cluster', 'unknown')}]:\n"
                    f"  Question: {failure.get('question', '')}\n"
                    f"  Expected: {failure.get('expected_text', '')} (index {failure.get('expected_idx', '')})\n"
                    f"  Predicted: {failure.get('predicted_text', '')} (index {failure.get('predicted_idx', '')})"
                )
        except Exception as e:
            logger.error(f"Error during QA probe evaluation logging: {e}")

    # 2. Cloze Probe Traces
    if cfg.probes.run_cloze:
        logger.info("Running Cloze probe to collect evaluation traces...")
        try:
            term_traces = get_cloze_probe_traces(
                model=model,
                tokenizer=tokenizer,
                vocab_cloze_path=cfg.data.cloze_set_path,
                top_k=cfg.probes.cloze_top_k,
                max_new_tokens=cfg.probes.cloze_max_new_tokens,
                device=str(device),
                generation_batch_size=cfg.probes.cloze_gen_batch_size,
                max_length=cfg.probes.cloze_max_seq_len,
                eval_num="final",
            )
            final_traces.extend(term_traces)
            term_failures = [s for s in term_traces if s.get("Result") == "Fail"]
            logger.info(f"Cloze Probe: {len(term_traces)} total evaluated ({len(term_failures)} failures found).")
            for idx, failure in enumerate(term_failures):
                logger.warning(
                    f"Cloze Failure #{idx+1} [Category: {failure.get('category', 'unknown')}]:\n"
                    f"  Prompt: {failure.get('prompt', '')}\n"
                    f"  Target Term: {failure.get('target_term', '')}\n"
                    f"  Completions: {failure.get('generated_completions', [])}"
                )
        except Exception as e:
            logger.error(f"Error during Cloze probe evaluation logging: {e}")

    # 3. Concept Probe Traces
    if cfg.probes.run_concept:
        logger.info("Running Concept probe to collect evaluation traces...")
        try:
            # We pass use_bertscore=True because it's the final high-fidelity evaluation
            ret_traces = get_concept_probe_traces(
                model=model,
                tokenizer=tokenizer,
                retrieval_prompts_path=cfg.data.concept_prompts_path,
                retrieval_references_path=cfg.data.concept_references_path,
                bertscore_model=cfg.probes.bertscore_model,
                max_new_tokens=cfg.probes.concept_max_new_tokens,
                device=str(device),
                use_bertscore=True,
                generation_batch_size=cfg.probes.concept_gen_batch_size,
                bertscore_batch_size=cfg.probes.concept_bertscore_batch_size,
                max_length=cfg.probes.concept_max_seq_len,
                failure_threshold=cfg.gates.concept_threshold,
                eval_num="final",
            )
            final_traces.extend(ret_traces)
            ret_failures = [s for s in ret_traces if s.get("Result") == "Fail"]
            logger.info(f"Concept Probe: {len(ret_traces)} total evaluated ({len(ret_failures)} failures found).")
            for idx, failure in enumerate(ret_failures):
                logger.warning(
                    f"Concept Failure #{idx+1} [F1 Score: {failure.get('score', 0.0):.4f}]:\n"
                    f"  Prompt: {failure.get('prompt', '')}\n"
                    f"  Reference: {failure.get('reference', '')}\n"
                    f"  Generated: {failure.get('generated', '')}"
                )
        except Exception as e:
            logger.error(f"Error during Concept probe evaluation logging: {e}")

    # Save to file
    try:
        append_eval_traces_to_csv(eval_traces_path, final_traces)
        logger.info(f"Detailed evaluation probe traces (Pass/Fail) saved to {eval_traces_path}")
    except Exception as e:
        logger.error(f"Failed to write evaluation probe traces CSV file: {e}")

