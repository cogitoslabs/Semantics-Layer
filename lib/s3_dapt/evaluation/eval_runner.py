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
from lib.s3_dapt.probes.qa_probe         import eval_qa_accuracy
from lib.s3_dapt.probes.perplexity_probe import eval_perplexity
from lib.s3_dapt.probes.cloze_probe import eval_cloze_coverage
from lib.s3_dapt.probes.concept_probe  import eval_concept_precision, clear_scorer_cache
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
                max_length=cfg.probes.concept_max_seq_len,
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

    total_elapsed = time.time() - eval_start

    # Calculate perplexity improvement if possible
    ppl_improvement_pct = None
    if len(state["perplexity_history"]) > 0:
        from lib.s3_dapt.probes.perplexity_probe import compute_ppl_improvement
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
            "cloze_coverage"    : term_result["coverage"],
            "concept_precision" : ret_result["precision"],
        },
        "per_cluster_qa"   : qa_result.get("per_cluster_accuracy", {}),
        "per_category_cloze": term_result.get("per_category", {}),
        "probe_elapsed_sec": {
            "perplexity"  : round(ppl_elapsed, 1),
            "qa"          : round(qa_elapsed, 1),
            "cloze"       : round(term_elapsed, 1),
            "concept"     : round(ret_elapsed, 1),
            "total"       : round(total_elapsed, 1),
        },
    }

    # ── Update rolling state histories ────────────────────────────────────────
    if cfg.probes.run_perplexity:
        state["perplexity_history"].append(ppl_result["perplexity"])
    if cfg.probes.run_qa:
        state["qa_acc_history"].append(qa_result["accuracy"])
    if cfg.probes.run_cloze:
        state["cloze_cov_history"].append(term_result["coverage"])
    if cfg.probes.run_concept:
        state["concept_prec_history"].append(ret_result["precision"])

    # ── Write to JSONL ────────────────────────────────────────────────────────
    metrics_writer.write(metrics)

    # ── Log to Weights & Biases ───────────────────────────────────────────────
    if cfg.wandb.enabled:
        try:
            import wandb
            log_data = {}
            if cfg.probes.run_perplexity:
                log_data["eval/perplexity"] = float(ppl_result["perplexity"])
            if cfg.probes.run_qa:
                log_data["eval/qa_accuracy"] = float(qa_result["accuracy"])
            if cfg.probes.run_cloze:
                log_data["eval/cloze_coverage"] = float(term_result["coverage"])
            if cfg.probes.run_concept:
                log_data["eval/concept_precision"] = float(ret_result["precision"])
            log_data.update({
                "eval/eval_count": int(state["eval_count"]),
                "eval/tokens_processed": int(state["tokens_processed"]),
            })
            wandb.log(log_data)
        except Exception as e:
            logger.warning(f"Failed to log evaluation metrics to wandb: {e}")

    logger.info(
        f"\n  Eval #{state['eval_count']} summary:\n"
        f"    PPL={ppl_result['perplexity']:.3f}  "
        f"QA={qa_result['accuracy']:.3f}  "
        f"Cloze={term_result['coverage']:.3f}  "
        f"Concept={ret_result['precision']:.3f}  "
        f"(elapsed: {total_elapsed:.0f}s)\n"
    )

    # ── Compile and save failed evaluations if using high-fidelity SciBERT (final evaluation) ──
    if use_bertscore:
        failed_runs = {}
        if cfg.probes.run_qa and "failures" in qa_result:
            failed_runs["qa"] = qa_result["failures"]
        if cfg.probes.run_cloze and "failures" in term_result:
            failed_runs["cloze"] = term_result["failures"]
        if cfg.probes.run_concept and "failures" in ret_result:
            failed_runs["concept"] = ret_result["failures"]

        failures_json_path = cfg.logging.log_dir / "failed_evals.json"
        try:
            import os
            import json
            os.makedirs(cfg.logging.log_dir, exist_ok=True)
            with open(failures_json_path, "w", encoding="utf-8") as f:
                json.dump(failed_runs, f, indent=2)
            logger.info(f"Failed evaluations automatically saved to {failures_json_path}")
        except Exception as e:
            logger.error(f"Failed to write failed evaluations JSON file: {e}")

    # Clear cached scorers to reclaim VRAM
    clear_scorer_cache()

    return metrics


def run_inference_and_log_failures(cfg: DAPTConfig) -> None:
    """
    Load the final model and tokenizer from cfg.model.checkpoint_dir,
    run inference on active evaluation probes, log all failed samples,
    and save them to a structured JSON file.
    """
    failures_json_path = cfg.logging.log_dir / "failed_evals.json"
    if failures_json_path.exists():
        logger.info(f"Failed evaluations already exist at {failures_json_path}. Skipping redundant final inference.")
        return
    import os
    import json
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from lib.s3_dapt.probes.qa_probe import get_failed_qa_samples
    from lib.s3_dapt.probes.cloze_probe import get_failed_cloze_samples
    from lib.s3_dapt.probes.concept_probe import get_failed_concept_samples

    model_dir = cfg.model.checkpoint_dir
    if not model_dir.exists():
        logger.error(f"Saved model directory not found at {model_dir}. Cannot run final inference.")
        return

    logger.info(f"Loading final model from {model_dir} for final inference and failure logging...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        attn_implementation = "sdpa"
    else:
        torch_dtype = torch.float32
        attn_implementation = "eager"

    try:
        if getattr(cfg.model, "peft_dapt", False):
            from peft import PeftModel
            base_model = AutoModelForCausalLM.from_pretrained(
                cfg.model.base_model_name,
                torch_dtype=torch_dtype,
                attn_implementation=attn_implementation
            )
            model = PeftModel.from_pretrained(base_model, str(model_dir))
        else:
            model = AutoModelForCausalLM.from_pretrained(
                str(model_dir),
                torch_dtype=torch_dtype,
                attn_implementation=attn_implementation
            )
        model.to(device)
        tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    except Exception as e:
        logger.error(f"Failed to load final model or tokenizer from {model_dir}: {e}")
        return

    failed_runs = {}

    # 1. QA Probe Failures
    if cfg.probes.run_qa:
        logger.info("Running QA probe to collect failed samples...")
        try:
            qa_batch_size = cfg.probes.qa_batch_size or cfg.optimizer.eval_batch_size
            qa_failures = get_failed_qa_samples(
                model=model,
                tokenizer=tokenizer,
                qa_probe_path=cfg.data.qa_probe_path,
                device=str(device),
                batch_size=qa_batch_size,
                max_length=cfg.probes.qa_max_seq_len,
            )
            failed_runs["qa"] = qa_failures
            logger.info(f"QA Probe: {len(qa_failures)} failures found.")
            for idx, failure in enumerate(qa_failures):
                logger.warning(
                    f"QA Failure #{idx+1} [Cluster: {failure['cluster']}]:\n"
                    f"  Question: {failure['question']}\n"
                    f"  Expected: {failure['expected_text']} (index {failure['expected_idx']})\n"
                    f"  Predicted: {failure['predicted_text']} (index {failure['predicted_idx']})"
                )
        except Exception as e:
            logger.error(f"Error during QA probe failure logging: {e}")

    # 2. Cloze Probe Failures
    if cfg.probes.run_cloze:
        logger.info("Running Cloze probe to collect failed samples...")
        try:
            term_failures = get_failed_cloze_samples(
                model=model,
                tokenizer=tokenizer,
                vocab_cloze_path=cfg.data.cloze_set_path,
                top_k=cfg.probes.cloze_top_k,
                max_new_tokens=cfg.probes.cloze_max_new_tokens,
                device=str(device),
                generation_batch_size=cfg.probes.cloze_gen_batch_size,
                max_length=cfg.probes.cloze_max_seq_len,
            )
            failed_runs["cloze"] = term_failures
            logger.info(f"Cloze Probe: {len(term_failures)} failures found.")
            for idx, failure in enumerate(term_failures):
                logger.warning(
                    f"Cloze Failure #{idx+1} [Category: {failure['category']}]:\n"
                    f"  Prompt: {failure['prompt']}\n"
                    f"  Target Term: {failure['target_term']}\n"
                    f"  Completions: {failure['generated_completions']}"
                )
        except Exception as e:
            logger.error(f"Error during Cloze probe failure logging: {e}")

    # 3. Concept Probe Failures
    if cfg.probes.run_concept:
        logger.info("Running Concept probe to collect failed samples...")
        try:
            # We pass use_bertscore=True because it's the final high-fidelity evaluation
            ret_failures = get_failed_concept_samples(
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
                failure_threshold=cfg.gates.concept_threshold
            )
            failed_runs["concept"] = ret_failures
            logger.info(f"Concept Probe: {len(ret_failures)} failures found.")
            for idx, failure in enumerate(ret_failures):
                logger.warning(
                    f"Concept Failure #{idx+1} [F1 Score: {failure['score']:.4f}]:\n"
                    f"  Prompt: {failure['prompt']}\n"
                    f"  Reference: {failure['reference']}\n"
                    f"  Generated: {failure['generated']}"
                )
        except Exception as e:
            logger.error(f"Error during Concept probe failure logging: {e}")

    # Save to file
    failures_json_path = cfg.logging.log_dir / "failed_evals.json"
    try:
        os.makedirs(cfg.logging.log_dir, exist_ok=True)
        with open(failures_json_path, "w", encoding="utf-8") as f:
            json.dump(failed_runs, f, indent=2)
        logger.info(f"Detailed failed evaluations saved to {failures_json_path}")
    except Exception as e:
        logger.error(f"Failed to write failed evaluations JSON file: {e}")

