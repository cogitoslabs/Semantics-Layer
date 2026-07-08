import os
import gc
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from lib.utils import DAPTConfig
from lib.s3_dapt.dataset import MemmapDataset
from lib.s3_dapt.model_utils import load_model_and_tokenizer
from lib.s3_dapt.metrics_compat import evaluate_perplexity, evaluate_qa_accuracy
from lib.s3_dapt.evaluation.eval_runner import run_all_probes
from lib.s3_dapt.training_helpers import (
    setup_training_environment,
    verify_eval_files,
    init_optimizer_scheduler,
    handle_evaluation_cycle,
    run_final_eval,
)
from lib.s3_dapt.evaluation.gate_logic import DAPTDecision, handle_hard_cap
from lib.utils.logger import setup_logger, get_logger, MetricsWriter

logger = get_logger(__name__)


# ── Main pipeline entry point ─────────────────────────────────────────────────

def run_dapt_pipeline(cfg: DAPTConfig) -> None:
    resources = {}
    try:
        _run_dapt_pipeline_impl(
            cfg=cfg,
            resources=resources
        )
    finally:
        if "mmapped_tokens" in resources:
            tok = resources["mmapped_tokens"]
            if hasattr(tok, "_mmap") and tok._mmap is not None:
                try:
                    tok._mmap.close()
                except Exception:
                    pass


def _run_dapt_pipeline_impl(
    cfg: DAPTConfig,
    resources: Optional[Dict[str, Any]] = None
) -> None:

    output_dir = str(cfg.model.checkpoint_dir)

    # Set up logger
    import sys
    setup_logger(
        f"{__name__}.{sys._getframe().f_code.co_name}",
        cfg.logging,
    )
    global logger
    logger = get_logger(f"{__name__}.{sys._getframe().f_code.co_name}")

    logger.info(cfg.summary())

    # Device config
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Set up training environment (seeds, wandb init)
    setup_training_environment(cfg, device)

    # Load model & tokenizer
    model, tokenizer = load_model_and_tokenizer(cfg, device)

    # Check that all required evaluation files exist
    verify_eval_files(cfg)

    # 3. Load pre-tokenized training dataset using mmap
    logger.info(f"Loading memory-mapped training tokens from {cfg.data.pretokenized_bin_path}")
    if not cfg.data.pretokenized_bin_path.exists():
        raise FileNotFoundError(
            f"Pretokenized training tokens file not found: {cfg.data.pretokenized_bin_path}. "
            "Please run step 1.5 (pre-tokenization) first."
        )
    
    # Load using mmap_mode='r' to map file on disk
    mmapped_tokens = np.load(cfg.data.pretokenized_bin_path, mmap_mode='r')
    if resources is not None:
        resources["mmapped_tokens"] = mmapped_tokens

    dataset = MemmapDataset(mmapped_tokens, block_size=cfg.model.max_seq_len)
    logger.info(f"Loaded {len(dataset):,} training blocks of size {cfg.model.max_seq_len}.")

    num_workers = 4 if device.type == "cuda" else 0

    train_dataloader = DataLoader(
        dataset,
        batch_size=cfg.optimizer.train_batch_size,
        shuffle=True,
        pin_memory=(device.type == "cuda"),
        num_workers=num_workers,
        drop_last=True,
    )

    # Build Optimizer & Scheduler
    optimizer, scheduler = init_optimizer_scheduler(cfg, model, len(train_dataloader))

    # Initialize State
    state = {
        "tokens_processed" : 0,
        "last_eval_at"     : 0,
        "eval_count"       : 0,
        "perplexity_history" : [],
        "qa_acc_history"     : [],
        "term_cov_history"   : [],
        "ret_prec_history"   : [],
        "eval_history"       : [],
        "convergence_met"  : False,
        "last_checkpoint"  : None,
        "steps_completed"  : 0,
    }

    metrics_writer = MetricsWriter(cfg.logging.metrics_log_file)
    last_checkpoint_path_ref = [None]

    # Mixed precision / autocast context setup
    use_cuda = (device.type == "cuda")
    scaler = None
    if use_cuda:
        torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        if torch_dtype == torch.float16:
            scaler = torch.amp.GradScaler("cuda") if hasattr(torch, "amp") else torch.cuda.amp.GradScaler()
        autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=torch_dtype) if hasattr(torch, "amp") else torch.cuda.amp.autocast(dtype=torch_dtype)
    else:
        class DummyCtx:
            def __enter__(self): pass
            def __exit__(self, exc_type, exc_val, exc_tb): pass
        autocast_ctx = DummyCtx()

    def count_tokens(b) -> int:
        if "attention_mask" in b:
            return int(b["attention_mask"].sum().item())
        return int(b["input_ids"].numel())

    # 4. Run baseline evaluation on the unmodified base model
    logger.info("Running baseline evaluation on the base model...")
    state["eval_count"] += 1
    state["last_eval_at"] = 0
    state["last_slow_eval_at"] = 0
    metrics = run_all_probes(
        model=model,
        tokenizer=tokenizer,
        cfg=cfg,
        state=state,
        metrics_writer=metrics_writer,
        device=str(device),
        run_slow_probes=True,
        use_bertscore=True,
    )
    state["eval_history"].append(metrics)

    # Free up memory used by baseline evaluation (SciBERT / BERTScore, generation cache)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 5. Training loop
    model.train()
    logger.info("Starting pretraining loop...")
    optimizer.zero_grad()
    step = 0
    for epoch in range(cfg.corpus.max_corpus_passes):
        logger.info(f"\n{'#'*60}\n  Corpus pass {epoch + 1}/{cfg.corpus.max_corpus_passes}\n{'#'*60}")

        for batch in train_dataloader:
            # Check hard cap
            if state["tokens_processed"] >= cfg.corpus.hard_stop_tokens:
                logger.warning("Hard cap token count reached inside epoch loop. Breaking.")
                break

            # Forward & Backward pass
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            attention_mask = batch.get("attention_mask", None)
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)

            with autocast_ctx:
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                loss = outputs.loss / cfg.optimizer.gradient_accumulation_steps

            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            # Increment tokens processed
            batch_tokens = count_tokens(batch)
            state["tokens_processed"] += batch_tokens

            # Step optimizer & scheduler after accumulating gradients
            if (step + 1) % cfg.optimizer.gradient_accumulation_steps == 0:
                if scaler is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.optimizer.max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.optimizer.max_grad_norm)
                    optimizer.step()

                scheduler.step()
                optimizer.zero_grad()

            step += 1
            state["steps_completed"] = step

            # Log to WandB at step intervals
            if cfg.wandb.enabled and state["steps_completed"] % cfg.wandb.log_interval_steps == 0:
                try:
                    import wandb
                    current_lr = optimizer.param_groups[0].get("lr")
                    if current_lr is None:
                        current_lr = 0.0
                    wandb.log({
                        "train/loss": float(loss.item() * cfg.optimizer.gradient_accumulation_steps),
                        "train/learning_rate": float(current_lr),
                        "train/tokens_processed": int(state["tokens_processed"]),
                        "train/step": int(state["steps_completed"]),
                    })
                except Exception as e:
                    logger.warning(f"Error logging to wandb during training: {e}")

            # Check evaluation interval
            tokens_since_eval = state["tokens_processed"] - state["last_eval_at"]
            if tokens_since_eval < cfg.corpus.eval_interval_tokens:
                continue

            decision, gate_details = handle_evaluation_cycle(
                model=model,
                tokenizer=tokenizer,
                cfg=cfg,
                state=state,
                optimizer=optimizer,
                metrics_writer=metrics_writer,
                device=device,
                last_checkpoint_path_ref=last_checkpoint_path_ref,
            )

            if decision == DAPTDecision.CONVERGED:
                logger.info("✅  DAPT CONVERGED. Selecting best checkpoint for Phase 0.5 hand-off.")
                state["convergence_met"] = True
                run_final_eval(model, tokenizer, cfg, state, metrics_writer, device)
                
                os.makedirs(output_dir, exist_ok=True)
                model.save_pretrained(output_dir)
                tokenizer.save_pretrained(output_dir)
                logger.info(f"Converged model saved to: {output_dir}")
                if cfg.wandb.enabled:
                    try:
                        import wandb
                        wandb.finish()
                    except Exception:
                        pass
                return

            elif decision == DAPTDecision.HARD_CAP:
                handle_hard_cap(
                    state=state,
                    gate_details=gate_details,
                    last_checkpoint_path  = last_checkpoint_path_ref[0],
                    risk_report_path      = cfg.logging.risk_report_path,
                    qa_acc_threshold      = cfg.gates.qa_acc_threshold,
                    qa_low_threshold      = cfg.gates.qa_low_threshold,
                    ppl_improvement_threshold = cfg.gates.ppl_improvement_threshold,
                    term_cov_threshold        = cfg.gates.term_cov_threshold,
                    ret_prec_threshold        = cfg.gates.ret_prec_threshold,
                    total_corpus_tokens   = cfg.corpus.total_corpus_tokens,
                )
                
                os.makedirs(output_dir, exist_ok=True)
                model.save_pretrained(output_dir)
                tokenizer.save_pretrained(output_dir)
                logger.warning(f"Model saved to output_dir after hard cap: {output_dir}. Non-convergence risk report generated.")
                if cfg.wandb.enabled:
                    try:
                        import wandb
                        wandb.finish()
                    except Exception:
                        pass
                return

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            model.train()

        # End of epoch
        logger.info(f"Completed corpus pass {epoch + 1}. Total tokens: {state['tokens_processed']/1e9:.2f}B")

    # 5. Final check if not converged by end of passes
    logger.warning("Training loop exhausted without a convergence decision. Running final gate check.")
    if state["tokens_processed"] > state["last_eval_at"]:
        decision, gate_details = handle_evaluation_cycle(
            model=model,
            tokenizer=tokenizer,
            cfg=cfg,
            state=state,
            optimizer=optimizer,
            metrics_writer=metrics_writer,
            device=device,
            last_checkpoint_path_ref=last_checkpoint_path_ref,
        )

    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    best_ckpt = run_final_eval(model, tokenizer, cfg, state, metrics_writer, device)
    
    if decision == DAPTDecision.CONVERGED:
        logger.info(f"DAPT training finished. Selected best checkpoint: {best_ckpt}")
    else:
        handle_hard_cap(
            state=state,
            gate_details=gate_details,
            last_checkpoint_path  = last_checkpoint_path_ref[0],
            risk_report_path      = cfg.logging.risk_report_path,
            qa_acc_threshold      = cfg.gates.qa_acc_threshold,
            qa_low_threshold      = cfg.gates.qa_low_threshold,
            ppl_improvement_threshold = cfg.gates.ppl_improvement_threshold,
            term_cov_threshold        = cfg.gates.term_cov_threshold,
            ret_prec_threshold        = cfg.gates.ret_prec_threshold,
            total_corpus_tokens   = cfg.corpus.total_corpus_tokens,
        )
        logger.warning("DAPT completed without convergence. Non-convergence risk report generated.")

    if cfg.wandb.enabled:
        try:
            import wandb
            wandb.finish()
        except Exception:
            pass
