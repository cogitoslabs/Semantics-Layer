from scipy._lib.array_api_compat.common import device
import os
import sys
import gc
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from lib.utils import DAPTConfig
from lib.utils.checkpoint import find_latest_checkpoint, load_checkpoint
from lib.s3_dapt.dataset import MemmapDataset
from lib.s3_dapt.model_utils import load_model_and_tokenizer
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
    import time
    start_time = time.time()
    try:
        run_dapt_pipeline_impl(
            cfg=cfg,
            resources=resources
        )
    finally:
        end_time = time.time()
        time_taken = end_time - start_time
        tokens_processed = 0
        if "state" in resources and "tokens_processed" in resources["state"]:
            tokens_processed = resources["state"]["tokens_processed"]
        
        print(f"Tokens processed: {tokens_processed}")
        print(f"Time taken for the function to run: {time_taken:.2f} seconds")
        logger.info(f"Tokens processed: {tokens_processed}")
        logger.info(f"Time taken for the function to run: {time_taken:.2f} seconds")

        if "mmapped_tokens" in resources:
            tok = resources["mmapped_tokens"]
            if hasattr(tok, "_mmap") and tok._mmap is not None:
                try:
                    tok._mmap.close()
                except Exception:
                    pass


def run_dapt_pipeline_impl(
    cfg: DAPTConfig,
    resources: Optional[Dict[str, Any]] = None
) -> None:

    device = init_logging_and_device(cfg)
    setup_training_environment(cfg, device)
    model, tokenizer = init_model_and_tokenizer(cfg, device)
    verify_eval_files(cfg)
    train_dataloader = load_dataloader(cfg, device, resources)
    optimizer, scheduler = init_optimizer_scheduler(cfg, model, len(train_dataloader))
    state = init_state(resources)   
    metrics_writer = MetricsWriter(cfg.logging.metrics_log_file)
    last_checkpoint_path_ref = [None]
    scaler, autocast_ctx = setup_mixed_precision(device)

    if cfg.model.restart_from_checkpoint:
        latest_ckpt = find_latest_checkpoint(cfg.model.checkpoint_dir)
        if latest_ckpt:
            logger.info(f"RESTART_TRAINING_FROM_CHECKPOINT is True. Resuming from checkpoint: {latest_ckpt}")
            restored_state = load_checkpoint(
                ckpt_path=latest_ckpt,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
            )
            if restored_state:
                state.update(restored_state)
                # Keep perplexity_history reference synchronized
                if "perplexity_history" in restored_state:
                    state["ppl_history"] = state["perplexity_history"]
                last_checkpoint_path_ref[0] = latest_ckpt
                logger.info(
                    f"Successfully restored checkpoint state: epoch={state.get('epoch', 0)}, "
                    f"epoch_step={state.get('epoch_step', 0)}, "
                    f"tokens_processed={state.get('tokens_processed', 0):,}, "
                    f"global_step={state.get('global_step', 0)}"
                )
        else:
            logger.warning(
                f"RESTART_TRAINING_FROM_CHECKPOINT is True, but no existing checkpoints were found in {cfg.model.checkpoint_dir}. "
                "Starting training from scratch on base model."
            )

    run_baseline_eval(model, tokenizer, cfg, state, metrics_writer, device)

    run_training_loop(
        model=model,
        tokenizer=tokenizer,
        cfg=cfg,
        state=state,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        autocast_ctx=autocast_ctx,
        device=device,
        train_dataloader=train_dataloader,
        metrics_writer=metrics_writer,
        last_checkpoint_path_ref=last_checkpoint_path_ref,
    )


def init_logging_and_device(cfg: DAPTConfig) -> torch.device:
    setup_logger(
        f"lib.s3_dapt.dapt.run_dapt_pipeline_impl",
        cfg.logging,
    )
    global logger
    logger = get_logger(f"lib.s3_dapt.dapt.run_dapt_pipeline_impl")

    # Load corpus tokens from the pre-tokenized training array
    bin_path = cfg.data.pretokenized_bin_path
    if bin_path and bin_path.exists():
        try:
            tokens_len = len(np.load(bin_path, mmap_mode='r'))
            cfg.corpus.total_corpus_tokens = tokens_len
            logger.info(f"Dynamically set total_corpus_tokens to {tokens_len:,} based on pretokenized binary size.")
        except Exception as e:
            logger.warning(f"Could not load pretokenized bin size dynamically: {e}")

    logger.info(cfg.summary())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    return device


def init_model_and_tokenizer(
    cfg: DAPTConfig,
    device: torch.device
) -> Tuple[torch.nn.Module, Any]:
    # Load model & tokenizer
    model, tokenizer = load_model_and_tokenizer(cfg, device)

    # Disable KV cache during training to prevent generating unused past_key_values
    if hasattr(model, "config"):
        model.config.use_cache = False

    # Fuse kernels for faster training; no numerical change
    if cfg.model.torch_compile and device.type == "cuda" and hasattr(torch, "compile"):
        logger.info("Applying torch.compile to model...")
        model = torch.compile(model)

    return model, tokenizer


def load_dataloader(
    cfg: DAPTConfig,
    device: torch.device,
    resources: Optional[Dict[str, Any]]
) -> DataLoader:
    logger.info(f"Loading memory-mapped training tokens from {cfg.data.pretokenized_bin_path}")
    if not cfg.data.pretokenized_bin_path.exists():
        raise FileNotFoundError(
            f"Pretokenized training tokens file not found: {cfg.data.pretokenized_bin_path}. "
            "Please run step 1.5 (pre-tokenization) first."
        )
    
    mmapped_tokens = np.load(cfg.data.pretokenized_bin_path, mmap_mode='r')
    if resources is not None:
        resources["mmapped_tokens"] = mmapped_tokens

    dataset = MemmapDataset(mmapped_tokens, block_size=cfg.model.max_seq_len)
    logger.info(f"Loaded {len(dataset):,} training blocks of size {cfg.model.max_seq_len}.")

    in_colab = "google.colab" in sys.modules
    num_cpus = os.cpu_count() or 1
    # Colab's Drive-backed mmap + forked workers causes contention; local runs benefit from workers
    num_workers = 0 if (device.type != "cuda" or in_colab) else max(1, num_cpus - 2)

    return DataLoader(
        dataset,
        batch_size=cfg.optimizer.train_batch_size,
        shuffle=True,
        pin_memory=(device.type == "cuda"),
        num_workers=num_workers,
        drop_last=True,
    )


def init_state(resources: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    perplexity_history: List[float] = []
    state = {
        "tokens_processed" : 0,
        "last_eval_at"     : 0,
        "last_slow_eval_at": 0,
        "eval_count"       : 0,
        "global_step"      : 0,
        "steps_completed"  : 0,
        "epoch"            : 0,
        "epoch_step"       : 0,
        "perplexity_history" : perplexity_history,
        "ppl_history"        : perplexity_history,
        "qa_acc_history"     : [],
        "cloze_cov_history"  : [],
        "concept_prec_history": [],
        "eval_history"       : [],
        "eval_timestamps"    : [],
        "tokens_history"     : [],
        "convergence_met"  : False,
        "last_checkpoint"  : None,
    }
    if resources is not None:
        resources["state"] = state
    return state


def setup_mixed_precision(device: torch.device) -> Tuple[Any, Any]:
    use_cuda = (device.type == "cuda")
    scaler = None
    if use_cuda:
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        if dtype == torch.float16:
            scaler = torch.amp.GradScaler("cuda") if hasattr(torch, "amp") else torch.cuda.amp.GradScaler()
        autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=dtype) if hasattr(torch, "amp") else torch.cuda.amp.autocast(dtype=dtype)
    else:
        class DummyCtx:
            def __enter__(self): pass
            def __exit__(self, exc_type, exc_val, exc_tb): pass
        autocast_ctx = DummyCtx()
    return scaler, autocast_ctx


def run_baseline_eval(
    model: torch.nn.Module,
    tokenizer: Any,
    cfg: DAPTConfig,
    state: Dict[str, Any],
    metrics_writer: MetricsWriter,
    device: torch.device,
) -> None:
    if state.get("eval_count", 0) > 0:
        logger.info(f"Skipping baseline evaluation (resumed checkpoint already has {state['eval_count']} evaluations logged).")
        return

    logger.info("Running baseline evaluation on the base model...")
    state["eval_count"] += 1
    state["last_eval_at"] = 0
    state["last_slow_eval_at"] = 0
    try:
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
        clear_gpu_cache()
    except (getattr(torch.cuda, "OutOfMemoryError", RuntimeError), RuntimeError) as e:
        if "out of memory" in str(e).lower():
            logger.warning("CUDA Out of Memory in baseline evaluation. Clearing cache and retrying...")
            e = None
            clear_gpu_cache()
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
        else:
            raise e



def clear_gpu_cache() -> None:
    """Clear Python garbage collector and PyTorch CUDA cache to recover memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_training_loop(
    model: torch.nn.Module,
    tokenizer: Any,
    cfg: DAPTConfig,
    state: Dict[str, Any],
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: Optional[Any],
    autocast_ctx: Any,
    device: torch.device,
    train_dataloader: DataLoader,
    metrics_writer: MetricsWriter,
    last_checkpoint_path_ref: List[Optional[Path]],
) -> None:
    model.train()
    logger.info("Starting pretraining loop...")
    optimizer.zero_grad()
    step = state.get("global_step", 0)
    start_epoch = state.get("epoch", 0)
    start_step_in_epoch = state.get("epoch_step", 0)
    decision = None
    gate_details = None

    for epoch in range(start_epoch, cfg.corpus.max_corpus_passes):
        state["epoch"] = epoch
        logger.info(f"\n{'#'*60}\n  Corpus pass {epoch + 1}/{cfg.corpus.max_corpus_passes}\n{'#'*60}")

        for batch_idx, batch in enumerate(train_dataloader):
            # Skip batches if resuming mid-epoch
            if epoch == start_epoch and batch_idx < start_step_in_epoch:
                continue

            state["epoch_step"] = batch_idx

            # Check hard cap
            if state["tokens_processed"] >= cfg.corpus.hard_stop_tokens:
                logger.warning("Hard cap token count reached inside epoch loop. Breaking.")
                break

            # Execute a single training step
            loss_val = train_step(
                batch=batch,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                autocast_ctx=autocast_ctx,
                device=device,
                cfg=cfg,
                step=step,
                state=state,
            )

            step += 1
            state["steps_completed"] = step
            state["global_step"] = step

            # Log to WandB at step intervals
            log_wandb_training(cfg, loss_val, optimizer, state)

            # Check evaluation interval
            tokens_since_eval = state["tokens_processed"] - state["last_eval_at"]
            if tokens_since_eval < cfg.corpus.eval_interval_tokens:
                continue

            decision, gate_details = run_evaluation_cycle(
                model=model,
                tokenizer=tokenizer,
                cfg=cfg,
                state=state,
                optimizer=optimizer,
                scheduler=scheduler,
                metrics_writer=metrics_writer,
                device=device,
                last_checkpoint_path_ref=last_checkpoint_path_ref,
            )

            if handle_decision_action(
                decision=decision,
                gate_details=gate_details,
                model=model,
                tokenizer=tokenizer,
                cfg=cfg,
                state=state,
                metrics_writer=metrics_writer,
                device=device,
                last_checkpoint_path_ref=last_checkpoint_path_ref,
            ):
                return

            model.train()

        # Reset start_step_in_epoch for subsequent epochs
        start_step_in_epoch = 0
        logger.info(f"Completed corpus pass {epoch + 1}. Total tokens: {state['tokens_processed']/1e3:.2f}K")

    # 5. Final check if not converged by end of passes
    handle_final_check(
        model=model,
        tokenizer=tokenizer,
        cfg=cfg,
        state=state,
        optimizer=optimizer,
        scheduler=scheduler,
        metrics_writer=metrics_writer,
        device=device,
        last_checkpoint_path_ref=last_checkpoint_path_ref,
        last_decision=decision,
        last_gate_details=gate_details,
    )


def train_step(
    batch: Dict[str, torch.Tensor],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: Optional[Any],
    autocast_ctx: Any,
    device: torch.device,
    cfg: DAPTConfig,
    step: int,
    state: Dict[str, Any],
) -> float:
    """
    Runs one training step. On CUDA OOM, frees the failed attempt's tensors,
    clears the allocator cache, and retries once with the batch split in half
    (processed as two accumulated sub-steps) instead of blindly resubmitting
    the same batch.
    """
    try:
        return run_train_step(
            batch=batch,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            autocast_ctx=autocast_ctx,
            device=device,
            cfg=cfg,
            step=step,
            state=state,
        )
    except (getattr(torch.cuda, "OutOfMemoryError", RuntimeError), RuntimeError) as e:
        if "out of memory" not in str(e).lower():
            raise

        logger.warning("CUDA OOM in training step. Freeing memory and retrying with a split batch...")

        # Drop any references to the failed attempt's tensors/graph so empty_cache() can reclaim them
        optimizer.zero_grad(set_to_none=True)
        gc.collect()
        torch.cuda.synchronize()
        clear_gpu_cache()

        batch_size = batch["input_ids"].shape[0]
        if batch_size < 2:
            logger.error("OOM on a batch of size 1; cannot split further.")
            raise

        logger.warning(f"Splitting batch of size {batch_size} into two halves and retrying.")

        half = batch_size // 2
        batch_a = {k: v[:half] for k, v in batch.items() if isinstance(v, torch.Tensor)}
        batch_b = {k: v[half:] for k, v in batch.items() if isinstance(v, torch.Tensor)}

        # Both halves are treated as sub-steps of the same accumulation step.
        # Gradients from A and B add together correctly. We scale the loss of each sub-step
        # by 0.5 to keep the total gradient mathematically identical to a single pass.
        # We only step the optimizer/scheduler on the second half (skip_optimizer_step=False).
        loss_a = run_train_step(
            batch=batch_a,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            autocast_ctx=autocast_ctx,
            device=device,
            cfg=cfg,
            step=step,
            state=state,
            skip_optimizer_step=True,
            loss_scale=0.5,
        )
        torch.cuda.synchronize()
        clear_gpu_cache()

        loss_b = run_train_step(
            batch=batch_b,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            autocast_ctx=autocast_ctx,
            device=device,
            cfg=cfg,
            step=step,
            state=state,
            skip_optimizer_step=False,
            loss_scale=0.5,
        )

        return loss_a + loss_b


def run_train_step(
    batch: Dict[str, torch.Tensor],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: Optional[Any],
    autocast_ctx: Any,
    device: torch.device,
    cfg: DAPTConfig,
    step: int,
    state: Dict[str, Any],
    skip_optimizer_step: bool = False,
    loss_scale: float = 1.0,
) -> float:
    input_ids = batch["input_ids"].to(device, non_blocking=True)
    labels = batch["labels"].to(device, non_blocking=True)
    attention_mask = batch.get("attention_mask", None)
    if attention_mask is not None:
        attention_mask = attention_mask.to(device, non_blocking=True)

    try:
        with autocast_ctx:
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            # Apply both gradient accumulation divisor and split-batch loss scaling factor
            loss = (outputs.loss / cfg.optimizer.gradient_accumulation_steps) * loss_scale

        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        batch_tokens = count_tokens(batch)
        state["tokens_processed"] += batch_tokens

        do_step = (not skip_optimizer_step) and (step + 1) % cfg.optimizer.gradient_accumulation_steps == 0
        if do_step:
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

        return float(loss.item() * cfg.optimizer.gradient_accumulation_steps)

    finally:
        # Explicitly drop references to prevent lingers on exceptions/OOMs
        if "input_ids" in locals():
            del input_ids
        if "labels" in locals():
            del labels
        if "attention_mask" in locals():
            del attention_mask
        if "outputs" in locals():
            del outputs
        if "loss" in locals():
            del loss


def count_tokens(batch: Dict[str, torch.Tensor]) -> int:
    if "attention_mask" in batch:
        return int(batch["attention_mask"].sum().item())
    return int(batch["input_ids"].numel())


def log_wandb_training(
    cfg: DAPTConfig,
    loss_val: float,
    optimizer: torch.optim.Optimizer,
    state: Dict[str, Any],
) -> None:
    if not cfg.wandb.enabled:
        return
    if state["steps_completed"] % cfg.wandb.log_interval_steps != 0:
        return
    try:
        import wandb
        current_lr = optimizer.param_groups[0].get("lr")
        if current_lr is None:
            current_lr = 0.0
        wandb.log({
            "train/loss": loss_val,
            "train/learning_rate": float(current_lr),
            "train/tokens_processed": int(state["tokens_processed"]),
            "train/step": int(state["steps_completed"]),
        })
    except Exception as e:
        logger.warning(f"Error logging to wandb during training: {e}")


def run_evaluation_cycle(
    model: torch.nn.Module,
    tokenizer: Any,
    cfg: DAPTConfig,
    state: Dict[str, Any],
    optimizer: torch.optim.Optimizer,
    metrics_writer: MetricsWriter,
    device: torch.device,
    last_checkpoint_path_ref: List[Optional[Path]],
    scheduler: Optional[Any] = None,
) -> Tuple[Optional[DAPTDecision], Optional[Dict[str, Any]]]:
    try:
        decision, gate_details = handle_evaluation_cycle(
            model=model,
            tokenizer=tokenizer,
            cfg=cfg,
            state=state,
            optimizer=optimizer,
            scheduler=scheduler,
            metrics_writer=metrics_writer,
            device=device,
            last_checkpoint_path_ref=last_checkpoint_path_ref,
        )
        clear_gpu_cache()
    except (getattr(torch.cuda, "OutOfMemoryError", RuntimeError), RuntimeError) as e:
        if "out of memory" in str(e).lower():
            logger.warning("CUDA Out of Memory in evaluation cycle. Clearing cache and retrying...")
            e = None
            clear_gpu_cache()
            decision, gate_details = handle_evaluation_cycle(
                model=model,
                tokenizer=tokenizer,
                cfg=cfg,
                state=state,
                optimizer=optimizer,
                scheduler=scheduler,
                metrics_writer=metrics_writer,
                device=device,
                last_checkpoint_path_ref=last_checkpoint_path_ref,
            )
        else:
            raise e
    return decision, gate_details


def handle_decision_action(
    decision: DAPTDecision,
    gate_details: Dict[str, Any],
    model: torch.nn.Module,
    tokenizer: Any,
    cfg: DAPTConfig,
    state: Dict[str, Any],
    metrics_writer: MetricsWriter,
    device: torch.device,
    last_checkpoint_path_ref: List[Optional[Path]],
) -> bool:
    """Handles converged or hard cap decisions. Returns True if pipeline should terminate."""
    output_dir = str(cfg.model.checkpoint_dir)
    if decision == DAPTDecision.CONVERGED:
        logger.info("✅  DAPT CONVERGED. Selecting best checkpoint for Phase 0.5 hand-off.")
        state["convergence_met"] = True
        run_final_eval(model, tokenizer, cfg, state, metrics_writer, device)
        
        os.makedirs(output_dir, exist_ok=True)
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        logger.info(f"Converged model saved to: {output_dir}")
        print_probe_history(state)
        finish_wandb(cfg)
        return True

    elif decision == DAPTDecision.HARD_CAP:
        handle_hard_cap(
            state=state,
            gate_details=gate_details,
            last_checkpoint_path  = last_checkpoint_path_ref[0],
            risk_report_path      = cfg.logging.risk_report_path,
            qa_acc_threshold      = cfg.gates.qa_acc_threshold,
            qa_low_threshold      = cfg.gates.qa_low_threshold,
            ppl_improvement_threshold = cfg.gates.ppl_improvement_threshold,
            cloze_threshold       = cfg.gates.cloze_threshold,
            concept_threshold     = cfg.gates.concept_threshold,
            total_corpus_tokens   = cfg.corpus.total_corpus_tokens,
            run_qa                = cfg.probes.run_qa,
            run_perplexity        = cfg.probes.run_perplexity,
            run_cloze             = cfg.probes.run_cloze,
            run_concept           = cfg.probes.run_concept,
        )
        
        os.makedirs(output_dir, exist_ok=True)
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        logger.warning(f"Model saved to output_dir after hard cap: {output_dir}. Non-convergence risk report generated.")
        print_probe_history(state)
        finish_wandb(cfg)
        return True

    return False


def handle_final_check(
    model: torch.nn.Module,
    tokenizer: Any,
    cfg: DAPTConfig,
    state: Dict[str, Any],
    optimizer: torch.optim.Optimizer,
    metrics_writer: MetricsWriter,
    device: torch.device,
    last_checkpoint_path_ref: List[Optional[Path]],
    last_decision: Optional[DAPTDecision],
    last_gate_details: Optional[Dict[str, Any]],
    scheduler: Optional[Any] = None,
) -> None:
    logger.warning("Training loop exhausted without a convergence decision. Running final gate check.")
    output_dir = str(cfg.model.checkpoint_dir)
    decision = last_decision
    gate_details = last_gate_details
    if state["tokens_processed"] > state["last_eval_at"]:
        decision, gate_details = handle_evaluation_cycle(
            model=model,
            tokenizer=tokenizer,
            cfg=cfg,
            state=state,
            optimizer=optimizer,
            scheduler=scheduler,
            metrics_writer=metrics_writer,
            device=device,
            last_checkpoint_path_ref=last_checkpoint_path_ref,
        )

    best_ckpt = run_final_eval(model, tokenizer, cfg, state, metrics_writer, device)

    # Save the reloaded best checkpoint to output_dir so output_dir always holds the best model
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

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
            cloze_threshold       = cfg.gates.cloze_threshold,
            concept_threshold     = cfg.gates.concept_threshold,
            total_corpus_tokens   = cfg.corpus.total_corpus_tokens,
            run_qa                = cfg.probes.run_qa,
            run_perplexity        = cfg.probes.run_perplexity,
            run_cloze             = cfg.probes.run_cloze,
            run_concept           = cfg.probes.run_concept,
        )
        logger.warning("DAPT completed without convergence. Non-convergence risk report generated.")

    print_probe_history(state)
    finish_wandb(cfg)


def print_probe_history(state: Dict[str, Any]) -> None:
    ppl_history = state.get("perplexity_history", [])
    qa_history = state.get("qa_acc_history", [])
    cloze_history = state.get("cloze_cov_history", [])
    concept_history = state.get("concept_prec_history", [])

    ppl_vals = ", ".join(f"{p:.3f}" for p in ppl_history) if ppl_history else "n/a"
    qa_vals = ", ".join(f"{q:.4f}" for q in qa_history) if qa_history else "n/a"
    cloze_vals = ", ".join(f"{c:.4f}" for c in cloze_history) if cloze_history else "n/a"
    concept_vals = ", ".join(f"{c:.4f}" for c in concept_history) if concept_history else "n/a"

    logger.info(
        f"\n"
        f"============================================================\n"
        f"  DAPT Probe History Summary\n"
        f"============================================================\n"
        f"  Perplexity probe - {ppl_vals}\n"
        f"  QA probe - {qa_vals}\n"
        f"  Cloze probe - {cloze_vals}\n"
        f"  Concept probe - {concept_vals}\n"
        f"============================================================\n"
    )


def finish_wandb(cfg: DAPTConfig) -> None:
    if cfg.wandb.enabled:
        try:
            import wandb
            wandb.finish()
        except Exception:
            pass
