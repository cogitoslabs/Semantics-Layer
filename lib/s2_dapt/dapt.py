import json
import math
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup

from lib.utils import DAPTConfig
from lib.s2_dapt.evaluation.eval_runner import run_all_probes
from lib.s2_dapt.evaluation.gate_logic import (
    DAPTDecision,
    check_convergence_gates,
    handle_hard_cap,
    log_gate_status,
)
from lib.utils.checkpoint import save_checkpoint, select_best_checkpoint
from lib.utils.logger import setup_logger, get_logger, MetricsWriter

logger = get_logger("dapt")


# ── Backward-compatible helper functions ──────────────────────────────────────

def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(file_path):
        return []
    with open(file_path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def evaluate_perplexity(
    model: Any,
    tokenizer: Any,
    dataset: List[Dict[str, Any]],
    block_size: int = 512
) -> float:
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    # Prevent warning by temporarily increasing model_max_length during evaluation
    orig_max_len = getattr(tokenizer, "model_max_length", None)
    tokenizer.model_max_length = 100_000_000

    try:
        with torch.no_grad():
            for item in dataset:
                text = item.get("text", "")
                if not text.strip():
                    continue

                tokens = tokenizer.encode(
                    text,
                    add_special_tokens=False
                )

                if tokenizer.eos_token_id is not None:
                    tokens.append(tokenizer.eos_token_id)

                for start in range(0, len(tokens), block_size):
                    chunk = tokens[start:start + block_size]

                    if len(chunk) < 2:
                        continue

                    input_ids = torch.tensor(
                        [chunk],
                        dtype=torch.long,
                        device=model.device
                    )

                    attention_mask = torch.ones_like(input_ids)

                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=input_ids
                    )

                    num_tokens = len(chunk) - 1

                    total_loss += outputs.loss.item() * num_tokens
                    total_tokens += num_tokens
    finally:
        if orig_max_len is not None:
            tokenizer.model_max_length = orig_max_len

    if total_tokens == 0:
        return float("inf")

    avg_nll = total_loss / total_tokens
    return math.exp(avg_nll)


def evaluate_qa_accuracy(model: Any, tokenizer: Any, probe_questions: List[Dict[str, Any]]) -> float:
    model.eval()
    correct = 0
    total = 0
    options = ["A", "B", "C", "D"]
    
    with torch.no_grad():
        for q in probe_questions:
            prompt = f"Question: {q['question']}\nAnswer:"
            inputs = tokenizer(prompt, return_tensors="pt")
            input_ids = inputs["input_ids"].to(model.device)
            attention_mask = inputs.get("attention_mask", torch.ones_like(input_ids)).to(model.device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            next_token_logits = outputs.logits[0, -1, :]
            next_token_probs = torch.softmax(next_token_logits, dim=-1)
            
            option_probs = {}
            for opt in options:
                opt_token_ids = tokenizer.encode(" " + opt, add_special_tokens=False)
                if len(opt_token_ids) > 0:
                    opt_token_id = opt_token_ids[-1]
                    option_probs[opt] = next_token_probs[opt_token_id].item()
                else:
                    option_probs[opt] = 0.0
            
            best_option = max(option_probs, key=option_probs.get)
            if best_option == q.get("answer"):
                correct += 1
            total += 1
            
    return (correct / total) * 100 if total > 0 else 0.0


def prepare_training_blocks(tokenizer: Any, train_docs: List[Dict[str, Any]], block_size: int = 512) -> List[List[int]]:
    train_chunks = []
    
    eos_id = tokenizer.eos_token_id
    if eos_id is None or not isinstance(eos_id, int):
        eos_id = 0
        
    orig_max_len = getattr(tokenizer, "model_max_length", None)
    tokenizer.model_max_length = 100_000_000

    try:
        for doc in train_docs:
            text = doc.get("text", "")
            if not text.strip():
                continue
            tokens = tokenizer.encode(text, add_special_tokens=False)
            train_chunks.extend(tokens)
            train_chunks.append(eos_id)
    finally:
        if orig_max_len is not None:
            tokenizer.model_max_length = orig_max_len
        
    tokenized_train_blocks = []
    for i in range(0, len(train_chunks) - block_size + 1, block_size):
        tokenized_train_blocks.append(train_chunks[i : i + block_size])
        
    if not tokenized_train_blocks and train_chunks:
        tokenized_train_blocks.append(train_chunks)
        
    return tokenized_train_blocks

# ── Model/Tokenizer loader ────────────────────────────────────────────────────

def load_model_and_tokenizer(cfg: DAPTConfig, device: torch.device):
    """Load model and tokenizer with GPU optimizations if applicable."""
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.base_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if device.type == "cuda":
        torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        attn_implementation = "sdpa"
    else:
        torch_dtype = torch.float32
        attn_implementation = "eager"

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model.base_model_name,
        dtype=torch_dtype,
        attn_implementation=attn_implementation
    )
    model.to(device)
    return model, tokenizer


# ── Main pipeline entry point ─────────────────────────────────────────────────

def run_dapt_pipeline(
    model_name: str,
    corpus_path: str,
    probe_qa_path: str,
    epochs: int = 3,
    lr: float = 5e-5,
    batch_size: int = 2,
    output_dir: str = "./outputs/dapt_model"
) -> None:
    # 1. Load config and override defaults with parameters
    cfg = DAPTConfig()
    if model_name:
        cfg.model.base_model_name = model_name
    if probe_qa_path:
        cfg.data.qa_probe_path = Path(probe_qa_path)
    if epochs:
        cfg.corpus.max_corpus_passes = epochs
    if lr:
        cfg.optimizer.learning_rate = lr
    if batch_size:
        cfg.optimizer.train_batch_size = batch_size
    if output_dir:
        cfg.storage.checkpoint_dir = Path(output_dir)

    # Validate config and build directories
    cfg.validate()
    cfg.ensure_dirs()

    # Set up logger
    setup_logger(
        name="dapt",
        log_dir=cfg.storage.log_dir,
        level=cfg.misc.log_level,
    )
    global logger
    logger = get_logger("dapt.pipeline")

    logger.info(cfg.summary())

    # Set up Weights & Biases if enabled
    if cfg.wandb.enabled:
        try:
            import wandb
            # API Key Login
            if cfg.wandb.api_key and cfg.wandb.api_key != "your_api_key_here":
                wandb.login(key=cfg.wandb.api_key)
            else:
                os.environ.pop("WANDB_API_KEY", None)
                logger.warning(
                    "WANDB_ENABLED is True, but no valid WANDB_API_KEY was found. "
                    "Setting W&B to offline mode to prevent interactive prompt blocking."
                )
                os.environ["WANDB_MODE"] = "offline"

            wandb.init(
                project=cfg.wandb.project,
                entity=cfg.wandb.entity,
                name=cfg.wandb.run_name,
                config={
                    "base_model": cfg.model.base_model_name,
                    "dtype": cfg.model.model_dtype,
                    "learning_rate": cfg.optimizer.learning_rate,
                    "batch_size": cfg.optimizer.train_batch_size,
                    "warmup_steps": cfg.optimizer.warmup_steps,
                    "max_seq_len": cfg.model.max_seq_len,
                    "total_corpus_tokens": cfg.corpus.total_corpus_tokens,
                    "eval_interval_tokens": cfg.corpus.eval_interval_tokens,
                    "qa_acc_threshold": cfg.gates.qa_acc_threshold,
                    "ppl_improvement_threshold": cfg.gates.ppl_improvement_threshold,
                }
            )
            logger.info(f"Initialized Weights & Biases run: {wandb.run.name if wandb.run else ''}")
        except ImportError:
            logger.warning("wandb is not installed. Disabling Weights & Biases logging.")
            cfg.wandb.enabled = False

    # Set seed
    random.seed(cfg.misc.seed)
    np.random.seed(cfg.misc.seed)
    torch.manual_seed(cfg.misc.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.misc.seed)

    # Device config
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load model & tokenizer
    model, tokenizer = load_model_and_tokenizer(cfg, device)

    # 2. Check that all required evaluation files exist
    missing_files = []
    for name, path in [
        ("QA probe", cfg.data.qa_probe_path),
        ("PPL corpus", cfg.data.ppl_corpus_path),
        ("Vocab cloze", cfg.data.vocab_cloze_path),
        ("Anatomical prompts", cfg.data.anatomical_prompts_path),
        ("Anatomical references", cfg.data.anatomical_references_path),
    ]:
        if not path.exists():
            missing_files.append(f"{name} file not found: {path}")

    if missing_files:
        error_msg = "Required evaluation files are missing:\n" + "\n".join(missing_files)
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)

    # 3. Load pretraining corpus and split 80/20 into train/validation splits
    logger.info(f"Loading pretraining corpus from {corpus_path}")
    corpus_docs = load_jsonl(corpus_path)
    if not corpus_docs:
        logger.warning(f"Pretraining corpus at '{corpus_path}' is empty or not found. Cannot proceed.")
        return

    val_size = max(2, int(len(corpus_docs) * 0.20))
    # Make sure we don't try to split more than exists
    if len(corpus_docs) <= val_size:
        val_size = max(1, len(corpus_docs) // 2)

    train_docs = corpus_docs[:-val_size]
    val_docs = corpus_docs[-val_size:]
    logger.info(f"Corpus split: {len(train_docs)} training docs, {len(val_docs)} validation docs.")

    # Write validation texts to ppl_corpus_path for perplexity evaluation
    val_text = "\n".join(doc.get("text", "") for doc in val_docs if doc.get("text", "").strip())
    with open(cfg.data.ppl_corpus_path, "w", encoding="utf-8") as f:
        f.write(val_text)
    logger.info(f"Written validation split text to {cfg.data.ppl_corpus_path} for perplexity probe.")

    # Tokenize and chunk training documents
    logger.info("Tokenizing and chunking training documents...")
    tokenized_train_blocks = prepare_training_blocks(tokenizer, train_docs, block_size=cfg.model.max_seq_len)
    logger.info(f"Created {len(tokenized_train_blocks)} training blocks of size {cfg.model.max_seq_len}.")

    if not tokenized_train_blocks:
        logger.warning("No training blocks were created. Check the training documents.")
        return

    # Build Dataloader
    input_ids_tensor = torch.tensor(tokenized_train_blocks, dtype=torch.long)
    dataset = TensorDataset(input_ids_tensor)

    def collate_fn(batch):
        input_ids_list = [b[0] for b in batch]
        max_len = max(len(ids) for ids in input_ids_list)
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        
        padded_ids = []
        attention_mask = []
        for ids in input_ids_list:
            pad_len = max_len - len(ids)
            padded_ids.append(torch.cat([ids, torch.full((pad_len,), pad_id, dtype=torch.long)]))
            attention_mask.append(torch.cat([torch.ones(len(ids), dtype=torch.long), torch.zeros(pad_len, dtype=torch.long)]))
            
        padded_ids = torch.stack(padded_ids)
        attention_mask = torch.stack(attention_mask)
        
        labels = padded_ids.clone()
        labels[attention_mask == 0] = -100
        
        return {
            "input_ids": padded_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

    train_dataloader = DataLoader(
        dataset,
        batch_size=cfg.optimizer.train_batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        pin_memory=(device.type == "cuda"),
        num_workers=0,
    )

    # Build Optimizer & Scheduler
    estimated_steps = len(train_dataloader) * cfg.corpus.max_corpus_passes
    optimizer = AdamW(
        model.parameters(),
        lr=cfg.optimizer.learning_rate,
        weight_decay=cfg.optimizer.weight_decay,
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=cfg.optimizer.warmup_steps,
        num_training_steps=estimated_steps,
    )

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

    metrics_writer = MetricsWriter(cfg.storage.metrics_log_file)
    last_checkpoint_path: Optional[Path] = None

    use_cuda = (device.type == "cuda")
    scaler = None
    if use_cuda:
        torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        if torch_dtype == torch.float16:
            if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
                scaler = torch.amp.GradScaler("cuda")
            else:
                scaler = torch.cuda.amp.GradScaler()
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
    metrics = run_all_probes(
        model=model,
        tokenizer=tokenizer,
        cfg=cfg,
        state=state,
        metrics_writer=metrics_writer,
        device=str(device),
    )
    state["eval_history"].append(metrics)

    # 5. Training loop
    model.train()
    logger.info("Starting pretraining loop...")
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

            optimizer.zero_grad()
            with autocast_ctx:
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                loss = outputs.loss

            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.optimizer.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.optimizer.max_grad_norm)
                optimizer.step()

            scheduler.step()

            # Increment steps completed
            state["steps_completed"] = state.get("steps_completed", 0) + 1

            # Log to WandB at step intervals
            if cfg.wandb.enabled and state["steps_completed"] % cfg.wandb.log_interval_steps == 0:
                try:
                    import wandb
                    current_lr = optimizer.param_groups[0]["lr"]
                    wandb.log({
                        "train/loss": float(loss.item()),
                        "train/learning_rate": float(current_lr),
                        "train/tokens_processed": int(state["tokens_processed"]),
                        "train/step": int(state["steps_completed"]),
                    })
                except Exception as e:
                    logger.warning(f"Error logging to wandb during training: {e}")

            # Increment tokens processed
            batch_tokens = count_tokens(batch)
            state["tokens_processed"] += batch_tokens

            # Check evaluation interval
            tokens_since_eval = state["tokens_processed"] - state["last_eval_at"]
            if tokens_since_eval < cfg.corpus.eval_interval_tokens:
                continue

            # Run evaluation
            state["eval_count"] += 1
            state["last_eval_at"] = state["tokens_processed"]

            # Save checkpoint
            last_checkpoint_path = save_checkpoint(
                model=model,
                optimizer=optimizer,
                state=state,
                checkpoint_dir=cfg.storage.checkpoint_dir,
                keep_last=cfg.storage.checkpoint_keep_last,
            )
            state["last_checkpoint"] = str(last_checkpoint_path)

            # Evaluate probes
            metrics = run_all_probes(
                model=model,
                tokenizer=tokenizer,
                cfg=cfg,
                state=state,
                metrics_writer=metrics_writer,
                device=str(device),
            )
            state["eval_history"].append(metrics)

            model.train()

            # Check convergence gates
            decision, gate_details = check_convergence_gates(
                state=state,
                qa_acc_threshold          = cfg.gates.qa_acc_threshold,
                ppl_improvement_threshold = cfg.gates.ppl_improvement_threshold,
                ppl_plateau_window        = cfg.gates.ppl_plateau_window,
                term_cov_threshold        = cfg.gates.term_cov_threshold,
                ret_prec_threshold        = cfg.gates.ret_prec_threshold,
                hard_stop_tokens          = cfg.corpus.hard_stop_tokens,
                total_corpus_tokens       = cfg.corpus.total_corpus_tokens,
            )

            log_gate_status(
                state=state,
                gate_details=gate_details,
                decision=decision,
                qa_threshold   = cfg.gates.qa_acc_threshold,
                ppl_threshold  = cfg.gates.ppl_improvement_threshold,
                ppl_window     = cfg.gates.ppl_plateau_window,
                term_threshold = cfg.gates.term_cov_threshold,
                ret_threshold  = cfg.gates.ret_prec_threshold,
                total_corpus_tokens = cfg.corpus.total_corpus_tokens,
            )

            if cfg.wandb.enabled:
                try:
                    import wandb
                    wandb.log({
                        "gate/qa_gate": int(gate_details["qa_gate_passed"]),
                        "gate/ppl_gate": int(gate_details["ppl_gate_passed"]),
                        "gate/secondary_gate": int(gate_details["secondary_gate_passed"]),
                        "gate/decision": decision.name,
                    })
                except Exception as e:
                    logger.warning(f"Error logging gate status to wandb: {e}")

            if decision == DAPTDecision.CONVERGED:
                logger.info("✅  DAPT CONVERGED. Selecting best checkpoint for Phase 0.5 hand-off.")
                state["convergence_met"] = True
                best_ckpt = select_best_checkpoint(
                    eval_history              = state["eval_history"],
                    checkpoint_dir            = cfg.storage.checkpoint_dir,
                    ppl_improvement_threshold = cfg.gates.ppl_improvement_threshold,
                    manifest_path             = cfg.storage.best_checkpoint_manifest,
                )
                
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
                _, risk_report = handle_hard_cap(
                    state=state,
                    gate_details=gate_details,
                    last_checkpoint_path  = last_checkpoint_path,
                    risk_report_path      = cfg.storage.risk_report_path,
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

        # End of epoch
        logger.info(f"Completed corpus pass {epoch + 1}. Total tokens: {state['tokens_processed']/1e9:.2f}B")

    # 5. Final check if not converged by end of passes
    logger.warning("Training loop exhausted without a convergence decision. Running final gate check.")
    if state["tokens_processed"] > state["last_eval_at"]:
        state["eval_count"] += 1
        state["last_eval_at"] = state["tokens_processed"]
        last_checkpoint_path = save_checkpoint(
            model=model,
            optimizer=optimizer,
            state=state,
            checkpoint_dir=cfg.storage.checkpoint_dir,
            keep_last=cfg.storage.checkpoint_keep_last,
        )
        state["last_checkpoint"] = str(last_checkpoint_path)
        metrics = run_all_probes(
            model=model,
            tokenizer=tokenizer,
            cfg=cfg,
            state=state,
            metrics_writer=metrics_writer,
            device=str(device),
        )
        state["eval_history"].append(metrics)

    decision, gate_details = check_convergence_gates(
        state=state,
        qa_acc_threshold          = cfg.gates.qa_acc_threshold,
        ppl_improvement_threshold = cfg.gates.ppl_improvement_threshold,
        ppl_plateau_window        = cfg.gates.ppl_plateau_window,
        term_cov_threshold        = cfg.gates.term_cov_threshold,
        ret_prec_threshold        = cfg.gates.ret_prec_threshold,
        hard_stop_tokens          = cfg.corpus.hard_stop_tokens,
        total_corpus_tokens       = cfg.corpus.total_corpus_tokens,
    )

    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    if decision == DAPTDecision.CONVERGED:
        best_ckpt = select_best_checkpoint(
            eval_history              = state["eval_history"],
            checkpoint_dir            = cfg.storage.checkpoint_dir,
            ppl_improvement_threshold = cfg.gates.ppl_improvement_threshold,
            manifest_path             = cfg.storage.best_checkpoint_manifest,
        )
        logger.info(f"DAPT training finished. Selected best checkpoint: {best_ckpt}")
    else:
        _, risk_report = handle_hard_cap(
            state=state,
            gate_details=gate_details,
            last_checkpoint_path  = last_checkpoint_path,
            risk_report_path      = cfg.storage.risk_report_path,
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
