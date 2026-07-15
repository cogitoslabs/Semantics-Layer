import os
import random
import numpy as np
import torch
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from lib.utils import DAPTConfig
from lib.utils.logger import get_logger, MetricsWriter, save_json
from lib.utils.checkpoint import save_checkpoint, select_best_checkpoint, load_checkpoint
from lib.s3_dapt.evaluation.eval_runner import run_all_probes
from lib.s3_dapt.evaluation.gate_logic import (
    DAPTDecision,
    check_convergence_gates,
    log_gate_status,
)

logger = get_logger("dapt.training_helpers")

def setup_training_environment(cfg: DAPTConfig, device: torch.device):
    """Set seeds and configure Weights & Biases if enabled."""
    random.seed(cfg.misc.seed)
    np.random.seed(cfg.misc.seed)
    torch.manual_seed(cfg.misc.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.misc.seed)
    
    # Initialize Weights & Biases
    if cfg.wandb.enabled:
        try:
            import wandb
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

def verify_eval_files(cfg: DAPTConfig) -> None:
    """Verify that all files needed for probe evaluation exist on disk."""
    missing_files = []
    checks = []
    if cfg.probes.run_qa:
        checks.append(("Probe 2 - QA probe", cfg.data.qa_probe_path))
    if cfg.probes.run_perplexity:
        checks.append(("Probe 1 - Perplexity probe corpus", cfg.data.ppl_corpus_path))
    if cfg.probes.run_cloze:
        checks.append(("Probe 3 - Cloze probe set", cfg.data.cloze_set_path))
    if cfg.probes.run_concept:
        checks.append(("Probe 4 - Concept Probe prompts", cfg.data.concept_prompts_path))
        checks.append(("Probe 4 - Concept Probe references", cfg.data.concept_references_path))

    for name, path in checks:
        if not path.exists():
            missing_files.append(f"{name} file not found: {path}")

    if missing_files:
        error_msg = "Required evaluation files are missing:\n" + "\n".join(missing_files)
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)

def init_optimizer_scheduler(cfg: DAPTConfig, model: torch.nn.Module, train_dataloader_len: int):
    """Initialize optimizer and scheduler (Adafactor for CPU, AdamW for GPU)."""
    estimated_steps = train_dataloader_len * cfg.corpus.max_corpus_passes
    
    is_cpu = True
    for param in model.parameters():
        is_cpu = (param.device.type == "cpu")
        break
    
    # Filter for trainable parameters only (essential for PEFT memory savings)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    
    if is_cpu:
        from transformers import Adafactor
        logger.info("Using Adafactor optimizer for CPU training.")
        optimizer = Adafactor(
            trainable_params,
            scale_parameter=True,
            relative_step=True,
            warmup_init=True,
            lr=None, 
        )
        
        class DummyScheduler:
            def __init__(self, opt):
                self.optimizer = opt
            def step(self):
                pass
                
        scheduler = DummyScheduler(optimizer)
    else:
        logger.info("Using AdamW optimizer for GPU training.")
        optimizer = AdamW(
            trainable_params,
            lr=cfg.optimizer.learning_rate,
            weight_decay=cfg.optimizer.weight_decay,
        )
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=cfg.optimizer.warmup_steps,
            num_training_steps=estimated_steps,
        )
        
    return optimizer, scheduler

def handle_evaluation_cycle(
    model,
    tokenizer,
    cfg,
    state,
    optimizer,
    metrics_writer,
    device,
    last_checkpoint_path_ref,
) -> tuple:
    """Run probe evaluations, save progress checkpoint, check gates and log metrics."""
    state["eval_count"] += 1
    state["last_eval_at"] = state["tokens_processed"]

    # Save checkpoint
    last_checkpoint_path = save_checkpoint(
        model=model,
        optimizer=optimizer if cfg.model.save_optimizer_state else None,
        state=state,
        checkpoint_dir=cfg.model.checkpoint_dir,
        keep_last=cfg.model.checkpoint_keep_last,
    )
    state["last_checkpoint"] = str(last_checkpoint_path)
    last_checkpoint_path_ref[0] = last_checkpoint_path

    # Determine whether to run slow probes
    run_slow = False
    tokens_since_slow_eval = state["tokens_processed"] - state.get("last_slow_eval_at", 0)
    if tokens_since_slow_eval >= cfg.corpus.slow_eval_interval_tokens:
        run_slow = True
        state["last_slow_eval_at"] = state["tokens_processed"]

    # Evaluate probes
    metrics = run_all_probes(
        model=model,
        tokenizer=tokenizer,
        cfg=cfg,
        state=state,
        metrics_writer=metrics_writer,
        device=str(device),
        run_slow_probes=run_slow,
        use_bertscore=run_slow,
    )
    state["eval_history"].append(metrics)

    # Check convergence gates
    decision, gate_details = check_convergence_gates(
        state=state,
        qa_acc_threshold          = cfg.gates.qa_acc_threshold,
        ppl_improvement_threshold = cfg.gates.ppl_improvement_threshold,
        ppl_plateau_window        = cfg.gates.ppl_plateau_window,
        cloze_threshold           = cfg.gates.cloze_threshold,
        concept_threshold         = cfg.gates.concept_threshold,
        hard_stop_tokens          = cfg.corpus.hard_stop_tokens,
        total_corpus_tokens       = cfg.corpus.total_corpus_tokens,
        run_qa                    = cfg.probes.run_qa,
        run_perplexity            = cfg.probes.run_perplexity,
        run_cloze                 = cfg.probes.run_cloze,
        run_concept               = cfg.probes.run_concept,
    )

    log_gate_status(
        state=state,
        gate_details=gate_details,
        decision=decision,
        qa_threshold   = cfg.gates.qa_acc_threshold,
        ppl_threshold  = cfg.gates.ppl_improvement_threshold,
        ppl_window     = cfg.gates.ppl_plateau_window,
        cloze_threshold = cfg.gates.cloze_threshold,
        concept_threshold = cfg.gates.concept_threshold,
        total_corpus_tokens = cfg.corpus.total_corpus_tokens,
        run_qa                = cfg.probes.run_qa,
        run_perplexity        = cfg.probes.run_perplexity,
        run_cloze             = cfg.probes.run_cloze,
        run_concept           = cfg.probes.run_concept,
    )

    if cfg.wandb.enabled:
        try:
            import wandb
            wandb.log({
                "gate/qa_gate": int(gate_details["qa_gate"]),
                "gate/ppl_gate": int(gate_details["ppl_gate"]),
                "gate/secondary_gate": int(gate_details["secondary_gate"]),
                "gate/decision": decision.name,
            })
        except Exception as e:
            logger.warning(f"Error logging gate status to wandb: {e}")

    return decision, gate_details

def run_final_eval(model, tokenizer, cfg, state, metrics_writer, device):
    """Loads best checkpoint and performs high-fidelity BERTScore evaluation."""
    best_ckpt = select_best_checkpoint(
        eval_history              = state["eval_history"],
        checkpoint_dir            = cfg.model.checkpoint_dir,
        ppl_improvement_threshold = cfg.gates.ppl_improvement_threshold,
        manifest_path             = cfg.model.best_checkpoint_manifest,
        run_qa                    = cfg.probes.run_qa,
        run_perplexity            = cfg.probes.run_perplexity,
        run_cloze                 = cfg.probes.run_cloze,
        run_concept               = cfg.probes.run_concept,
    )
    
    if best_ckpt:
        logger.info(f"Loading best checkpoint for final SciBERT evaluation: {best_ckpt}")
        load_checkpoint(best_ckpt, model)
        state["eval_count"] += 1
        final_metrics = run_all_probes(
            model=model,
            tokenizer=tokenizer,
            cfg=cfg,
            state=state,
            metrics_writer=metrics_writer,
            device=str(device),
            run_slow_probes=True,
            use_bertscore=True,
        )
        manifest = {
            "best_eval_id": state["eval_history"][-1]["eval_id"] if state["eval_history"] else 0,
            "best_checkpoint_path": str(best_ckpt),
            "selected_from": "final_evaluation_run",
            "metrics": final_metrics["metrics"],
        }
        save_json(manifest, cfg.model.best_checkpoint_manifest)
    return best_ckpt
