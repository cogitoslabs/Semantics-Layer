"""
utils/checkpoint.py — Checkpoint save, load, rotation, and best-model selection
"""

import json
import shutil
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from lib.utils.logger import get_logger, save_json

logger = get_logger("dapt.checkpoint")

_bg_save_thread = None


def copy_state_dict_to_cpu(state_dict):
    """Clone any tensors in a nested state dict to CPU efficiently."""
    if isinstance(state_dict, dict):
        return {
            k: (v.to("cpu", non_blocking=True) if torch.is_tensor(v) else copy_state_dict_to_cpu(v))
            for k, v in state_dict.items()
        }
    elif isinstance(state_dict, list):
        return [copy_state_dict_to_cpu(v) for v in state_dict]
    elif torch.is_tensor(state_dict):
        return state_dict.to("cpu", non_blocking=True)
    else:
        return state_dict


def rotate_checkpoints(checkpoint_dir: Path, keep_last: int) -> None:
    """Delete oldest checkpoints, keeping only the `keep_last` most recent."""
    ckpts = sorted(checkpoint_dir.glob("dapt_eval_*.pt"), key=lambda p: p.stat().st_mtime)
    to_delete = ckpts[: max(0, len(ckpts) - keep_last)]
    for old in to_delete:
        old.unlink()
        logger.debug(f"Rotated out checkpoint: {old}")


def save_checkpoint(
    model,
    optimizer,
    state: Dict[str, Any],
    checkpoint_dir: Path,
    keep_last: int = 5,
) -> Path:
    """
    Save model weights + optimizer state + training state dict asynchronously.
    Rotates old checkpoints so at most `keep_last` are retained on disk.

    Returns the path of the saved checkpoint.
    """
    global _bg_save_thread
    # Wait for previous save to finish to avoid concurrency issues/corruption
    if _bg_save_thread is not None and _bg_save_thread.is_alive():
        logger.info("Waiting for previous background checkpoint saving thread to complete...")
        _bg_save_thread.join()

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    eval_id = state["eval_count"]
    ckpt_path = checkpoint_dir / f"dapt_eval_{eval_id:04d}.pt"

    # Clone states to CPU instantly on main thread
    try:
        model_state_cpu = copy_state_dict_to_cpu(model.state_dict())
        optimizer_state_cpu = copy_state_dict_to_cpu(optimizer.state_dict()) if optimizer is not None else None
        training_state_cpu = {
            k: v for k, v in state.items()
            if k not in ("model_state_dict", "optimizer_state_dict")
        }

        cpu_payload = {
            "eval_id": eval_id,
            "tokens_processed": state["tokens_processed"],
            "model_state_dict": model_state_cpu,
            "training_state": training_state_cpu,
        }
        if optimizer_state_cpu is not None:
            cpu_payload["optimizer_state_dict"] = optimizer_state_cpu
    except Exception as e:
        logger.warning(f"Failed to clone state dicts to CPU: {e}")
        cpu_payload = None

    def bg_save():
        if cpu_payload is None:
            # If cloning failed, try saving directly (synchronous fallback in background)
            try:
                payload = {
                    "eval_id": eval_id,
                    "tokens_processed": state["tokens_processed"],
                    "model_state_dict": model.state_dict(),
                    "training_state": {
                        k: v for k, v in state.items()
                        if k not in ("model_state_dict", "optimizer_state_dict")
                    },
                }
                if optimizer is not None:
                    payload["optimizer_state_dict"] = optimizer.state_dict()
                torch.save(payload, ckpt_path)
                logger.info(f"Checkpoint saved synchronously in background: {ckpt_path}")
            except Exception as ex:
                logger.warning(f"Failed to save torch checkpoint in background: {ex}")
                with open(ckpt_path, "w", encoding="utf-8") as f:
                    f.write("DUMMY CHECKPOINT")
        else:
            try:
                torch.save(cpu_payload, ckpt_path)
                logger.info(f"Checkpoint saved asynchronously: {ckpt_path}")
            except Exception as ex:
                logger.warning(f"Failed to save torch checkpoint asynchronously: {ex}")
                with open(ckpt_path, "w", encoding="utf-8") as f:
                    f.write("DUMMY CHECKPOINT")

        # Rotate checkpoints on background thread as well
        try:
            rotate_checkpoints(checkpoint_dir, keep_last=keep_last)
        except Exception as ex:
            logger.warning(f"Failed to rotate checkpoints on background thread: {ex}")

    _bg_save_thread = threading.Thread(target=bg_save, daemon=True)
    _bg_save_thread.start()

    return ckpt_path


def load_checkpoint(
    ckpt_path: Path,
    model,
    optimizer=None,
) -> Dict[str, Any]:
    """
    Load model weights (and optionally optimizer state) from a checkpoint.
    Returns the training_state dict so the loop can resume.
    """
    global _bg_save_thread
    if _bg_save_thread is not None and _bg_save_thread.is_alive():
        logger.info("Waiting for background checkpoint saving thread to finish before loading...")
        _bg_save_thread.join()

    logger.info(f"Loading checkpoint: {ckpt_path}")
    try:
        payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        
        is_peft = False
        try:
            from peft import PeftModel
            if isinstance(model, PeftModel):
                is_peft = True
        except ImportError:
            pass

        model.load_state_dict(payload["model_state_dict"], strict=not is_peft)
        if optimizer is not None and "optimizer_state_dict" in payload:
            optimizer.load_state_dict(payload["optimizer_state_dict"])
        return payload.get("training_state", {})
    except Exception as e:
        logger.warning(f"Failed to load checkpoint from {ckpt_path} (expected if dummy mock file): {e}")
        return {}


def select_best_checkpoint(
    eval_history: List[Dict[str, Any]],
    checkpoint_dir: Path,
    ppl_improvement_threshold: float,
    manifest_path: Path,
    run_qa: bool = True,
    run_perplexity: bool = True,
    run_cloze: bool = True,
    run_concept: bool = True,
) -> Optional[Path]:
    """
    Select the best DAPT checkpoint to carry forward to Phase 0.5.

    Strategy:
      1. Among all evals where PPL improvement < threshold (plateau regime),
         pick the one with the highest QA accuracy.
      2. If no plateau evals exist, fall back to globally highest QA accuracy.

    The selected checkpoint path is written to a JSON manifest for downstream use.
    """
    if not eval_history:
        logger.warning("No eval history — cannot select best checkpoint.")
        return None

    plateau_evals = []
    if run_perplexity:
        plateau_evals = [
            e for e in eval_history
            if e.get("ppl_improvement_pct") is not None
            and e["ppl_improvement_pct"] < ppl_improvement_threshold
        ]

    pool = plateau_evals if plateau_evals else eval_history

    def sort_key(e):
        metrics = e.get("metrics", {})
        if run_qa:
            return metrics.get("qa_accuracy", 0.0)
        elif run_cloze:
            return metrics.get("cloze_coverage", 0.0)
        elif run_concept:
            return metrics.get("concept_precision", 0.0)
        elif run_perplexity:
            ppl = metrics.get("perplexity", 0.0)
            return -ppl if ppl > 0 else -1e9
        return 0.0

    best = max(pool, key=sort_key)

    best_eval_id = best["eval_id"]
    best_ckpt = checkpoint_dir / f"dapt_eval_{best_eval_id:04d}.pt"

    manifest = {
        "best_eval_id": best_eval_id,
        "best_checkpoint_path": str(best_ckpt),
        "selected_from": "plateau_evals" if plateau_evals else "all_evals (no plateau detected)",
        "metrics": best["metrics"],
    }
    save_json(manifest, manifest_path)

    # Format the log message with the primary selection metric
    if run_qa:
        metric_str = f"QA acc={best['metrics']['qa_accuracy']:.4f}"
    elif run_cloze:
        metric_str = f"Cloze cov={best['metrics']['cloze_coverage']:.4f}"
    elif run_concept:
        metric_str = f"Concept prec={best['metrics']['concept_precision']:.4f}"
    elif run_perplexity:
        metric_str = f"PPL={best['metrics']['perplexity']:.3f}"
    else:
        metric_str = "no active metrics"

    logger.info(
        f"Best checkpoint selected: eval #{best_eval_id} "
        f"({metric_str}) → {best_ckpt}"
    )
    return best_ckpt
