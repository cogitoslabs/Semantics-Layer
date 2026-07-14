"""
probes/perplexity_probe.py — Probe 1 - Perplexity probe
"""

import contextlib
import math
from pathlib import Path
from typing import Dict, Any, List, Optional, Iterator

import os
import numpy as np
import torch
import torch.nn.functional as F

from lib.utils.logger import get_logger

logger = get_logger("dapt.probe.perplexity")


def token_batch_iter(
    token_ids: List[int],
    seq_len: int,
    batch_size: int,
) -> Iterator[Dict[str, torch.Tensor]]:
    """
    Yields non-overlapping sliding-window batches from a flat token list.
    Each batch has input_ids and labels (labels = input_ids shifted by 1).
    """
    chunk_size = seq_len + 1
    total_seqs = len(token_ids) // chunk_size

    for i in range(0, total_seqs, batch_size):
        batch_inputs = []
        for j in range(i, min(i + batch_size, total_seqs)):
            start_idx = j * chunk_size
            seq = token_ids[start_idx : start_idx + chunk_size]
            batch_inputs.append(seq)

        if not batch_inputs:
            break

        input_tensor = torch.tensor(
            [seq[:-1] for seq in batch_inputs], dtype=torch.long
        )
        label_tensor = torch.tensor(
            [seq[1:] for seq in batch_inputs], dtype=torch.long
        )
        yield {"input_ids": input_tensor, "labels": label_tensor}


def load_ppl_corpus(
    ppl_corpus_path: Path,
    tokenizer,
    max_tokens: int,
    seq_len: int,
) -> List[int]:
    """
    Load the pre-tokenized held-out perplexity corpus.
    """
    logger.info(f"Loading pre-tokenized PPL corpus from: {ppl_corpus_path} (max {max_tokens/1e6:.1f}M tokens)")

    if not os.path.exists(ppl_corpus_path):
        raise FileNotFoundError(f"Pre-tokenized perplexity corpus not found at: {ppl_corpus_path}")

    # Load from NumPy array directly
    token_ids = np.load(ppl_corpus_path).tolist()

    if len(token_ids) > max_tokens:
        logger.info(
            f"Truncating PPL corpus from {len(token_ids)/1e6:.1f}M to {max_tokens/1e6:.1f}M tokens"
        )
        token_ids = token_ids[:max_tokens]

    usable = (len(token_ids) // (seq_len + 1)) * (seq_len + 1)
    token_ids = token_ids[:usable]

    logger.info(f"PPL corpus ready: {len(token_ids)/1e3:.2f}K tokens, {usable // seq_len} sequences")
    return token_ids


_tensor_cache: Dict[str, torch.Tensor] = {}


def eval_perplexity(
    model,
    tokenizer,
    ppl_corpus_path: Path,
    max_tokens: int,
    seq_len: int,
    batch_size: int,
    device: str = "cuda",
    _token_cache: Dict[str, List[int]] = {},
) -> Dict[str, Any]:
    """
    Run Probe 1 - Perplexity probe and return perplexity on the held-out corpus.
    """
    cache_key = str(ppl_corpus_path)
    if cache_key not in _token_cache:
        _token_cache[cache_key] = load_ppl_corpus(ppl_corpus_path, tokenizer, max_tokens, seq_len)

    token_ids = _token_cache[cache_key]

    # Build a 2D CPU tensor once per corpus, then copy to target device and cache it
    cache_key_dev = f"{cache_key}_{device}"
    if cache_key_dev not in _tensor_cache:
        if cache_key not in _tensor_cache:
            n_seqs = len(token_ids) // (seq_len + 1)
            usable = n_seqs * (seq_len + 1)
            _tensor_cache[cache_key] = torch.tensor(
                token_ids[:usable], dtype=torch.long
            ).view(n_seqs, seq_len + 1)
        _tensor_cache[cache_key_dev] = _tensor_cache[cache_key].to(device)
    seqs_2d = _tensor_cache[cache_key_dev]  # shape: (N, seq_len+1)

    model.eval()
    total_nll   = 0.0
    total_tokens = 0
    batch_count = 0

    # Detect model parameter dtype dynamically
    is_cuda = "cuda" in str(device)
    try:
        model_dtype = next(iter(model.parameters())).dtype
    except (StopIteration, TypeError, AttributeError):
        model_dtype = torch.float32

    autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=model_dtype) if is_cuda and hasattr(torch, "amp") else contextlib.nullcontext()

    n_seqs = seqs_2d.shape[0]
    with torch.inference_mode():
        with autocast_ctx:
            for start in range(0, n_seqs - batch_size + 1, batch_size):
                chunk = seqs_2d[start : start + batch_size]  # (B, seq_len+1)
                input_ids = chunk[:, :-1]
                labels    = chunk[:, 1:]

                outputs = model(input_ids=input_ids)
                logits  = outputs.logits

                B, T, V = logits.shape
                loss = F.cross_entropy(
                    logits.reshape(B * T, V),
                    labels.reshape(B * T),
                    reduction="sum",
                )
                total_nll    += loss.item()
                total_tokens += B * T
                batch_count  += 1

                if batch_count % 50 == 0:
                    running_ppl = math.exp(total_nll / total_tokens)
                    logger.debug(f"  PPL probe: {total_tokens/1e6:.1f}M tokens processed, running PPL={running_ppl:.2f}")

    avg_nll = total_nll / total_tokens if total_tokens > 0 else float("inf")
    ppl     = math.exp(avg_nll)

    logger.info(f"Probe 1 - Perplexity probe: {ppl:.4f}  (NLL: {avg_nll:.4f} nats, tokens: {total_tokens})")

    return {
        "perplexity"   : ppl,
        "avg_nll_nats" : avg_nll,
        "total_tokens" : total_tokens,
    }


def compute_ppl_improvement(ppl_prev: float, ppl_curr: float) -> float:
    if ppl_prev <= 0:
        return 0.0
    return (ppl_prev - ppl_curr) / ppl_prev * 100.0


def check_ppl_plateau(
    ppl_history: List[float],
    improvement_threshold: float,
    window: int,
) -> tuple:
    if len(ppl_history) < window + 1:
        return False, []

    recent = ppl_history[-(window + 1):]
    improvements = [
        compute_ppl_improvement(recent[i], recent[i + 1])
        for i in range(window)
    ]
    plateau = all(imp < improvement_threshold for imp in improvements)
    return plateau, improvements
