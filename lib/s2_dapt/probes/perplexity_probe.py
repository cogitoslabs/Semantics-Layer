"""
probes/perplexity_probe.py — Probe B: Held-Out Perplexity
"""

import math
from pathlib import Path
from typing import Dict, Any, List, Optional, Iterator

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
    total = len(token_ids)
    stride = seq_len

    for start in range(0, total - seq_len, stride * batch_size):
        batch_inputs = []
        for b in range(batch_size):
            seg_start = start + b * stride
            seg_end   = seg_start + seq_len + 1
            if seg_end > total:
                break
            batch_inputs.append(token_ids[seg_start:seg_end])

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
    Load and tokenize the held-out perplexity corpus.
    """
    logger.info(f"Loading PPL corpus from: {ppl_corpus_path} (max {max_tokens/1e6:.1f}M tokens)")

    with open(ppl_corpus_path, "r", encoding="utf-8") as f:
        text = f.read()

    token_ids = tokenizer.encode(text, add_special_tokens=False)

    if len(token_ids) > max_tokens:
        logger.info(
            f"Truncating PPL corpus from {len(token_ids)/1e6:.1f}M to {max_tokens/1e6:.1f}M tokens"
        )
        token_ids = token_ids[:max_tokens]

    usable = (len(token_ids) // (seq_len + 1)) * (seq_len + 1)
    token_ids = token_ids[:usable]

    logger.info(f"PPL corpus ready: {len(token_ids)/1e6:.2f}M tokens, {usable // seq_len} sequences")
    return token_ids


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
    Run Probe B and return perplexity on the held-out corpus.
    """
    cache_key = str(ppl_corpus_path)
    if cache_key not in _token_cache:
        _token_cache[cache_key] = load_ppl_corpus(ppl_corpus_path, tokenizer, max_tokens, seq_len)

    token_ids = _token_cache[cache_key]

    model.eval()
    total_nll   = 0.0
    total_tokens = 0
    batch_count = 0

    with torch.no_grad():
        for batch in token_batch_iter(token_ids, seq_len=seq_len, batch_size=batch_size):
            input_ids = batch["input_ids"].to(device)
            labels    = batch["labels"].to(device)

            outputs = model(input_ids=input_ids)
            logits  = outputs.logits

            B, T, V = logits.shape
            loss = F.cross_entropy(
                logits.view(B * T, V),
                labels.view(B * T),
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

    logger.info(f"Probe B — Perplexity: {ppl:.4f}  (NLL: {avg_nll:.4f} nats, tokens: {total_tokens})")

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
