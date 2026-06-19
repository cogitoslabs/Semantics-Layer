"""
probes/terminology_probe.py — Probe C: Neuroscience Terminology Coverage
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional

import torch

from lib.utils.logger import get_logger

logger = get_logger("dapt.probe.terminology")


def generate_topk_completions(
    model,
    tokenizer,
    prompt: str,
    k: int,
    max_new_tokens: int,
    device: str = "cuda",
) -> List[str]:
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=256,
    )
    if hasattr(inputs, "to"):
        inputs = inputs.to(device)
    else:
        inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}

    # Ensure input_ids is not empty to avoid a shape mismatch/reshape crash on sequence length 0
    if inputs["input_ids"].shape[1] == 0:
        fallback_id = tokenizer.bos_token_id or tokenizer.eos_token_id or 0
        inputs["input_ids"] = torch.tensor([[fallback_id]], dtype=torch.long, device=device)
        if "attention_mask" in inputs:
            inputs["attention_mask"] = torch.tensor([[1]], dtype=torch.long, device=device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=(k > 1),
            top_k=50 if k > 1 else None,
            top_p=0.95 if k > 1 else None,
            num_return_sequences=k,
            pad_token_id=tokenizer.eos_token_id,
        )

    prompt_len = inputs["input_ids"].shape[1]
    completions = []
    for seq in outputs:
        generated_ids = seq[prompt_len:]
        text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        completions.append(text)

    return completions


def term_found_in_completions(target_term: str, completions: List[str]) -> bool:
    target_lower = target_term.lower()
    return any(target_lower in c.lower() for c in completions)


def eval_terminology_coverage(
    model,
    tokenizer,
    vocab_cloze_path: Path,
    top_k: int,
    max_new_tokens: int,
    device: str = "cuda",
    max_samples: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Run Probe C and return overall and per-category terminology coverage.
    """
    with open(vocab_cloze_path, "r", encoding="utf-8") as f:
        cloze_items: List[Dict[str, Any]] = json.load(f)

    if max_samples is not None:
        cloze_items = cloze_items[:max_samples]

    model.eval()
    covered = 0
    category_stats: Dict[str, Dict[str, int]] = {}
    missed_terms: List[str] = []

    for i, item in enumerate(cloze_items):
        prompt      = item["prompt"]
        target_term = item["target_term"]
        category    = item.get("category", "unknown")

        # Causal language models predict next tokens after the prompt.
        # To fill the blank, we must feed the prefix of the prompt up to the placeholder "___".
        eval_prompt = prompt.split("___")[0] if "___" in prompt else prompt
        if not eval_prompt.strip():
            # If the prefix is empty, fallback to tokenizer BOS/EOS or a space to avoid empty input
            eval_prompt = tokenizer.bos_token or tokenizer.eos_token or " "

        completions = generate_topk_completions(
            model, tokenizer, eval_prompt,
            k=top_k, max_new_tokens=max_new_tokens, device=device
        )

        hit = term_found_in_completions(target_term, completions)
        covered += int(hit)

        if not hit:
            missed_terms.append(target_term)

        if category not in category_stats:
            category_stats[category] = {"covered": 0, "total": 0}
        category_stats[category]["covered"] += int(hit)
        category_stats[category]["total"] += 1

        if (i + 1) % 50 == 0:
            running_cov = covered / (i + 1)
            logger.debug(
                f"  Terminology probe: {i+1}/{len(cloze_items)} evaluated, "
                f"running coverage={running_cov:.3f}"
            )

    total    = len(cloze_items)
    coverage = covered / total if total > 0 else 0.0

    per_category = {
        cat: stats["covered"] / stats["total"]
        for cat, stats in category_stats.items()
        if stats["total"] > 0
    }

    logger.info(
        f"Probe C — Terminology Coverage: {coverage:.4f} ({covered}/{total})\n"
        f"  Per-category: {per_category}\n"
        f"  Sample missed terms: {missed_terms[:10]}"
    )

    return {
        "coverage"     : coverage,
        "covered"      : covered,
        "total"        : total,
        "per_category" : per_category,
        "missed_terms" : missed_terms,
    }
