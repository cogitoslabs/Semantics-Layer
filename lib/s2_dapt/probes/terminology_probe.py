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
            use_cache=True,
        )

    prompt_len = inputs["input_ids"].shape[1]
    completions = []
    for seq in outputs:
        generated_ids = seq[prompt_len:]
        text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        completions.append(text)

    return completions


def generate_topk_completions_batch(
    model,
    tokenizer,
    prompts: List[str],
    k: int,
    max_new_tokens: int,
    device: str = "cuda",
    batch_size: int = 16,
) -> List[List[str]]:
    if not prompts:
        return []

    # Sort prompts by length to minimize padding matrix overhead and save GPU VRAM (bucket by padding)
    indexed_prompts = sorted(enumerate(prompts), key=lambda x: len(x[1]))
    sorted_prompts = [p for _, p in indexed_prompts]
    orig_indices = [idx for idx, _ in indexed_prompts]

    sorted_completions = []
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else (tokenizer.eos_token_id or 0)

    orig_padding_side = getattr(tokenizer, "padding_side", "right")
    tokenizer.padding_side = "left"

    try:
        for start_idx in range(0, len(sorted_prompts), batch_size):
            batch_prompts = sorted_prompts[start_idx : start_idx + batch_size]

            # Tokenize each prompt individually to handle mock tokenizers in tests
            batch_inputs = []
            for prompt in batch_prompts:
                inputs = tokenizer(
                    prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=256,
                )
                if inputs["input_ids"].shape[1] == 0:
                    fallback_id = tokenizer.bos_token_id or tokenizer.eos_token_id or 0
                    inputs["input_ids"] = torch.tensor([[fallback_id]], dtype=torch.long)
                    if "attention_mask" in inputs:
                        inputs["attention_mask"] = torch.tensor([[1]], dtype=torch.long)
                batch_inputs.append(inputs)

            max_len = max(inputs["input_ids"].shape[1] for inputs in batch_inputs)

            padded_input_ids = []
            padded_attention_masks = []
            input_device = torch.device(device)

            for inputs in batch_inputs:
                input_ids = inputs["input_ids"]
                seq_len = input_ids.shape[1]
                pad_len = max_len - seq_len

                if pad_len > 0:
                    pad_tensor = torch.full((1, pad_len), pad_id, dtype=input_ids.dtype, device=input_ids.device)
                    new_input_ids = torch.cat([pad_tensor, input_ids], dim=1)

                    if "attention_mask" in inputs:
                        mask = inputs["attention_mask"]
                        mask_pad = torch.zeros((1, pad_len), dtype=mask.dtype, device=mask.device)
                        new_mask = torch.cat([mask_pad, mask], dim=1)
                    else:
                        mask_pad = torch.zeros((1, pad_len), dtype=torch.long, device=input_ids.device)
                        mask_one = torch.ones((1, seq_len), dtype=torch.long, device=input_ids.device)
                        new_mask = torch.cat([mask_pad, mask_one], dim=1)
                else:
                    new_input_ids = input_ids
                    if "attention_mask" in inputs:
                        new_mask = inputs["attention_mask"]
                    else:
                        new_mask = torch.ones_like(input_ids)

                padded_input_ids.append(new_input_ids)
                padded_attention_masks.append(new_mask)

            stacked_input_ids = torch.cat(padded_input_ids, dim=0).to(input_device)
            stacked_attention_masks = torch.cat(padded_attention_masks, dim=0).to(input_device)

            with torch.no_grad():
                outputs = model.generate(
                    input_ids=stacked_input_ids,
                    attention_mask=stacked_attention_masks,
                    max_new_tokens=max_new_tokens,
                    do_sample=(k > 1),
                    top_k=50 if k > 1 else None,
                    top_p=0.95 if k > 1 else None,
                    num_return_sequences=k,
                    pad_token_id=tokenizer.eos_token_id,
                    use_cache=True,
                )

            is_tensor = isinstance(outputs, torch.Tensor)

            for i in range(len(batch_prompts)):
                prompt_completions = []
                for seq_idx in range(k):
                    flat_idx = i * k + seq_idx
                    if is_tensor:
                        batch_idx = flat_idx if flat_idx < outputs.shape[0] else 0
                        seq = outputs[batch_idx]
                        generated_ids = seq[max_len:]
                    else:
                        try:
                            generated_ids = outputs[flat_idx, max_len:]
                        except Exception:
                            generated_ids = outputs
                    text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
                    prompt_completions.append(text)
                sorted_completions.append(prompt_completions)
    finally:
        tokenizer.padding_side = orig_padding_side

    # Restore original order
    all_completions = [None] * len(prompts)
    for orig_idx, completions in zip(orig_indices, sorted_completions):
        all_completions[orig_idx] = completions

    return all_completions


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
    generation_batch_size: int = 16,
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

    # Causal language models predict next tokens after the prompt.
    # To fill the blank, we must feed the prefix of the prompt up to the placeholder "___".
    eval_prompts = []
    for item in cloze_items:
        prompt = item["prompt"]
        eval_prompt = prompt.split("___")[0] if "___" in prompt else prompt
        if not eval_prompt.strip():
            # If the prefix is empty, fallback to tokenizer BOS/EOS or a space to avoid empty input
            eval_prompt = tokenizer.bos_token or tokenizer.eos_token or " "
        eval_prompts.append(eval_prompt)

    logger.info(f"Generating completions for {len(cloze_items)} cloze questions (greedy, batch_size={generation_batch_size})...")
    batch_completions = generate_topk_completions_batch(
        model=model,
        tokenizer=tokenizer,
        prompts=eval_prompts,
        k=top_k,
        max_new_tokens=max_new_tokens,
        device=device,
        batch_size=generation_batch_size,
    )

    failures = []
    for i, item in enumerate(cloze_items):
        target_term = item["target_term"]
        category    = item.get("category", "unknown")
        completions = batch_completions[i]

        hit = term_found_in_completions(target_term, completions)
        covered += int(hit)

        if not hit:
            missed_terms.append(target_term)
            failures.append({
                "prompt": item["prompt"],
                "target_term": target_term,
                "generated_completions": completions,
                "category": category
            })

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
        "failures"     : failures,
    }


def get_failed_terminology_samples(
    model,
    tokenizer,
    vocab_cloze_path: Path,
    top_k: int,
    max_new_tokens: int,
    device: str = "cuda",
    generation_batch_size: int = 16,
) -> List[Dict[str, Any]]:
    """
    Run Probe C and return detailed list of failed samples where target term was not covered.
    """
    with open(vocab_cloze_path, "r", encoding="utf-8") as f:
        cloze_items: List[Dict[str, Any]] = json.load(f)

    model.eval()
    eval_prompts = []
    for item in cloze_items:
        prompt = item["prompt"]
        eval_prompt = prompt.split("___")[0] if "___" in prompt else prompt
        if not eval_prompt.strip():
            eval_prompt = tokenizer.bos_token or tokenizer.eos_token or " "
        eval_prompts.append(eval_prompt)

    batch_completions = generate_topk_completions_batch(
        model=model,
        tokenizer=tokenizer,
        prompts=eval_prompts,
        k=top_k,
        max_new_tokens=max_new_tokens,
        device=device,
        batch_size=generation_batch_size,
    )

    failures = []
    for i, item in enumerate(cloze_items):
        target_term = item["target_term"]
        completions = batch_completions[i]
        hit = term_found_in_completions(target_term, completions)
        if not hit:
            failures.append({
                "prompt": item["prompt"],
                "target_term": target_term,
                "generated_completions": completions,
                "category": item.get("category", "unknown")
            })
    return failures

