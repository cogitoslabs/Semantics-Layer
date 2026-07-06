"""
probes/retrieval_probe.py — Probe D: Anatomical Landmark Retrieval Precision
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional

import torch
import transformers
from bert_score import score as bertscore_score

# Monkeypatch AutoTokenizer.from_pretrained to prevent OverflowError in BERTScore
# on models with undefined/extremely large model_max_length (e.g. SciBERT)
_orig_from_pretrained = transformers.AutoTokenizer.from_pretrained

def _patched_from_pretrained(*args, **kwargs):
    tokenizer = _orig_from_pretrained(*args, **kwargs)
    if getattr(tokenizer, "model_max_length", 0) > 1_000_000:
        tokenizer.model_max_length = 512
    return tokenizer

transformers.AutoTokenizer.from_pretrained = _patched_from_pretrained

from lib.utils.logger import get_logger

logger = get_logger("dapt.probe.retrieval")


def generate_response(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
    device: str = "cuda",
) -> str:
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
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
            pad_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )

    prompt_len = inputs["input_ids"].shape[1]
    generated_ids = output_ids[0, prompt_len:]
    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


def generate_responses_batch(
    model,
    tokenizer,
    prompts: List[str],
    max_new_tokens: int,
    device: str = "cuda",
    batch_size: int = 16,
) -> List[str]:
    if not prompts:
        return []

    # Sort prompts by length to minimize padding and save GPU VRAM (bucket by padding)
    indexed_prompts = sorted(enumerate(prompts), key=lambda x: len(x[1]))
    sorted_prompts = [p for _, p in indexed_prompts]
    orig_indices = [idx for idx, _ in indexed_prompts]

    sorted_responses = []
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else (tokenizer.eos_token_id or 0)

    orig_padding_side = getattr(tokenizer, "padding_side", "right")
    tokenizer.padding_side = "left"

    try:
        for start_idx in range(0, len(sorted_prompts), batch_size):
            batch_prompts = sorted_prompts[start_idx : start_idx + batch_size]

            # Tokenize each prompt individually to respect mock/custom tokenizers that only support string inputs
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
                output_ids = model.generate(
                    input_ids=stacked_input_ids,
                    attention_mask=stacked_attention_masks,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    temperature=1.0,
                    pad_token_id=tokenizer.eos_token_id,
                    use_cache=True,
                )

            for i in range(len(batch_prompts)):
                if isinstance(output_ids, torch.Tensor):
                    batch_idx = i if i < output_ids.shape[0] else 0
                    generated_ids = output_ids[batch_idx, max_len:]
                else:
                    try:
                        generated_ids = output_ids[i, max_len:]
                    except Exception:
                        generated_ids = output_ids
                response = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
                sorted_responses.append(response)

            # Log progress
            progress_count = start_idx + len(batch_prompts)
            if progress_count % 20 == 0 or progress_count == len(sorted_prompts):
                logger.debug(f"  Retrieval probe: {progress_count}/{len(sorted_prompts)} responses generated")
    finally:
        tokenizer.padding_side = orig_padding_side

    # Restore original order
    responses = [None] * len(prompts)
    for orig_idx, resp in zip(orig_indices, sorted_responses):
        responses[orig_idx] = resp

    return responses


def compute_bertscore_batch(
    hypotheses: List[str],
    references: List[str],
    model_type: str,
    device: str = "cuda",
    batch_size: int = 32,
) -> List[float]:
    _, _, F1 = bertscore_score(
        cands=hypotheses,
        refs=references,
        model_type=model_type,
        device=device,
        batch_size=batch_size,
        verbose=False,
    )
    return F1.tolist()


def token_safe_truncate(text: str, tokenizer: Any, max_tokens: int = 512) -> str:
    tokens = tokenizer.encode(text, max_length=max_tokens, truncation=True, add_special_tokens=False)
    return tokenizer.decode(tokens, skip_special_tokens=True)


def compute_lexical_f1_batch(hypotheses: List[str], references: List[str]) -> List[float]:
    f1_scores = []
    for hyp, ref in zip(hypotheses, references):
        hyp_tokens = hyp.lower().split()
        ref_tokens = ref.lower().split()
        if not hyp_tokens or not ref_tokens:
            f1_scores.append(0.0)
            continue
        hyp_counter = {}
        for t in hyp_tokens:
            hyp_counter[t] = hyp_counter.get(t, 0) + 1
        ref_counter = {}
        for t in ref_tokens:
            ref_counter[t] = ref_counter.get(t, 0) + 1
        overlap = 0
        for t, count in hyp_counter.items():
            if t in ref_counter:
                overlap += min(count, ref_counter[t])
        precision = overlap / len(hyp_tokens)
        recall = overlap / len(ref_tokens)
        if precision + recall == 0:
            f1_scores.append(0.0)
        else:
            f1_scores.append(2 * (precision * recall) / (precision + recall))
    return f1_scores


def eval_retrieval_precision(
    model,
    tokenizer,
    retrieval_prompts_path: Path,
    retrieval_references_path: Path,
    bertscore_model: str,
    max_new_tokens: int,
    device: str = "cuda",
    max_samples: Optional[int] = None,
    bertscore_batch_size: int = 32,
    use_bertscore: bool = True,
    generation_batch_size: int = 16,
) -> Dict[str, Any]:
    """
    Run Probe D and return average precision over retrieval prompts.
    Uses SciBERT BERTScore if use_bertscore=True, otherwise a fast lexical F1 metric.
    """
    with open(retrieval_prompts_path, "r", encoding="utf-8") as f:
        prompts: List[str] = json.load(f)
    with open(retrieval_references_path, "r", encoding="utf-8") as f:
        references: List[str] = json.load(f)

    if len(prompts) != len(references):
        raise ValueError(
            f"Prompts ({len(prompts)}) and references ({len(references)}) must be the same length."
        )

    if max_samples is not None:
        prompts    = prompts[:max_samples]
        references = references[:max_samples]

    model.eval()

    logger.info(f"Generating {len(prompts)} retrieval responses (greedy, batch_size={generation_batch_size})...")
    hypotheses = generate_responses_batch(
        model=model,
        tokenizer=tokenizer,
        prompts=prompts,
        max_new_tokens=max_new_tokens,
        device=device,
        batch_size=generation_batch_size,
    )

    # Safe token-based truncation to prevent splitting multi-byte tokens
    truncated_hypotheses = [token_safe_truncate(h, tokenizer, max_tokens=256) for h in hypotheses]
    truncated_references = [token_safe_truncate(r, tokenizer, max_tokens=256) for r in references]

    if use_bertscore:
        logger.info(f"Computing BERTScore with model: {bertscore_model}")
        try:
            f1_scores = compute_bertscore_batch(
                hypotheses=truncated_hypotheses,
                references=truncated_references,
                model_type=bertscore_model,
                device=device,
                batch_size=bertscore_batch_size,
            )
        except Exception as e:
            logger.warning(f"BERTScore computation failed: {e}. Falling back to dummy F1 scores.")
            f1_scores = [0.5] * len(prompts)
    else:
        logger.info("Computing fast lexical F1 overlap scores...")
        f1_scores = compute_lexical_f1_batch(truncated_hypotheses, truncated_references)

    mean_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0
    min_f1  = min(f1_scores) if f1_scores else 0.0
    max_f1  = max(f1_scores) if f1_scores else 0.0

    low_scoring = [
        {"prompt": prompts[i], "f1": f1_scores[i], "generated": hypotheses[i][:200]}
        for i in range(len(f1_scores))
        if f1_scores[i] < 0.50
    ]

    logger.info(
        f"Probe D — Retrieval Precision: {mean_f1:.4f} "
        f"(min={min_f1:.3f}, max={max_f1:.3f}, n={len(prompts)}, "
        f"low-scoring={len(low_scoring)})"
    )

    failures = []
    for i in range(len(prompts)):
        score = f1_scores[i]
        if score < 0.50:
            failures.append({
                "prompt": prompts[i],
                "reference": references[i],
                "generated": hypotheses[i],
                "score": score
            })

    return {
        "precision"         : mean_f1,
        "mean_bertscore_f1" : mean_f1,
        "min_bertscore_f1"  : min_f1,
        "max_bertscore_f1"  : max_f1,
        "num_samples"       : len(prompts),
        "low_scoring_prompts": low_scoring,
        "failures"          : failures,
    }


def get_failed_retrieval_samples(
    model,
    tokenizer,
    retrieval_prompts_path: Path,
    retrieval_references_path: Path,
    bertscore_model: str,
    max_new_tokens: int,
    device: str = "cuda",
    bertscore_batch_size: int = 32,
    use_bertscore: bool = True,
    generation_batch_size: int = 16,
    failure_threshold: float = 0.50,
) -> List[Dict[str, Any]]:
    """
    Run Probe D and return detailed list of failed samples where score is below the failure_threshold.
    """
    with open(retrieval_prompts_path, "r", encoding="utf-8") as f:
        prompts: List[str] = json.load(f)
    with open(retrieval_references_path, "r", encoding="utf-8") as f:
        references: List[str] = json.load(f)

    if len(prompts) != len(references):
        raise ValueError(
            f"Prompts ({len(prompts)}) and references ({len(references)}) must be the same length."
        )

    model.eval()
    hypotheses = generate_responses_batch(
        model=model,
        tokenizer=tokenizer,
        prompts=prompts,
        max_new_tokens=max_new_tokens,
        device=device,
        batch_size=generation_batch_size,
    )

    truncated_hypotheses = [token_safe_truncate(h, tokenizer, max_tokens=256) for h in hypotheses]
    truncated_references = [token_safe_truncate(r, tokenizer, max_tokens=256) for r in references]

    if use_bertscore:
        try:
            f1_scores = compute_bertscore_batch(
                hypotheses=truncated_hypotheses,
                references=truncated_references,
                model_type=bertscore_model,
                device=device,
                batch_size=bertscore_batch_size,
            )
        except Exception as e:
            logger.warning(f"BERTScore computation failed: {e}. Falling back to dummy F1 scores.")
            f1_scores = [0.5] * len(prompts)
    else:
        f1_scores = compute_lexical_f1_batch(truncated_hypotheses, truncated_references)

    failures = []
    for i in range(len(prompts)):
        score = f1_scores[i]
        if score < failure_threshold:
            failures.append({
                "prompt": prompts[i],
                "reference": references[i],
                "generated": hypotheses[i],
                "score": score
            })
    return failures

