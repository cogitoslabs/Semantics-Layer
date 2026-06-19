"""
probes/retrieval_probe.py — Probe D: Anatomical Landmark Retrieval Precision
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional

import torch
import bert_score.utils
from bert_score import score as bertscore_score

# Monkeypatch get_tokenizer to prevent OverflowError on models with undefined model_max_length
_orig_get_tokenizer = bert_score.utils.get_tokenizer

def _patched_get_tokenizer(*args, **kwargs):
    tokenizer = _orig_get_tokenizer(*args, **kwargs)
    if getattr(tokenizer, "model_max_length", 0) > 1_000_000:
        tokenizer.model_max_length = 512
    return tokenizer

bert_score.utils.get_tokenizer = _patched_get_tokenizer

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
        )

    prompt_len = inputs["input_ids"].shape[1]
    generated_ids = output_ids[0, prompt_len:]
    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


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

    logger.info(f"Generating {len(prompts)} retrieval responses (greedy)...")
    hypotheses = []
    for i, prompt in enumerate(prompts):
        response = generate_response(model, tokenizer, prompt, max_new_tokens, device=device)
        hypotheses.append(response)
        if (i + 1) % 20 == 0:
            logger.debug(f"  Retrieval probe: {i+1}/{len(prompts)} responses generated")

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

    return {
        "precision"         : mean_f1,
        "mean_bertscore_f1" : mean_f1,
        "min_bertscore_f1"  : min_f1,
        "max_bertscore_f1"  : max_f1,
        "num_samples"       : len(prompts),
        "low_scoring_prompts": low_scoring,
    }
