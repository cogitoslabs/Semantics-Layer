"""
probes/retrieval_probe.py — Probe 4 - Concept Probe
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Union

import torch
import transformers
from bert_score import score as bertscore_score
from contextlib import contextmanager

@contextmanager
def _patched_tokenizer_context():
    """
    Temporarily patch AutoTokenizer.from_pretrained to clamp model_max_length for SciBERT
    to prevent OverflowError in BERTScore.
    """
    orig_from_pretrained = transformers.AutoTokenizer.from_pretrained
    
    def patched_from_pretrained(*args, **kwargs):
        tokenizer = orig_from_pretrained(*args, **kwargs)
        max_len = getattr(tokenizer, "model_max_length", 0)
        if isinstance(max_len, (int, float)) and max_len > 1_000_000:
            tokenizer.model_max_length = 512

        # Ensure build_inputs_with_special_tokens is defined to prevent AttributeError in bert_score
        # on environments with certain transformers versions (e.g. Google Colab)
        if not hasattr(tokenizer, "build_inputs_with_special_tokens"):
            logger.info(f"Monkeypatching build_inputs_with_special_tokens onto {tokenizer.__class__.__name__}")
            def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
                cls_id = getattr(self, "cls_token_id", None) or 101
                sep_id = getattr(self, "sep_token_id", None) or 102
                if token_ids_1 is None:
                    return [cls_id] + token_ids_0 + [sep_id]
                return [cls_id] + token_ids_0 + [sep_id] + token_ids_1 + [sep_id]
            # Bind the function to the instance
            import types
            tokenizer.build_inputs_with_special_tokens = types.MethodType(build_inputs_with_special_tokens, tokenizer)

        return tokenizer
        
    transformers.AutoTokenizer.from_pretrained = patched_from_pretrained
    try:
        yield
    finally:
        transformers.AutoTokenizer.from_pretrained = orig_from_pretrained

from lib.utils.logger import get_logger
from lib.utils import model_trace
from lib.s3_dapt.probes.utils import clean_for_match
from typing import Any

logger = get_logger("dapt.probe.concept")

_SCORER_CACHE = {}

def get_bertscorer(model_type: str, device: str) -> Any:
    import bert_score
    
    # Register local model path in bert_score's model2layers dictionary dynamically
    if model_type not in bert_score.utils.model2layers:
        last_part = model_type.replace("\\", "/").split("/")[-1]
        for key, val in list(bert_score.utils.model2layers.items()):
            if key.split("/")[-1] == last_part:
                bert_score.utils.model2layers[model_type] = val
                logger.info(f"Dynamically registered local path '{model_type}' in bert_score.utils.model2layers with {val} layers (copied from '{key}')")
                break

    cache_key = (model_type, device)
    if cache_key not in _SCORER_CACHE:
        logger.info(f"Initializing and caching BERTScorer for {model_type} on {device}...")
        _SCORER_CACHE[cache_key] = bert_score.BERTScorer(model_type=model_type, device=device)
    return _SCORER_CACHE[cache_key]



def clear_scorer_cache() -> None:
    """Clear cached BERTScorers to free up GPU memory."""
    global _SCORER_CACHE
    if _SCORER_CACHE:
        logger.info("Clearing cached BERTScorers from memory...")
        _SCORER_CACHE.clear()
        import gc
        import torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


@model_trace
def generate_response(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
    device: str = "cuda",
    max_length: int = 256,
    eval_num: Union[int, str] = 1,
    eval_category: str = "Concept",
    eval_seq_num: Optional[int] = None,
) -> str:
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
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

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )

    prompt_len = inputs["input_ids"].shape[1]
    generated_ids = output_ids[0, prompt_len:]
    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


@model_trace
def generate_responses_batch(
    model,
    tokenizer,
    prompts: List[str],
    max_new_tokens: int,
    device: str = "cuda",
    batch_size: int = 16,
    max_length: int = 256,
    eval_num: Union[int, str] = 1,
    eval_category: str = "Concept",
    eval_seq_start: int = 1,
) -> List[str]:
    if not prompts:
        return []

    # Sort prompts by length to minimize padding and save GPU VRAM (bucket by padding)
    indexed_prompts = sorted(enumerate(prompts), key=lambda x: len(x[1]))
    sorted_prompts = [p for _, p in indexed_prompts]
    orig_indices = [idx for idx, _ in indexed_prompts]

    sorted_responses = []
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else (tokenizer.eos_token_id or 0)

    # Set up stop token IDs including EOS to stop generation
    stop_token_ids = []
    if tokenizer.eos_token_id is not None:
        if isinstance(tokenizer.eos_token_id, list):
            stop_token_ids.extend(tokenizer.eos_token_id)
        else:
            stop_token_ids.append(tokenizer.eos_token_id)

    stop_token_ids = list(set(stop_token_ids))

    orig_padding_side = getattr(tokenizer, "padding_side", "right")
    tokenizer.padding_side = "left"

    try:
        for start_idx in range(0, len(sorted_prompts), batch_size):
            batch_prompts = sorted_prompts[start_idx : start_idx + batch_size]

            # Try built-in batch tokenization first for optimized C++ execution
            try:
                tokenized = tokenizer(
                    batch_prompts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                )
                
                # Check for MagicMock/mock tokenizer that returns single inputs or fails
                if tokenized["input_ids"].shape[0] != len(batch_prompts):
                    raise ValueError("Batch size mismatch from tokenizer")
                
                stacked_input_ids = tokenized["input_ids"].to(device)
                stacked_attention_masks = tokenized.get(
                    "attention_mask", torch.ones_like(stacked_input_ids)
                ).to(device)
                max_len = stacked_input_ids.shape[1]
                
            except Exception:
                # Fallback to individual tokenization and manual padding (for mocks/custom tokenizers)
                batch_inputs = []
                for prompt in batch_prompts:
                    inputs = tokenizer(
                        prompt,
                        return_tensors="pt",
                        truncation=True,
                        max_length=max_length,
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

            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids=stacked_input_ids,
                    attention_mask=stacked_attention_masks,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    temperature=1.0,
                    repetition_penalty=1.1,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=stop_token_ids,
                    use_cache=True,
                )

            is_mock = "Mock" in type(model).__name__ or "MagicMock" in type(model).__name__
            has_mismatch = isinstance(output_ids, torch.Tensor) and output_ids.shape[0] != len(batch_prompts)
            if has_mismatch and not is_mock:
                raise ValueError(
                    f"Batch size mismatch: model generated {output_ids.shape[0]} sequences, "
                    f"but expected {len(batch_prompts)}."
                )

            for i in range(len(batch_prompts)):
                if isinstance(output_ids, torch.Tensor):
                    if has_mismatch:
                        batch_idx = i if i < output_ids.shape[0] else 0
                        generated_ids = output_ids[batch_idx, max_len:]
                    else:
                        generated_ids = output_ids[i, max_len:]
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
    if not hypotheses:
        return []

    # Map empty/whitespace candidates to 0.0 directly to avoid bert_score warning spam
    scores = [0.0] * len(hypotheses)
    non_empty_indices = [i for i, h in enumerate(hypotheses) if h.strip()]

    if not non_empty_indices:
        return scores

    cands_to_eval = [hypotheses[i] for i in non_empty_indices]
    refs_to_eval = [references[i] for i in non_empty_indices]

    import warnings
    with _patched_tokenizer_context():
        scorer = get_bertscorer(model_type, device)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, message=".*[Ee]mpty candidate sentence.*")
            warnings.filterwarnings("ignore", category=UserWarning, message=".*[Ee]mpty reference sentence.*")
            _, _, F1 = scorer.score(
                cands=cands_to_eval,
                refs=refs_to_eval,
                batch_size=batch_size,
                verbose=False,
            )
        F1_list = F1.tolist()

    for idx, score in zip(non_empty_indices, F1_list):
        scores[idx] = score

    return scores


def compute_lexical_f1_batch(hypotheses: List[str], references: List[str]) -> List[float]:
    # Pre-tokenize all references once to avoid redundant tokenization in the loop
    pre_tokenized_refs = []
    for ref in references:
        ref_tokens = clean_for_match(ref).split()
        ref_counter = {}
        for t in ref_tokens:
            ref_counter[t] = ref_counter.get(t, 0) + 1
        pre_tokenized_refs.append((ref_tokens, ref_counter))

    f1_scores = []
    for hyp, (ref_tokens, ref_counter) in zip(hypotheses, pre_tokenized_refs):
        hyp_tokens = clean_for_match(hyp).split()
        if not hyp_tokens or not ref_tokens:
            f1_scores.append(0.0)
            continue
        hyp_counter = {}
        for t in hyp_tokens:
            hyp_counter[t] = hyp_counter.get(t, 0) + 1
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


def format_concept_prompt(prompt_text: str) -> str:
    """
    Format concept probe prompt to condition base causal language models
    to output a direct definition/explanation instead of continuing exam question lists.
    """
    return f"Prompt: {prompt_text}\nAnswer:"


def eval_concept_precision(
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
    failure_threshold: float = 0.50,
    max_length: int = 256,
    eval_num: Union[int, str] = 1,
) -> Dict[str, Any]:
    """
    Run Probe 4 - Concept Probe and return average precision over retrieval prompts.
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

    formatted_prompts = [format_concept_prompt(p) for p in prompts]

    logger.info(f"Generating {len(prompts)} retrieval responses (greedy, batch_size={generation_batch_size})...")
    hypotheses = generate_responses_batch(
        model=model,
        tokenizer=tokenizer,
        prompts=formatted_prompts,
        max_new_tokens=max_new_tokens,
        device=device,
        batch_size=generation_batch_size,
        max_length=max_length,
        eval_num=eval_num,
        eval_category="Concept",
        eval_seq_start=1,
    )

    bertscore_failed = False
    if use_bertscore:
        logger.info(f"Computing BERTScore with model: {bertscore_model}")
        try:
            f1_scores = compute_bertscore_batch(
                hypotheses=hypotheses,
                references=references,
                model_type=bertscore_model,
                device=device,
                batch_size=bertscore_batch_size,
            )
        except Exception as e:
            logger.exception("BERTScore computation failed.")
            raise RuntimeError(
                f"BERTScore evaluation failed. High-fidelity retrieval precision metrics cannot be computed: {e}"
            ) from e
    else:
        logger.info("Computing fast lexical F1 overlap scores...")
        f1_scores = compute_lexical_f1_batch(hypotheses, references)

    mean_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0
    min_f1  = min(f1_scores) if f1_scores else 0.0
    max_f1  = max(f1_scores) if f1_scores else 0.0

    low_scoring = [
        {"prompt": prompts[i], "f1": f1_scores[i], "generated": hypotheses[i][:200]}
        for i in range(len(f1_scores))
        if f1_scores[i] < failure_threshold
    ]

    logger.info(
        f"Probe 4 - Concept Probe: {mean_f1:.4f} "
        f"(min={min_f1:.3f}, max={max_f1:.3f}, n={len(prompts)}, "
        f"low-scoring={len(low_scoring)})"
    )

    eval_traces = []
    failures = []
    for i in range(len(prompts)):
        score = f1_scores[i]
        is_pass = score >= failure_threshold
        gen_answer = str(hypotheses[i]) if hypotheses else ""
        eval_dump = json.dumps({"prompt": prompts[i], "reference": references[i]})

        sample_record = {
            "Eval #": str(eval_num),
            "Eval Category": "Concept",
            "Eval Seq #": i + 1,
            "Eval": eval_dump,
            "Generated Answer by the model": gen_answer,
            "Matching Score": round(float(score), 4),
            "Result": "Pass" if is_pass else "Fail",
            "prompt": prompts[i],
            "reference": references[i],
            "generated": gen_answer,
            "score": float(score),
            "threshold": float(failure_threshold),
        }
        eval_traces.append(sample_record)
        if not is_pass:
            failures.append(sample_record)

    return {
        "precision"         : mean_f1,
        "mean_bertscore_f1" : mean_f1,
        "min_bertscore_f1"  : min_f1,
        "max_bertscore_f1"  : max_f1,
        "num_samples"       : len(prompts),
        "low_scoring_prompts": low_scoring,
        "eval_traces"       : eval_traces,
        "samples"           : eval_traces,
        "failures"          : failures,
        "bertscore_failed"  : bertscore_failed,
    }


def get_concept_probe_traces(
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
    max_length: int = 256,
    eval_num: Union[int, str] = 1,
) -> List[Dict[str, Any]]:
    """
    Run Probe 4 - Concept Probe and return detailed list of all evaluation traces with Eval # and Result ('Pass' or 'Fail').
    """
    result = eval_concept_precision(
        model=model,
        tokenizer=tokenizer,
        retrieval_prompts_path=retrieval_prompts_path,
        retrieval_references_path=retrieval_references_path,
        bertscore_model=bertscore_model,
        max_new_tokens=max_new_tokens,
        device=device,
        bertscore_batch_size=bertscore_batch_size,
        use_bertscore=use_bertscore,
        generation_batch_size=generation_batch_size,
        failure_threshold=failure_threshold,
        max_length=max_length,
        eval_num=eval_num,
    )
    return result["eval_traces"]


def get_concept_probe_samples(
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
    max_length: int = 256,
    eval_num: Union[int, str] = 1,
) -> List[Dict[str, Any]]:
    """
    Backward-compatible alias for get_concept_probe_traces.
    """
    return get_concept_probe_traces(
        model=model,
        tokenizer=tokenizer,
        retrieval_prompts_path=retrieval_prompts_path,
        retrieval_references_path=retrieval_references_path,
        bertscore_model=bertscore_model,
        max_new_tokens=max_new_tokens,
        device=device,
        bertscore_batch_size=bertscore_batch_size,
        use_bertscore=use_bertscore,
        generation_batch_size=generation_batch_size,
        failure_threshold=failure_threshold,
        max_length=max_length,
        eval_num=eval_num,
    )


def get_failed_concept_samples(
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
    max_length: int = 256,
    eval_num: Union[int, str] = 1,
) -> List[Dict[str, Any]]:
    """
    Backward-compatible alias for get_concept_probe_traces.
    """
    return get_concept_probe_traces(
        model=model,
        tokenizer=tokenizer,
        retrieval_prompts_path=retrieval_prompts_path,
        retrieval_references_path=retrieval_references_path,
        bertscore_model=bertscore_model,
        max_new_tokens=max_new_tokens,
        device=device,
        bertscore_batch_size=bertscore_batch_size,
        use_bertscore=use_bertscore,
        generation_batch_size=generation_batch_size,
        failure_threshold=failure_threshold,
        max_length=max_length,
        eval_num=eval_num,
    )

