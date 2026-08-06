"""
probes/qa_probe.py — Probe 2 - QA probe
"""

import contextlib
import json
from functools import lru_cache
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, TypedDict, Union, overload

import torch
import torch.nn.functional as F

from lib.utils.logger import get_logger

logger = get_logger("dapt.probe.qa")

DEFAULT_MAX_LENGTH = 512
LOG_INTERVAL = 100


class QAEvalResults(TypedDict):
    accuracy: float
    correct: int
    total: int
    per_cluster_accuracy: Dict[str, float]
    failures: List[Dict[str, Any]]


def load_qa_items(
    qa_probe_path: Path,
    max_samples: Optional[int] = None
) -> Tuple[List[Dict[str, Any]], int]:
    """
    Load raw items from a JSONL file. Returns (items, skipped_json_lines_count).
    """
    qa_items: List[Dict[str, Any]] = []
    skipped_json_count = 0
    try:
        with open(qa_probe_path, "r", encoding="utf-8") as f:
            for line in f:
                line_str = line.strip()
                if not line_str:
                    continue
                try:
                    qa_items.append(json.loads(line_str))
                    if max_samples is not None and len(qa_items) >= max_samples:
                        break
                except json.JSONDecodeError:
                    skipped_json_count += 1
    except FileNotFoundError:
        logger.error(f"QA probe file not found at {qa_probe_path}")

    return qa_items, skipped_json_count


def validate_items(qa_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Filter and validate items. Ensures:
    1. 'question' is not None.
    2. 'choices' list exists, is not empty.
    3. 'answer_idx' exists and is a valid index within the choices.
    """
    valid_items = []
    for item in qa_items:
        question = item.get("question")
        choices = item.get("choices")
        answer_idx = item.get("answer_idx")
        if question is not None and choices is not None and answer_idx is not None:
            if isinstance(choices, list) and len(choices) > 0:
                if isinstance(answer_idx, int) and 0 <= answer_idx < len(choices):
                    valid_items.append(item)
                    continue
        logger.warning(
            f"Skipping malformed or out-of-bounds QA item: {item}"
        )
    return valid_items


@overload
def score_choices_by_logprob(
    model: torch.nn.Module,
    tokenizer: Any,
    prompt: str,
    choices: List[str],
    device: str = "cuda",
    use_pmi: bool = True,
    use_length_norm: bool = True,
    max_length: int = 512,
) -> int:
    ...


@overload
def score_choices_by_logprob(
    model: torch.nn.Module,
    tokenizer: Any,
    prompt: List[str],
    choices: List[List[str]],
    device: str = "cuda",
    use_pmi: bool = True,
    use_length_norm: bool = True,
    max_length: int = 512,
) -> List[int]:
    ...


@lru_cache(maxsize=8)
def _detect_bos(tokenizer_id: int, tokenizer_ref) -> Tuple[bool, str]:
    """Return (has_bos, uncond_prefix) for a tokenizer; result cached by tokenizer identity."""
    bos_id = tokenizer_ref.bos_token_id
    has_bos = False
    if bos_id is not None:
        try:
            test_res = tokenizer_ref("test")
            test_ids = test_res.get("input_ids", [])
            if isinstance(test_ids, torch.Tensor):
                test_list = test_ids.view(-1).tolist()
            elif isinstance(test_ids, list):
                test_list = test_ids[0] if (len(test_ids) > 0 and isinstance(test_ids[0], list)) else test_ids
            else:
                test_list = []
            if len(test_list) > 0 and test_list[0] == bos_id:
                has_bos = True
        except Exception:
            pass

    uncond_prefix = ""
    if not has_bos:
        try:
            tok = tokenizer_ref.bos_token
            if isinstance(tok, str):
                uncond_prefix = tok
        except Exception:
            pass
        if not uncond_prefix:
            try:
                tok = tokenizer_ref.eos_token
                if isinstance(tok, str):
                    uncond_prefix = tok
            except Exception:
                pass

    return has_bos, uncond_prefix


def score_choices_by_logprob(
    model: torch.nn.Module,
    tokenizer: Any,
    prompt: Union[str, List[str]],
    choices: Union[List[str], List[List[str]]],
    device: str = "cuda",
    use_pmi: bool = True,
    use_length_norm: bool = True,
    max_length: int = 512,
) -> Union[int, List[int]]:
    is_batched = not isinstance(prompt, str)

    if not is_batched:
        # Prompt is a string, choices is a List[str]
        prompts = [prompt]  # type: ignore
        choices_list = [choices]  # type: ignore
    else:
        # Prompt is List[str], choices is List[List[str]]
        prompts = prompt  # type: ignore
        choices_list = choices  # type: ignore

    if not prompts:
        return [] if is_batched else -1

    # Validate that choices are not empty (defense-in-depth for direct API calls)
    for chs in choices_list:
        if not chs:
            raise ValueError("Empty choices list provided to score_choices_by_logprob.")

    dev = torch.device(device)

    # Verify that model parameter device matches the requested device (prevent silent side effects)
    try:
        model_device = next(iter(model.parameters())).device
        if model_device.type != dev.type:
            raise ValueError(
                f"Model device ({model_device}) does not match requested device ({dev}). "
                "Moving the model inside scoring functions is disabled to prevent training side effects."
            )
    except StopIteration:
        pass

    # Ensure pad token is set
    # Note: pad_token_id fallback to 0 is safe because pad tokens are fully masked out in the attention mask
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # BOS detection is cached by tokenizer identity to avoid re-tokenizing "test" on every call
    has_bos, uncond_prefix = _detect_bos(id(tokenizer), tokenizer)

    # Build flat texts for conditional (prompt + " " + choice) and unconditional (choice)
    flat_cond_texts: List[str] = []
    flat_uncond_texts: List[str] = []
    prompts_for_flat: List[str] = []  # Keep track of the original prompt string to compute its char length

    for j, (p, chs) in enumerate(zip(prompts, choices_list)):
        for choice in chs:
            flat_cond_texts.append(p + " " + choice)
            flat_uncond_texts.append(uncond_prefix + choice)
            prompts_for_flat.append(p)

    N = len(flat_cond_texts)

    # 2. Hoist model dtype and autocast configuration
    try:
        model_dtype = next(iter(model.parameters())).dtype
    except (StopIteration, TypeError, AttributeError):
        model_dtype = torch.float32

    is_cuda = "cuda" in str(device)
    autocast_ctx = (
        torch.amp.autocast(device_type="cuda", dtype=model_dtype)
        if is_cuda and hasattr(torch, "amp")
        else contextlib.nullcontext()
    )    # Placeholders for scores
    scores = [0.0] * (2 * N)
    truncation_count = 0
    prompt_len_cache: Dict[str, int] = {}

    # Temporarily set tokenizer truncation to 'left' to preserve answer choices at the end
    original_truncation_side = tokenizer.truncation_side
    tokenizer.truncation_side = "left"

    def run_forward(texts):
        # Try batch tokenization first
        try:
            tokenized = tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
                return_offsets_mapping=True,
            )
            # Verify that the returned batch size matches the input list length
            if tokenized["input_ids"].shape[0] != len(texts):
                raise ValueError("Batch size mismatch from tokenizer")
        except Exception:
            # Fallback to loop-based tokenization for mock/slow tokenizers
            tokenized_inputs = []
            for t in texts:
                t_input = tokenizer(
                    t,
                    return_tensors="pt",
                    truncation=True,
                    max_length=max_length,
                )
                tokenized_inputs.append(t_input)
            
            # Manually pad and reconstruct tokenized dict
            max_len = max(t["input_ids"].shape[1] for t in tokenized_inputs)
            pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else (tokenizer.eos_token_id or 0)
            
            input_ids_list = []
            attention_mask_list = []
            for t in tokenized_inputs:
                ids = t["input_ids"][0]
                mask = t["attention_mask"][0]
                curr_len = len(ids)
                pad_len = max_len - curr_len
                
                padded_ids = torch.cat([ids, torch.tensor([pad_id] * pad_len, dtype=ids.dtype, device=ids.device)])
                padded_mask = torch.cat([mask, torch.tensor([0] * pad_len, dtype=mask.dtype, device=mask.device)])
                
                input_ids_list.append(padded_ids)
                attention_mask_list.append(padded_mask)
                
            tokenized = {
                "input_ids": torch.stack(input_ids_list),
                "attention_mask": torch.stack(attention_mask_list),
            }

        input_ids = tokenized["input_ids"].to(dev)
        attention_mask = tokenized["attention_mask"].to(dev)

        # Compute active lengths once to avoid redundant GPU transfers and calculations
        seq_lens = attention_mask.sum(dim=1).tolist()

        # Single forward pass for this batch
        with torch.inference_mode():
            with autocast_ctx:
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits

        return input_ids, logits, seq_lens, tokenized.get("offset_mapping")

    try:
        # Run conditional forward pass
        cond_input_ids, cond_logits, cond_seq_lens, cond_offsets = run_forward(flat_cond_texts)

        # Run unconditional forward pass (only if use_pmi is True)
        if use_pmi:
            uncond_input_ids, uncond_logits, uncond_seq_lens, uncond_offsets = run_forward(flat_uncond_texts)

        # Compute logprobs for conditional sequences
        for idx in range(N):
            seq_len_idx = cond_seq_lens[idx]
            if seq_len_idx >= max_length:
                truncation_count += 1

            prompt_str = prompts_for_flat[idx]
            prompt_char_len = len(prompt_str)

            # Find token boundary where the choice starts using character offset mapping if available
            if cond_offsets is not None:
                seq_offsets = cond_offsets[idx]
                split_idx = seq_len_idx
                for t_i, (start, end) in enumerate(seq_offsets):
                    if t_i >= seq_len_idx:
                        break
                    if start >= prompt_char_len:
                        split_idx = t_i
                        break
            else:
                if prompt_str not in prompt_len_cache:
                    prompt_tokenized = tokenizer(prompt_str, return_tensors="pt")
                    prompt_len_cache[prompt_str] = prompt_tokenized["input_ids"].shape[1]
                split_idx = prompt_len_cache[prompt_str]

            if seq_len_idx <= split_idx:
                scores[idx] = 0.0
                continue

            labels = cond_input_ids[idx, split_idx:seq_len_idx]
            logits_start = max(0, split_idx - 1)
            choice_logits = cond_logits[idx, logits_start : seq_len_idx - 1, :]

            if split_idx == 0:
                labels = labels[1:]

            log_probs = F.log_softmax(choice_logits, dim=-1)
            choice_len = len(labels)
            if choice_len == 0:
                scores[idx] = 0.0
                continue

            token_log_probs = log_probs[torch.arange(choice_len, device=dev), labels]
            sum_log_prob = token_log_probs.sum().item()

            if use_length_norm:
                scores[idx] = sum_log_prob / choice_len
            else:
                scores[idx] = sum_log_prob

        # Compute logprobs for unconditional sequences (only if use_pmi is True)
        if use_pmi:
            for idx in range(N):
                seq_len_idx = uncond_seq_lens[idx]
                if seq_len_idx >= max_length:
                    truncation_count += 1

                # Unconditional: Check if first token is BOS or a prepended prefix
                split_idx = 1 if (has_bos or len(uncond_prefix) > 0) else 0

                if seq_len_idx <= split_idx:
                    scores[N + idx] = 0.0
                    continue

                labels = uncond_input_ids[idx, split_idx:seq_len_idx]
                logits_start = max(0, split_idx - 1)
                choice_logits = uncond_logits[idx, logits_start : seq_len_idx - 1, :]

                if split_idx == 0:
                    labels = labels[1:]

                log_probs = F.log_softmax(choice_logits, dim=-1)
                choice_len = len(labels)
                if choice_len == 0:
                    scores[N + idx] = 0.0
                    continue

                token_log_probs = log_probs[torch.arange(choice_len, device=dev), labels]
                sum_log_prob = token_log_probs.sum().item()

                if use_length_norm:
                    scores[N + idx] = sum_log_prob / choice_len
                else:
                    scores[N + idx] = sum_log_prob

    finally:
        # Restore the original truncation side
        tokenizer.truncation_side = original_truncation_side

    # Log truncation warnings consolidated for the batch
    if truncation_count > 0:
        logger.warning(
            f"Truncated {truncation_count} sequence(s) from the left to max_length={max_length}. "
            "This can impact choice prediction accuracy."
        )

    # 4. Group scores back to identify predicted choice index for each prompt
    predicted_indices = []
    curr_flat_idx = 0
    for chs in choices_list:
        num_choices = len(chs)
        choice_scores = []
        for i in range(num_choices):
            idx = curr_flat_idx + i
            cond_score = scores[idx]
            uncond_score = scores[N + idx]
            if use_pmi:
                pmi_score = cond_score - uncond_score
                choice_scores.append(pmi_score)
            else:
                choice_scores.append(cond_score)
        predicted_idx = int(torch.tensor(choice_scores).argmax().item())
        predicted_indices.append(predicted_idx)
        curr_flat_idx += num_choices

    return predicted_indices if is_batched else predicted_indices[0]


def eval_qa_accuracy(
    model,
    tokenizer,
    qa_probe_path: Path,
    device: str = "cuda",
    max_samples: Optional[int] = None,
    batch_size: int = 32,
    use_pmi: bool = True,
    use_length_norm: bool = True,
    max_length: int = 512,
    eval_num: Union[int, str] = 1,
) -> QAEvalResults:
    """
    Run Probe 2 - QA probe and return accuracy plus per-cluster diagnostics.
    """
    # 1. Load and validate QA items
    qa_items, skipped_json_count = load_qa_items(qa_probe_path, max_samples)
    if skipped_json_count > 0:
        logger.warning(f"Skipped {skipped_json_count} malformed JSONL lines in QA probe file.")

    if not qa_items:
        logger.warning(f"No QA probe questions found at {qa_probe_path}")
        return {
            "accuracy": 0.0,
            "correct": 0,
            "total": 0,
            "per_cluster_accuracy": {},
            "failures": [],
        }

    # 2. Filter and validate items
    valid_items = validate_items(qa_items)
    if not valid_items:
        logger.warning(f"No valid QA probe questions after filtering at {qa_probe_path}")
        return {
            "accuracy": 0.0,
            "correct": 0,
            "total": 0,
            "per_cluster_accuracy": {},
            "failures": [],
        }

    # Save training mode and set to eval
    original_mode = model.training
    model.eval()

    try:
        correct = 0
        cluster_stats: Dict[str, Dict[str, int]] = {}
        eval_traces = []
        failures = []

        # Process in batches
        for b_idx in range(0, len(valid_items), batch_size):
            batch_items = valid_items[b_idx : b_idx + batch_size]
            prompts = []
            choices_list = []
            for item in batch_items:
                prompts.append(f"Question: {item['question']}\nAnswer:")
                choices_list.append(item["choices"])

            predicted_indices = score_choices_by_logprob(
                model,
                tokenizer,
                prompts,
                choices_list,
                device=device,
                use_pmi=use_pmi,
                use_length_norm=use_length_norm,
                max_length=max_length,
            )

            for rel_idx, (item, predicted) in enumerate(zip(batch_items, predicted_indices)):
                seq_idx = b_idx + rel_idx + 1
                answer_idx = item["answer_idx"]
                cluster = item.get("cluster", "unknown")
                is_correct = int(predicted == answer_idx)
                correct += is_correct
                choices = item["choices"]
                predicted_text = choices[predicted] if predicted < len(choices) else "unknown"

                sample_record = {
                    "Eval #": str(eval_num),
                    "Eval Category": "QA",
                    "Eval Seq #": item.get("seq_num", seq_idx),
                    "Eval": json.dumps(item),
                    "Generated Answer by the model": predicted_text,
                    "Matching Score": 1.0 if is_correct else 0.0,
                    "Result": "Pass" if is_correct else "Fail",
                    "question": item["question"],
                    "choices": choices,
                    "expected_idx": answer_idx,
                    "expected_text": choices[answer_idx] if answer_idx < len(choices) else "unknown",
                    "predicted_idx": predicted,
                    "predicted_text": predicted_text,
                    "cluster": cluster,
                }
                eval_traces.append(sample_record)

                if not is_correct:
                    failures.append(sample_record)

                if cluster not in cluster_stats:
                    cluster_stats[cluster] = {"correct": 0, "total": 0}
                cluster_stats[cluster]["correct"] += is_correct
                cluster_stats[cluster]["total"] += 1

            processed_count = b_idx + len(batch_items)
            if (b_idx // LOG_INTERVAL) < (processed_count // LOG_INTERVAL) or processed_count == len(valid_items):
                running_acc = correct / processed_count
                logger.debug(f"  QA probe: {processed_count}/{len(valid_items)} evaluated, running acc={running_acc:.3f}")

        total = len(valid_items)
        accuracy = correct / total if total > 0 else 0.0

        per_cluster = {
            cluster: stats["correct"] / stats["total"]
            for cluster, stats in cluster_stats.items()
            if stats["total"] > 0
        }

        logger.info(f"Probe 2 - QA probe: {accuracy:.4f} ({correct}/{total})")

        return {
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
            "per_cluster_accuracy": per_cluster,
            "eval_traces": eval_traces,
            "samples": eval_traces,
            "failures": failures,
        }

    finally:
        # Restore training mode if it was originally True
        if original_mode:
            model.train()


def get_qa_probe_traces(
    model,
    tokenizer,
    qa_probe_path: Path,
    device: str = "cuda",
    max_samples: Optional[int] = None,
    batch_size: int = 32,
    use_pmi: bool = True,
    use_length_norm: bool = True,
    max_length: int = 512,
    eval_num: Union[int, str] = 1,
) -> List[Dict[str, Any]]:
    """
    Run Probe 2 - QA probe and return list of all evaluation traces with Eval # and Result ('Pass' or 'Fail').
    """
    result = eval_qa_accuracy(
        model=model,
        tokenizer=tokenizer,
        qa_probe_path=qa_probe_path,
        device=device,
        max_samples=max_samples,
        batch_size=batch_size,
        use_pmi=use_pmi,
        use_length_norm=use_length_norm,
        max_length=max_length,
        eval_num=eval_num,
    )
    return result["eval_traces"]


def get_qa_probe_samples(
    model,
    tokenizer,
    qa_probe_path: Path,
    device: str = "cuda",
    max_samples: Optional[int] = None,
    batch_size: int = 32,
    use_pmi: bool = True,
    use_length_norm: bool = True,
    max_length: int = 512,
    eval_num: Union[int, str] = 1,
) -> List[Dict[str, Any]]:
    """
    Backward-compatible alias for get_qa_probe_traces.
    """
    return get_qa_probe_traces(
        model=model,
        tokenizer=tokenizer,
        qa_probe_path=qa_probe_path,
        device=device,
        max_samples=max_samples,
        batch_size=batch_size,
        use_pmi=use_pmi,
        use_length_norm=use_length_norm,
        max_length=max_length,
        eval_num=eval_num,
    )


def get_failed_qa_samples(
    model,
    tokenizer,
    qa_probe_path: Path,
    device: str = "cuda",
    max_samples: Optional[int] = None,
    batch_size: int = 32,
    use_pmi: bool = True,
    use_length_norm: bool = True,
    max_length: int = 512,
    eval_num: Union[int, str] = 1,
) -> List[Dict[str, Any]]:
    """
    Backward-compatible alias for get_qa_probe_traces.
    """
    return get_qa_probe_traces(
        model=model,
        tokenizer=tokenizer,
        qa_probe_path=qa_probe_path,
        device=device,
        max_samples=max_samples,
        batch_size=batch_size,
        use_pmi=use_pmi,
        use_length_norm=use_length_norm,
        max_length=max_length,
        eval_num=eval_num,
    )
