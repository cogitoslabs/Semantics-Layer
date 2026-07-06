"""
probes/qa_probe.py — Probe A: Neuroscience QA Accuracy
"""

import contextlib
import json
from pathlib import Path
from typing import List, Dict, Any, Optional

import torch
import torch.nn.functional as F

from lib.utils.logger import get_logger

logger = get_logger("dapt.probe.qa")

MCQ_LABELS = ["A", "B", "C", "D", "E"]


def format_mcq_prompt(question: str, choices: List[str]) -> str:
    lines = [f"Question: {question}\n"]
    for i, choice in enumerate(choices):
        label = MCQ_LABELS[i] if i < len(MCQ_LABELS) else str(i)
        lines.append(f"{label}. {choice}")
    lines.append("\nAnswer:")
    return "\n".join(lines)


def score_choices_by_logprob(
    model,
    tokenizer,
    prompt: Any,
    choices: Any,
    device: str = "cuda",
) -> Any:
    is_batched = not isinstance(prompt, str)

    if not is_batched:
        prompts = [prompt]
        choices_list = [choices]
    else:
        prompts = prompt
        choices_list = choices

    if not prompts:
        return [] if is_batched else -1

    dev = torch.device(device)

    # Ensure pad token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 1. Tokenize all prompts to get their token lengths
    prompt_lens = []
    for p in prompts:
        p_inputs = tokenizer(
            p,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )
        prompt_lens.append(p_inputs["input_ids"].shape[1])

    # 2. Build flat texts for conditional (prompt + choice) and unconditional (choice)
    flat_cond_texts = []
    flat_cond_prompt_lens = []
    flat_uncond_texts = []

    for j, (p, chs) in enumerate(zip(prompts, choices_list)):
        p_len = prompt_lens[j]
        for choice in chs:
            flat_cond_texts.append(p + " " + choice)
            flat_cond_prompt_lens.append(p_len)
            flat_uncond_texts.append(choice)

    # 3. Helper to tokenize, run forward pass, and compute log probabilities
    def compute_logprobs(texts, split_lens=None):
        tokenized_inputs = []
        for text in texts:
            t_input = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=512,
            )
            tokenized_inputs.append(t_input)

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

        input_ids = torch.stack(input_ids_list).to(dev)
        attention_mask = torch.stack(attention_mask_list).to(dev)

        try:
            model_dtype = next(iter(model.parameters())).dtype
        except (StopIteration, TypeError, AttributeError):
            model_dtype = torch.float32

        is_cuda = "cuda" in str(device)
        autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=model_dtype) if is_cuda and hasattr(torch, "amp") else contextlib.nullcontext()

        with torch.no_grad():
            with autocast_ctx:
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits

        scores = []
        for idx in range(len(texts)):
            if split_lens is not None:
                p_len = split_lens[idx]
            else:
                p_len = 1  # For unconditional, split after BOS token

            seq_len_idx = attention_mask[idx].sum().item()

            if seq_len_idx <= p_len:
                scores.append(0.0)
                continue

            labels = input_ids[idx, p_len:seq_len_idx]
            choice_logits = logits[idx, p_len - 1 : seq_len_idx - 1, :]

            log_probs = F.log_softmax(choice_logits, dim=-1)
            choice_len = len(labels)
            if choice_len == 0:
                scores.append(0.0)
                continue
            token_log_probs = log_probs[torch.arange(choice_len, device=dev), labels]
            scores.append(token_log_probs.sum().item())
        return scores

    cond_scores = compute_logprobs(flat_cond_texts, flat_cond_prompt_lens)
    uncond_scores = compute_logprobs(flat_uncond_texts, None)

    # 4. Group scores back to identify predicted choice index for each prompt using PMI (cond - uncond)
    predicted_indices = []
    curr_flat_idx = 0
    for chs in choices_list:
        num_choices = len(chs)
        choice_scores = []
        for i in range(num_choices):
            idx = curr_flat_idx + i
            pmi_score = cond_scores[idx] - uncond_scores[idx]
            choice_scores.append(pmi_score)
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
) -> Dict[str, Any]:
    """
    Run Probe A and return accuracy plus per-cluster diagnostics.
    """
    # Load from JSONL format with early-stopping for max_samples
    qa_items: List[Dict[str, Any]] = []
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
                    pass
    except FileNotFoundError:
        logger.error(f"QA probe file not found at {qa_probe_path}")

    if not qa_items:
        logger.warning(f"No QA probe questions found at {qa_probe_path}")
        return {"accuracy": 0.0, "correct": 0, "total": 0, "per_cluster_accuracy": {}, "failures": []}

    model.eval()
    correct = 0
    cluster_stats: Dict[str, Dict[str, int]] = {}

    # Filter out malformed items first to make batching clean
    valid_items = []
    for item in qa_items:
        question = item.get("question")
        if "choices" in item and "answer_idx" in item and question is not None:
            valid_items.append(item)
        else:
            logger.warning(
                f"Skipping malformed or unsupported QA item (must be New format with 'choices' and 'answer_idx'): {item}"
            )

    # Process in batches
    failures = []
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
        )

        for item, predicted in zip(batch_items, predicted_indices):
            answer_idx = item["answer_idx"]
            cluster = item.get("cluster", "unknown")
            is_correct = int(predicted == answer_idx)
            correct += is_correct

            if not is_correct:
                choices = item["choices"]
                failures.append({
                    "question": item["question"],
                    "choices": choices,
                    "expected_idx": answer_idx,
                    "expected_text": choices[answer_idx] if answer_idx < len(choices) else "unknown",
                    "predicted_idx": predicted,
                    "predicted_text": choices[predicted] if predicted < len(choices) else "unknown",
                    "cluster": cluster
                })

            if cluster not in cluster_stats:
                cluster_stats[cluster] = {"correct": 0, "total": 0}
            cluster_stats[cluster]["correct"] += is_correct
            cluster_stats[cluster]["total"] += 1

        processed_count = b_idx + len(batch_items)
        if (b_idx // 100) < (processed_count // 100) or processed_count == len(valid_items):
            running_acc = correct / processed_count
            logger.debug(f"  QA probe: {processed_count}/{len(valid_items)} evaluated, running acc={running_acc:.3f}")

    total = len(valid_items)
    accuracy = correct / total if total > 0 else 0.0

    per_cluster = {
        cluster: stats["correct"] / stats["total"]
        for cluster, stats in cluster_stats.items()
        if stats["total"] > 0
    }

    logger.info(f"Probe A — QA Accuracy: {accuracy:.4f} ({correct}/{total})")

    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "per_cluster_accuracy": per_cluster,
        "failures": failures,
    }


def get_failed_qa_samples(
    model,
    tokenizer,
    qa_probe_path: Path,
    device: str = "cuda",
    batch_size: int = 32,
) -> List[Dict[str, Any]]:
    """
    Run Probe A and return a list of failed samples.
    """
    qa_items: List[Dict[str, Any]] = []
    try:
        with open(qa_probe_path, "r", encoding="utf-8") as f:
            for line in f:
                line_str = line.strip()
                if not line_str:
                    continue
                try:
                    qa_items.append(json.loads(line_str))
                except json.JSONDecodeError:
                    pass
    except FileNotFoundError:
        logger.error(f"QA probe file not found at {qa_probe_path}")
        return []

    model.eval()
    failures = []

    valid_items = []
    for item in qa_items:
        question = item.get("question")
        if "choices" in item and "answer_idx" in item and question is not None:
            valid_items.append(item)

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
        )

        for item, predicted in zip(batch_items, predicted_indices):
            answer_idx = item["answer_idx"]
            if predicted != answer_idx:
                choices = item["choices"]
                failures.append({
                    "question": item["question"],
                    "choices": choices,
                    "expected_idx": answer_idx,
                    "expected_text": choices[answer_idx] if answer_idx < len(choices) else "unknown",
                    "predicted_idx": predicted,
                    "predicted_text": choices[predicted] if predicted < len(choices) else "unknown",
                    "cluster": item.get("cluster", "unknown")
                })
    return failures


