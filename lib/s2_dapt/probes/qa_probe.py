"""
probes/qa_probe.py — Probe A: Neuroscience QA Accuracy
"""

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
    prompt: str,
    choices: List[str],
    device: str = "cuda",
) -> int:
    scores = []

    for choice in choices:
        full_text = prompt + " " + choice
        inputs = tokenizer(
            full_text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )
        if hasattr(inputs, "to"):
            inputs = inputs.to(device)
        else:
            inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}

        prompt_ids_inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )
        if hasattr(prompt_ids_inputs, "to"):
            prompt_ids_inputs = prompt_ids_inputs.to(device)
        else:
            prompt_ids_inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in prompt_ids_inputs.items()}

        prompt_ids = prompt_ids_inputs["input_ids"]

        prompt_len = prompt_ids.shape[1]

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits

        labels = inputs["input_ids"][0, prompt_len:].cpu()
        choice_logits = logits[0, prompt_len - 1 : -1, :].cpu()

        if len(labels) == 0:
            scores.append(-float("inf"))
            continue

        log_probs = F.log_softmax(choice_logits, dim=-1)
        choice_log_prob = sum(
            log_probs[t, labels[t]].item() for t in range(len(labels))
        )
        scores.append(choice_log_prob / max(len(labels), 1))

    return int(torch.tensor(scores).argmax().item())


def eval_qa_accuracy(
    model,
    tokenizer,
    qa_probe_path: Path,
    device: str = "cuda",
    max_samples: Optional[int] = None,
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
        return {"accuracy": 0.0, "correct": 0, "total": 0, "per_cluster_accuracy": {}}

    model.eval()
    correct = 0
    cluster_stats: Dict[str, Dict[str, int]] = {}

    for i, item in enumerate(qa_items):
        question = item.get("question")
        cluster = item.get("cluster", item.get("id", "unknown"))

        # Determine MCQ format: limit to New format (choices list + answer index)
        if "choices" in item and "answer_idx" in item and question is not None:
            choices = item["choices"]
            answer_idx = item["answer_idx"]
            prompt = format_mcq_prompt(question, choices)
            predicted = score_choices_by_logprob(model, tokenizer, prompt, choices, device=device)
            is_correct = int(predicted == answer_idx)
        else:
            logger.warning(
                f"Skipping malformed or unsupported QA item (must be New format with 'choices' and 'answer_idx'): {item}"
            )
            continue

        correct += is_correct

        if cluster not in cluster_stats:
            cluster_stats[cluster] = {"correct": 0, "total": 0}
        cluster_stats[cluster]["correct"] += is_correct
        cluster_stats[cluster]["total"] += 1

        if (i + 1) % 100 == 0:
            running_acc = correct / (i + 1)
            logger.debug(f"  QA probe: {i+1}/{len(qa_items)} evaluated, running acc={running_acc:.3f}")

    total = len(qa_items)
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
    }

