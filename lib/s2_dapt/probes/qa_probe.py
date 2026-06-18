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
    # Dynamically load from either JSON or JSONL format
    qa_items: List[Dict[str, Any]] = []
    try:
        with open(qa_probe_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                qa_items = data
            elif isinstance(data, dict):
                qa_items = [data]
    except (json.JSONDecodeError, TypeError, ValueError):
        # Fallback to JSONL
        qa_items = []
        with open(qa_probe_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        qa_items.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass

    if not qa_items:
        logger.warning(f"No QA probe questions found at {qa_probe_path}")
        return {"accuracy": 0.0, "correct": 0, "total": 0, "per_cluster_accuracy": {}}

    if max_samples is not None:
        qa_items = qa_items[:max_samples]

    model.eval()
    correct = 0
    cluster_stats: Dict[str, Dict[str, int]] = {}

    for i, item in enumerate(qa_items):
        question = item["question"]
        cluster = item.get("cluster", item.get("id", "unknown"))

        # Determine MCQ format
        if "choices" in item and "answer_idx" in item:
            # New format (choices list + index)
            choices = item["choices"]
            answer_idx = item["answer_idx"]
            prompt = format_mcq_prompt(question, choices)
            predicted = score_choices_by_logprob(model, tokenizer, prompt, choices, device=device)
            is_correct = int(predicted == answer_idx)
        elif "answer" in item:
            # Old format (next token letter classification)
            correct_answer_char = item["answer"].strip()
            prompt = f"Question: {question}\nAnswer:"
            inputs = tokenizer(prompt, return_tensors="pt")
            if hasattr(inputs, "to"):
                inputs = inputs.to(device)
            else:
                inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}
            input_ids = inputs["input_ids"]
            attention_mask = inputs.get("attention_mask", torch.ones_like(input_ids))

            with torch.no_grad():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                next_token_logits = outputs.logits[0, -1, :]
                next_token_probs = torch.softmax(next_token_logits, dim=-1)

            options = ["A", "B", "C", "D", "E"]
            option_probs = {}
            for opt in options:
                opt_token_ids = tokenizer.encode(" " + opt, add_special_tokens=False)
                if len(opt_token_ids) > 0:
                    opt_token_id = opt_token_ids[-1]
                    option_probs[opt] = next_token_probs[opt_token_id].item()
                else:
                    option_probs[opt] = 0.0

            best_option = max(option_probs, key=option_probs.get)
            is_correct = int(best_option == correct_answer_char)
        else:
            logger.warning(f"Skipping malformed QA item: {item}")
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

