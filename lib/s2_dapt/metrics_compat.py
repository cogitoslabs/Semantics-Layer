import math
import os
import torch
from typing import Any, Dict, List

def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(file_path):
        return []
    with open(file_path, "r", encoding="utf-8") as f:
        import json
        return [json.loads(line) for line in f if line.strip()]

def evaluate_perplexity(
    model: Any,
    tokenizer: Any,
    dataset: List[Dict[str, Any]],
    block_size: int = 512
) -> float:
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    # Prevent warning by temporarily increasing model_max_length during evaluation
    orig_max_len = getattr(tokenizer, "model_max_length", None)
    tokenizer.model_max_length = 100_000_000

    try:
        with torch.no_grad():
            for item in dataset:
                text = item.get("text", "")
                if not text.strip():
                    continue

                tokens = tokenizer.encode(
                    text,
                    add_special_tokens=False
                )

                if tokenizer.eos_token_id is not None:
                    tokens.append(tokenizer.eos_token_id)

                for start in range(0, len(tokens), block_size):
                    chunk = tokens[start:start + block_size]

                    if len(chunk) < 2:
                        continue

                    input_ids = torch.tensor(
                        [chunk],
                        dtype=torch.long,
                        device=model.device
                    )

                    attention_mask = torch.ones_like(input_ids)

                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=input_ids
                    )

                    num_tokens = len(chunk) - 1

                    total_loss += outputs.loss.item() * num_tokens
                    total_tokens += num_tokens
    finally:
        if orig_max_len is not None:
            tokenizer.model_max_length = orig_max_len

    if total_tokens == 0:
        return float("inf")

    avg_nll = total_loss / total_tokens
    return math.exp(avg_nll)

def evaluate_qa_accuracy(model: Any, tokenizer: Any, probe_questions: List[Dict[str, Any]]) -> float:
    model.eval()
    correct = 0
    total = 0
    options = ["A", "B", "C", "D"]
    
    with torch.no_grad():
        for q in probe_questions:
            prompt = f"Question: {q['question']}\nAnswer:"
            inputs = tokenizer(prompt, return_tensors="pt")
            input_ids = inputs["input_ids"].to(model.device)
            attention_mask = inputs.get("attention_mask", torch.ones_like(input_ids)).to(model.device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            next_token_logits = outputs.logits[0, -1, :]
            next_token_probs = torch.softmax(next_token_logits, dim=-1)
            
            option_probs = {}
            for opt in options:
                opt_token_ids = tokenizer.encode(" " + opt, add_special_tokens=False)
                if len(opt_token_ids) > 0:
                    opt_token_id = opt_token_ids[-1]
                    option_probs[opt] = next_token_probs[opt_token_id].item()
                else:
                    option_probs[opt] = 0.0
            
            best_option = max(option_probs, key=option_probs.get)
            if best_option == q.get("answer"):
                correct += 1
            total += 1
            
    return (correct / total) * 100 if total > 0 else 0.0
