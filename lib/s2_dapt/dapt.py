import json
import math
import os
from typing import Any, Dict, List
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(file_path):
        return []
    with open(file_path, "r", encoding="utf-8") as f:
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
            # We construct the question prompt ending with colon (no trailing space)
            prompt = f"Question: {q['question']}\nAnswer:"
            inputs = tokenizer(prompt, return_tensors="pt")
            input_ids = inputs["input_ids"].to(model.device)
            attention_mask = inputs.get("attention_mask", torch.ones_like(input_ids)).to(model.device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            # Next token logits is the last position of the output sequence
            next_token_logits = outputs.logits[0, -1, :]
            next_token_probs = torch.softmax(next_token_logits, dim=-1)
            
            option_probs = {}
            for opt in options:
                # Tokenize the option with a leading space (e.g., " A")
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


def prepare_training_blocks(tokenizer: Any, train_docs: List[Dict[str, Any]], block_size: int = 512) -> List[List[int]]:
    train_chunks = []
    
    eos_id = tokenizer.eos_token_id
    if eos_id is None or not isinstance(eos_id, int):
        eos_id = 0
        
    # Prevent warning by temporarily increasing model_max_length during tokenization
    orig_max_len = getattr(tokenizer, "model_max_length", None)
    tokenizer.model_max_length = 100_000_000

    try:
        for doc in train_docs:
            text = doc.get("text", "")
            if not text.strip():
                continue
            tokens = tokenizer.encode(text, add_special_tokens=False)
            train_chunks.extend(tokens)
            train_chunks.append(eos_id)
    finally:
        if orig_max_len is not None:
            tokenizer.model_max_length = orig_max_len
        
    tokenized_train_blocks = []
    for i in range(0, len(train_chunks) - block_size + 1, block_size):
        tokenized_train_blocks.append(train_chunks[i : i + block_size])
        
    if not tokenized_train_blocks and train_chunks:
        tokenized_train_blocks.append(train_chunks)
        
    return tokenized_train_blocks


def train_epoch(
    model: Any,
    tokenizer: Any,
    optimizer: Any,
    batches: List[List[List[int]]],
    device: torch.device,
    eos_id: int
) -> float:
    model.train()
    epoch_loss = 0.0
    num_batches = 0
    
    use_cuda = device.type == "cuda"
    dtype = torch.bfloat16 if (use_cuda and torch.cuda.is_bf16_supported()) else torch.float16
    
    if use_cuda and dtype == torch.float16:
        if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
            scaler = torch.amp.GradScaler("cuda")
        else:
            scaler = torch.cuda.amp.GradScaler()
    else:
        scaler = None

    if use_cuda:
        if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
            autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=dtype)
        else:
            autocast_ctx = torch.cuda.amp.autocast(dtype=dtype)
    else:
        # Dummy context manager for CPU
        class DummyCtx:
            def __enter__(self): pass
            def __exit__(self, exc_type, exc_val, exc_tb): pass
        autocast_ctx = DummyCtx()
        
    for batch in batches:
        optimizer.zero_grad()
        
        max_batch_len = max(len(block) for block in batch)
        
        pad_id = tokenizer.pad_token_id
        if pad_id is None or not isinstance(pad_id, int):
            pad_id = eos_id
            
        input_ids_list = []
        attention_mask_list = []
        for block in batch:
            pad_len = max_batch_len - len(block)
            padded_block = block + [pad_id] * pad_len
            input_ids_list.append(padded_block)
            attention_mask_list.append([1] * len(block) + [0] * pad_len)
            
        input_ids = torch.tensor(input_ids_list, dtype=torch.long).to(device)
        attention_mask = torch.tensor(attention_mask_list, dtype=torch.long).to(device)
        
        if input_ids.size(1) <= 1:
            continue
            
        labels = input_ids.clone()
        if attention_mask is not None:
            labels[attention_mask == 0] = -100
            
        with autocast_ctx:
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
        epoch_loss += loss.item()
        num_batches += 1
        
    return epoch_loss / num_batches if num_batches > 0 else 0.0


def log_improvement_summary(
    initial_ppl: float,
    initial_qa_acc: float,
    final_ppl: float,
    final_qa_acc: float
) -> None:
    print("\n=== DAPT IMPROVEMENT SUMMARY ===")
    ppl_diff = initial_ppl - final_ppl
    qa_diff = final_qa_acc - initial_qa_acc
    print(f"Perplexity: {initial_ppl:.4f} -> {final_ppl:.4f} (Change: {-ppl_diff/initial_ppl * 100:.2f}%)")
    print(f"QA Accuracy: {initial_qa_acc:.2f}% -> {final_qa_acc:.2f}% (Change: {qa_diff:+.2f}%)")


def run_dapt_pipeline(
    model_name: str,
    corpus_path: str,
    probe_qa_path: str,
    epochs: int = 3,
    lr: float = 5e-5,
    batch_size: int = 2,
    output_dir: str = "./outputs/dapt_model"
) -> None:
    print(f"Loading base model and tokenizer: {model_name}")
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Enable mixed-precision loading and optimized attention on GPU
    if device.type == "cuda":
        torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        attn_implementation = "sdpa"
    else:
        torch_dtype = torch.float32
        attn_implementation = "eager"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        attn_implementation=attn_implementation
    )
    model.to(device)
    print(f"Model loaded successfully on device: {device} (dtype: {torch_dtype}, attn: {attn_implementation})")
    
    # Load datasets
    corpus_docs = load_jsonl(corpus_path)
    probe_questions = load_jsonl(probe_qa_path)
    
    if not corpus_docs:
        print(f"[Warning] Pretraining corpus at '{corpus_path}' is empty or not found.")
        return
    if not probe_questions:
        print(f"[Warning] Probe QA dataset at '{probe_qa_path}' is empty or not found.")
        
    # Split corpus into Train (80%) and Validation (20%)
    val_size = max(2, int(len(corpus_docs) * 0.20))
    train_docs = corpus_docs[:-val_size]
    val_docs = corpus_docs[-val_size:]
    print(f"Corpus split: {len(train_docs)} training docs, {len(val_docs)} validation docs.")
    
    # Evaluate BEFORE CPT
    print("\n=== EVALUATION BEFORE CPT ===")
    initial_ppl = evaluate_perplexity(model, tokenizer, val_docs)
    initial_qa_acc = evaluate_qa_accuracy(model, tokenizer, probe_questions)
    print(f"Initial Perplexity: {initial_ppl:.4f}")
    print(f"Initial QA Accuracy: {initial_qa_acc:.2f}%")
    
    # Prepare training chunks by tokenizing the entire text of each document without truncation
    print("Tokenizing and chunking training documents...")
    tokenized_train_blocks = prepare_training_blocks(tokenizer, train_docs)
    print(f"Created {len(tokenized_train_blocks)} training blocks of size 512.")
    
    # Run CPT / DAPT Training
    print("\n=== STARTING CONTINUED PRETRAINING (CPT/DAPT) ===")
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    
    # Batch inputs
    batches = []
    for i in range(0, len(tokenized_train_blocks), batch_size):
        batches.append(tokenized_train_blocks[i : i + batch_size])
        
    eos_id = tokenizer.eos_token_id
    if eos_id is None or not isinstance(eos_id, int):
        eos_id = 0
        
    for epoch in range(epochs):
        avg_loss = train_epoch(model, tokenizer, optimizer, batches, device, eos_id)
        print(f"Epoch {epoch+1}/{epochs} | Average Loss: {avg_loss:.4f}")
        
    # Save model and tokenizer
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Trained model saved to: {output_dir}")
    
    # Evaluate AFTER CPT
    print("\n=== EVALUATION AFTER CPT ===")
    final_ppl = evaluate_perplexity(model, tokenizer, val_docs)
    final_qa_acc = evaluate_qa_accuracy(model, tokenizer, probe_questions)
    print(f"Final Perplexity: {final_ppl:.4f}")
    print(f"Final QA Accuracy: {final_qa_acc:.2f}%")
    
    # Comparison Summary
    log_improvement_summary(initial_ppl, initial_qa_acc, final_ppl, final_qa_acc)
