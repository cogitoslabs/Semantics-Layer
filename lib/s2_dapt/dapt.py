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


def evaluate_perplexity(model: Any, tokenizer: Any, dataset: List[Dict[str, Any]]) -> float:
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    max_length = 512
    
    # Enable padding configuration
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    with torch.no_grad():
        for item in dataset:
            text = item.get("text", "")
            if not text.strip():
                continue
            # Tokenize the entire text without truncation
            inputs = tokenizer(text, return_tensors="pt")
            input_ids = inputs["input_ids"]
            
            seq_len = input_ids.size(1)
            if seq_len <= 1:
                continue
                
            # Split input_ids into chunks of size max_length
            for i in range(0, seq_len, max_length):
                chunk_ids = input_ids[:, i : i + max_length].to(model.device)
                
                if chunk_ids.size(1) <= 1:
                    continue
                    
                chunk_labels = chunk_ids.clone()
                chunk_attention_mask = torch.ones_like(chunk_ids).to(model.device)
                
                outputs = model(input_ids=chunk_ids, attention_mask=chunk_attention_mask, labels=chunk_labels)
                loss = outputs.loss
                num_tokens = chunk_ids.size(1) - 1
                total_loss += loss.item() * num_tokens
                total_tokens += num_tokens
            
    if total_tokens == 0:
        return float("inf")
    return math.exp(total_loss / total_tokens)


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
        
    model = AutoModelForCausalLM.from_pretrained(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Model loaded successfully on device: {device}")
    
    # Load datasets
    corpus_docs = load_jsonl(corpus_path)
    probe_questions = load_jsonl(probe_qa_path)
    
    if not corpus_docs:
        print(f"[Warning] Pretraining corpus at '{corpus_path}' is empty or not found.")
        return
    if not probe_questions:
        print(f"[Warning] Probe QA dataset at '{probe_qa_path}' is empty or not found.")
        
    # Split corpus into Train (75%) and Validation (25%)
    val_size = max(1, int(len(corpus_docs) * 0.25))
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
    train_chunks = []
    
    # Safely get EOS token ID
    eos_id = tokenizer.eos_token_id
    if eos_id is None or not isinstance(eos_id, int):
        eos_id = 0
        
    for doc in train_docs:
        text = doc.get("text", "")
        if not text.strip():
            continue
        tokens = tokenizer.encode(text, add_special_tokens=False)
        train_chunks.extend(tokens)
        train_chunks.append(eos_id)
        
    # Split training tokens into blocks of 512 tokens
    block_size = 512
    tokenized_train_blocks = []
    for i in range(0, len(train_chunks) - block_size + 1, block_size):
        tokenized_train_blocks.append(train_chunks[i : i + block_size])
        
    # Fallback to whatever tokens we have if the corpus is too small to form a full block
    if not tokenized_train_blocks and train_chunks:
        tokenized_train_blocks.append(train_chunks)
        
    print(f"Created {len(tokenized_train_blocks)} training blocks of size {block_size}.")
    
    # Run CPT / DAPT Training
    print("\n=== STARTING CONTINUED PRETRAINING (CPT/DAPT) ===")
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    
    # Batch inputs
    batches = []
    for i in range(0, len(tokenized_train_blocks), batch_size):
        batches.append(tokenized_train_blocks[i : i + batch_size])
        
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        num_batches = 0
        
        for batch in batches:
            optimizer.zero_grad()
            
            # Since blocks might be of different sizes in the fallback/partial cases,
            # we pad dynamically to max length in the batch.
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
                
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            
            # Gradient clipping to stabilize training
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
            
        avg_loss = epoch_loss / num_batches if num_batches > 0 else 0.0
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
    print("\n=== DAPT IMPROVEMENT SUMMARY ===")
    ppl_diff = initial_ppl - final_ppl
    qa_diff = final_qa_acc - initial_qa_acc
    print(f"Perplexity: {initial_ppl:.4f} -> {final_ppl:.4f} (Change: {-ppl_diff/initial_ppl * 100:.2f}%)")
    print(f"QA Accuracy: {initial_qa_acc:.2f}% -> {final_qa_acc:.2f}% (Change: {qa_diff:+.2f}%)")
