import os
import sys
import shutil
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer

def main():
    target_tokens = 614400  # 600 blocks of 1024 tokens = 614.4K tokens
    model_name = "HuggingFaceTB/SmolLM2-135M"
    dataset_name = "HuggingFaceFW/fineweb-edu"
    data_dir = "data/dapt"
    
    fineweb_path = os.path.join(data_dir, "fineweb_token.npy")
    train_tokens_path = os.path.join(data_dir, "train_tokens.npy")
    train_tokens_backup_path = os.path.join(data_dir, "train_tokens_old.npy")
    
    print("=============================================================")
    print("   PREPARE REPLAY DATA: FINEWEB-EDU -> DAPT TOKENS")
    print("=============================================================")
    
    # 1. Load tokenizer
    print(f"Loading tokenizer for: {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    eos_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0
    print(f"Loaded tokenizer. EOS token ID: {eos_token_id}")
    
    # 2. Download and stream FineWeb-Edu
    print(f"Streaming dataset from Hugging Face: {dataset_name} (sample-10BT)...")
    dataset = load_dataset(dataset_name, name="sample-10BT", split="train", streaming=True)
    
    print("Tokenizing FineWeb-Edu text to get 600K tokens...")
    tokens = []
    processed_docs = 0
    
    for row in dataset:
        text = row.get("text", "")
        if not text:
            continue
            
        doc_tokens = tokenizer.encode(text)
        tokens.extend(doc_tokens)
        tokens.append(eos_token_id)
        
        processed_docs += 1
        if len(tokens) >= target_tokens:
            break
            
        if processed_docs % 50 == 0:
            print(f"  Processed {processed_docs} documents, collected {len(tokens):,} tokens...")
            
    # Truncate to exact target count
    tokens = tokens[:target_tokens]
    print(f"Successfully collected exactly {len(tokens):,} tokens from {processed_docs} documents.")
    
    # Save as numpy array
    fineweb_tokens_arr = np.array(tokens, dtype=np.int32)
    os.makedirs(data_dir, exist_ok=True)
    np.save(fineweb_path, fineweb_tokens_arr)
    print(f"Saved tokenized replay data to: {fineweb_path}")
    
    # 3. Create backup of train_tokens.npy
    if os.path.exists(train_tokens_path):
        print(f"Creating backup of original train_tokens.npy -> {train_tokens_backup_path}...")
        shutil.copy2(train_tokens_path, train_tokens_backup_path)
        # Also create a file named train_token_old.npy as requested by the user explicitly
        user_backup_path = os.path.join(data_dir, "train_token_old.npy")
        shutil.copy2(train_tokens_path, user_backup_path)
        print("Backup copies created successfully.")
    else:
        print(f"Warning: Original train_tokens.npy not found at {train_tokens_path}. Merge will skip concatenation.")
        return
        
    # 4. Merge fineweb_token.npy into train_tokens.npy
    print("Merging replay tokens into original pretraining corpus...")
    original_tokens = np.load(train_tokens_backup_path)
    print(f"  Original tokens: {len(original_tokens):,} tokens ({len(original_tokens)//1024} blocks)")
    print(f"  Replay tokens  : {len(fineweb_tokens_arr):,} tokens ({len(fineweb_tokens_arr)//1024} blocks)")
    
    merged_tokens = np.concatenate([original_tokens, fineweb_tokens_arr])
    print(f"  Merged tokens  : {len(merged_tokens):,} tokens ({len(merged_tokens)//1024} blocks)")
    
    np.save(train_tokens_path, merged_tokens)
    print(f"Successfully saved merged corpus to: {train_tokens_path}")
    print("Replay data preparation complete.")

if __name__ == "__main__":
    main()
