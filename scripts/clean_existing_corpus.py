import os
import sys
import json
import tempfile
import tiktoken
from pathlib import Path

# Add root directory to python path for imports
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from lib.utils import clean_corpus_text
from lib.s1_build_corpus.worker import MIN_CONTENT_LENGTH


def main():
    corpus_path = ROOT_DIR / "data" / "dapt" / "domain_dapt_corpus.jsonl"
    if not corpus_path.exists():
        print(f"Error: Corpus file not found at {corpus_path}")
        sys.exit(1)
        
    print(f"Streaming and cleaning corpus from: {corpus_path}")
    
    # Load tiktoken tokenizer for token counting
    tokenizer = tiktoken.get_encoding("cl100k_base")
    
    orig_total_tokens = 0
    clean_total_tokens = 0
    orig_docs = 0
    clean_docs = 0
    skipped_docs = 0
    
    # Create a temporary file to write cleaned data
    temp_dir = corpus_path.parent
    fd, temp_path_str = tempfile.mkstemp(dir=temp_dir, prefix="clean_", suffix=".jsonl")
    temp_path = Path(temp_path_str)
    
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as out:
            with open(corpus_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if not line.strip():
                        continue
                    orig_docs += 1
                    
                    doc = json.loads(line)
                    text = doc.get("text", "")
                    orig_token_count = doc.get("token_count", 0)
                    orig_total_tokens += orig_token_count
                    
                    # Clean the text
                    cleaned_text = clean_corpus_text(text)
                    
                    # If content is too short after cleaning, skip it
                    if not cleaned_text or len(cleaned_text.strip()) < MIN_CONTENT_LENGTH:
                        skipped_docs += 1
                        continue
                        
                    # Recompute token count
                    new_token_count = len(tokenizer.encode(cleaned_text))
                    clean_total_tokens += new_token_count
                    
                    # Write updated record
                    doc["text"] = cleaned_text
                    doc["token_count"] = new_token_count
                    out.write(json.dumps(doc) + "\n")
                    clean_docs += 1
                    
        # Replace the original corpus file with the cleaned temporary file
        if corpus_path.exists():
            corpus_path.unlink()
        temp_path.rename(corpus_path)
        
        print("\n" + "=" * 50)
        print("Corpus Cleaning Completed Successfully!")
        print("=" * 50)
        print(f"Original documents   : {orig_docs}")
        print(f"Cleaned documents    : {clean_docs} (Skipped {skipped_docs} short chunks)")
        print(f"Original tokens      : {orig_total_tokens:,}")
        print(f"Cleaned tokens       : {clean_total_tokens:,}")
        print(f"Token reduction      : {orig_total_tokens - clean_total_tokens:,} tokens "
              f"({(orig_total_tokens - clean_total_tokens)/orig_total_tokens * 100:.2f}%)")
        print("=" * 50 + "\n")
        
    except Exception as e:
        print(f"An error occurred during cleaning: {e}")
        if temp_path.exists():
            temp_path.unlink()
        sys.exit(1)


if __name__ == "__main__":
    main()
