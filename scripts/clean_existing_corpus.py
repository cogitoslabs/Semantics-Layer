import os
import sys
import json
import tempfile
import tiktoken
from pathlib import Path
from collections import defaultdict, Counter

# Add root directory to python path for imports
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from lib.utils import clean_corpus_text
from lib.s1_build_corpus.worker import MIN_CONTENT_LENGTH


def main():
    if len(sys.argv) > 1:
        corpus_path = Path(sys.argv[1]).resolve()
    else:
        in_path = ROOT_DIR / "data" / "dapt" / "in" / "domain_dapt_corpus.jsonl"
        root_path = ROOT_DIR / "data" / "dapt" / "domain_dapt_corpus.jsonl"
        if in_path.exists():
            corpus_path = in_path
        elif root_path.exists():
            corpus_path = root_path
        else:
            print(f"Error: Corpus file not found at {in_path} or {root_path}")
            sys.exit(1)
        
    print(f"Streaming and cleaning corpus from: {corpus_path}")
    
    # Load tiktoken tokenizer for token counting
    tokenizer = tiktoken.get_encoding("cl100k_base")
    
    orig_total_tokens = 0
    clean_total_tokens = 0
    orig_docs = 0
    
    docs_by_source = defaultdict(list)
    
    # 1. Load all records into memory and group by source_file
    with open(corpus_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            orig_docs += 1
            doc = json.loads(line)
            orig_token_count = doc.get("token_count", 0)
            orig_total_tokens += orig_token_count
            
            # Run initial cleaning pass (handles dehyphenation, reference stripping, metadata removal, and index/bib check)
            cleaned_text = clean_corpus_text(doc.get("text", ""), doc.get("source_file", ""))
            if cleaned_text.strip() and len(cleaned_text.strip()) >= MIN_CONTENT_LENGTH:
                doc["text"] = cleaned_text
                docs_by_source[doc.get("source_file", "")].append(doc)
                
    # Create a temporary file to write cleaned data
    temp_dir = corpus_path.parent
    fd, temp_path_str = tempfile.mkstemp(dir=temp_dir, prefix="clean_", suffix=".jsonl")
    temp_path = Path(temp_path_str)
    
    clean_docs = 0
    
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as out:
            # 2. Perform document-level running header/footer stripping for each source group
            for source_file, group_docs in docs_by_source.items():
                if len(group_docs) >= 3:
                    # Count line frequencies across all docs in this source group
                    line_counts = Counter()
                    for doc in group_docs:
                        for line in doc["text"].split('\n'):
                            stripped = line.strip()
                            if stripped and len(stripped) < 80:
                                line_counts[stripped] += 1
                                
                    # Identify boilerplates
                    num_docs = len(group_docs)
                    threshold = min(15, max(3, int(num_docs * 0.10)))
                    boilerplates = {line for line, count in line_counts.items() if count >= threshold}
                    
                    if boilerplates:
                        print(f"Source: {source_file} - Identified {len(boilerplates)} running headers/footers (threshold={threshold})")
                        # Strip lines from the documents in the group
                        for doc in group_docs:
                            lines = doc["text"].split('\n')
                            cleaned_lines = [l for l in lines if l.strip() not in boilerplates]
                            doc["text"] = '\n'.join(cleaned_lines)
                            
                # Recompute token count and write valid documents
                for doc in group_docs:
                    cleaned_text = doc["text"]
                    if cleaned_text.strip() and len(cleaned_text.strip()) >= MIN_CONTENT_LENGTH:
                        new_token_count = len(tokenizer.encode(cleaned_text))
                        clean_total_tokens += new_token_count
                        doc["token_count"] = new_token_count
                        out.write(json.dumps(doc) + "\n")
                        clean_docs += 1
                        
        # Replace the original corpus file with the cleaned temporary file
        if corpus_path.exists():
            corpus_path.unlink()
        temp_path.rename(corpus_path)
        
        # Clean the perplexity held-out corpus in-place if it exists
        ppl_path = ROOT_DIR / "evals" / "dapt" / "ppl_held_out.txt"
        if ppl_path.exists():
            print(f"Cleaning perplexity held-out corpus at: {ppl_path}")
            with open(ppl_path, "r", encoding="utf-8") as f:
                ppl_content = f.read()
            cleaned_ppl = clean_corpus_text(ppl_content)
            with open(ppl_path, "w", encoding="utf-8") as f:
                f.write(cleaned_ppl)
            print("Perplexity held-out corpus cleaned successfully!")
            
        skipped_docs = orig_docs - clean_docs
        
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
