import os
import json
from pathlib import Path
from datasets import load_dataset
from transformers import AutoTokenizer
from lib.utils import get_logger, PipelineConfig

logger = get_logger(__name__)

def run_replay_corpus(cfg: PipelineConfig) -> None:
    """
    Streams FineWeb-Edu documents from HuggingFace, counts tokens to gather
    exactly ~600 blocks (~614.4K tokens), and saves the raw text documents
    to dapt/in/fineweb_replay.jsonl for downstream merging and pre-tokenization.
    """
    # 1. Output location: cfg.data.dapt_in_dir / "fineweb_replay.jsonl"
    out_dir = Path(cfg.data.dapt_in_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = out_dir / "fineweb_replay.jsonl"
    
    target_tokens = 614400  # 600 blocks of 1024 tokens = 614.4K tokens
    model_name = cfg.model.base_model_name
    dataset_name = "HuggingFaceFW/fineweb-edu"
    
    logger.info("=============================================================")
    logger.info("   REPLAY CORPUS GENERATION: FINEWEB-EDU -> RAW JSONL        ")
    logger.info("=============================================================")
    logger.info(f"Target tokens  : {target_tokens:,}")
    logger.info(f"Output file    : {output_path}")
    logger.info(f"Base model     : {model_name}")
    
    # 1. Load tokenizer
    logger.info(f"Loading tokenizer for: {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # 2. Download and stream FineWeb-Edu
    logger.info(f"Streaming dataset from Hugging Face: {dataset_name} (sample-10BT)...")
    dataset = load_dataset(dataset_name, name="sample-10BT", split="train", streaming=True)
    
    tokens_accumulated = 0
    doc_count = 0
    
    with open(output_path, "w", encoding="utf-8") as out_f:
        for row in dataset:
            text = row.get("text", "")
            if not text or not text.strip():
                continue
                
            # Encode text to count tokens
            doc_tokens = tokenizer.encode(text, add_special_tokens=False)
            doc_tokens_count = len(doc_tokens)
            
            if doc_tokens_count == 0:
                continue
                
            # Format and write JSONL record
            record = {
                "id": f"replay_doc_{doc_count:06d}",
                "source_file": "fineweb-edu",
                "text": text,
                "token_count": doc_tokens_count
            }
            out_f.write(json.dumps(record) + "\n")
            
            tokens_accumulated += doc_tokens_count
            doc_count += 1
            
            if tokens_accumulated >= target_tokens:
                break
                
            if doc_count % 50 == 0:
                logger.info(f"  Processed {doc_count} documents, collected {tokens_accumulated:,} tokens...")
                
    logger.info(f"Successfully wrote {doc_count} replay documents ({tokens_accumulated:,} tokens) to: {output_path}")

if __name__ == "__main__":
    # Test execution
    from lib.utils import PipelineConfig
    cfg = PipelineConfig()
    run_replay_corpus(cfg)
