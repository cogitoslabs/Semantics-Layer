import os
import json
from pathlib import Path
from lib.utils import get_logger, PipelineConfig

logger = get_logger(__name__)

def run_merge_corpus(cfg: PipelineConfig) -> None:
    """
    Scans the directory cfg.data.dapt_in_dir for all .jsonl files,
    merges them line-by-line, re-indexes the document IDs sequentially,
    and writes the unified corpus to cfg.build.output_path.
    """
    import sys
    from lib.utils import setup_logger
    setup_logger(
        f"{__name__}.{sys._getframe().f_code.co_name}",
        cfg.logging,
    )
    global logger
    logger = get_logger(f"{__name__}.{sys._getframe().f_code.co_name}")

    in_dir = Path(cfg.data.dapt_in_dir)
    output_path = Path(cfg.build.output_path)
    
    logger.info("=============================================================")
    logger.info("   MERGING CORPUS: DAPT/IN/*.JSONL -> UNIFIED CORPUS         ")
    logger.info("=============================================================")
    logger.info(f"Input directory : {in_dir}")
    logger.info(f"Output file     : {output_path}")
    
    if not in_dir.exists():
        logger.warning(f"Input directory {in_dir} does not exist. Nothing to merge.")
        return
        
    jsonl_files = sorted(list(in_dir.glob("*.jsonl")))
    if not jsonl_files:
        logger.warning(f"No .jsonl files found in {in_dir}. Nothing to merge.")
        return
        
    logger.info(f"Found {len(jsonl_files)} files to merge:")
    for f in jsonl_files:
        logger.info(f"  - {f.name} ({f.stat().st_size / 1024 / 1024:.2f} MB)")
        
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    deduplicator = None
    if cfg.build.minhash_enabled:
        from .minhash_lsh import MinHashLSHDeduplicator
        deduplicator = MinHashLSHDeduplicator(
            num_perm=cfg.build.minhash_num_perm,
            num_bands=cfg.build.minhash_num_bands,
            jaccard_threshold=cfg.build.minhash_jaccard_threshold,
            ngram_size=cfg.build.minhash_ngram_size,
        )
        logger.info(
            f"MinHash LSH Deduplication Enabled | Threshold={cfg.build.minhash_jaccard_threshold} | "
            f"Bands={cfg.build.minhash_num_bands} | Permutations={cfg.build.minhash_num_perm} | N-Gram={cfg.build.minhash_ngram_size}"
        )

    total_chunks_processed = 0
    merged_count = 0
    duplicate_count = 0
    total_tokens = 0
    dropped_tokens = 0
    
    with open(output_path, "w", encoding="utf-8") as out_f:
        for file_path in jsonl_files:
            logger.info(f"Processing: {file_path.name}...")
            file_doc_count = 0
            file_dup_count = 0
            with open(file_path, "r", encoding="utf-8") as in_f:
                for line in in_f:
                    if not line.strip():
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError as e:
                        logger.warning(f"Skipping invalid JSON line in {file_path.name}: {e}")
                        continue
                        
                    text = record.get("text", "")
                    source_file = record.get("source_file", file_path.name)
                    chunk_id = record.get("chunk_id", None)
                    page_range = record.get("page_range", None)
                    token_count = record.get("token_count", 0)
                    total_chunks_processed += 1

                    if deduplicator is not None:
                        cand_id = f"{source_file}:{chunk_id}:{merged_count + duplicate_count}"
                        if deduplicator.is_duplicate_and_add(cand_id, text):
                            duplicate_count += 1
                            file_dup_count += 1
                            dropped_tokens += token_count
                            continue

                    merged_record = {
                        "id": f"domain_doc_{merged_count:06d}",
                        "source_file": source_file,
                        "chunk_id": chunk_id,
                        "page_range": page_range,
                        "text": text,
                        "token_count": token_count
                    }
                    
                    out_f.write(json.dumps(merged_record) + "\n")
                    merged_count += 1
                    file_doc_count += 1
                    total_tokens += token_count
            logger.info(f"  Merged {file_doc_count} unique documents from {file_path.name} ({file_dup_count} duplicates dropped)")
            
    logger.info("=============================================================")
    logger.info("   MERGE & DEDUPLICATION SUMMARY                             ")
    logger.info("=============================================================")
    logger.info(f"Total Chunks Processed : {total_chunks_processed:,}")
    logger.info(f"Unique Chunks Merged   : {merged_count:,}")
    logger.info(f"Duplicate Chunks Dropped: {duplicate_count:,}")
    logger.info(f"Unique Tokens Kept     : {total_tokens:,}")
    logger.info(f"Duplicate Tokens Dropped: {dropped_tokens:,}")
    logger.info(f"Unified Corpus Saved To: {output_path}")

if __name__ == "__main__":
    from lib.utils import PipelineConfig
    cfg = PipelineConfig()
    run_merge_corpus(cfg)
