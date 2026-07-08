"""
Offline Pre-tokenization Step (s1.5)
- Stream-reads the parsed JSONL corpus line-by-line.
- Splits into 80% train and 20% validation.
- Writes validation texts directly to disk for Probe B (Perplexity).
- Tokenizes training documents and saves them to a flat NumPy array file on disk.
"""

import json
import os
from pathlib import Path
from typing import Optional

import numpy as np
from transformers import AutoTokenizer

from lib.utils.logger import get_logger, setup_logger
from lib.utils import PipelineConfig

logger = get_logger(__name__)


def run_pretokenization(
    cfg: PipelineConfig,
    val_ratio: float = 0.20,
) -> None:
    import sys
    setup_logger(
        f"{__name__}.{sys._getframe().f_code.co_name}",
        cfg.logging,
    )
    global logger
    logger = get_logger(f"{__name__}.{sys._getframe().f_code.co_name}")
    
    corpus_path = str(cfg.build.output_path)
    base_model_name = cfg.model.base_model_name
    output_bin_path = str(cfg.data.pretokenized_bin_path)
    ppl_corpus_path = str(cfg.data.ppl_corpus_path)

    logger.info(f"Starting pre-tokenization of {corpus_path}")
    logger.info(f"Base model tokenizer: {base_model_name}")
    logger.info(f"Output bin path: {output_bin_path}")
    logger.info(f"Validation ppl corpus path: {ppl_corpus_path}")

    if not os.path.exists(corpus_path):
        raise FileNotFoundError(f"Corpus file not found at: {corpus_path}")

    # Count lines first without loading them into memory to prevent OOM
    total_docs = 0
    with open(corpus_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                total_docs += 1

    if total_docs == 0:
        logger.warning(f"The corpus file {corpus_path} is empty. Nothing to process.")
        return

    val_size = max(2, int(total_docs * val_ratio))
    if total_docs <= val_size:
        val_size = max(1, total_docs // 2)

    train_size = total_docs - val_size
    logger.info(f"Corpus split: {train_size} training docs, {val_size} validation docs.")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    tokenizer.model_max_length = 100_000_000
    eos_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0

    # Read and process validation split streamingly, writing directly to ppl_corpus_path
    Path(ppl_corpus_path).parent.mkdir(parents=True, exist_ok=True)
    val_count = 0
    with open(ppl_corpus_path, "w", encoding="utf-8") as val_out:
        with open(corpus_path, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                if not line.strip():
                    continue
                # If we are in the validation split (last val_size docs)
                if idx >= train_size:
                    doc = json.loads(line)
                    text = doc.get("text", "").strip()
                    if text:
                        val_out.write(text + "\n")
                        val_count += 1

    logger.info(f"Successfully wrote {val_count} validation documents to {ppl_corpus_path}")

    # Tokenize training documents streamingly and write to a list/array
    train_tokens = []
    processed_train_docs = 0

    Path(output_bin_path).parent.mkdir(parents=True, exist_ok=True)

    with open(corpus_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if not line.strip():
                continue
            if idx >= train_size:
                # Validation docs are handled above, we can stop
                break

            doc = json.loads(line)
            text = doc.get("text", "")
            if not text.strip():
                continue

            tokens = tokenizer.encode(text, add_special_tokens=False)
            train_tokens.extend(tokens)
            train_tokens.append(eos_id)

            processed_train_docs += 1
            if processed_train_docs % 100 == 0:
                logger.info(f"Tokenized {processed_train_docs}/{train_size} training documents...")

    # Convert to NumPy array and save to disk
    token_arr = np.array(train_tokens, dtype=np.int32)
    np.save(output_bin_path, token_arr)
    logger.info(f"Successfully saved {len(token_arr):,} training tokens to {output_bin_path}")
