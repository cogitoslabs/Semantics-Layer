import json
import logging
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Generator, List
from transformers import AutoTokenizer
from lib.utils import PipelineConfig

logger = logging.getLogger(__name__)

@dataclass
class Chunk:
    chunk_id: str      # "{doc_id}_{chunk_index}"
    doc_id: str
    doc_type: str      # "long_form" | "abstract"
    text: str
    token_count: int


def chunk_document(doc_id: str, text: str, doc_type: str, tokenizer, chunk_size: int, overlap_size: int) -> List[Chunk]:
    """Tokenize and split document text into chunks with specified size and overlap."""
    tokens = tokenizer.encode(text, add_special_tokens=False)
    if not tokens:
        return []

    chunks = []
    step = chunk_size - overlap_size
    if step <= 0:
        step = chunk_size

    chunk_index = 0
    start_idx = 0

    while start_idx < len(tokens):
        end_idx = min(start_idx + chunk_size, len(tokens))
        chunk_tokens = tokens[start_idx:end_idx]
        chunk_text = tokenizer.decode(chunk_tokens, skip_special_tokens=True)

        chunks.append(Chunk(
            chunk_id=f"{doc_id}_{chunk_index}",
            doc_id=doc_id,
            doc_type=doc_type,
            text=chunk_text,
            token_count=len(chunk_tokens)
        ))

        chunk_index += 1
        if end_idx == len(tokens):
            break
        start_idx += step

    return chunks


def run_chunking(cfg: PipelineConfig) -> List[Chunk]:
    """Read the retrieval corpus, chunk all documents, and save the chunks to chunks.jsonl."""
    logger.info(f"Loading tokenizer from model: {cfg.model.base_model_name}")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.base_model_name)

    corpus_path = Path(cfg.rad.retrieval_corpus_path)
    chunks_path = Path(cfg.rad.chunks_path)

    if not corpus_path.exists():
        raise FileNotFoundError(f"Retrieval corpus file not found at {corpus_path}")

    chunks_path.parent.mkdir(parents=True, exist_ok=True)

    all_chunks = []
    logger.info(f"Processing corpus from {corpus_path}")

    with open(corpus_path, "r", encoding="utf-8") as f_in, open(chunks_path, "w", encoding="utf-8") as f_out:
        for idx, line in enumerate(f_in):
            if not line.strip():
                continue
            record = json.loads(line)
            text = record.get("text", "")
            doc_type = record.get("doc_type", "long_form")
            doc_id = record.get("doc_id", f"doc_{idx}")

            if doc_type == "long_form":
                chunk_size = cfg.rad.long_form_chunk_tokens
                overlap = cfg.rad.long_form_overlap_tokens
            elif doc_type == "abstract":
                chunk_size = cfg.rad.abstract_chunk_tokens
                overlap = cfg.rad.abstract_overlap_tokens
            else:
                logger.warning(f"Unknown doc_type {doc_type!r} for doc {doc_id}, defaulting to long_form params")
                chunk_size = cfg.rad.long_form_chunk_tokens
                overlap = cfg.rad.long_form_overlap_tokens

            doc_chunks = chunk_document(doc_id, text, doc_type, tokenizer, chunk_size, overlap)
            for chunk in doc_chunks:
                f_out.write(json.dumps(asdict(chunk)) + "\n")
                all_chunks.append(chunk)

    logger.info(f"Successfully chunked corpus: generated {len(all_chunks)} chunks and saved to {chunks_path}")
    return all_chunks
