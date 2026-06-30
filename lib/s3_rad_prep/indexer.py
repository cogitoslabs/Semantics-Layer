import json
import logging
import time
from pathlib import Path
from typing import List, Optional
import numpy as np
import torch
import faiss
from transformers import AutoTokenizer, AutoModel
from lib.utils import PipelineConfig
from lib.s3_rad_prep.chunker import Chunk

logger = logging.getLogger(__name__)

class DenseEmbedder:
    def __init__(self, model_key: str, device: Optional[str] = None):
        if model_key == "biolinkbert" or model_key == "hybrid":
            self.model_name = "michiyasunaga/BioLinkBERT-large"
        elif model_key == "pubmedbert":
            self.model_name = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract"
        else:
            self.model_name = model_key

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Initializing DenseEmbedder using {self.model_name} on device {self.device}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name).to(self.device)
        self.model.eval()

    def embed_batch(self, texts: List[str]) -> np.ndarray:
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            attention_mask = inputs["attention_mask"].unsqueeze(-1)
            token_embeddings = outputs.last_hidden_state
            sum_embeddings = (token_embeddings * attention_mask).sum(dim=1)
            sum_mask = attention_mask.sum(dim=1)
            sum_mask = torch.clamp(sum_mask, min=1e-9)
            embeddings = sum_embeddings / sum_mask

        embeddings_np = embeddings.cpu().numpy()
        norms = np.linalg.norm(embeddings_np, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1e-9, norms)
        return embeddings_np / norms


def run_indexing(cfg: PipelineConfig, chunks: Optional[List[Chunk]] = None) -> None:
    """Embed all chunks and construct/save a FAISS IndexFlatIP."""
    chunks_path = Path(cfg.rad.chunks_path)
    index_dir = Path(cfg.rad.index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)

    if chunks is None:
        if not chunks_path.exists():
            raise FileNotFoundError(f"Chunks file not found at {chunks_path}. Run chunking first.")
        logger.info(f"Loading chunks from {chunks_path}")
        chunks = []
        with open(chunks_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    chunks.append(Chunk(**data))

    if not chunks:
        logger.warning("No chunks to index.")
        return

    logger.info(f"Embedding {len(chunks)} chunks with batch size {cfg.rad.embed_batch_size}")
    embedder = DenseEmbedder(cfg.rad.embedding_model)

    all_embeddings = []
    texts = [chunk.text for chunk in chunks]
    batch_size = cfg.rad.embed_batch_size

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        batch_embeddings = embedder.embed_batch(batch_texts)
        all_embeddings.append(batch_embeddings)

    embeddings_matrix = np.vstack(all_embeddings).astype("float32")

    dimension = embeddings_matrix.shape[1]
    logger.info(f"Building FAISS IndexFlatIP with dimension {dimension}")
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings_matrix)

    index_file = index_dir / "index.faiss"
    logger.info(f"Saving FAISS index to {index_file}")
    faiss.write_index(index, str(index_file))

    # Save parallel chunks metadata (no text, just IDs and doc_type)
    metadata_file = index_dir / "chunks_metadata.jsonl"
    logger.info(f"Saving chunk metadata to {metadata_file}")
    with open(metadata_file, "w", encoding="utf-8") as f:
        for chunk in chunks:
            meta = {
                "chunk_id": chunk.chunk_id,
                "doc_id": chunk.doc_id,
                "doc_type": chunk.doc_type,
                "token_count": chunk.token_count
            }
            f.write(json.dumps(meta) + "\n")

    # Save index manifest
    manifest_file = index_dir / "index_manifest.json"
    total_tokens = sum(chunk.token_count for chunk in chunks)
    manifest = {
        "model": cfg.rad.embedding_model,
        "chunk_count": len(chunks),
        "total_tokens": total_tokens,
        "build_timestamp": time.time()
    }
    logger.info(f"Saving manifest to {manifest_file}")
    with open(manifest_file, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    logger.info("Indexing complete.")
