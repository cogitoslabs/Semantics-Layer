<!-- Goals -->
**Hybrid retrieval (Dense + Sparse) & Cross-Encoder Reranking strategy**  
- **Dense Embedding**: Added support for **BGE-Large** (`BAAI/bge-large-en-v1.5`) alongside BioLinkBERT and PubMedBERT.
- **Two-Stage Retrieval Pipeline**:
  - **Stage 1 (Candidate Retrieval)**: Dense (BGE-Large) + Sparse (BM25Okapi) candidate retrieval (top-$N$ candidate pool, default 50).
  - **Stage 2 (Cross-Encoder Reranking)**: Rerank top candidates using **Cross-Encoder** (`BAAI/bge-reranker-large`).
  - **Stage 3 (Relevance Gate & Routing)**: Apply relevance threshold gating ($\ge 0.65$) on normalized reranker scores and route low-context queries ($<2$ chunks) to the no-retrieval track.

<!-- Notes -->
Feature Specification: [hybrid-retrieval-reranking.md](file:///e:/Projects/cnd/Semantics/context/feature-specs/hybrid-retrieval-reranking.md)

