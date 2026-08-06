# Feature Spec: MinHash LSH Chunk Deduplication

## Objective

Integrate MinHash Locality-Sensitive Hashing (LSH) into `merge_corpus.py` to identify and eliminate duplicate or near-duplicate text chunks (e.g., overlapping journal abstracts, republished paper segments, or duplicated textbook sections across PubMed, PMC, and bioRxiv) before producing the final DAPT training corpus.

---

## Scope

This specification defines the additions and changes required to implement chunk-level MinHash LSH deduplication during the corpus merge step:
1. **Environment & Configuration**: Add `MINHASH_ENABLED`, `MINHASH_JACCARD_THRESHOLD`, `MINHASH_NUM_PERM`, `MINHASH_NGRAM_SIZE`, and `MINHASH_NUM_BANDS` options in `.env.common` and `lib/utils/config.py`.
2. **MinHash LSH Engine (`minhash_lsh.py`)**: Implement a zero-dependency Python module `lib/s1_build_corpus/minhash_lsh.py` that computes 128 64-bit MinHash permutations over word 5-grams, partitions signatures into 16 LSH bands of 8 rows each, and performs exact Jaccard verification for bucket collision candidates.
3. **Corpus Merging Integration (`merge_corpus.py`)**: Update `run_merge_corpus` to process streamed JSONL records through `MinHashLSHDeduplicator`, filtering duplicate chunks and reporting pre-/post-deduplication chunk and token metrics.
4. **Testing**: Add a unit test suite `tests/test_minhash_lsh.py` covering MinHash fingerprinting, band hashing, exact/near-duplicate detection, distinct document preservation, and full `merge_corpus` pipeline integration.

---

## Technical Specifications

### 1. Configuration (`lib/utils/config.py` & `.env.common`)
Add fields to `CorpusBuildConfig`:
- `minhash_enabled`: `bool` (default: `True`)
- `minhash_jaccard_threshold`: `float` (default: `0.85`)
- `minhash_num_perm`: `int` (default: `128`)
- `minhash_ngram_size`: `int` (default: `5`)
- `minhash_num_bands`: `int` (default: `16`)

### 2. MinHash LSH Engine (`lib/s1_build_corpus/minhash_lsh.py`)
- **Class `MinHashLSHDeduplicator`**:
  - `__init__(num_perm=128, num_bands=16, jaccard_threshold=0.85, ngram_size=5, seed=42)`: Precomputes linear hash parameters $(a_i, b_i) \pmod M$.
  - `_get_shingles(text)`: Converts text to lowercase, strips non-alphanumeric noise, extracts word $n$-grams.
  - `_compute_minhash(shingles)`: Computes the 128 MinHash signature vector $S = [h_1, \dots, h_{128}]$.
  - `is_duplicate_and_add(doc_id, text)`:
    1. Extracts shingles and computes MinHash signature.
    2. Identifies candidate document IDs from matching LSH band buckets.
    3. Computes exact Jaccard similarity $J(A, B) = \frac{|A \cap B|}{|A \cup B|}$ for candidate shingle sets.
    4. If $J \ge \text{jaccard\_threshold}$, returns `True` (duplicate detected, not added to index).
    5. If $J < \text{jaccard\_threshold}$, inserts chunk signature into band index and stores shingle set, returning `False` (unique chunk).

### 3. Pipeline Integration (`lib/s1_build_corpus/merge_corpus.py`)
In `run_merge_corpus(cfg)`:
- Instantiate `MinHashLSHDeduplicator` if `cfg.build.minhash_enabled` is `True`.
- Process each document chunk line-by-line.
- If deduplicator flags chunk as duplicate:
  - Increment `duplicate_count` and `dropped_tokens`.
  - Skip writing chunk to output.
- If chunk is unique:
  - Increment `merged_count` and `total_tokens`.
  - Re-index chunk `id` as `domain_doc_XXXXXX`.
  - Write record to `cfg.build.output_path`.
- Log detailed statistics (total chunks, unique chunks, dropped duplicate chunks, unique tokens, dropped duplicate tokens).
