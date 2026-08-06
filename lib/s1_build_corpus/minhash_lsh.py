"""
Zero-dependency MinHash Locality-Sensitive Hashing (LSH) deduplicator
for text chunk deduplication.
"""

import hashlib
import random
import re
from collections import defaultdict
from typing import Dict, List, Set, Tuple


class MinHashLSHDeduplicator:
    """
    MinHash LSH index for detecting near-duplicate text chunks based on
    word n-gram Jaccard similarity.
    """

    MERSENNE_PRIME = (1 << 61) - 1  # 2305843009213693951

    def __init__(
        self,
        num_perm: int = 128,
        num_bands: int = 16,
        jaccard_threshold: float = 0.85,
        ngram_size: int = 5,
        seed: int = 42,
    ):
        if num_perm % num_bands != 0:
            raise ValueError(f"num_perm ({num_perm}) must be divisible by num_bands ({num_bands})")

        self.num_perm = num_perm
        self.num_bands = num_bands
        self.rows_per_band = num_perm // num_bands
        self.jaccard_threshold = jaccard_threshold
        self.ngram_size = ngram_size

        # Precompute random linear hash parameters (a_i * x + b_i) % MERSENNE_PRIME
        rng = random.Random(seed)
        self.a_coeffs = [rng.randint(1, self.MERSENNE_PRIME - 1) for _ in range(num_perm)]
        self.b_coeffs = [rng.randint(0, self.MERSENNE_PRIME - 1) for _ in range(num_perm)]

        # LSH index: list of dicts, one per band
        # band_index[band_idx][band_hash] = list of doc_ids
        self.band_index: List[Dict[int, List[str]]] = [
            defaultdict(list) for _ in range(self.num_bands)
        ]

        # In-memory storage for verification
        self.doc_shingles: Dict[str, Set[str]] = {}

    def extract_shingles(self, text: str) -> Set[str]:
        """Tokenize text into lowercase word n-grams (shingles)."""
        words = re.findall(r"\b\w+\b", text.lower())
        if not words:
            return set()
        if len(words) < self.ngram_size:
            return {" ".join(words)}
        return {
            " ".join(words[i : i + self.ngram_size])
            for i in range(len(words) - self.ngram_size + 1)
        }

    def _hash_shingle(self, shingle: str) -> int:
        """Hash string shingle to 64-bit unsigned integer."""
        digest = hashlib.md5(shingle.encode("utf-8")).digest()
        return int.from_bytes(digest[:8], byteorder="little")

    def compute_minhash(self, shingles: Set[str]) -> List[int]:
        """Compute MinHash signature vector for a set of shingles."""
        if not shingles:
            return [0] * self.num_perm

        shingle_hashes = [self._hash_shingle(s) for s in shingles]
        signature = []
        for a, b in zip(self.a_coeffs, self.b_coeffs):
            min_val = min((a * h + b) % self.MERSENNE_PRIME for h in shingle_hashes)
            signature.append(min_val)
        return signature

    def is_duplicate_and_add(self, doc_id: str, text: str) -> bool:
        """
        Check if text chunk is a near-duplicate of any existing indexed chunk.
        If duplicate (Jaccard similarity >= threshold), returns True (not added).
        If unique, indexes chunk and returns False.
        """
        shingles = self.extract_shingles(text)
        if not shingles:
            return False

        signature = self.compute_minhash(shingles)

        # 1. Collect candidate document IDs from LSH band buckets
        candidates: Set[str] = set()
        for b in range(self.num_bands):
            start = b * self.rows_per_band
            end = start + self.rows_per_band
            band_tuple = tuple(signature[start:end])
            band_hash = hash(band_tuple)
            matching_ids = self.band_index[b].get(band_hash, [])
            candidates.update(matching_ids)

        # 2. Verify exact Jaccard similarity against candidate shingle sets
        for cand_id in candidates:
            cand_shingles = self.doc_shingles.get(cand_id)
            if not cand_shingles:
                continue

            intersection = len(shingles.intersection(cand_shingles))
            union = len(shingles.union(cand_shingles))
            if union > 0:
                jaccard = intersection / union
                if jaccard >= self.jaccard_threshold:
                    return True

        # 3. Unique chunk: index into LSH band buckets and store shingles
        self.doc_shingles[doc_id] = shingles
        for b in range(self.num_bands):
            start = b * self.rows_per_band
            end = start + self.rows_per_band
            band_tuple = tuple(signature[start:end])
            band_hash = hash(band_tuple)
            self.band_index[b][band_hash].append(doc_id)

        return False
