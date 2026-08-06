import json
import pytest
from pathlib import Path
from lib.s1_build_corpus.minhash_lsh import MinHashLSHDeduplicator
from lib.s1_build_corpus.merge_corpus import run_merge_corpus
from lib.utils.config import PipelineConfig


def test_minhash_exact_duplicate():
    dedup = MinHashLSHDeduplicator(num_perm=128, num_bands=16, jaccard_threshold=0.85, ngram_size=5)
    text1 = "Long-term potentiation is a persistent strengthening of synapses based on recent patterns of activity in neuroscience."
    text2 = "Long-term potentiation is a persistent strengthening of synapses based on recent patterns of activity in neuroscience."

    assert dedup.is_duplicate_and_add("doc_1", text1) is False
    assert dedup.is_duplicate_and_add("doc_2", text2) is True


def test_minhash_near_duplicate():
    dedup = MinHashLSHDeduplicator(num_perm=128, num_bands=16, jaccard_threshold=0.85, ngram_size=5)
    # ~100-word paragraph with only 1 word altered (Jaccard similarity ~0.90 for 5-grams)
    text1 = (
        "Long term potentiation is widely considered one of the primary cellular mechanisms underlying learning "
        "and memory formation in the central nervous system. High frequency stimulation of excitatory pathways leads to a "
        "persistent enhancement of synaptic strength between neurons across multiple brain regions. This long lasting "
        "increase in signal transmission involves both presynaptic neurotransmitter release and postsynaptic receptor "
        "insertion, establishing robust functional connectivity within neural networks. Experimental evidence indicates "
        "that blockade of NMDA receptors impairs both long term potentiation induction and spatial memory acquisition."
    )
    text2 = (
        "Long term potentiation is widely considered one of the primary cellular mechanisms underlying learning "
        "and memory formation in the central nervous system. High frequency stimulation of excitatory pathways leads to a "
        "durable enhancement of synaptic strength between neurons across multiple brain regions. This long lasting "
        "increase in signal transmission involves both presynaptic neurotransmitter release and postsynaptic receptor "
        "insertion, establishing robust functional connectivity within neural networks. Experimental evidence indicates "
        "that blockade of NMDA receptors impairs both long term potentiation induction and spatial memory acquisition."
    )

    assert dedup.is_duplicate_and_add("doc_1", text1) is False
    assert dedup.is_duplicate_and_add("doc_2", text2) is True


def test_minhash_distinct_texts():
    dedup = MinHashLSHDeduplicator(num_perm=128, num_bands=16, jaccard_threshold=0.85, ngram_size=5)
    text1 = "Action potentials propagate along axonal membranes via voltage-gated sodium channels."
    text2 = "Astrocytes regulate extracellular potassium concentration and support blood brain barrier integrity."

    assert dedup.is_duplicate_and_add("doc_1", text1) is False
    assert dedup.is_duplicate_and_add("doc_2", text2) is False


def test_minhash_short_and_empty():
    dedup = MinHashLSHDeduplicator(num_perm=128, num_bands=16, jaccard_threshold=0.85, ngram_size=5)

    assert dedup.is_duplicate_and_add("doc_empty", "") is False
    assert dedup.is_duplicate_and_add("doc_short", "Short text") is False


def test_merge_corpus_minhash_integration(tmp_path: Path):
    in_dir = tmp_path / "in"
    in_dir.mkdir(parents=True, exist_ok=True)
    out_file = tmp_path / "out" / "domain_dapt_corpus.jsonl"

    file1 = in_dir / "corpus1.jsonl"
    file2 = in_dir / "corpus2.jsonl"

    chunk_unique_1 = {
        "text": "Dopaminergic neurons in the substantia nigra pars compacta project heavily to the dorsal striatum.",
        "source_file": "paper1.pdf",
        "chunk_id": 0,
        "token_count": 20,
    }
    chunk_duplicate_1 = {
        "text": "Dopaminergic neurons in the substantia nigra pars compacta project heavily to the dorsal striatum.",
        "source_file": "paper2.pdf",
        "chunk_id": 0,
        "token_count": 20,
    }
    chunk_unique_2 = {
        "text": "Microglial activation triggers neuroinflammatory responses following ischemic cerebrovascular stroke.",
        "source_file": "paper3.pdf",
        "chunk_id": 0,
        "token_count": 15,
    }

    with open(file1, "w", encoding="utf-8") as f:
        f.write(json.dumps(chunk_unique_1) + "\n")
        f.write(json.dumps(chunk_unique_2) + "\n")

    with open(file2, "w", encoding="utf-8") as f:
        f.write(json.dumps(chunk_duplicate_1) + "\n")

    cfg = PipelineConfig()
    cfg.data.dapt_in_dir = str(in_dir)
    cfg.build.output_path = out_file
    cfg.build.minhash_enabled = True
    cfg.build.minhash_jaccard_threshold = 0.85

    run_merge_corpus(cfg)

    assert out_file.exists()
    lines = out_file.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 2

    records = [json.loads(line) for line in lines]
    assert records[0]["id"] == "domain_doc_000000"
    assert records[1]["id"] == "domain_doc_000001"
    assert records[0]["text"] == chunk_unique_1["text"]
    assert records[1]["text"] == chunk_unique_2["text"]
