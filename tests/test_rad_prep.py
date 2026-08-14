import json
import os
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest
import numpy as np
import torch
import faiss

from lib.utils import PipelineConfig
from lib.s4_rad_prep.chunker import Chunk, chunk_document, run_chunking
from lib.s4_rad_prep.indexer import run_indexing
from lib.s4_rad_prep.retriever import Retriever, RetrievalResult
from lib.s4_rad_prep.no_retrieval_router import NoRetrievalRouter
from lib.s4_rad_prep.prompt_generator import format_prompt, PromptGenerator



class SimpleMockTokenizer:
    def __init__(self):
        self.pad_token = "<pad>"
        self.eos_token = "</s>"
        self.eos_token_id = 2
        self.pad_token_id = 1

    def encode(self, text, add_special_tokens=False):
        # Treat each word as a token
        return [ord(w[0]) for w in text.split() if w]

    def decode(self, tokens, skip_special_tokens=True):
        return " ".join(chr(t) for t in tokens)

    def tokenize(self, text):
        return text.split()

    def __call__(self, text, *args, **kwargs):
        from transformers import BatchEncoding
        if isinstance(text, str):
            n = 1
        else:
            n = len(text)
        return BatchEncoding({
            "input_ids": torch.ones((n, 4), dtype=torch.long),
            "attention_mask": torch.ones((n, 4), dtype=torch.long),
        })



@pytest.fixture
def mock_tokenizer():
    return SimpleMockTokenizer()


@pytest.fixture
def test_cfg():
    cfg = PipelineConfig()
    # Use small validation settings
    cfg.rad.long_form_chunk_tokens = 10
    cfg.rad.long_form_overlap_tokens = 2
    cfg.rad.abstract_chunk_tokens = 5
    cfg.rad.abstract_overlap_tokens = 1
    cfg.rad.top_k = 3
    cfg.rad.min_grounded_pct = 0.50
    return cfg




def test_chunker_long_form(mock_tokenizer):
    text = "worda wordb wordc wordd worde wordf wordg wordh wordi wordj wordk wordl wordm wordn wordo"
    # Total 15 words (tokens)
    # chunk_size = 10, overlap = 2 -> step = 8
    # Chunk 0: index 0 to 10
    # Chunk 1: index 8 to 15 (end of list)
    chunks = chunk_document("doc1", text, "long_form", mock_tokenizer, chunk_size=10, overlap_size=2)
    assert len(chunks) == 2
    assert chunks[0].chunk_id == "doc1_0"
    assert chunks[0].doc_type == "long_form"
    assert chunks[0].token_count == 10
    assert chunks[1].chunk_id == "doc1_1"
    assert chunks[1].token_count == 7


def test_chunker_abstract(mock_tokenizer):
    text = "worda wordb wordc wordd worde wordf wordg wordh wordi wordj"
    # Total 10 words (tokens)
    # chunk_size = 5, overlap = 1 -> step = 4
    # Chunk 0: index 0 to 5
    # Chunk 1: index 4 to 9
    # Chunk 2: index 8 to 10
    chunks = chunk_document("doc2", text, "abstract", mock_tokenizer, chunk_size=5, overlap_size=1)
    assert len(chunks) == 3
    assert chunks[0].doc_type == "abstract"
    assert chunks[0].token_count == 5
    assert chunks[1].token_count == 5
    assert chunks[2].token_count == 2


def test_chunker_short_doc(mock_tokenizer):
    text = "worda wordb wordc"
    chunks = chunk_document("doc3", text, "long_form", mock_tokenizer, chunk_size=10, overlap_size=2)
    assert len(chunks) == 1
    assert chunks[0].token_count == 3


def test_indexer_build_and_load(test_cfg):
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        test_cfg.rad.index_dir = tmp_path / "index"
        test_cfg.rad.chunks_path = tmp_path / "chunks.jsonl"

        chunks = [
            Chunk(chunk_id="c1", doc_id="d1", doc_type="long_form", text="biomedical data science", token_count=3),
            Chunk(chunk_id="c2", doc_id="d1", doc_type="long_form", text="neural systems neuroscience", token_count=3),
        ]

        mock_embeddings = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0]
        ], dtype="float32")

        with patch("lib.s4_rad_prep.indexer.DenseEmbedder") as mock_embedder_class:
            mock_embedder = MagicMock()
            mock_embedder.embed_batch.return_value = mock_embeddings
            mock_embedder_class.return_value = mock_embedder

            run_indexing(test_cfg, chunks)

        assert (test_cfg.rad.index_dir / "index.faiss").exists()
        assert (test_cfg.rad.index_dir / "chunks_metadata.jsonl").exists()
        assert (test_cfg.rad.index_dir / "index_manifest.json").exists()

        with open(test_cfg.rad.index_dir / "index_manifest.json", "r") as f:
            manifest = json.load(f)
            assert manifest["chunk_count"] == 2
            assert manifest["total_tokens"] == 6


def test_dense_retrieval(test_cfg):
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        test_cfg.rad.index_dir = tmp_path / "index"
        test_cfg.rad.chunks_path = tmp_path / "chunks.jsonl"

        chunks = [
            Chunk(chunk_id="c1", doc_id="d1", doc_type="long_form", text="cognitive neuroscience", token_count=2),
            Chunk(chunk_id="c2", doc_id="d2", doc_type="long_form", text="visual cortex brain", token_count=3),
        ]

        with open(test_cfg.rad.chunks_path, "w") as f:
            for c in chunks:
                f.write(json.dumps(c.__dict__) + "\n")

        # Create mock faiss index
        index = faiss.IndexFlatIP(4)
        index.add(np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype="float32"))
        test_cfg.rad.index_dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(index, str(test_cfg.rad.index_dir / "index.faiss"))

        mock_query_emb = np.array([[0.9, 0.1, 0, 0]], dtype="float32")

        with patch("lib.s4_rad_prep.retriever.DenseEmbedder") as mock_embedder_class, \
             patch("transformers.AutoTokenizer.from_pretrained") as mock_tok_class:
            mock_embedder = MagicMock()
            mock_embedder.embed_batch.return_value = mock_query_emb
            mock_embedder_class.return_value = mock_embedder
            mock_tok_class.return_value = SimpleMockTokenizer()

            test_cfg.rad.retrieval_mode = "dense"
            test_cfg.rad.relevance_threshold = 0.5
            retriever = Retriever(test_cfg)
            result = retriever.retrieve("cognitive neuroscience")

            assert len(result.chunks) >= 1
            assert result.chunks[0].chunk_id == "c1"
            assert result.scores[0] == pytest.approx(0.9, rel=1e-2)


def test_relevance_threshold_gate(test_cfg):
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        test_cfg.rad.index_dir = tmp_path / "index"
        test_cfg.rad.chunks_path = tmp_path / "chunks.jsonl"

        chunks = [
            Chunk(chunk_id="c1", doc_id="d1", doc_type="long_form", text="brain visual system", token_count=3),
            Chunk(chunk_id="c2", doc_id="d2", doc_type="long_form", text="cardiac cycle blood", token_count=3),
        ]
        with open(test_cfg.rad.chunks_path, "w") as f:
            for c in chunks:
                f.write(json.dumps(c.__dict__) + "\n")

        index = faiss.IndexFlatIP(4)
        index.add(np.array([[0.9, 0.1, 0, 0], [0.1, 0.9, 0, 0]], dtype="float32"))
        test_cfg.rad.index_dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(index, str(test_cfg.rad.index_dir / "index.faiss"))

        mock_query_emb = np.array([[0.95, 0.05, 0, 0]], dtype="float32")

        with patch("lib.s4_rad_prep.retriever.DenseEmbedder") as mock_embedder_class, \
             patch("transformers.AutoTokenizer.from_pretrained") as mock_tok_class:
            mock_embedder = MagicMock()
            mock_embedder.embed_batch.return_value = mock_query_emb
            mock_embedder_class.return_value = mock_embedder
            mock_tok_class.return_value = SimpleMockTokenizer()

            test_cfg.rad.retrieval_mode = "dense"
            test_cfg.rad.relevance_threshold = 0.8
            retriever = Retriever(test_cfg)
            result = retriever.retrieve("brain visual system")

            # c1 should pass (similarity ~0.85+), c2 should be gated (similarity ~0.14)
            assert len(result.chunks) == 1
            assert result.chunks[0].chunk_id == "c1"
            assert result.passed_threshold == 1


def test_no_retrieval_router(test_cfg):
    with tempfile.TemporaryDirectory() as tmpdir:
        log_file = Path(tmpdir) / "no_retrieval_rates.jsonl"
        router = NoRetrievalRouter(log_file, min_passed_chunks=2)

        decision1 = router.route_sample("s1", "cognitive", passed_chunks=1)
        assert decision1.no_retrieval is True
        assert decision1.reason.startswith("Insufficient")

        decision2 = router.route_sample("s2", "cognitive", passed_chunks=3)
        assert decision2.no_retrieval is False

        router.flush_batch()

        stats = router.get_aggregate_stats()
        assert stats["total_samples"] == 2
        assert stats["no_retrieval_count"] == 1
        assert stats["overall_rate"] == 0.5
        assert stats["by_cluster"]["cognitive"]["no_retrieval"] == 1


def test_hybrid_fusion(test_cfg):
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        test_cfg.rad.index_dir = tmp_path / "index"
        test_cfg.rad.chunks_path = tmp_path / "chunks.jsonl"

        chunks = [
            Chunk(chunk_id="c1", doc_id="d1", doc_type="long_form", text="hippocampal theta oscillations", token_count=3),
            Chunk(chunk_id="c2", doc_id="d2", doc_type="long_form", text="cerebellar Purkinje cells motor control", token_count=5),
        ]
        with open(test_cfg.rad.chunks_path, "w") as f:
            for c in chunks:
                f.write(json.dumps(c.__dict__) + "\n")

        index = faiss.IndexFlatIP(4)
        index.add(np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype="float32"))
        test_cfg.rad.index_dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(index, str(test_cfg.rad.index_dir / "index.faiss"))

        with patch("lib.s4_rad_prep.retriever.DenseEmbedder") as mock_embedder_class, \
             patch("transformers.AutoTokenizer.from_pretrained") as mock_tok_class:
            mock_embedder = MagicMock()
            mock_embedder.embed_batch.return_value = np.array([[0.9, 0.1, 0, 0]], dtype="float32")
            mock_embedder_class.return_value = mock_embedder
            mock_tok_class.return_value = SimpleMockTokenizer()

            test_cfg.rad.retrieval_mode = "hybrid"
            test_cfg.rad.use_reranker = False
            test_cfg.rad.relevance_threshold = 0.1
            retriever = Retriever(test_cfg)
            result = retriever.retrieve("hippocampal motor control")

            assert len(result.chunks) > 0
            assert result.retrieval_mode == "hybrid"


def test_trace_generator_grounded(test_cfg):
    chunks = [Chunk(chunk_id="c1", doc_id="d1", doc_type="long_form", text="hippocampus functions in memory", token_count=4)]
    prompt = format_prompt("What is hippocampus?", "Memory function", chunks, no_retrieval=False)
    assert "[CONTEXT]" in prompt
    assert "hippocampus functions in memory" in prompt
    assert "\\boxed{}" in prompt


def test_trace_generator_no_retrieval(test_cfg):
    prompt = format_prompt("What is hippocampus?", "Memory function", [], no_retrieval=True)
    assert "[NO CONTEXT AVAILABLE]" in prompt
    assert "[CONTEXT]" not in prompt
    assert "\\boxed{}" in prompt


def test_trace_prompt_generator(test_cfg):
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        test_cfg.rad.traces_dir = tmp_path / "traces"
        test_cfg.logging.log_dir = tmp_path / "logs"

        samples = [
            {"question": "Q1", "answer": "A1", "sample_id": "s1", "cluster": "c1"},
            {"question": "Q2", "answer": "A2", "sample_id": "s2", "cluster": "c1"},
            {"question": "Q3", "answer": "A3", "sample_id": "s3", "cluster": "c2"}
        ]

        ret_results = [
            RetrievalResult(chunks=[Chunk(chunk_id="c1", doc_id="d1", doc_type="abstract", text="context 1", token_count=2)], scores=[0.9], passed_threshold=2, retrieval_mode="dense"),
            RetrievalResult(chunks=[Chunk(chunk_id="c2", doc_id="d2", doc_type="abstract", text="context 2", token_count=2)], scores=[0.95], passed_threshold=2, retrieval_mode="dense"),
            RetrievalResult(chunks=[], scores=[], passed_threshold=0, retrieval_mode="dense")
        ]

        router = NoRetrievalRouter(tmp_path / "logs" / "rad_prep" / "no_retrieval_rates.jsonl")

        generator = PromptGenerator(test_cfg)
        generator.generate_prompts(samples, ret_results, router)


        # Check files
        grounded_file = test_cfg.rad.traces_dir / "grounded_traces.jsonl"
        no_ret_file = test_cfg.rad.traces_dir / "no_retrieval_traces.jsonl"

        assert grounded_file.exists()
        assert no_ret_file.exists()

        with open(grounded_file, "r") as f:
            grounded_lines = [json.loads(line) for line in f]
        with open(no_ret_file, "r") as f:
            no_ret_lines = [json.loads(line) for line in f]

        assert len(grounded_lines) == 2
        assert grounded_lines[0]["sample_id"] == "s1"
        assert grounded_lines[1]["sample_id"] == "s2"

        assert len(no_ret_lines) == 1
        assert no_ret_lines[0]["sample_id"] == "s3"


def test_pipeline_end_to_end(test_cfg):
    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            tmp_path = Path(tmpdir)
            test_cfg.rad.use_reranker = False
            test_cfg.rad.retrieval_corpus_path = tmp_path / "retrieval_corpus.jsonl"
            test_cfg.rad.chunks_path = tmp_path / "chunks.jsonl"
            test_cfg.rad.index_dir = tmp_path / "index"
            test_cfg.rad.traces_dir = tmp_path / "traces"
            test_cfg.rad.qa_samples_path = tmp_path / "probe_qa.jsonl"
            test_cfg.logging.log_dir = tmp_path / "logs"

            # Write mock corpus
            with open(test_cfg.rad.retrieval_corpus_path, "w") as f:
                f.write(json.dumps({"text": "the human brain consists of cortex and cerebellum", "doc_type": "long_form", "doc_id": "d1"}) + "\n")
                f.write(json.dumps({"text": "neural networks learn patterns from data representations", "doc_type": "abstract", "doc_id": "d2"}) + "\n")

            # Write mock QA samples
            with open(test_cfg.rad.qa_samples_path, "w") as f:
                f.write(json.dumps({"question": "cortex cerebellum location?", "answer": "brain", "cluster": "neuro"}) + "\n")
                f.write(json.dumps({"choices": ["neural", "blood"], "answer_idx": 0, "question": "network representation?", "cluster": "cs"}) + "\n")

            # Mock tokenizers and embeddings
            with patch("transformers.AutoTokenizer.from_pretrained") as mock_tok_class, \
                 patch("lib.s4_rad_prep.indexer.DenseEmbedder") as mock_embedder_class, \
                 patch("lib.s4_rad_prep.retriever.DenseEmbedder") as mock_ret_embedder_class:

                mock_tok = SimpleMockTokenizer()
                mock_tok_class.return_value = mock_tok

                mock_emb = MagicMock()
                mock_emb.embed_batch.side_effect = lambda texts, **kwargs: np.tile(
                    np.array([1.0, 0.0, 0.0, 0.0], dtype="float32"), (len(texts), 1)
                )
                mock_embedder_class.return_value = mock_emb
                mock_ret_embedder_class.return_value = mock_emb

                from lib.s4_rad_prep import run_rad_prep_pipeline
                run_rad_prep_pipeline(test_cfg)

                # Check all artifacts
                assert test_cfg.rad.chunks_path.exists()
                assert (test_cfg.rad.index_dir / "index.faiss").exists()
                assert (test_cfg.rad.traces_dir / "grounded_traces.jsonl").exists()
                assert (test_cfg.logging.log_dir / "rad_prep" / "phase_manifest.json").exists()
                
                with open(test_cfg.logging.log_dir / "rad_prep" / "phase_manifest.json", "r") as f:
                    manifest = json.load(f)
                    assert manifest["status"] == "complete"
                    assert manifest["metrics"]["grounded_trace_count"] == 2
        finally:
            from lib.utils.logger import close_loggers
            close_loggers()


def test_trace_generator_bedrock(test_cfg):
    test_cfg.benchmarking.teacher_backend = "bedrock"
    test_cfg.benchmarking.teacher_model_name = "meta.llama3"


    mock_response = {
        "output": {
            "message": {
                "content": [{"text": "This is a response from Llama on Bedrock."}]
            }
        }
    }

    with patch.dict("os.environ", {"AWS_ACCESS_KEY_ID": "test_id", "AWS_SECRET_ACCESS_KEY": "test_secret"}), \
         patch("boto3.client") as mock_boto_client:
        mock_client_instance = MagicMock()
        mock_client_instance.converse.return_value = mock_response
        mock_boto_client.return_value = mock_client_instance
        from lib.utils.teacher_backend import BedrockBackend
        backend = BedrockBackend(test_cfg)
        traces = backend.generate_batch(["Test prompt"])

        assert len(traces) == 1
        assert traces[0] == "This is a response from Llama on Bedrock."
        mock_client_instance.converse.assert_called_once()



def test_trace_generator_bedrock_missing_credentials(test_cfg):
    test_cfg.benchmarking.teacher_backend = "aws"
    from lib.utils.teacher_backend import BedrockBackend
    with patch.dict("os.environ", {}, clear=True):
        with pytest.raises(RuntimeError, match="SEVERE ERROR: Missing AWS credentials"):
            BedrockBackend(test_cfg)


def test_trace_generator_bedrock_api_error(test_cfg):
    test_cfg.benchmarking.teacher_backend = "bedrock"

    from lib.utils.teacher_backend import BedrockBackend
    with patch.dict("os.environ", {"AWS_ACCESS_KEY_ID": "test_id", "AWS_SECRET_ACCESS_KEY": "test_secret"}), \
         patch("boto3.client") as mock_boto_client:
        mock_client_instance = MagicMock()
        mock_client_instance.converse.side_effect = Exception("Access Denied / Unable to locate credentials")
        mock_boto_client.return_value = mock_client_instance

        backend = BedrockBackend(test_cfg)
        with pytest.raises(RuntimeError, match="SEVERE ERROR: AWS Bedrock API call failed"):
            backend.generate_batch(["Test prompt"])


def test_bge_large_dense_embedder():
    from lib.s4_rad_prep.indexer import DenseEmbedder

    mock_outputs = MagicMock()
    # last_hidden_state shape (1, 3, 4)
    mock_outputs.last_hidden_state = torch.ones((1, 3, 4))

    with patch("transformers.AutoTokenizer.from_pretrained") as mock_tok, \
         patch("transformers.AutoModel.from_pretrained") as mock_model:

        tok_instance = MagicMock()
        from transformers import BatchEncoding
        tok_instance.return_value = BatchEncoding({
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]])
        })
        mock_tok.return_value = tok_instance
        model_instance = MagicMock()
        model_instance.return_value = mock_outputs
        model_instance.to.return_value = model_instance
        mock_model.return_value = model_instance

        embedder = DenseEmbedder("bge-large", device="cpu")
        assert embedder.model_name == "BAAI/bge-large-en-v1.5"
        assert embedder.is_bge is True

        res = embedder.embed_batch(["neuroscience research"], is_query=True)
        assert res.shape == (1, 4)
        # Check query instruction prefix was added
        called_args = tok_instance.call_args[0][0]
        assert called_args[0].startswith("Represent this sentence for searching relevant passages:")


def test_query_variant_extraction(test_cfg):
    test_cfg.rad.chunks_path = Path("nonexistent_chunks.jsonl")
    test_cfg.rad.index_dir = Path("nonexistent_index")

    with patch("lib.s4_rad_prep.retriever.DenseEmbedder"), \
         patch("transformers.AutoTokenizer.from_pretrained"):
        retriever = object.__new__(Retriever)
        retriever.cfg = test_cfg

        variants = retriever._extract_query_variants("Which definition best describes 'Sulcus'?")
        assert "Sulcus definition" in variants
        assert "Sulcus" in variants
        assert "Which definition best describes 'Sulcus'?" in variants


def test_dense_embedder_backward_compatibility():
    from lib.s4_rad_prep.indexer import DenseEmbedder

    mock_outputs = MagicMock()
    mock_outputs.last_hidden_state = torch.ones((1, 3, 4))

    with patch("transformers.AutoTokenizer.from_pretrained") as mock_tok, \
         patch("transformers.AutoModel.from_pretrained") as mock_model:

        tok_instance = MagicMock()
        from transformers import BatchEncoding
        tok_instance.return_value = BatchEncoding({
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]])
        })
        mock_tok.return_value = tok_instance
        model_instance = MagicMock()
        model_instance.return_value = mock_outputs
        model_instance.to.return_value = model_instance
        mock_model.return_value = model_instance

        embedder_bio = DenseEmbedder("biolinkbert", device="cpu")
        assert embedder_bio.model_name == "michiyasunaga/BioLinkBERT-large"
        assert embedder_bio.is_bge is False

        embedder_pub = DenseEmbedder("pubmedbert", device="cpu")
        assert embedder_pub.model_name == "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract"
        assert embedder_pub.is_bge is False


def test_cross_encoder_reranker():
    from lib.s4_rad_prep.reranker import CrossEncoderReranker

    chunks = [
        Chunk(chunk_id="c1", doc_id="d1", doc_type="long_form", text="cortex visual system", token_count=3),
        Chunk(chunk_id="c2", doc_id="d2", doc_type="long_form", text="synaptic plasticity memory", token_count=3),
    ]

    mock_logits = torch.tensor([[2.0], [-1.0]])  # sigmoid(2.0)~0.88, sigmoid(-1.0)~0.27
    mock_outputs = MagicMock()
    mock_outputs.logits = mock_logits

    with patch("transformers.AutoTokenizer.from_pretrained") as mock_tok, \
         patch("transformers.AutoModelForSequenceClassification.from_pretrained") as mock_model:

        tok_instance = MagicMock()
        from transformers import BatchEncoding
        tok_instance.return_value = BatchEncoding({
            "input_ids": torch.tensor([[1, 2], [3, 4]]),
            "attention_mask": torch.tensor([[1, 1], [1, 1]])
        })
        mock_tok.return_value = tok_instance

        model_instance = MagicMock()
        model_instance.return_value = mock_outputs
        model_instance.to.return_value = model_instance
        mock_model.return_value = model_instance

        reranker = CrossEncoderReranker("BAAI/bge-reranker-large", device="cpu", batch_size=32)
        reranked_chunks, reranked_scores = reranker.rerank("visual cortex", chunks, top_k=2)

        assert len(reranked_chunks) == 2
        assert reranked_chunks[0].chunk_id == "c1"
        assert reranked_scores[0] > reranked_scores[1]
        assert reranked_scores[0] == pytest.approx(0.88, rel=1e-1)


def test_hybrid_retrieval_with_reranker(test_cfg):
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        test_cfg.rad.index_dir = tmp_path / "index"
        test_cfg.rad.chunks_path = tmp_path / "chunks.jsonl"
        test_cfg.rad.retrieval_mode = "hybrid"
        test_cfg.rad.use_reranker = True
        test_cfg.rad.rerank_candidate_k = 10
        test_cfg.rad.top_k = 2
        test_cfg.rad.relevance_threshold = 0.50

        chunks = [
            Chunk(chunk_id="c1", doc_id="d1", doc_type="long_form", text="visual cortex occipital lobe", token_count=4),
            Chunk(chunk_id="c2", doc_id="d2", doc_type="long_form", text="motor cortex frontal lobe", token_count=4),
        ]
        with open(test_cfg.rad.chunks_path, "w") as f:
            for c in chunks:
                f.write(json.dumps(c.__dict__) + "\n")

        index = faiss.IndexFlatIP(4)
        index.add(np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype="float32"))
        test_cfg.rad.index_dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(index, str(test_cfg.rad.index_dir / "index.faiss"))

        mock_query_emb = np.array([[0.9, 0.1, 0, 0]], dtype="float32")

        mock_reranker = MagicMock()
        mock_reranker.rerank.return_value = ([chunks[0]], [0.85])

        with patch("lib.s4_rad_prep.retriever.DenseEmbedder") as mock_embedder_class, \
             patch("transformers.AutoTokenizer.from_pretrained") as mock_tok_class:

            mock_embedder = MagicMock()
            mock_embedder.embed_batch.return_value = mock_query_emb
            mock_embedder_class.return_value = mock_embedder
            mock_tok_class.return_value = SimpleMockTokenizer()

            retriever = Retriever(test_cfg, reranker=mock_reranker)
            result = retriever.retrieve("visual cortex occipital")

            assert len(result.chunks) == 1
            assert result.chunks[0].chunk_id == "c1"
            assert result.scores[0] == pytest.approx(0.85, rel=1e-2)
            mock_reranker.rerank.assert_called_once()


def test_reranker_disabled_fallback(test_cfg):
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        test_cfg.rad.index_dir = tmp_path / "index"
        test_cfg.rad.chunks_path = tmp_path / "chunks.jsonl"
        test_cfg.rad.retrieval_mode = "hybrid"
        test_cfg.rad.use_reranker = False
        test_cfg.rad.top_k = 2
        test_cfg.rad.relevance_threshold = 0.00  # allow RRF scores to pass

        chunks = [
            Chunk(chunk_id="c1", doc_id="d1", doc_type="long_form", text="hippocampus memory formation", token_count=3),
            Chunk(chunk_id="c2", doc_id="d2", doc_type="long_form", text="amygdala emotional processing", token_count=3),
        ]
        with open(test_cfg.rad.chunks_path, "w") as f:
            for c in chunks:
                f.write(json.dumps(c.__dict__) + "\n")

        index = faiss.IndexFlatIP(4)
        index.add(np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype="float32"))
        test_cfg.rad.index_dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(index, str(test_cfg.rad.index_dir / "index.faiss"))

        mock_query_emb = np.array([[0.9, 0.1, 0, 0]], dtype="float32")

        with patch("lib.s4_rad_prep.retriever.DenseEmbedder") as mock_embedder_class, \
             patch("transformers.AutoTokenizer.from_pretrained") as mock_tok_class:

            mock_embedder = MagicMock()
            mock_embedder.embed_batch.return_value = mock_query_emb
            mock_embedder_class.return_value = mock_embedder
            mock_tok_class.return_value = SimpleMockTokenizer()

            retriever = Retriever(test_cfg)
            assert retriever.reranker is None
            result = retriever.retrieve("hippocampus memory")

            assert len(result.chunks) == 2
            assert result.chunks[0].chunk_id == "c1"



