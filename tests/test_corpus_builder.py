import json
import os
import tempfile
from unittest.mock import MagicMock, patch
import pytest

from lib.s1_build_corpus.storage import StorageAdapter
from lib.s1_build_corpus.worker import worker_init, worker_task, ExtractionResult
from lib.s1_build_corpus.build_corpus import CorpusBuilder, run_corpus_builder


class DummyStorageAdapter(StorageAdapter):
    def __init__(self, pdf_paths):
        self.pdf_paths = pdf_paths

    def stream_pdfs(self):
        for path in self.pdf_paths:
            yield os.path.basename(path), path, False


@pytest.fixture
def mock_gpu_queue():
    queue = MagicMock()
    queue.get.return_value = 0
    return queue


@pytest.fixture
def mock_docling():
    with patch("docling.document_converter.DocumentConverter") as mock_converter_cls:
        mock_instance = MagicMock()
        mock_converter_cls.return_value = mock_instance
        
        # Mock the convert chain: converter.convert(path).document.export_to_markdown()
        mock_doc = MagicMock()
        # Create a string longer than MIN_CONTENT_LENGTH (300)
        dummy_text = "This is a neuroscience reasoning and retrieval test document. " * 10
        mock_doc.export_to_markdown.return_value = dummy_text
        
        mock_result = MagicMock()
        mock_result.document = mock_doc
        mock_instance.convert.return_value = mock_result
        
        yield mock_converter_cls, mock_instance


def test_worker_init_and_task(mock_gpu_queue, mock_docling):
    # Initialize worker
    worker_init(mock_gpu_queue)
    
    # Run task
    result = worker_task("test.pdf", "dummy_path.pdf")
    
    assert result.succeeded
    assert result.filename == "test.pdf"
    assert "neuroscience" in result.text
    assert result.token_count > 0


def test_corpus_builder_pipeline(mock_gpu_queue, mock_docling):
    # Initialize worker
    worker_init(mock_gpu_queue)

    with tempfile.TemporaryDirectory() as tmpdir:
        pdf_path = os.path.join(tmpdir, "paper.pdf")
        with open(pdf_path, "w") as f:
            f.write("%PDF-1.4 dummy contents")

        output_jsonl = os.path.join(tmpdir, "output.jsonl")
        storage = DummyStorageAdapter([pdf_path])

        # Patch ProcessPoolExecutor to run synchronously or mock worker pool
        # To test the builder cleanly, we mock ProcessPoolExecutor to return our dummy worker_task results
        with patch("lib.s1_build_corpus.build_corpus.ProcessPoolExecutor") as mock_executor_cls:
            mock_executor = MagicMock()
            mock_executor_cls.return_value.__enter__.return_value = mock_executor
            
            # Create a mock future
            mock_future = MagicMock()
            mock_executor.submit.return_value = mock_future
            
            # Mock as_completed to yield our future
            with patch("lib.s1_build_corpus.build_corpus.as_completed") as mock_as_completed:
                mock_as_completed.return_value = [mock_future]
                
                # Mock future.result() to return a successful ExtractionResult
                dummy_text = "This is a neuroscience reasoning and retrieval test document. " * 10
                mock_future.result.return_value = ExtractionResult(
                    filename="paper.pdf",
                    text=dummy_text,
                    token_count=50,
                    status="SUCCESS"
                )

                builder = CorpusBuilder(
                    storage=storage,
                    output_path=output_jsonl,
                    available_gpus="0",
                    workers_per_gpu=1
                )
                builder.build()

        # Verify output
        assert os.path.exists(output_jsonl)
        with open(output_jsonl, "r", encoding="utf-8") as f:
            lines = f.readlines()
            assert len(lines) == 1
            record = json.loads(lines[0])
            assert record["source_file"] == "paper.pdf"
            assert record["token_count"] == 50
            assert "neuroscience" in record["text"]
