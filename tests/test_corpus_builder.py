import json
import os
import tempfile
from unittest.mock import MagicMock, patch
import pytest

from lib.utils import StorageAdapter
from lib.s1_build_corpus.worker import worker_init, worker_task, ExtractionResult
from lib.s1_build_corpus.build_corpus import CorpusBuilder, run_corpus_builder


class DummyStorageAdapter(StorageAdapter):
    def __init__(self, doc_paths):
        self.doc_paths = doc_paths

    def stream_documents(self):
        for path in self.doc_paths:
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


@pytest.fixture(autouse=True)
def mock_pdfium():
    with patch("pypdfium2.PdfDocument") as mock_pdf_cls:
        mock_doc = MagicMock()
        mock_doc.__len__.return_value = 5
        mock_doc.__enter__.return_value = mock_doc
        mock_pdf_cls.return_value = mock_doc
        yield mock_pdf_cls


def test_worker_init_and_task(mock_gpu_queue, mock_docling):
    # Initialize worker
    worker_init(mock_gpu_queue)
    
    # Run task with chunk_size=3 to verify overlap chunking works
    # mock_pdfium returns 5 pages.
    # With chunk_size=3, stride = 2:
    # Chunk 0: (1, 3)
    # Chunk 1: (3, 5)
    result = worker_task("test.pdf", "dummy_path.pdf", chunk_size=3)
    
    assert result.succeeded
    assert result.filename == "test.pdf"
    assert len(result.chunks) == 2
    assert result.chunks[0].chunk_index == 0
    assert result.chunks[0].page_range == (1, 3)
    assert "neuroscience" in result.chunks[0].text
    assert result.chunks[0].token_count > 0
    assert result.chunks[1].chunk_index == 1
    assert result.chunks[1].page_range == (3, 5)
    assert "neuroscience" in result.chunks[1].text
    assert result.chunks[1].token_count > 0


def test_corpus_builder_pipeline(mock_gpu_queue, mock_docling):
    # Initialize worker
    worker_init(mock_gpu_queue)

    with tempfile.TemporaryDirectory() as tmpdir:
        pdf_path = os.path.join(tmpdir, "paper.pdf")
        with open(pdf_path, "w") as f:
            f.write("%PDF-1.4 dummy contents")

        output_jsonl = os.path.join(tmpdir, "output.jsonl")
        storage = DummyStorageAdapter([pdf_path])

        # Patch multiprocessing.Pool to mock worker pool
        with patch("lib.s1_build_corpus.build_corpus.multiprocessing.Pool") as mock_pool_cls:
            mock_pool = MagicMock()
            mock_pool_cls.return_value = mock_pool
            
            # Mock apply_async to return a mock task whose get() returns our successful ExtractionResult
            from lib.s1_build_corpus.worker import ChunkResult
            dummy_text = "This is a neuroscience reasoning and retrieval test document. " * 10
            
            mock_task = MagicMock()
            mock_task.get.return_value = ExtractionResult(
                filename="paper.pdf",
                chunks=[
                    ChunkResult(
                        chunk_index=0,
                        text=dummy_text,
                        token_count=50,
                        page_range=(1, 3)
                    )
                ],
                status="SUCCESS"
            )
            mock_pool.apply_async.return_value = mock_task

            from pathlib import Path
            from lib.utils import CorpusBuildConfig
            from lib.utils.config import LoggingConfig
            cfg_build = CorpusBuildConfig(
                output_path=Path(output_jsonl),
                extracted_output_path=Path(output_jsonl),
                available_gpus="0",
                workers_per_gpu=1,
                chunk_size=3
            )
            builder = CorpusBuilder(
                storage=storage,
                cfg=cfg_build,
                logging_cfg=LoggingConfig(),
            )
            builder.build()

        # Verify output
        assert os.path.exists(output_jsonl)
        with open(output_jsonl, "r", encoding="utf-8") as f:
            lines = f.readlines()
            assert len(lines) == 2
            record = json.loads(lines[0])
            assert record["source_file"] == "paper.pdf"
            assert record["chunk_id"] == 0
            assert record["page_range"] == [1, 3]
            assert record["token_count"] == 50
            assert "neuroscience" in record["text"]


def test_worker_task_txt_html(mock_gpu_queue, mock_docling):
    # Initialize worker
    worker_init(mock_gpu_queue)
    
    # Run task for TXT (no start/end pages, chunk_size=2)
    # The dummy text is short enough to fit in a single chunk
    result_txt = worker_task("test.txt", "dummy_path.txt", chunk_size=2)
    assert result_txt.succeeded
    assert result_txt.filename == "test.txt"
    assert len(result_txt.chunks) == 1
    assert result_txt.chunks[0].chunk_index == 0
    assert result_txt.chunks[0].page_range is None
    assert "neuroscience" in result_txt.chunks[0].text
    
    # Run task for HTML (no start/end pages, chunk_size=2)
    result_html = worker_task("test.html", "dummy_path.html", chunk_size=2)
    assert result_html.succeeded
    assert result_html.filename == "test.html"
    assert len(result_html.chunks) == 1
    assert result_html.chunks[0].chunk_index == 0
    assert result_html.chunks[0].page_range is None
    assert "neuroscience" in result_html.chunks[0].text


def test_worker_task_sliding_window(mock_gpu_queue, mock_docling):
    # Initialize worker
    worker_init(mock_gpu_queue)
    
    # Set converter mock to return a long string
    long_text = "This is a long neuroscience document block. " * 100 # ~1000 tokens
    _, mock_instance = mock_docling
    mock_instance.convert.return_value.document.export_to_markdown.return_value = long_text
    
    # Run task with chunk_size=1 (target_tokens = 400, overlap = 40)
    result = worker_task("test.txt", "dummy_path.txt", chunk_size=1)
    
    assert result.succeeded
    assert len(result.chunks) > 1
    for i, chunk in enumerate(result.chunks):
        assert chunk.chunk_index == i
        assert chunk.page_range is None
        assert chunk.token_count > 0


def test_corpus_builder_auto_resolution():
    from pathlib import Path
    from lib.utils import CorpusBuildConfig
    from lib.utils.config import LoggingConfig

    # 1. Test GPU Mode Auto Resolution
    with patch("lib.utils.system_detection.get_available_gpus_and_vram") as mock_gpus, \
         patch("lib.utils.system_detection.get_cpu_cores") as mock_cpus, \
         patch("lib.utils.system_detection.get_available_system_ram_gb") as mock_ram:
        
        mock_gpus.return_value = [(0, 16.0)]
        mock_cpus.return_value = 8
        mock_ram.return_value = 32.0

        cfg = CorpusBuildConfig(
            output_path=Path("dummy.jsonl"),
            available_gpus="AUTO",
            workers_per_gpu="AUTO",
            chunk_size="AUTO",
            maxtasksperchild="AUTO"
        )
        builder = CorpusBuilder(
            storage=MagicMock(),
            cfg=cfg,
            logging_cfg=LoggingConfig(),
        )

        assert builder.gpu_ids == [0]
        assert builder.workers_per_gpu == 4
        assert builder.total_workers == 4
        assert builder.chunk_size == 10
        assert builder.maxtasksperchild == 8

    # 2. Test CPU Fallback Mode Auto Resolution
    with patch("lib.utils.system_detection.get_available_gpus_and_vram") as mock_gpus, \
         patch("lib.utils.system_detection.get_cpu_cores") as mock_cpus, \
         patch("lib.utils.system_detection.get_available_system_ram_gb") as mock_ram:
        
        mock_gpus.return_value = []
        mock_cpus.return_value = 8
        mock_ram.return_value = 32.0

        cfg = CorpusBuildConfig(
            output_path=Path("dummy.jsonl"),
            available_gpus="AUTO",
            workers_per_gpu="AUTO",
            chunk_size="AUTO",
            maxtasksperchild="AUTO"
        )
        builder = CorpusBuilder(
            storage=MagicMock(),
            cfg=cfg,
            logging_cfg=LoggingConfig(),
        )

        assert builder.gpu_ids == [-1]
        assert builder.workers_per_gpu == 4
        assert builder.total_workers == 4
        assert builder.chunk_size == 10
        assert builder.maxtasksperchild == 8

