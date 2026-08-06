"""
test_pdf_utils.py — Unit tests for PDF main text extractor helper function.
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from lib.utils.pdf_utils import is_toc_page, extract_main_text_from_pdfs


def test_is_toc_page():
    # Positive case: contains TOC heading and lines with dots/page numbers
    toc_text = (
        "Table of Contents\n"
        "1. Introduction ...................................... 1\n"
        "2. Background ........................................ 5\n"
        "3. Experimental Setup ................................ 12\n"
        "4. Results and Analysis .............................. 24\n"
    )
    assert is_toc_page(toc_text) is True

    # Positive case: lines starting with Chapter/Section and page numbers
    toc_text_2 = (
        "Contents\n"
        "Chapter 1: Neuroanatomy and Physiology 10\n"
        "Chapter 2: Synaptic Transmission 45\n"
        "Chapter 3: Cognitive Systems 85\n"
    )
    assert is_toc_page(toc_text_2) is True

    # Negative case: standard narrative text
    narrative_text = (
        "Introduction\n"
        "The human brain is the central organ of the human nervous system.\n"
        "It consists of the cerebrum, the brainstem, and the cerebellum.\n"
        "These structures control most of the activities of the body, processing,\n"
        "integrating, and coordinating the information it receives.\n"
    )
    assert is_toc_page(narrative_text) is False


# Define Mock Classes for Docling elements
class MockProvenance:
    def __init__(self, page_no):
        self.page_no = page_no


class MockItem:
    def __init__(self, text, page_no):
        self.text = text
        self.prov = [MockProvenance(page_no)]


class TextItem(MockItem):
    label = "text"


class ListItem(MockItem):
    label = "list_item"


class CodeItem(MockItem):
    label = "code"


class SectionHeaderItem(MockItem):
    label = "section_header"
    def __init__(self, text, page_no, level):
        super().__init__(text, page_no)
        self.level = level


@pytest.fixture
def mock_docling_result():
    # Setup mock items representing a book with:
    # Page 1: Copyright / Cover Page (should be skipped)
    # Page 2: Table of Contents (should be skipped)
    # Page 3: Main text, with a Questions section (should be skipped) and next section (should resume)
    # Page 4: Main text with repeating header/footer to test furniture stripping
    items = [
        # Page 1: Copyright Page
        TextItem("Principles of Neural Science", 1),
        TextItem("Copyright © 2026 by McGraw-Hill Education.", 1),
        TextItem("All rights reserved. Printed in the United States.", 1),
        
        # Page 2: Table of Contents
        TextItem("Running Header A", 2),
        SectionHeaderItem("Contents", 2, 1),
        TextItem("Chapter 1: The Brain ..................... 1", 2),
        TextItem("Chapter 2: Synapses ..................... 15", 2),
        TextItem("Running Footer B", 2),
        
        # Page 3: Main Content
        TextItem("Running Header A", 3),  # Header/footer candidate (repeats on Page 4)
        SectionHeaderItem("Chapter 1: The Brain", 3, 1),
        TextItem("This is the main introduction narrative text.", 3),
        ListItem("First key point about neural pathways.", 3),
        
        # Questions section (should be skipped)
        SectionHeaderItem("Review Questions", 3, 2),
        TextItem("Q1: Describe the function of the hippocampus.", 3),
        ListItem("a) Memory consolidation", 3),
        
        # Resuming section (same level or higher level than the skipped section)
        SectionHeaderItem("Methodology", 3, 2),
        TextItem("We conducted experiments using optogenetics.", 3),
        TextItem("Running Footer B", 3),  # Header/footer candidate (repeats on Page 4)
        
        # Page 4: More Content
        TextItem("Running Header A", 4),  # Header/footer candidate
        SectionHeaderItem("Chapter 2: Synapses", 4, 1),
        TextItem("Synaptic transmission is the process of neurotransmitter release.", 4),
        CodeItem("print('synapse active')", 4),
        TextItem("Running Footer B", 4),  # Header/footer candidate
    ]
    
    mock_doc = MagicMock()
    mock_doc.iterate_items.return_value = [(item, None) for item in items]
    
    mock_result = MagicMock()
    mock_result.document = mock_doc
    return mock_result


@patch("lib.utils.pdf_utils.DocumentConverter")
def test_extract_main_text_from_pdfs_consolidated(mock_converter_class, mock_docling_result):
    # Setup mock converter instance
    mock_converter = MagicMock()
    mock_converter.convert.return_value = mock_docling_result
    mock_converter_class.return_value = mock_converter
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create input directory and dummy pdf
        input_dir = Path(tmpdir) / "input"
        input_dir.mkdir()
        dummy_pdf = input_dir / "test_book.pdf"
        dummy_pdf.write_bytes(b"%PDF-1.4 mock pdf data")
        
        output_file = Path(tmpdir) / "consolidated_output.txt"
        
        # Execute extractor
        extract_main_text_from_pdfs(
            input_dir=input_dir,
            output_path=output_file,
            recursive=False
        )
        
        # Read result
        assert output_file.exists()
        result_text = output_file.read_text(encoding="utf-8")
        
        # Assertions
        # 1. Page 1 (Copyright) should be skipped
        assert "McGraw-Hill Education" not in result_text
        
        # 2. Page 2 (TOC) should be skipped
        assert "Chapter 1: The Brain ....................." not in result_text
        
        # 3. Running headers and footers should be stripped
        assert "Running Header A" not in result_text
        assert "Running Footer B" not in result_text
        
        # 4. Main content should be present
        assert "# Chapter 1: The Brain" in result_text
        assert "This is the main introduction narrative text." in result_text
        assert "- First key point about neural pathways." in result_text
        
        # 5. Review Questions section (and its questions/lists) should be skipped
        assert "Review Questions" not in result_text
        assert "Q1: Describe the function" not in result_text
        assert "a) Memory consolidation" not in result_text
        
        # 6. Parser should resume at Methodology (sibling section of Review Questions)
        assert "## Methodology" in result_text
        assert "We conducted experiments using optogenetics." in result_text
        # 7. Chapter 2 content should be present
        assert "# Chapter 2: Synapses" in result_text
        assert "Synaptic transmission is the process of neurotransmitter release." in result_text
        assert "print('synapse active')" in result_text


@patch("lib.utils.pdf_utils.DocumentConverter")
def test_extract_main_text_from_pdfs_individual_files(mock_converter_class, mock_docling_result):
    # Setup mock converter instance
    mock_converter = MagicMock()
    mock_converter.convert.return_value = mock_docling_result
    mock_converter_class.return_value = mock_converter
    
    with tempfile.TemporaryDirectory() as tmpdir:
        input_dir = Path(tmpdir) / "input"
        input_dir.mkdir()
        
        # Write two dummy PDF files
        pdf1 = input_dir / "book1.pdf"
        pdf1.write_bytes(b"%PDF-1.4 data")
        pdf2 = input_dir / "book2.pdf"
        pdf2.write_bytes(b"%PDF-1.4 data")
        
        output_dir = Path(tmpdir) / "output"
        
        # Execute extractor
        extract_main_text_from_pdfs(
            input_dir=input_dir,
            output_path=output_dir,
            recursive=False
        )
        
        # Verify output files
        out1 = output_dir / "book1.txt"
        out2 = output_dir / "book2.txt"
        
        assert out1.exists()
        assert out2.exists()
        
        text1 = out1.read_text(encoding="utf-8")
        text2 = out2.read_text(encoding="utf-8")
        
        # Check filtered elements in book1.txt
        assert "# Chapter 1: The Brain" in text1
        assert "Review Questions" not in text1
        assert "Running Header A" not in text1
        
        # Check filtered elements in book2.txt
        assert "# Chapter 1: The Brain" in text2
        assert "Review Questions" not in text2
        assert "Running Header A" not in text2
