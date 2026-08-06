"""
pdf_utils.py — Helper utilities for scanning directories, extracting main narrative text 
from PDFs using Docling, and writing clean text outputs while filtering out unwanted sections.
"""

import os
import re
from pathlib import Path
from collections import Counter
from typing import List, Dict, Set, Any

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend

from lib.utils.logger import get_logger
from lib.utils.clean_text import clean_corpus_text, is_standalone_index_or_bibliography, is_copyright_or_front_matter

logger = get_logger(__name__)

# Pattern to identify Table of Contents pages
TOC_HEADING_PAT = re.compile(r'^\s*#*\s*(table of\s+)?contents\b', re.IGNORECASE)

# Pattern to identify section headers that should be excluded
EXCLUDED_SECTIONS_PAT = re.compile(
    r'\b(questions|exercises|problems|study questions|review questions|discussion questions|'
    r'review exercises|practice problems|activities|bibliography|references|selected reading|'
    r'further reading|suggested reading|literature cited|index|subject index|author index)\b',
    re.IGNORECASE
)

# Pattern to identify Figure and Table captions
CAPTION_PAT = re.compile(r'^\s*(figure|fig\.|table|tbl\.)\s+\d+', re.IGNORECASE)


def is_caption_or_non_narrative(item: Any) -> bool:
    """Check if an item is a figure/table caption or non-narrative element."""
    label = str(getattr(item, "label", "")).lower()
    if label in ("caption", "picture", "table", "footnote"):
        return True
    text = getattr(item, "text", "").strip()
    if CAPTION_PAT.search(text):
        return True
    return False


def is_toc_page(text: str) -> bool:
    """
    Heuristic to determine if a page's text represents a Table of Contents.
    Checks for TOC-like headings and a significant ratio of lines ending with page numbers
    preceded by dots or spaces.
    """
    text_stripped = text.strip()
    if not text_stripped:
        return False
        
    text_lower = text_stripped.lower()
    if not ("contents" in text_lower or "list of" in text_lower or "directory" in text_lower):
        return False
        
    lines = [line.strip() for line in text_stripped.split("\n") if line.strip()]
    if not lines:
        return False
        
    toc_lines = 0
    for line in lines:
        # Check if line ends with a number (page reference) preceded by dots, spaces, or dashes
        if re.search(r'(\.|\s|-)\s*\d+$', line) or re.search(r'^\s*(chapter|section|part|unit)\s+\d+', line, re.IGNORECASE):
            toc_lines += 1
            
    # If a significant ratio of lines are TOC-like, or we have at least 4 TOC lines
    ratio = toc_lines / len(lines)
    return ratio > 0.20 or toc_lines >= 4


def is_index_page(text: str) -> bool:
    """
    Heuristic to detect index pages based on density of page number references (e.g. ', 123').
    Since Docling can merge index columns into long paragraphs, line-level checks might fail.
    """
    char_len = len(text)
    if char_len < 100:
        return False
    matches = len(re.findall(r',\s*\d+', text))
    return (matches / char_len) > 0.003


def extract_main_text_from_pdfs(
    input_dir: str | Path,
    output_path: str | Path,
    recursive: bool = True,
    exclude_title_pages: bool = True,
    exclude_toc: bool = True,
    exclude_questions_exercises: bool = True,
    exclude_index: bool = True,
    exclude_headers_footers: bool = True,
    exclude_bibliography: bool = True,
) -> None:
    """
    Reads all PDF files from input_dir, extracts their main narrative text, excludes
    title/copyright pages, Table of Contents, questions, exercises, index, running headers, 
    footers, and bibliography, and writes the output to text files.

    Args:
        input_dir: Path to directory containing PDF files.
        output_path: Path to target output file (.txt) or directory. If a file path is provided,
                     all extracted texts are consolidated into that file. If a directory path is
                     provided, clean text is written into individual .txt files for each PDF.
        recursive: Whether to scan directories recursively.
        exclude_title_pages: Whether to exclude front matter, copyright pages, or covers.
        exclude_toc: Whether to exclude Table of Contents pages.
        exclude_questions_exercises: Whether to exclude questions, exercises, problems, and activities.
        exclude_index: Whether to exclude index sections.
        exclude_headers_footers: Whether to filter out running headers, footers, and page number lines.
        exclude_bibliography: Whether to exclude bibliographies and references.
    """
    input_path = Path(input_dir)
    target_path = Path(output_path)
    
    if not input_path.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
        
    # Determine output mode
    if target_path.suffix.lower() == ".txt":
        is_dir_mode = False
        target_path.parent.mkdir(parents=True, exist_ok=True)
        # Clear/initialize the consolidated output file
        with open(target_path, "w", encoding="utf-8") as _:
            pass
        logger.info(f"Consolidating extracted text from {input_dir} into file: {target_path}")
    else:
        is_dir_mode = True
        target_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Extracting texts from {input_dir} into directory: {target_path}")
        
    # Scan for PDF files
    pattern = "**/*.pdf" if recursive else "*.pdf"
    pdf_files = sorted(list(input_path.glob(pattern)))
    
    if not pdf_files:
        logger.warning(f"No PDF files found in {input_dir}")
        return
        
    logger.info(f"Found {len(pdf_files)} PDF files to process.")
    
    # Initialize DocumentConverter once for efficiency
    # Disabling OCR and using PyPdfiumDocumentBackend makes conversion extremely fast and lightweight
    options = PdfPipelineOptions()
    options.do_ocr = False
    options.do_table_structure = False
    options.generate_page_images = False
    options.generate_picture_images = False
    
    pdf_format_option = PdfFormatOption(
        pipeline_options=options,
        backend=PyPdfiumDocumentBackend
    )
    converter = DocumentConverter(
        format_options={InputFormat.PDF: pdf_format_option}
    )
    
    for idx, pdf_file in enumerate(pdf_files):
        filename = pdf_file.name
        logger.info(f"[{idx+1}/{len(pdf_files)}] Converting {filename}...")
        
        try:
            result = converter.convert(pdf_file)
            doc = result.document
        except Exception as e:
            logger.error(f"Failed to convert {filename}: {e}")
            continue
            
        # 1. Group text/heading/list elements by page number
        pages_dict: Dict[int, List[Any]] = {}
        for item, _ in doc.iterate_items():
            item_type = type(item).__name__
            if item_type not in ("TextItem", "SectionHeaderItem", "ListItem", "CodeItem"):
                continue
            
            # Identify page number from provenance
            page_no = 1
            if getattr(item, "prov", None) and len(item.prov) > 0:
                page_no = item.prov[0].page_no
            pages_dict.setdefault(page_no, []).append(item)
            
        if not pages_dict:
            logger.warning(f"Skipped {filename} (no readable text elements found)")
            continue
            
        # 2. Heuristic Running Headers & Footers Stripping
        # Build frequency table of short single-line text elements across all pages
        headers_footers: Set[str] = set()
        if exclude_headers_footers:
            item_freqs: Counter = Counter()
            for page_no, items in pages_dict.items():
                seen_on_page = set()
                for item in items:
                    text_stripped = item.text.strip()
                    # Candidate running furniture is short (< 120 chars) and single-line
                    if text_stripped and len(text_stripped) < 120 and "\n" not in text_stripped:
                        seen_on_page.add(text_stripped)
                for text_stripped in seen_on_page:
                    item_freqs[text_stripped] += 1
                    
            total_pages = len(pages_dict)
            # A line is repeating if it appears on at least 3 pages, or >=15% of pages
            hf_threshold = max(3, int(total_pages * 0.15))
            headers_footers = {text for text, freq in item_freqs.items() if freq >= hf_threshold}
            if headers_footers:
                logger.info(f"Stripping {len(headers_footers)} running headers/footers in {filename}")
                
        # 3. Filter pages at the Page Level
        valid_items: List[Any] = []
        for page_no in sorted(pages_dict.keys()):
            items = pages_dict[page_no]
            page_text = "\n".join(item.text for item in items)
            
            # Check Title / Copyright pages
            if exclude_title_pages and is_copyright_or_front_matter(page_text, filename):
                logger.debug(f"Skipping Page {page_no} of {filename}: Copyright or front matter")
                continue
                
            # Check Standalone Index / Bibliography pages
            is_standalone = is_standalone_index_or_bibliography(page_text, filename)
            # Add density check for index pages (since Docling joins columns into paragraphs)
            is_idx_density = exclude_index and is_index_page(page_text)
            
            if is_standalone or is_idx_density:
                is_index = is_idx_density or "index" in page_text.lower()
                if (is_index and exclude_index) or (not is_index and exclude_bibliography):
                    logger.debug(f"Skipping Page {page_no} of {filename}: Standalone index/bibliography page")
                    continue
                    
            # Check Table of Contents pages
            if exclude_toc and is_toc_page(page_text):
                logger.debug(f"Skipping Page {page_no} of {filename}: Table of Contents page")
                continue
                
            # Keep items on this page, filtering out running headers/footers and captions
            for item in items:
                if exclude_headers_footers and item.text.strip() in headers_footers:
                    continue
                if is_caption_or_non_narrative(item):
                    continue
                valid_items.append(item)
                
        # 4. Filter sections (Questions, Exercises, Bibliography, Index) using Hierarchical State-Machine
        skipping = False
        skip_level = None
        final_items: List[Any] = []
        
        for item in valid_items:
            item_type = type(item).__name__
            if item_type == "SectionHeaderItem":
                header_text = item.text.strip()
                level = getattr(item, "level", 1)
                
                # Exclude check based on configuration flags
                is_excluded = False
                if EXCLUDED_SECTIONS_PAT.search(header_text):
                    lower_hdr = header_text.lower()
                    
                    # Distinguish section categories
                    is_ex = any(x in lower_hdr for x in ("questions", "exercises", "problems", "activities"))
                    is_bib = any(x in lower_hdr for x in ("bibliography", "references", "reading", "cited"))
                    is_idx = "index" in lower_hdr
                    
                    if (is_ex and exclude_questions_exercises) or \
                       (is_bib and exclude_bibliography) or \
                       (is_idx and exclude_index):
                        is_excluded = True
                        
                if skipping:
                    if level <= skip_level:
                        if is_excluded:
                            # Update target skip level if nested/sequential excluded section
                            skip_level = level
                        else:
                            # Resume normal text collection
                            skipping = False
                            skip_level = None
                            final_items.append(item)
                    else:
                        # Skip sub-header/content of skipped section
                        pass
                else:
                    if is_excluded:
                        skipping = True
                        skip_level = level
                        logger.debug(f"Skipping section starting at: {repr(header_text)} (level {level})")
                    else:
                        final_items.append(item)
            else:
                if not skipping:
                    final_items.append(item)
                    
        # 5. Format remaining content elements and unwrap broken paragraph splits
        lines: List[str] = []
        for item in final_items:
            item_type = type(item).__name__
            text = item.text.strip()
            if not text:
                continue

            if item_type == "SectionHeaderItem":
                level = getattr(item, "level", 1)
                hashes = "#" * min(level, 6)
                lines.append(f"\n{hashes} {text}\n")
            elif item_type == "ListItem":
                lines.append(f"- {text}")
            elif item_type == "CodeItem":
                lines.append(f"```\n{text}\n```")
            else:
                # Narrative text block: check if it should join with previous paragraph
                if lines and not lines[-1].startswith(("#", "-", "```")) and not lines[-1].endswith("\n"):
                    prev_text = lines[-1].strip()
                    if prev_text and (not prev_text[-1] in ".!?:;" or text[0].islower()):
                        if prev_text.endswith("-"):
                            lines[-1] = prev_text[:-1] + text
                        else:
                            lines[-1] = prev_text + " " + text
                        continue
                lines.append(text)
                
        raw_text = "\n".join(lines)
        
        # 6. Apply final text repair & normalization
        cleaned_text = clean_corpus_text(raw_text, filename)
        
        # 7. Write to output
        if is_dir_mode:
            out_file = target_path / f"{pdf_file.stem}.txt"
            out_file.write_text(cleaned_text, encoding="utf-8")
            logger.info(f"Wrote clean text of {filename} -> {out_file} ({len(cleaned_text):,} chars)")
        else:
            with open(target_path, "a", encoding="utf-8") as out:
                if os.path.getsize(target_path) > 0:
                    out.write("\n\n")
                out.write(f"--- Document: {filename} ---\n\n")
                out.write(cleaned_text)
            logger.info(f"Appended clean text of {filename} -> {target_path} ({len(cleaned_text):,} chars)")
            
    logger.info(f"Completed extraction pipeline for directory: {input_dir}")
