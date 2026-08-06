# Feature Spec: DAPT Pretraining Corpus Cleaning Enhancements

## Objective

Filter out all non-prose noise, layout artifacts, broken hyphens, control characters, and publisher metadata from `data/dapt/domain_dapt_corpus.jsonl` to ensure maximum data quality for domain-adaptive pretraining (DAPT).

---

## Scope

This specification defines the additions and improvements to:
1. `lib/utils/clean_text.py`: Core cleaning functions for text normalization, dehyphenation, control char removal, table handling, and metadata stripping.
2. `scripts/clean_existing_corpus.py`: Batch streaming script to clean `domain_dapt_corpus.jsonl` in-place and re-compute token metrics.
3. `tests/test_dapt_corpus_cleaning.py`: Unit and end-to-end pytest suite verifying zero regression and 100% compliance with cleaning rules.

---

## Data Flow & Processing Pipeline

```
Raw domain_dapt_corpus.jsonl (1,608 chunks)
            ↓
    Control Char & Non-Printable Byte Stripping (\x00-\x1f, \xad, \ufeff, etc.)
            ↓
    HTML Unescape & Comment/Placeholder Stripping (<!-- formula-not-decoded -->, <!-- image -->)
            ↓
    Private Use Area (PUA) Glyph Normalization (\ue000-\uf8ff)
            ↓
    Space-Padded Dehyphenation & Ligature Repair ("con -form" → "conform")
            ↓
    Markdown Table Stripping / Conversion (Remove |---| and pipe-table blocks)
            ↓
    Metadata & Proof Line Stripping (ISBNs, DOIs, .indd, copyright, page numbers)
            ↓
    Isolated Callout / Short Line Removal ("M", "H", "=", "1.0")
            ↓
    Standalone Index & Bibliography Chunks Removal
            ↓
    Min Prose Content Length & Quality Gate (>= 300 chars, alpha ratio >= 0.70)
            ↓
Updated domain_dapt_corpus.jsonl
```

---

## Technical Specifications

### 1. Control Characters & PUA Removal
- Regex pattern: `re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f\xad\ufeff]')`
- PUA pattern: `re.compile(r'[\ue000-\uf8ff]')`
- All control bytes are stripped. PUA math brackets `\uf6ae` / `\uf6af` are mapped to `(` / `)`, and other PUA codes are replaced with space.

### 2. Space-Padded Dehyphenation
- Match patterns:
  - `re.compile(r'\b([a-zA-Z]{2,})\s+-\s*([a-zA-Z]{2,})\b')`
  - `re.compile(r'\b([a-zA-Z]{2,})\s*-\s+([a-zA-Z]{2,})\b')`
- Merges split words (e.g. `con -form` → `conform`, `motiva -tion` → `motivation`, `evalu -ating` → `evaluating`).

### 3. Markdown Table & Non-Prose Filtering
- Pipe table lines (`^\s*\|.*\|\s*$`) and separator lines (`|---|`) are stripped from prose chunks.
- If a chunk consists of > 40% table lines, it is flagged as non-prose and filtered out.

### 4. Diagram Callouts & Short Line Filtering
- Standalone lines matching isolated symbols or 1-2 character OCR artifacts (e.g. `^\s*(?:[A-Z0-9]{1,2}|=|--|1\.0|2\.0)\s*$`) are stripped.

### 6. Figure Captions & Inline Figure Callout Stripping
- Standalone figure/table caption lines matching `re.compile(r'^\s*(?:FIGURE|Figure|Fig\.|TABLE|Table|BOX|Box)\s+\d+.*$', re.I)` are stripped.
- Inline parenthetical figure references matching `re.compile(r'\s*\(\s*(?:see\s+)?(?:Figure|Fig\.|Table|Box)\s+[\d\.\-A-Za-z\s,;]+\)', re.I)` are removed from sentences.

### 7. TOC Page-Number, Unheadinged Citation & Archive Header Filtering
- Standalone TOC listing lines matching title-page pairs (`^\s*(?:##?\s*)?\d+\.\s+.*?\b\d{2,3}\s*$`) or lines with dot leaders (`\.\.\.\.\.\s*\d+`) are stripped.
- Chunks where $\ge 35\%$ of lines match academic citation format (`- Author, A. B. (Year)...`) are identified as unheadinged bibliography pages and dropped.
- Archive & foundation header lines (`Digitized by the Internet Archive...`) are stripped.

---

## Verification Plan

### Automated Tests
- `pytest tests/test_dapt_corpus_cleaning.py`
- Run `python scripts/clean_existing_corpus.py` and inspect token count reduction, broken hyphen count, control char count.

### Quality Criteria
- Zero remaining ASCII control characters (`\x00-\x1f`).
- < 100 remaining space-padded hyphens (down from 7,674).
- Zero `<!-- formula-not-decoded -->` or `<!-- image -->` placeholders.
- Clean pretraining corpus saved to `data/dapt/domain_dapt_corpus.jsonl`.
