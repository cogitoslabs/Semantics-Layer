# Feature Spec: Corpus Cleaning Enhancements

## Objective

Enhance the post-extraction corpus cleaning logic in the DAPT pipeline to remove non-textual layout structures (indexes, bibliographies), correct encoding/unescape issues, and remove styling/layout placeholders.

---

## Scope

This specification defines the additions and improvements to the corpus cleaning routine in `scripts/clean_existing_corpus.py`.

Four primary enhancements:
1. **Remove Standalone Index/Bibliography Chunks**: Detect and filter out entire chunks representing standalone indexes or bibliographies.
2. **Inline References Removal**: Split and truncate chunks from target books (Principles of Neural Science, Neuroscience: Exploring the Brain, and Fundamentals of Cognitive Neuroscience) at the start of inline reference/selected reading sections, keeping only the narrative prose.
3. **HTML Unescape**: Unescape HTML entities across all remaining corpus text to ensure raw Unicode characters are restored.
4. **Placeholder/Fence Stripping**: Remove `<!-- image -->` placeholders and stray triple-backtick fences (` ``` `).

---

## Data Flow

```
Raw domain_dapt_corpus.jsonl
            ↓
    HTML Unescape
            ↓
 Strip Placeholders & Fences
            ↓
  Inline References Removal (PNS/Neuroscience/Fundamentals)
            ↓
  Standalone Index & Bibliography Filter
            ↓
  Min Content Length Check (>= 300 chars)
            ↓
Updated domain_dapt_corpus.jsonl
```

---

## Component Specifications

### 1. Standalone Index & Bibliography Filter

Chunks containing lists of index entries or bibliography lists are skipped entirely.

- **Index detection**:
  - Starts with an index heading (e.g. `## Index`, `## I N D E X`, `## Subject Index`, `## Author Index`).
  - Or, has a high proportion (> 45%) of lines matching the index entry page list pattern: a comma followed by a page list at the end of the line (supporting Roman numerals and figures/tables/boxes page suffixes like `143f`, `147t`, `125b`).
- **Bibliography detection**:
  - Starts with a bibliography heading (e.g. `## Bibliography`, `## References`, `## Selected Reading`, `## Further Reading`) and contains few lines (< 15).
  - Or, has a high proportion of citation patterns: lines containing publication years (19xx or 20xx) paired with bibliography keywords (e.g., "press", "journal", "pp.", "ed.", "university") or lines starting with author names (e.g., "Author, A. B.").

### 2. Inline References Removal

For documents originating from:
- `Principles of Neural Science.pdf`
- `Neuroscience exploring the brain.pdf`
- `Fundamentals of Cognitive Neuroscience A Beginner’s Guide.pdf` (and variants)

Identify the inline reference sections starting with headings such as `## Selected Reading`, `## Suggested Reading`, `## References`, `## Further Reading`, or `## Bibliography`. Truncate the chunk text before the matching heading, preserving only the narrative portion.

### 3. HTML Unescape

Apply `html.unescape()` to the entire chunk text to clean up HTML entities like `&amp;`, `&lt;`, `&gt;`.

### 4. Placeholder and Stray Fence Stripping

- Remove `<!-- image -->` (and variants like `<!--image-->`).
- Remove lines containing only triple backticks (` ``` `) which act as formatting fences.
