import re
import html
import nltk
from nltk.corpus import words

# Load English words from NLTK corpus for dictionary-based joining
try:
    ENGLISH_WORDS = set(word.lower() for word in words.words())
except Exception:
    ENGLISH_WORDS = set()

# Common standalone words that should not be merged as a prefix (to prevent false joins like "a cross" -> "across")
COMMON_WORDS = {
    'a', 'i', 'an', 'the', 'in', 'on', 'at', 'by', 'to', 'of', 'for', 'with', 'from',
    'up', 'down', 'he', 'she', 'it', 'we', 'they', 'his', 'her', 'its', 'our',
    'their', 'is', 'are', 'was', 'were', 'be', 'been', 'has', 'have', 'had',
    'do', 'does', 'did', 'but', 'and', 'or', 'so', 'if', 'as', 'that', 'this',
    'these', 'those', 'who', 'whom', 'which', 'what', 'why', 'how'
}

# Scientific prefixes commonly split from their base words in PDF extraction
SCIENTIFIC_PREFIXES = {
    'neuro', 'opto', 'pre', 'post', 'synap', 'hyper', 'hypo', 'retro', 'intra',
    'inter', 'extra', 'multi', 'micro', 'macro', 'bio', 'patho', 'psycho',
    'electro', 'chemo', 'auto', 'somato', 'vaso', 'photo', 'mono', 'poly'
}

# Common inflected word suffixes to reconstruct base lemmas
COMMON_SUFFIXES = {'ing', 'ed', 's', 'es', 'er', 'est', 'ly', 'able', 'ible', 'al', 'ive', 'tion', 'ment', 'ty'}

# Layout, metadata, and publisher patterns
METADATA_PATTERNS = [
    # InDesign proof files: 08-Swanson_Ch-08.indd 177, Swanson_FM.indd, etc.
    re.compile(r'\b\d{2}-[\w\\-]+\.indd\b', re.I),
    # Timestamps: 5/28/2011 9:40:34 AM, 2026-07-20 23:30:00
    re.compile(r'\b\d{1,2}/\d{1,2}/\d{4}\s+\d{1,2}:\d{2}:\d{2}\s+(?:AM|PM)\b'),
    re.compile(r'\b\d{4}-\d{2}-\d{2}\s+\d{1,2}:\d{2}:\d{2}\b'),
    # Publisher / ISBN / DOI / Archive lines
    re.compile(r'^\s*(?:ISBN|DOI|ISSN)[:\s]\s*[\d\-X]+', re.I),
    re.compile(r'^\s*Typeset\s+in\s+.*$', re.I),
    re.compile(r'^\s*Library\s+of\s+Congress\s+Cataloging.*$', re.I),
    re.compile(r'^\s*Digitized\s+by\s+the\s+Internet\s+Archive.*$', re.I),
    # "This page intentionally left blank"
    re.compile(r'(?i)\bthis page intentionally left blank\b'),
]

# Standalone Table of Contents line patterns (e.g., "## 26. Memory and Amnesia 329" or dot leaders "...... 45" or multiple topic-page pairs)
TOC_LINE_PAT = re.compile(r'^\s*(?:##?\s*)?\d+\.\s+.*?\b\d{2,4}\s*$', re.I)
DOT_LEADER_PAT = re.compile(r'\.{4,}\s*\d+')
TOC_PAIR_PAT = re.compile(r'\b[A-Za-z\s,-]{3,}\s+\d{2,4}\b')

# Figure/Table/Box standalone caption pattern
FIG_CAPTION_PAT = re.compile(r'^\s*(?:FIGURE|Figure|Fig\.|TABLE|Table|BOX|Box)\s+\d+.*$', re.I)

# Inline parenthetical figure callout pattern (e.g. "(see Figure 4-2)", "(Fig. 3B)", "(Table 1.1)")
INLINE_FIG_REF_PAT = re.compile(r'\s*\(\s*(?:see\s+)?(?:Figure|Fig\.|Table|Box)\s+[\d\.\-A-Za-z\s,;]+\)', re.I)

# Control character pattern (excluding standard whitespace \n, \t, \r and \x01)
CONTROL_CHAR_PAT = re.compile(r'[\x02-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f\xad\ufeff]')

# PUA (Private Use Area) Unicode pattern
PUA_CHAR_PAT = re.compile(r'[\ue000-\uf8ff]')

# HTML comments and unparsed placeholders (e.g. <!-- formula-not-decoded -->, <!-- image -->)
HTML_COMMENT_PAT = re.compile(r'<!--\s*[\w\s\-]*\s*-->', re.I)

# Standalone ASCII Markdown table line pattern
TABLE_LINE_PAT = re.compile(r'^\s*\|.*\|\s*$')
TABLE_HEADER_SEP_PAT = re.compile(r'^\s*\|?[\s:\-]+\|[\s:\-|]+\s*$')

# Page number standalone line pattern
PAGE_NUM_LINE_PATTERN = re.compile(r'^\s*([ivxldcm]+|\d+|\w\s+\w)\s*$')

# Isolated callout / short symbol line pattern (e.g. isolated lines with "M", "H", "=", "1.0")
ISOLATED_CALLOUT_PAT = re.compile(r'^\s*(?:[A-Za-z0-9]{1,2}|=|--|1\.0|2\.0)\s*$')

# Index line pattern: comma followed by page number list at the end of the line
INDEX_LINE_PAT = re.compile(
    r',\s*(?:[ivxldcm]+|\d{1,4}[ftb]?)(?:\s*[-–]\s*(?:[ivxldcm]+|\d{1,4}[ftb]?))?'
    r'(?:\s*,\s*(?:[ivxldcm]+|\d{1,4}[ftb]?)(?:\s*[-–]\s*(?:[ivxldcm]+|\d{1,4}[ftb]?))?)*\s*(?:\||\.?)?$',
    re.I
)

# Year pattern: e.g. 1999 or 2011
YEAR_PAT = re.compile(r'\b(19|20)\d{2}\b')

# Bibliography keywords
BIB_KEYWORDS = re.compile(r'\b(press|journal|pp\.|ed\.|eds\.|vol\.|university|publisher|proceedings|symposium|monograph|encyclopedia|in:)\b', re.I)

# Author name prefix pattern: e.g., "Sanes, J. R.", "Sanes JR", or spaced "Gallese , V ."
AUTHOR_PREFIX_PAT = re.compile(r'^(?:[-\s*•]*)\s*[A-Z][a-zA-Z\s]+(?:\s*,|\s+and)\s+[A-Z](?:\s*\.\s*[A-Z])*(?:\s*,|\s+and)?', re.I)

# Inline reference headings pattern
INLINE_REF_HEADINGS = re.compile(
    r'^#+\s+(?:References|Selected Readings?|Suggested Readings?|Further Readings?|Bibliography)\b.*$',
    re.I | re.M
)

# Front-matter and copyright keywords
FRONT_MATTER_KEYWORDS = [
    r'reprinted with permission',
    r'reproduced by permission',
    r'in the public domain',
    r'all rights reserved',
    r'no part of this publication may be reproduced',
    r'copyright \d{4}',
    r'copyright ©',
    r'library of congress cataloging',
    r'cataloging-in-publication data',
    r'printed in the united states',
]
FRONT_MATTER_PATTERNS = [re.compile(p, re.I) for p in FRONT_MATTER_KEYWORDS]


def dehyphenate_text(text: str) -> str:
    """
    Join hyphenated line breaks and space-padded hyphen split words back together,
    using dictionary-guarding to preserve compounds.
    """
    if not text:
        return ""
        
    def line_wrap_repl(match):
        w1 = match.group(1)
        w2 = match.group(2)
        concat = w1.lower() + w2.lower()
        if is_valid_inflected_word(concat):
            if w1.isupper():
                return concat.upper()
            elif w1[0].isupper():
                return concat.capitalize()
            return concat
        else:
            return w1 + "-" + w2

    # 1. Cross-line hyphen wrap: word- \n word
    text = re.sub(r'(\b[a-zA-Z]{2,})-\s*\n+\s*([a-zA-Z]{2,}\b)', line_wrap_repl, text)
    
    def same_line_repl(match):
        w1 = match.group(1)
        w2 = match.group(2)
        w1_l = w1.lower()
        w2_l = w2.lower()
        concat = w1_l + w2_l
        
        # If both w1 and w2 are independent valid English words (>3 chars), keep hyphen (e.g. corticotropin-releasing)
        if w1_l in ENGLISH_WORDS and w2_l in ENGLISH_WORDS and len(w1_l) > 3 and len(w2_l) > 3:
            return w1 + "-" + w2
            
        if is_valid_inflected_word(concat):
            if w1.isupper():
                return concat.upper()
            elif w1[0].isupper():
                return concat.capitalize()
            return concat
        else:
            return w1 + "-" + w2

    # 2. Same line hyphens with space around hyphen: word - word, word -word, word- word
    text = re.sub(r'\b([a-zA-Z]{2,})\s+-\s*([a-zA-Z]{2,})\b', same_line_repl, text)
    text = re.sub(r'\b([a-zA-Z]{2,})\s*-\s+([a-zA-Z]{2,})\b', same_line_repl, text)
    return text


def is_valid_inflected_word(concat: str) -> bool:
    """Check if the concatenation is a valid inflected form of a dictionary word."""
    if concat in ENGLISH_WORDS:
        return True
    for suffix in COMMON_SUFFIXES:
        if concat.endswith(suffix) and len(concat) > len(suffix):
            base = concat[:-len(suffix)]
            if base in ENGLISH_WORDS:
                return True
            if base + 'e' in ENGLISH_WORDS:
                return True
            if len(base) > 2 and base[-1] == base[-2] and base[:-1] in ENGLISH_WORDS:
                return True
            if suffix in {'ed', 'es', 'er', 'est', 'ly', 'able', 'al'}:
                if base.endswith('i') and base[:-1] + 'y' in ENGLISH_WORDS:
                    return True
    return False


def should_join_words(w1: str, w2: str) -> bool:
    """Determine if w1 and w2 should be merged based on dictionary and heuristic checks."""
    if len(w1) < 2 and w1.isupper():
        return False
        
    w1_lower = w1.lower()
    w2_lower = w2.lower()
    concat = w1_lower + w2_lower
    
    if w1_lower in COMMON_WORDS:
        return False
        
    if is_valid_inflected_word(concat):
        return True
        
    if w1_lower in SCIENTIFIC_PREFIXES and is_valid_inflected_word(w2_lower):
        return True
        
    return False


def join_ligatures_dict(text: str) -> str:
    """Fast linear split-word joiner to repair broken PDF ligatures."""
    if not text:
        return ""
        
    lines = text.split('\n')
    cleaned_lines = []
    
    for line in lines:
        curr_line = line
        passes = 0
        while passes < 3:
            words_list = curr_line.split()
            if len(words_list) < 2:
                break
                
            res = []
            i = 0
            joined_any = False
            while i < len(words_list):
                if i + 1 < len(words_list):
                    w1, w2 = words_list[i], words_list[i+1]
                    w1_clean = w1.strip('.,;:()[]"\'')
                    w2_clean = w2.strip('.,;:()[]"\'')
                    if w1_clean.isalpha() and w2_clean.isalpha() and should_join_words(w1_clean, w2_clean):
                        concat = w1_clean.lower() + w2_clean.lower()
                        if w1_clean.isupper():
                            rep = concat.upper()
                        elif w1_clean[0].isupper():
                            rep = concat.capitalize()
                        else:
                            rep = concat
                        res.append(w1.replace(w1_clean, rep) + (w2.replace(w2_clean, '') if w2 != w2_clean else ''))
                        i += 2
                        joined_any = True
                        continue
                res.append(words_list[i])
                i += 1
            curr_line = ' '.join(res)
            passes += 1
            if not joined_any:
                break
        cleaned_lines.append(curr_line)
        
    return '\n'.join(cleaned_lines)


def is_target_references_book(source_file: str) -> bool:
    """Deprecated allowlist check. Now generically allows all non-fineweb sources."""
    if not source_file:
        return False
    return source_file != "fineweb-edu"


def remove_inline_references(text: str, source_file: str) -> str:
    """Generically strips inline references for all non-fineweb sources."""
    if not text:
        return ""
    if is_target_references_book(source_file):
        match = INLINE_REF_HEADINGS.search(text)
        if match:
            return text[:match.start()].strip()
    return text


def is_standalone_index_or_bibliography(text: str, source_file: str) -> bool:
    """Identifies standalone index or bibliography pages using line heuristics."""
    if not text:
        return True
        
    if source_file == "fineweb-edu" or not source_file.lower().endswith(".pdf"):
        return False
        
    lines = [l.strip() for l in text.split('\n') if l.strip()]
    if not lines:
        return True
        
    first_few = [l for l in lines[:5]]
    is_index_heading = any(re.search(r'^#+\s+(index|subject index|author index|i\s+n\s+d\s+e\s+x)', l, re.I) for l in first_few)
    is_bib_heading = any(re.search(r'^#+\s+(bibliography|references|selected reading|further reading|suggested reading)', l, re.I) for l in first_few)
    
    page_num_lines = sum(1 for l in lines if re.match(r'^\s*([ivxldcm]+|\d+|\w\s+\w)\s*$', l, re.I))
    
    index_matches = sum(1 for l in lines if INDEX_LINE_PAT.search(l))
    year_matches = sum(1 for l in lines if YEAR_PAT.search(l))
    bib_key_matches = sum(1 for l in lines if BIB_KEYWORDS.search(l))
    author_matches = sum(1 for l in lines if AUTHOR_PREFIX_PAT.match(l))
    citation_format_matches = sum(1 for l in lines if re.search(r'\b(19|20)\d{2}\b', l) and re.search(r'\b(pp\.|vol\.|journal|press|university|ed\.|eds\.|publisher)\b', l, re.I))
    
    n_lines = len(lines)
    index_ratio = index_matches / n_lines
    year_ratio = year_matches / n_lines
    bib_key_ratio = bib_key_matches / n_lines
    author_ratio = author_matches / n_lines
    page_num_ratio = page_num_lines / n_lines
    citation_ratio = citation_format_matches / n_lines
    
    if is_index_heading:
        return True
    if is_bib_heading and n_lines < 15:
        return True
        
    if index_ratio > 0.45 or (index_ratio > 0.3 and page_num_ratio > 0.15):
        return True
        
    if (year_ratio > 0.4 and bib_key_ratio > 0.3) or (author_ratio > 0.3 and year_ratio > 0.4) or citation_ratio > 0.35:
        return True
        
    return False


def is_copyright_or_front_matter(text: str, source_file: str) -> bool:
    """Check if a chunk contains typical copyright/front-matter boilerplate."""
    if not text:
        return True
    if source_file == "fineweb-edu" or not source_file.lower().endswith(".pdf"):
        return False
        
    matches = sum(1 for pat in FRONT_MATTER_PATTERNS if pat.search(text))
    if matches >= 2:
        return True
    if re.search(r'library of congress cataloging-in-publication', text, re.I) or \
       re.search(r'no part of this publication may be reproduced', text, re.I) or \
       re.search(r'digitized by the internet archive', text, re.I):
        return True
    return False


def clean_corpus_text(text: str, source_file: str = "") -> str:
    """
    Clean page layout noise, print proof timestamps, InDesign metadata,
    control bytes, HTML comment tags, PUA glyphs, markdown tables,
    figure captions, inline figure references, TOC listings,
    dehyphenate line-wraps and space-padded hyphens, and repair word splits.
    """
    if not text:
        return ""
        
    # 1. Strip parenthetical inline figure/table/box references: (see Figure 4-2), (Fig. 3B), (Table 1.1)
    text = INLINE_FIG_REF_PAT.sub('', text)
    
    # 2. Spaced-out caps and digit sequences
    text = re.sub(r'\b(?:[A-ZÀ-ÖØ-Þ]\s){2,}[A-ZÀ-ÖØ-Þ]\b', lambda m: m.group(0).replace(" ", ""), text)
    text = re.sub(r'\b(?:\d\s){2,}\d\b', lambda m: m.group(0).replace(" ", ""), text)
    
    # 3. PUA Unicode replacement
    text = text.replace('\uf6ae', '(').replace('\uf6af', ')')
    def pua_repl(m):
        ch = m.group(0)
        if ch in ('\uf6ae', '\uf6af'):
            return ''
        return ' '
    text = PUA_CHAR_PAT.sub(pua_repl, text)
    text = re.sub(r'\(\s+', '(', text)
    text = re.sub(r'\s+\)', ')', text)
    
    # 4. HTML unescape
    text = html.unescape(text)
    
    # 5. Strip HTML comments & placeholders (<!-- image -->, <!-- formula-not-decoded -->, etc.)
    text = HTML_COMMENT_PAT.sub('', text)
    
    # 6. Strip stray triple backtick fences
    text = re.sub(r'^\s*```\w*\s*$', '', text, flags=re.M)
    
    # 7. Metadata, Figure Captions, TOC listings, Markdown tables, and line-level filtering
    lines = text.split('\n')
    cleaned_lines = []
    table_line_count = 0
    toc_line_count = 0
    
    for line in lines:
        stripped = line.strip()
        
        # Skip InDesign proof, ISBN, DOI, Internet Archive, and publisher metadata patterns
        if any(pat.search(stripped) for pat in METADATA_PATTERNS):
            continue
            
        # Skip standalone Figure / Table / Box caption lines
        if FIG_CAPTION_PAT.match(stripped):
            continue
            
        # Check and skip standalone Table of Contents lines or dot leaders
        if TOC_LINE_PAT.match(stripped) or DOT_LEADER_PAT.search(stripped) or len(TOC_PAIR_PAT.findall(stripped)) >= 2:
            toc_line_count += 1
            continue
            
        # Skip standalone page numbers or Roman numerals
        if PAGE_NUM_LINE_PATTERN.match(stripped):
            continue
            
        # Skip standalone isolated callout / diagram noise lines (e.g. isolated "M", "H", "=", "1.0")
        if ISOLATED_CALLOUT_PAT.match(stripped):
            continue
            
        # Check and strip ASCII Markdown table lines
        if TABLE_LINE_PAT.match(stripped) or TABLE_HEADER_SEP_PAT.match(stripped):
            table_line_count += 1
            continue
            
        cleaned_lines.append(line)
        
    # If chunk was dominated by Markdown table structure (> 40% table lines) or TOC lines (> 30%), drop it
    if len(lines) > 0 and ((table_line_count / len(lines)) > 0.40 or (toc_line_count / len(lines)) > 0.30):
        return ""
        
    cleaned_text = '\n'.join(cleaned_lines)
    
    # 8. Dehyphenation pass (handles cross-line and space-padded same-line hyphens)
    cleaned_text = dehyphenate_text(cleaned_text)
    
    # 9. Control character mapping (\x01 rule runs FIRST)
    cleaned_text = re.sub(r'\b(\d+)\s*\x01\s*(\d+)\b', r'\1 x \2', cleaned_text)
    cleaned_text = re.sub(r'\s*\x01\s*(\d+)', r' -\1', cleaned_text)
    cleaned_text = re.sub(r'\s*\x01\s*', '-', cleaned_text)
    
    # 10. Strip remaining non-printable ASCII control characters (\x00-\x1f, \xad, \ufeff)
    cleaned_text = CONTROL_CHAR_PAT.sub('', cleaned_text)
    
    # 11. If source_file is specified, truncate inline references
    if source_file:
        cleaned_text = remove_inline_references(cleaned_text, source_file)
        if not cleaned_text:
            return ""
            
    # 12. Dictionary-based ligature / split-word joining
    cleaned_text = join_ligatures_dict(cleaned_text)
    
    # 13. Check for standalone index, bibliography, or front-matter chunks
    if source_file:
        if is_standalone_index_or_bibliography(cleaned_text, source_file):
            return ""
        if is_copyright_or_front_matter(cleaned_text, source_file):
            return ""
            
    # Normalize multiple consecutive spaces
    cleaned_text = re.sub(r'  +', ' ', cleaned_text)
    
    return cleaned_text.strip()
