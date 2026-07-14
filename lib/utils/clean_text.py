import re

# Ligature regex rules (supports case preservation for words starting with T/t)
LIGATURE_RULES = [
    # th / Th splits
    (re.compile(r'\b([tT])h\s+(e|is|ey|ese|ere|eir|ink|inking|ird|ree|rough|roughout|ought|o|omas|ailand|read|readship)\b'), r'\1h\2'),
    # th splits within words (e.g. o th er -> other, whe th er -> whether)
    (re.compile(r'\b(\w+)\s*th\s+er(s|ly|n)?\b'), r'\1ther\2'),
    # ff splits
    (re.compile(r'\b(\w*)eff\s+ect(s|ive|ively)?\b'), r'\1effect\2'),
    (re.compile(r'\b(\w*)eff\s+ort(s)?\b'), r'\1effort\2'),
    (re.compile(r'\b(\w*)diff\s+er(s|ent|ently|ence|ences|ing)?\b'), r'\1differ\2'),
    (re.compile(r'\b(\w*)suff\s+er(s|ed|ing)?\b'), r'\1suffer\2'),
    (re.compile(r'\b(\w*)off\s+er(s|ed|ing)?\b'), r'\1offer\2'),
    # fl splits
    (re.compile(r'\b(\w*)refl\s+ect(s|ed|ing|ion|ions|ive)?\b'), r'\1reflect\2'),
    (re.compile(r'\b(\w*)refl\s+ex(es)?\b'), r'\1reflex\2'),
    (re.compile(r'\b(\w*)infl\s+uence(s|d)?\b'), r'\1influence\2'),
    (re.compile(r'\b(\w*)confl\s+ict(s)?\b'), r'\1conflict\2'),
    # fi splits
    (re.compile(r'\b(\w*)affi\s+nity\b'), r'\1affinity'),
    (re.compile(r'\b(\w*)offi\s+ce(s)?\b'), r'\1office\2'),
    (re.compile(r'\b(\w*)offi\s+cial(s|ly)?\b'), r'\1official\2'),
    (re.compile(r'\b(\w*)specifi\s+c(s|ally)?\b'), r'\1specific\2'),
    (re.compile(r'\b(\w*)identifi\s+e(d|s|r)?\b'), r'\1identifie\2'),
    (re.compile(r'\b(\w*)signifi\s+cant(ly)?\b'), r'\1significant\2'),
    (re.compile(r'\b(\w*)classifi\s+cation(s)?\b'), r'\1classification\2'),
    (re.compile(r'\b(\w*)modifi\s+cation(s)?\b'), r'\1modification\2'),
    (re.compile(r'\b(\w*)clari\s+ty\b'), r'\1clarity'),
    (re.compile(r'\b(\w*)diffi\s+cult(y|ies)?\b'), r'\1difficult\2'),
    (re.compile(r'\b(\w*)defi\s+ne(d|s|ing)?\b'), r'\1define\2'),
    (re.compile(r'\b(\w*)defi\s+nition(s)?\b'), r'\1definition\2'),
    (re.compile(r'\b(\w*)fi\s+rst\b'), r'\1first'),
    (re.compile(r'\b(\w*)fi\s+nd(s|ing)?\b'), r'\1find\2'),
    (re.compile(r'\b(\w*)fi\s+gure(s)?\b'), r'\1figure\2'),
    (re.compile(r'\b(\w*)fi\s+nally\b'), r'\1finally'),
    (re.compile(r'\b(\w*)fi\s+eld(s)?\b'), r'\1field\2'),
    (re.compile(r'\b(\w*)fi\s+ve\b'), r'\1five'),
    (re.compile(r'\b(\w*)fi\s+ber(s)?\b'), r'\1fiber\2'),
    (re.compile(r'\b(\w*)fi\s+lament(s)?\b'), r'\1filament\2'),
    (re.compile(r'\b(\w*)fl\s+ip\b'), r'\1flip'),
    # physiology / psychiatry splits
    (re.compile(r'\b(\w*)physi\s+ology\b'), r'\1physiology'),
    (re.compile(r'\b(\w*)physi\s+ological(ly)?\b'), r'\1physiological\2'),
    (re.compile(r'\b(\w*)psychi\s+atry\b'), r'\1psychiatry'),
    (re.compile(r'\b(\w*)psychi\s+atrist(s)?\b'), r'\1psychiatrist\2'),
]

# Layout and metadata patterns
METADATA_PATTERNS = [
    # InDesign proof files: 00-Swanson\_FM.indd vi, etc.
    re.compile(r'\b\d{2}-\w+(?:\\_)?\w*\.indd(?:\s+\w+|\s+Sec\d+:\d+)?\b'),
    # Timestamps: 5/28/2011 9:40:34 AM
    re.compile(r'\b\d{1,2}/\d{1,2}/\d{4}\s+\d{1,2}:\d{2}:\d{2}\s+(?:AM|PM)\b'),
    # "This page intentionally left blank"
    re.compile(r'(?i)\bthis page intentionally left blank\b'),
]

# Page number standalone line pattern
PAGE_NUM_LINE_PATTERN = re.compile(r'^\s*([ivxldcmIVXLDCM]+|\d+|\w\s+\w)\s*$')


import html

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

# Author name prefix pattern: e.g., "Sanes, J. R." or "Sanes JR"
AUTHOR_PREFIX_PAT = re.compile(r'^(?:[-\s*•]*)\s*[A-Z][a-zA-Z\s]+,\s+[A-Z](?:\.[A-Z])*(?:\s*,\s*|and\s+)?', re.I)

# Inline reference headings pattern
INLINE_REF_HEADINGS = re.compile(
    r'^#+\s+(?:References|Selected Readings?|Suggested Readings?|Further Readings?|Bibliography)\b.*$',
    re.I | re.M
)

def is_target_references_book(source_file: str) -> bool:
    if not source_file:
        return False
    lower_name = source_file.lower()
    return any(name in lower_name for name in [
        "principles of neural science",
        "neuroscience exploring",
        "fundamentals of cognitive neuroscience"
    ])

def remove_inline_references(text: str, source_file: str) -> str:
    if not text:
        return ""
    if is_target_references_book(source_file):
        match = INLINE_REF_HEADINGS.search(text)
        if match:
            # Keep only narrative before the heading
            return text[:match.start()].strip()
    return text

def is_standalone_index_or_bibliography(text: str, source_file: str) -> bool:
    if not text:
        return True
        
    lines = [l.strip() for l in text.split('\n') if l.strip()]
    if not lines:
        return True
        
    # Check if first few lines have clear Index or Bibliography headings
    first_few = [l for l in lines[:5]]
    is_index_heading = any(re.search(r'^#+\s+(index|subject index|author index|i\s+n\s+d\s+e\s+x)', l, re.I) for l in first_few)
    is_bib_heading = any(re.search(r'^#+\s+(bibliography|references|selected reading|further reading|suggested reading)', l, re.I) for l in first_few)
    
    # Page numbers lines pattern (lines that are just page numbers or roman numerals)
    page_num_lines = sum(1 for l in lines if re.match(r'^\s*([ivxldcm]+|\d+|\w\s+\w)\s*$', l, re.I))
    
    index_matches = sum(1 for l in lines if INDEX_LINE_PAT.search(l))
    year_matches = sum(1 for l in lines if YEAR_PAT.search(l))
    bib_key_matches = sum(1 for l in lines if BIB_KEYWORDS.search(l))
    author_matches = sum(1 for l in lines if AUTHOR_PREFIX_PAT.match(l))
    
    n_lines = len(lines)
    index_ratio = index_matches / n_lines
    year_ratio = year_matches / n_lines
    bib_key_ratio = bib_key_matches / n_lines
    author_ratio = author_matches / n_lines
    page_num_ratio = page_num_lines / n_lines
    
    if is_index_heading:
        return True
    if is_bib_heading and n_lines < 15: # if it has the heading but is short, it's probably standalone
        return True
        
    # If a high percentage of lines are index lines
    if index_ratio > 0.45 or (index_ratio > 0.3 and page_num_ratio > 0.15):
        return True
        
    # If a high percentage of lines contain citations
    # Bibliography chunks often have high year ratio and bib keyword ratio
    if (year_ratio > 0.4 and bib_key_ratio > 0.3) or (author_ratio > 0.3 and year_ratio > 0.4):
        return True
        
    return False

def clean_corpus_text(text: str) -> str:
    """
    Clean page layout noise, print proof timestamps, InDesign metadata,
    and ligature word splits from parsed PDF text chunks.
    """
    if not text:
        return ""
        
    # 1. HTML unescape
    text = html.unescape(text)
    
    # 2. Strip <!-- image --> placeholders
    text = re.sub(r'<!--\s*image\s*-->', '', text, flags=re.I)
    
    # 3. Strip stray triple backtick fences
    text = re.sub(r'^\s*```\w*\s*$', '', text, flags=re.M)
    
    lines = text.split('\n')
    cleaned_lines = []
    
    for line in lines:
        stripped = line.strip()
        # 4. Skip lines matching metadata patterns
        if any(pat.search(stripped) for pat in METADATA_PATTERNS):
            continue
        # 5. Skip lines that are just page numbers or roman numerals
        if PAGE_NUM_LINE_PATTERN.match(stripped):
            continue
            
        # 6. Apply ligature and split word repairs
        cleaned_line = line
        for pat, repl in LIGATURE_RULES:
            cleaned_line = pat.sub(repl, cleaned_line)
            
        cleaned_lines.append(cleaned_line)
        
    return '\n'.join(cleaned_lines)
