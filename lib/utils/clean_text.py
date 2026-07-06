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


def clean_corpus_text(text: str) -> str:
    """
    Clean page layout noise, print proof timestamps, InDesign metadata,
    and ligature word splits from parsed PDF text chunks.
    """
    if not text:
        return ""
        
    lines = text.split('\n')
    cleaned_lines = []
    
    for line in lines:
        stripped = line.strip()
        # 1. Skip lines matching metadata patterns
        if any(pat.search(stripped) for pat in METADATA_PATTERNS):
            continue
        # 2. Skip lines that are just page numbers or roman numerals
        if PAGE_NUM_LINE_PATTERN.match(stripped):
            continue
            
        # 3. Apply ligature and split word repairs
        cleaned_line = line
        for pat, repl in LIGATURE_RULES:
            cleaned_line = pat.sub(repl, cleaned_line)
            
        cleaned_lines.append(cleaned_line)
        
    return '\n'.join(cleaned_lines)
