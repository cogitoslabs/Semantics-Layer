import re
from typing import List, Tuple, Optional
import nltk
from nltk.tokenize import sent_tokenize

# Ensure punkt sentence tokenizer is available
try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    try:
        nltk.download("punkt", quiet=True)
    except Exception:
        pass

def character_jaccard(str1: str, str2: str) -> float:
    """Computes character-level Jaccard similarity between two strings using character trigrams."""
    def get_trigrams(s: str) -> set:
        cleaned = "".join(c for c in s.lower() if c.isalnum() or c.isspace())
        return {cleaned[i:i+3] for i in range(len(cleaned) - 2)}
        
    s1 = get_trigrams(str1)
    s2 = get_trigrams(str2)
    if not s1 or not s2:
        return 0.0
    return len(s1 & s2) / len(s1 | s2)

def extract_citations(trace: str) -> List[str]:
    """
    Extracts citations from a teacher trace using three heuristics:
    1. Quoted spans: "..." or '...' longer than 8 characters.
    2. Bracketed/Source markers: [1], [Context 1], [Passage 1], [Source: ...], (Source: ...).
    3. Reference phrases: sentences containing "according to", "as described in", "the context states",
       "the text mentions", "as stated in", "from the passage", "based on the context", "the context provides",
       "the passage states", "is supported by".
    """
    if not trace:
        return []
        
    citations = []
    
    # Heuristic 1: Quoted spans
    double_quotes = re.findall(r'"([^"\n]+)"', trace)
    single_quotes = re.findall(r"'([^'\n]+)'", trace)
    for q in double_quotes + single_quotes:
        if len(q.strip()) > 8:
            citations.append(q.strip())
            
    # Heuristic 2: Bracketed & Source markers
    bracket_matches = re.findall(r"(?:\[(?:Context|Passage|Source)?\s*\d*\]|\(Source:[^\)]+\))", trace, re.IGNORECASE)
    for bm in bracket_matches:
        if bm.strip():
            citations.append(bm.strip())
            
    # Heuristic 3: Expanded reference phrases
    try:
        sentences = sent_tokenize(trace)
    except Exception:
        sentences = re.split(r"[.?!]+", trace)
        
    triggers = [
        "according to", "as described in", "the context states", "the text mentions",
        "as stated in", "from the passage", "based on the context", "the context provides",
        "the passage states", "is supported by", "mentioned in"
    ]
    for i, sent in enumerate(sentences):
        sent_lower = sent.lower()
        if any(t in sent_lower for t in triggers):
            citation_text = sent.strip()
            extra_sents = []
            if i + 1 < len(sentences):
                extra_sents.append(sentences[i+1].strip())
            if extra_sents:
                citation_text += " " + " ".join(extra_sents)
            citations.append(citation_text.strip())
            
    return list(set(citations))

def score_citation_accuracy(
    trace: str, 
    retrieved_context: str, 
    no_retrieval: bool, 
    min_overlap: float = 0.20
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """
    Scores the citation accuracy of a trace against retrieved context.
    Returns: (citation_precision, citation_recall, citation_accuracy)
    All fields are None if no_retrieval is True.
    """
    if no_retrieval:
        return None, None, None
        
    # Split retrieved context back into chunks (they were joined with \n\n)
    chunks = [c.strip() for c in retrieved_context.split("\n\n") if c.strip()]
    if not chunks:
        return 0.0, 0.0, 0.0
        
    citations = extract_citations(trace)
    if not citations:
        # If no explicit citations extracted, check if trace sentences have character overlap with context
        sentences = [s.strip() for s in re.split(r"[.?!]+", trace) if len(s.strip()) > 15]
        grounded_sents = 0
        for sent in sentences:
            if any(character_jaccard(sent, chunk) >= min_overlap for chunk in chunks):
                grounded_sents += 1
        if sentences and grounded_sents > 0:
            citation_precision = grounded_sents / len(sentences)
            citation_recall = min(1.0, grounded_sents / len(chunks))
            citation_accuracy = (2 * citation_precision * citation_recall) / (citation_precision + citation_recall) if (citation_precision + citation_recall) > 0 else 0.0
            return round(citation_precision, 4), round(citation_recall, 4), round(citation_accuracy, 4)
        return 0.0, 0.0, 0.0
        
    supported_citations_count = 0
    chunks_cited = set()
    
    for citation in citations:
        citation_supported = False
        for chunk_idx, chunk in enumerate(chunks):
            sim = character_jaccard(citation, chunk)
            if sim >= min_overlap:
                citation_supported = True
                chunks_cited.add(chunk_idx)
        if citation_supported:
            supported_citations_count += 1
            
    citation_precision = supported_citations_count / len(citations)
    # Recall evaluates coverage against cited or target relevant context chunks
    effective_chunks_target = max(1, min(len(chunks), len(citations)))
    citation_recall = min(1.0, len(chunks_cited) / effective_chunks_target)
    
    if citation_precision + citation_recall > 0:
        citation_accuracy = (2 * citation_precision * citation_recall) / (citation_precision + citation_recall)
    else:
        citation_accuracy = 0.0
        
    return round(citation_precision, 4), round(citation_recall, 4), round(citation_accuracy, 4)
