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
    Extracts citations from a teacher trace using two heuristics:
    1. Quoted spans: "..." or '...' longer than 10 characters.
    2. Reference phrases: sentences containing "according to", "as described in", "the context states".
    """
    if not trace:
        return []
        
    citations = []
    
    # Heuristic 1: Quoted spans
    # Match double quotes and single quotes
    double_quotes = re.findall(r'"([^"\n]+)"', trace)
    single_quotes = re.findall(r"'([^'\n]+)'", trace)
    
    for q in double_quotes + single_quotes:
        if len(q.strip()) > 10:
            citations.append(q.strip())
            
    # Heuristic 2: Reference phrases
    # Tokenize trace into sentences
    try:
        sentences = sent_tokenize(trace)
    except Exception:
        sentences = re.split(r"[.?!]+", trace)
        
    triggers = ["according to", "as described in", "the context states"]
    for i, sent in enumerate(sentences):
        sent_lower = sent.lower()
        if any(t in sent_lower for t in triggers):
            # Extract this sentence and up to 2 subsequent sentences to capture the citation context
            citation_text = sent.strip()
            # Append next 1-2 sentences if they exist
            extra_sents = []
            if i + 1 < len(sentences):
                extra_sents.append(sentences[i+1].strip())
            if i + 2 < len(sentences):
                extra_sents.append(sentences[i+2].strip())
                
            if extra_sents:
                citation_text += " " + " ".join(extra_sents)
                
            citations.append(citation_text.strip())
            
    return citations

def score_citation_accuracy(
    trace: str, 
    retrieved_context: str, 
    no_retrieval: bool, 
    min_overlap: float = 0.30
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
        # Grounded trace but context is empty (should not happen, but handle it)
        return 0.0, 0.0, 0.0
        
    citations = extract_citations(trace)
    if not citations:
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
    citation_recall = len(chunks_cited) / len(chunks)
    
    if citation_precision + citation_recall > 0:
        citation_accuracy = (2 * citation_precision * citation_recall) / (citation_precision + citation_recall)
    else:
        citation_accuracy = 0.0
        
    return citation_precision, citation_recall, citation_accuracy
