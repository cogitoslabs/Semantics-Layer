import re
import string
from collections import Counter
from typing import List, Optional

def normalize_text(text: str) -> str:
    """Lowercases text, strips punctuation, and normalizes whitespace."""
    if not text:
        return ""
    text = text.lower().strip()
    # Remove punctuation
    translator = str.maketrans("", "", string.punctuation)
    text = text.translate(translator)
    # Normalize whitespace
    return " ".join(text.split())

def extract_answer(trace: str) -> str:
    """
    Extracts the answer from the trace.
    Finds the content of the last \boxed{...} block.
    If none found, falls back to the last non-empty sentence.
    """
    if not trace:
        return ""
        
    # Find all \boxed{...} occurrences. Handle balanced brackets in a simple way.
    # We find \boxed{ and match characters until the matching }
    # To handle nested braces, we can use a recursive regex or search from right to left.
    # Since we want the last one, let's search from the end.
    matches = re.findall(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}", trace)
    if matches:
        return matches[-1].strip()
        
    # Fallback: last non-empty sentence
    # Split by common sentence endings: . ? !
    sentences = re.split(r"[.?!]+", trace)
    for s in reversed(sentences):
        cleaned = s.strip()
        if cleaned:
            return cleaned
            
    return trace.strip()

def compute_mc_accuracy(extracted: str, ground_truth: str, choices: List[str]) -> float:
    """
    Computes accuracy for multiple choice questions.
    Returns 1.0 if matched, else 0.0.
    """
    norm_extracted = normalize_text(extracted)
    norm_gt = normalize_text(ground_truth)
    
    # Direct match on choice text
    if norm_extracted == norm_gt:
        return 1.0
        
    # Find correct choice index
    correct_idx = -1
    for idx, choice in enumerate(choices):
        if normalize_text(choice) == norm_gt:
            correct_idx = idx
            break
            
    if correct_idx != -1:
        correct_letter = chr(ord('a') + correct_idx)
        
        # Check if the extracted text is exactly the letter (e.g. "a", "b", "c")
        if norm_extracted == correct_letter:
            return 1.0
            
        # Check if it matches "option a", "choice a", "a answer", etc.
        if norm_extracted in [f"option {correct_letter}", f"choice {correct_letter}", f"answer {correct_letter}"]:
            return 1.0
            
    return 0.0

def compute_f1_overlap(extracted: str, ground_truth: str) -> float:
    """
    Computes token-level F1 overlap score (SQuAD style) between two texts.
    """
    norm_extracted = normalize_text(extracted)
    norm_gt = normalize_text(ground_truth)
    
    pred_tokens = norm_extracted.split()
    ref_tokens = norm_gt.split()
    
    if not pred_tokens or not ref_tokens:
        return 1.0 if pred_tokens == ref_tokens else 0.0
        
    common = Counter(pred_tokens) & Counter(ref_tokens)
    num_same = sum(common.values())
    
    if num_same == 0:
        return 0.0
        
    precision = num_same / len(pred_tokens)
    recall = num_same / len(ref_tokens)
    
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def score_answer_accuracy(trace: str, ground_truth: str, choices: Optional[List[str]] = None) -> float:
    """
    Scores the accuracy of the extracted answer vs ground truth.
    If choices are provided, treats as multiple-choice; otherwise, treats as free-form F1.
    """
    extracted = extract_answer(trace)
    if choices:
        return compute_mc_accuracy(extracted, ground_truth, choices)
    else:
        return compute_f1_overlap(extracted, ground_truth)
