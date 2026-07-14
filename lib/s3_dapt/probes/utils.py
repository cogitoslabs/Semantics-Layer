"""
probes/utils.py — Shared utility functions for evaluation probes
"""

def clean_for_match(text: str) -> str:
    """
    Standardize text by stripping punctuation and converting to lowercase
    for robust lexical match comparisons.
    """
    if not isinstance(text, str):
        return ""
    for char in "()[:,;.?!`'\"#-*|]":
        text = text.replace(char, "")
    return text.strip().lower()
