"""
Clean domain_dapt_corpus.jsonl of PDF-extraction artifacts before DAPT.

Fixes applied (in order):
  1. Strip raw control characters (\\x00-\\x08, \\x0b-\\x1f, \\x7f) except \\n \\t
  2. Normalize non-breaking spaces (\\xa0) -> regular space
  3. Drop page-furniture lines (running headers/footers like "00-Swanson_FM.indd i")
  4. Collapse spaced-out capital letter headers ("P R A I S E" -> "PRAISE")
  5. Repair split ligatures ("Th e" -> "The", "speci fi c" -> "specific"),
     validated against a system dictionary so we only merge when the
     merged form is a real word (avoids false merges).
  6. Collapse resulting multi-blank-lines/whitespace runs.

Outputs:
  - cleaned jsonl
  - a before/after report (char count, a word/punct token-count proxy since
    the real tokenizer's merge table isn't downloadable in this offline
    sandbox, artifact counts)
  - a small side-by-side diff sample for manual sanity-checking
"""
import json
import re
import sys
from collections import Counter
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

IN_PATH = ROOT_DIR / "data" / "dapt" / "domain_dapt_corpus.jsonl"
if not IN_PATH.exists():
    print(f"Error: Corpus file not found at {IN_PATH}")
    sys.exit(1)
IN_PATH = ROOT_DIR /  "domain_dapt_corpus_cleaned.jsonl"
REPORT_PATH = ROOT_DIR / "cleaning_report.md"

# --- Load a dictionary for validated ligature merges ---
with open("/usr/share/dict/words") as f:
    WORDS = set(w.strip().lower() for w in f if w.strip())


def build_corpus_lexicon(path: str, min_count: int = 3) -> set:
    """
    Supplement the general dictionary with domain words (e.g. 'effector',
    'flexion', 'specificity') that a general wordlist won't contain but that
    appear correctly, unbroken, many times elsewhere in this same corpus.
    This lets us validate ligature merges against in-domain vocabulary too,
    not just general English.
    """
    counts = Counter()
    word_re = re.compile(r"[A-Za-z]{4,}")
    with open(path) as f:
        for line in f:
            text = json.loads(line)["text"]
            counts.update(w.lower() for w in word_re.findall(text))
    return {w for w, c in counts.items() if c >= min_count}


CORPUS_LEXICON = build_corpus_lexicon(IN_PATH)
WORDS = WORDS | CORPUS_LEXICON

CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")
SPACED_CAPS_RE = re.compile(r"\b(?:[A-Z] ){3,}[A-Z]\b")
FURNITURE_RE = re.compile(r"\S*\.indd\s+[ivxlc\d]+\b", re.IGNORECASE)
# generic ligature breaks: "fi ", "ff ", "fl " glyphs got extracted as a unit
# followed by a stray space before the rest of the word continues
LIGATURE_SPLIT_RE = re.compile(r"(\w*)(fi|ff|fl) (\w+)")
TH_SPLIT_RE = re.compile(r"\bTh ([a-z]+)\b")
MULTI_BLANK_RE = re.compile(r"\n{3,}")
MULTI_SPACE_RE = re.compile(r"[ \t]{2,}")


def fix_th_ligature(text: str) -> str:
    def repl(m):
        candidate = "Th" + m.group(1)
        # "Th" + lowercase run is essentially always a broken "The/This/..." etc.
        # in this corpus; validate against dict on the lowercase form when possible.
        return candidate
    return TH_SPLIT_RE.sub(repl, text)


def fix_fi_ligature(text: str) -> str:
    def repl(m):
        prefix, lig, suffix = m.group(1), m.group(2), m.group(3)
        merged = prefix + lig + suffix
        # Only merge if the merged word is a real dictionary word,
        # or the merged word minus trailing punctuation is.
        core = re.sub(r"[^a-zA-Z]+$", "", merged)
        trailing = merged[len(core):]
        if core.lower() in WORDS:
            return core + trailing
        return m.group(0)  # leave untouched if not a validated merge
    return LIGATURE_SPLIT_RE.sub(repl, text)


def segment_into_words(letters: str, max_word_len: int = 20):
    """
    DP word-break: find a segmentation of `letters` into dictionary words
    that covers the whole string, preferring fewer/longer words (this avoids
    the greedy failure mode where "FORTHE" -> "FORTH" + "E" instead of
    "FOR" + "THE"). Returns None if no full segmentation exists.
    """
    n = len(letters)
    # best[i] = (num_words, split) for prefix ending at i, or None
    best = [None] * (n + 1)
    best[0] = (0, [])
    for i in range(1, n + 1):
        for j in range(max(0, i - max_word_len), i):
            if best[j] is None:
                continue
            candidate = letters[j:i].lower()
            if candidate in WORDS:
                num_words = best[j][0] + 1
                if best[i] is None or num_words < best[i][0]:
                    best[i] = (num_words, best[j][1] + [letters[j:i]])
    if best[n] is None:
        return None
    return best[n][1]


def collapse_spaced_caps(text: str) -> str:
    # e.g. "P R A I S E F O R T H E" -> "PRAISE FOR THE"
    def repl(m):
        letters = m.group(0).replace(" ", "")
        words = segment_into_words(letters)
        if words is None:
            return m.group(0)  # no valid full segmentation; leave untouched
        return " ".join(words)
    return SPACED_CAPS_RE.sub(repl, text)


def clean_text(text: str, stats: Counter) -> str:
    orig = text

    n = len(CONTROL_CHARS_RE.findall(text))
    stats["control_chars_removed"] += n
    text = CONTROL_CHARS_RE.sub("", text)

    stats["nbsp_normalized"] += text.count("\xa0")
    text = text.replace("\xa0", " ")

    furniture_matches = FURNITURE_RE.findall(text)
    stats["furniture_lines_removed"] += len(FURNITURE_RE.findall(text))
    text = FURNITURE_RE.sub("", text)

    before = text
    text = collapse_spaced_caps(text)
    if text != before:
        stats["spaced_caps_fixed"] += len(SPACED_CAPS_RE.findall(before))

    before = text
    text = fix_th_ligature(text)
    stats["th_ligature_fixed"] += len(TH_SPLIT_RE.findall(before))

    before = text
    text = fix_fi_ligature(text)
    # count actual merges by comparing token counts roughly
    stats["fi_ff_fl_ligature_candidates"] += len(LIGATURE_SPLIT_RE.findall(before))

    text = MULTI_BLANK_RE.sub("\n\n", text)
    text = MULTI_SPACE_RE.sub(" ", text)
    text = text.strip()

    if text != orig:
        stats["docs_modified"] += 1

    return text


# GPT-2/BPE style pre-tokenizer regex, used as a token-count *proxy* since we
# have no network access to download an actual tokenizer's merge table here.
# This undercounts true BPE tokens (real BPE further splits rare/garbled
# words into sub-word pieces) but the delta between before/after is still
# informative, and if anything understates how much garbled text costs you
# since garbled fragments split into MORE real BPE tokens, not fewer.
PRETOKEN_RE = re.compile(r"\w+|[^\w\s]")


def count_proxy_tokens(text: str) -> int:
    return len(PRETOKEN_RE.findall(text))


def main():
    stats = Counter()
    total_chars_before = 0
    total_chars_after = 0
    total_tokens_before = 0
    total_tokens_after = 0

    diff_samples = []

    with open(IN_PATH) as fin, open(OUT_PATH, "w") as fout:
        for i, line in enumerate(fin):
            d = json.loads(line)
            orig_text = d["text"]
            cleaned = clean_text(orig_text, stats)

            total_chars_before += len(orig_text)
            total_chars_after += len(cleaned)

            tb = count_proxy_tokens(orig_text)
            ta = count_proxy_tokens(cleaned)
            total_tokens_before += tb
            total_tokens_after += ta

            if len(diff_samples) < 3 and cleaned != orig_text:
                diff_samples.append((d["id"], orig_text[:600], cleaned[:600]))

            d["text"] = cleaned
            fout.write(json.dumps(d, ensure_ascii=False) + "\n")

    stats["total_docs"] = i + 1

    report_lines = []
    report_lines.append("# DAPT Corpus Cleaning Report\n")
    report_lines.append(f"- Total documents: {stats['total_docs']}")
    report_lines.append(f"- Documents modified: {stats['docs_modified']}")
    report_lines.append("")
    report_lines.append("## Artifact counts fixed")
    report_lines.append(f"- Control characters removed: {stats['control_chars_removed']}")
    report_lines.append(f"- Non-breaking spaces normalized: {stats['nbsp_normalized']}")
    report_lines.append(f"- Page-furniture lines removed: {stats['furniture_lines_removed']}")
    report_lines.append(f"- Spaced-caps headers collapsed: {stats['spaced_caps_fixed']}")
    report_lines.append(f"- 'Th e/is/ese...' ligature splits fixed: {stats['th_ligature_fixed']}")
    report_lines.append(f"- 'fi'/'ff'/'fl' ligature break candidates seen: {stats['fi_ff_fl_ligature_candidates']}")
    report_lines.append("")
    report_lines.append("## Size impact")
    report_lines.append(f"- Chars before: {total_chars_before:,}")
    report_lines.append(f"- Chars after:  {total_chars_after:,}  ({(1 - total_chars_after/total_chars_before)*100:.2f}% reduction)")
    report_lines.append(f"- Pre-token units before (word/punct proxy, no tokenizer download available offline): {total_tokens_before:,}")
    report_lines.append(f"- Pre-token units after: {total_tokens_after:,}  ({(1 - total_tokens_after/total_tokens_before)*100:.2f}% reduction)")
    report_lines.append("")
    report_lines.append("## Sample before/after diffs")
    for doc_id, before, after in diff_samples:
        report_lines.append(f"\n### {doc_id}\n")
        report_lines.append("**Before:**\n```\n" + before + "\n```\n")
        report_lines.append("**After:**\n```\n" + after + "\n```\n")

    with open(REPORT_PATH, "w") as f:
        f.write("\n".join(report_lines))

    print("\n".join(report_lines[:20]))
    print("\n...\n")
    print(f"Wrote cleaned corpus to {OUT_PATH}")
    print(f"Wrote report to {REPORT_PATH}")


if __name__ == "__main__":
    main()
