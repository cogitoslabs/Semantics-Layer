import json
import re
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent

prompts_path = ROOT_DIR / "evals" / "dapt" / "retrieval_prompts.json"
refs_path = ROOT_DIR / "evals" / "dapt" / "retrieval_references.json"
qa_path = ROOT_DIR / "evals" / "dapt" / "probe_qa.jsonl"
cloze_path = ROOT_DIR / "evals" / "dapt" / "vocab_cloze_set.json"


def clean_eval_text(text: str) -> str:
    if not text:
        return ""
    # 1. Strip pipe characters & markdown table divider artifacts
    text = re.sub(r'\|+', '', text)
    
    # 2. Fix spaced possessive / contraction apostrophes: "one ' s" -> "one's", "one ’ s" -> "one's", "Parkinson ' s" -> "Parkinson's"
    text = re.sub(r"\b([a-zA-Z]+)\s+['\u2019]\s+(s|t|d|re|ve|ll|m)\b", r"\1'\2", text, flags=re.I)
    text = re.sub(r"\b([a-zA-Z]+)\s+['\u2019]\s+([a-zA-Z]+)\b", r"\1'\2", text)
    
    # 3. Fix space-padded hyphens: "Cross - modal" -> "Cross-modal", "base - rate" -> "base-rate"
    text = re.sub(r'\b([a-zA-Z]{2,})\s+-\s*([a-zA-Z]{2,})\b', r'\1-\2', text)
    text = re.sub(r'\b([a-zA-Z]{2,})\s*-\s+([a-zA-Z]{2,})\b', r'\1-\2', text)
    
    # 4. Fix spaces before standard punctuation: "word ." -> "word.", "(e . g . ,)" -> "(e.g.,)"
    text = re.sub(r'\s+([.,;:\?!])', r'\1', text)
    text = re.sub(r'\(\s+', '(', text)
    text = re.sub(r'\s+\)', ')', text)
    
    # 5. Fix "e. g." -> "e.g." and "i. e." -> "i.e."
    text = re.sub(r'\be\s*\.\s*g\s*\.', 'e.g.', text)
    text = re.sub(r'\bi\s*\.\s*e\s*\.', 'i.e.', text)
    
    # 6. Fix spaces inside smart quotes “ word ” -> “word”
    text = re.sub(r'“\s+', '“', text)
    text = re.sub(r'\s+”', '”', text)
    
    # 7. Normalize multiple spaces
    text = re.sub(r'  +', ' ', text)
    return text.strip()


def main():
    print("Starting evaluation datasets cleaning...")
    
    # 1. Clean prompts & refs
    with open(prompts_path, 'r', encoding='utf-8') as f:
        prompts = json.load(f)
    with open(refs_path, 'r', encoding='utf-8') as f:
        refs = json.load(f)

    clean_prompts = [clean_eval_text(p) for p in prompts]
    clean_refs = [clean_eval_text(r) for r in refs]

    with open(prompts_path, 'w', encoding='utf-8') as f:
        json.dump(clean_prompts, f, indent=2, ensure_ascii=False)

    with open(refs_path, 'w', encoding='utf-8') as f:
        json.dump(clean_refs, f, indent=2, ensure_ascii=False)

    print(f"  [OK] Cleaned {len(clean_prompts)} prompts in {prompts_path}")
    print(f"  [OK] Cleaned {len(clean_refs)} references in {refs_path}")

    # 2. Clean QA probe
    clean_qa = []
    with open(qa_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            data["question"] = clean_eval_text(data["question"])
            data["choices"] = [clean_eval_text(c) for c in data.get("choices", [])]
            clean_qa.append(data)

    with open(qa_path, 'w', encoding='utf-8') as f:
        for item in clean_qa:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"  [OK] Cleaned {len(clean_qa)} QA probe items in {qa_path}")

    # 3. Clean Vocab Cloze
    with open(cloze_path, 'r', encoding='utf-8') as f:
        cloze_data = json.load(f)

    clean_cloze = []
    for item in cloze_data:
        new_item = dict(item)
        new_item["prompt"] = clean_eval_text(item.get("prompt", ""))
        new_item["target_term"] = clean_eval_text(item.get("target_term", ""))
        clean_cloze.append(new_item)

    with open(cloze_path, 'w', encoding='utf-8') as f:
        json.dump(clean_cloze, f, indent=2, ensure_ascii=False)

    print(f"  [OK] Cleaned {len(clean_cloze)} Cloze items in {cloze_path}")
    print("\nAll evaluation probe datasets cleaned successfully!")


if __name__ == "__main__":
    main()
