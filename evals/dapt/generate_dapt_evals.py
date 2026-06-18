"""
evals/dapt/generate_dapt_evals.py — Procedural DAPT Evaluation Dataset Generator

This script parses the pretraining corpus to generate domain-adapted evaluations:
1. Splits off a 20% validation text split for held-out perplexity evaluation.
2. Extracts sentences and masks key terms to build a Cloze dataset.
3. Generates multiple-choice QA items using the masked sentences and distractors.
4. Curates anatomical reference paragraphs and prompt descriptions.
"""

import json
import random
import re
from pathlib import Path

SEED = 42
random.seed(SEED)

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
CORPUS_PATH = ROOT_DIR / "data/dapt/domain_dapt_corpus.jsonl"
EVALS_DIR = ROOT_DIR / "evals/dapt"

KEYWORDS = [
    "hippocampus", "amygdala", "striatum", "prefrontal", "cerebellum",
    "thalamus", "cortex", "fMRI", "lesion", "dopamine", "neuropsychology",
    "neuroscience", "cognition", "perception", "attention", "Stroop",
    "modularity"
]

ANATOMY_KEYWORDS = [
    "hippocampus", "amygdala", "striatum", "cerebellum", "cortex", "thalamus"
]


def load_corpus(corpus_path: Path):
    if not corpus_path.exists():
        raise FileNotFoundError(f"Corpus not found at: {corpus_path}")
    docs = []
    with open(corpus_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                docs.append(json.loads(line))
    return docs


def split_sentences(text: str):
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip().replace("\n", " ") for s in sentences if len(s.strip()) > 20]


def mask_keyword(sentence: str, keyword: str):
    pattern = re.compile(r'\b' + re.escape(keyword) + r'\b', re.IGNORECASE)
    match = pattern.search(sentence)
    if not match:
        return None
    return pattern.sub("___", sentence, count=1)


def main():
    print(f"Loading corpus from {CORPUS_PATH}...")
    docs = load_corpus(CORPUS_PATH)
    print(f"Loaded {len(docs)} document chunks.")

    # 1. 20% Validation split for held-out perplexity
    val_size = max(2, int(len(docs) * 0.20))
    if len(docs) <= val_size:
        val_size = max(1, len(docs) // 2)

    val_docs = docs[-val_size:]
    train_docs = docs[:-val_size]

    val_text = "\n".join(doc.get("text", "") for doc in val_docs if doc.get("text", "").strip())
    
    EVALS_DIR.mkdir(parents=True, exist_ok=True)

    # Helper to write to evals/dapt/
    def write_file(filename: str, content: str):
        p_eval = EVALS_DIR / filename
        p_eval.write_text(content, encoding="utf-8")

    # 1. 20% Validation split for held-out perplexity
    write_file("ppl_held_out.txt", val_text)
    print(f"[SUCCESS] Generated ppl_held_out.txt ({len(val_text):,} characters)")

    # Extract sentences
    print("Extracting sentences from pretraining corpus...")
    all_sentences = []
    for doc in train_docs:
        all_sentences.extend(split_sentences(doc.get("text", "")))
    print(f"Extracted {len(all_sentences):,} sentences.")

    # 2. Generate Cloze items & MCQ QA items
    cloze_items = []
    qa_items = []
    used_sentences = set()

    for keyword in KEYWORDS:
        count = 0
        for sentence in all_sentences:
            if sentence in used_sentences:
                continue
            masked = mask_keyword(sentence, keyword)
            if masked:
                cloze_items.append({
                    "prompt": masked,
                    "target_term": keyword,
                    "category": "anatomy" if keyword in ANATOMY_KEYWORDS else "cognitive_psychology"
                })
                used_sentences.add(sentence)

                distractors = [k for k in KEYWORDS if k != keyword]
                selected_distractors = random.sample(distractors, 3)
                choices = [keyword] + selected_distractors
                random.shuffle(choices)
                answer_idx = choices.index(keyword)

                qa_items.append({
                    "question": f"What term completes the following description: '{masked}'?",
                    "choices": choices,
                    "answer_idx": answer_idx,
                    "cluster": keyword
                })

                count += 1
                if count >= 3:
                    break

    write_file("vocab_cloze_set.json", json.dumps(cloze_items, indent=2))
    print(f"[SUCCESS] Generated vocab_cloze_set.json ({len(cloze_items)} items)")

    qa_jsonl_content = "".join(json.dumps(item) + "\n" for item in qa_items)
    write_file("probe_qa.jsonl", qa_jsonl_content)
    print(f"[SUCCESS] Generated probe_qa.jsonl ({len(qa_items)} items)")

    # 3. Generate Anatomical Prompts & References
    anatomical_prompts = []
    anatomical_references = []
    anatomy_chunks = []

    for doc in train_docs:
        text = doc.get("text", "")
        paras = [p.strip() for p in text.split("\n\n") if len(p.strip()) > 100]
        for p in paras:
            for kw in ANATOMY_KEYWORDS:
                if re.search(r'\b' + re.escape(kw) + r'\b', p, re.IGNORECASE):
                    if any(w in p.lower() for w in ["located", "situated", "structure", "anatomy", "connect"]):
                        anatomy_chunks.append((kw, p))
                        break

    random.shuffle(anatomy_chunks)
    selected_anatomy = []
    seen_kw = set()
    for kw, p in anatomy_chunks:
        if kw not in seen_kw:
            selected_anatomy.append((kw, p))
            seen_kw.add(kw)
        if len(selected_anatomy) >= 5:
            break

    if len(selected_anatomy) < 3:
        selected_anatomy = [
            ("hippocampus", "The hippocampus is a C-shaped structure located within the hippocampal formation in the medial temporal lobe. It is critical for memory consolidation and spatial navigation."),
            ("cerebellum", "The cerebellum is situated at the back of the brain, underlying the occipital and temporal lobes. It is responsible for motor control and motor learning."),
            ("amygdala", "The amygdala is a collection of nuclei located deep within the temporal lobe. It plays a primary role in processing emotional responses, particularly fear and pleasure.")
        ]

    for kw, ref in selected_anatomy:
        anatomical_prompts.append(f"Describe the anatomical location and primary functions of the {kw}.")
        clean_ref = re.sub(r'<!--.*?-->', '', ref).strip()
        anatomical_references.append(clean_ref)

    write_file("anatomical_prompts.json", json.dumps(anatomical_prompts, indent=2))
    write_file("anatomical_references.json", json.dumps(anatomical_references, indent=2))
    print(f"[SUCCESS] Generated anatomical_prompts.json & anatomical_references.json ({len(anatomical_prompts)} items)")


if __name__ == "__main__":
    main()
