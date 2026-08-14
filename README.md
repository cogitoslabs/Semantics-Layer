# Semantics Layer Pipeline

This repository implements the Semantics Layer Pipeline for building a neuroscience-specialist Small Language Model (SLM; 1B–4B parameters) deployable on edge devices (laptop/mobile via INT4 quantization) through distillation of fine-tuned Teacher LLMs on CND projection layer outputs.

---

## 1. Objectives of the Project

1. **Neuroscience Knowledge Substrate**: Build a comprehensive domain knowledge substrate via Domain Adaptive Pretraining (DAPT) on 10B–50B tokens of scientific literature (PubMed Central, bioRxiv, neuroscience textbooks, lecture notes, and atlas descriptions).
2. **Knowledge Before Reasoning**: Establish solid factual recall, terminology familiarity, and retrieval precision in the student model before attempting reasoning distillation.
3. **Retrieval-Augmented Distillation (RAD Prep)**: Index reference corpora using hybrid dense/sparse retrieval and query Teacher LLMs (e.g. Qwen3-1.7B, DeepSeek, AWS Bedrock) to generate grounded Chain-of-Thought (CoT) reasoning traces.
4. **Corpus Engineering & Micro-Clustering**: Discover latent neuroscience sub-domains (micro-clusters) via sentence embeddings, PCA dimensionality reduction, and HDBSCAN clustering to analyze domain coverage and prevent distribution imbalance.
5. **Multi-Teacher Benchmarking & Election**: Evaluate candidate teacher models per micro-cluster across 4 dimensions (answer accuracy, reasoning quality via LLM judge, citation accuracy, and NLI hallucination detection) to elect the optimal teacher per sub-domain.
6. **Quantization-Aware Deployment**: Deliver a calibrated, hallucination-resistant, INT4-quantized SLM optimized for scientific reasoning and research assistance.

---

## 2. Execution Pipeline Steps

The pipeline consists of 6 sequential steps executed via [pipeline.py](file:///e:/Projects/cnd/Semantics/pipeline.py) or `pipeline.ipynb`:

| Step | Pipeline Step & Module | Core Responsibilities & Description | Documentation |
| :---: | :--- | :--- | :---: |
| **s1** | **Corpus Construction & Deduplication**<br>[`lib/s1_build_corpus`](lib/s1_build_corpus) | Extracts document text from scientific PDFs in parallel using GPU/CPU Docling workers, strips running headers/footers, streams general web replay data, and merges files with zero-dependency MinHash LSH deduplication. | [`S1_BUILD_CORPUS.md`](docs/S1_BUILD_CORPUS.md) |
| **s2** | **Offline Pre-tokenization**<br>[`lib/s2_pretokenize`](lib/s2_pretokenize) | Shuffles the unified corpus reproducibly, partitions documents into 95% training and 5% validation splits, and tokenizes text using the student model tokenizer to export flat 32-bit integer NumPy arrays. | [`S2_PRETOKENIZE.md`](docs/S2_PRETOKENIZE.md) |
| **s3** | **Domain Adaptive Pretraining (DAPT)**<br>[`lib/s3_dapt`](lib/s3_dapt) | Performs continued causal LM pretraining (full or PEFT-LoRA) on the student model while running online multi-probe evaluations to enforce dynamic multi-gate early stopping decisions. | [`S3_DAPT.md`](docs/S3_DAPT.md) |
| **s4** | **Retrieval-Augmented Distillation Prep**<br>[`lib/s4_rad_prep`](lib/s4_rad_prep) | Chunks reference documents, builds hybrid FAISS/BM25 vector indices, routes QA samples based on context relevance, and queries Teacher LLMs to generate Chain-of-Thought reasoning traces. | [`S4_RAD_PREP.md`](docs/S4_RAD_PREP.md) |
| **s5** | **Corpus Engineering & Micro-Clustering**<br>[`lib/s5_clustering`](lib/s5_clustering) | Generates document embeddings, applies PCA dimensionality reduction, and runs HDBSCAN clustering to discover latent neuroscience micro-clusters while partitioning per-cluster 3-way splits (dev/val/sealed). | [`S5_CLUSTERING.md`](docs/S5_CLUSTERING.md) |
| **s6** | **Teacher Benchmarking**<br>[`lib/s6_teacher_benchmarking`](lib/s6_teacher_benchmarking) | Benchmarks candidate teacher models across micro-clusters on 4 dimensions (answer accuracy, reasoning quality via LLM judge, citation accuracy, NLI hallucination rate) and computes Cohen's Kappa calibration. | [`S6_TEACHER_BENCHMARKING.md`](docs/S6_TEACHER_BENCHMARKING.md) |

---

## 3. Directory & Module Structure

```
Semantics/
├── .agents/                 # Workspace-scoped agent skills and customized rules
├── .env                     # Local environment configuration file
├── .env.example             # Example environment configuration template
├── CLAUDE.md                # Claude AI developer guidelines and environment specs
├── GEMINI.md                # Gemini AI agent rules and context guidelines
├── README.md                # Root project documentation (this file)
├── pyproject.toml           # Python dependencies and build configuration (uv / pip)
├── pipeline.py              # Central Command Line Interface (CLI) entry point for executing steps
├── pipeline.ipynb           # Jupyter Notebook / Google Colab interactive pipeline execution harness
├── context/                 # Program specs, coding standards, change history, and interaction guidelines
│   ├── ai-interaction.md    # Workflow standards, branching rules, and communication rules
│   ├── change-history.md    # Historical log of implemented features and codebase modifications
│   ├── current-project-overview.md  # Detailed technical spec for all pipeline phases
│   ├── python-coding-standards.md   # PEP8 and project-specific Python coding conventions
│   └── feature-specs/       # Specification documents for individual pipeline features
├── data/                    # Pipeline input/output data artifacts (gitignored binary/JSONL files)
│   ├── dapt/                # Raw PDFs, extracted JSONL, tokenized arrays (train_tokens.npy, etc.)
│   ├── rad_prep/            # Reference chunks, FAISS vector index, and generated CoT traces
│   ├── clustering/          # Embedding arrays, cluster assignments, and splits.json
│   └── benchmarking/        # Per-teacher x cluster evaluation scores and human ground-truth labels
├── docs/                    # Step-by-step technical documentation specs
│   ├── S1_BUILD_CORPUS.md   # Step 1 specification: Corpus Construction & MinHash LSH
│   ├── S2_PRETOKENIZE.md    # Step 2 specification: Offline Pre-tokenization
│   ├── S3_DAPT.md           # Step 3 specification: Domain Adaptive Pretraining & Multi-Probe Gates
│   ├── S4_RAD_PREP.md       # Step 4 specification: RAD Prep, Vector Indexing & Trace Generation
│   ├── S5_CLUSTERING.md     # Step 5 specification: Micro-Clustering & 3-Way Splits
│   └── S6_TEACHER_BENCHMARKING.md # Step 6 specification: Teacher Evaluation & Cohen's Kappa
├── evals/                   # Evaluation probe datasets and held-out validation benchmarks
│   └── dapt/                # QA probes, cloze sets, and concept retrieval prompts/references
├── lib/                     # Core Python source packages for all pipeline steps
│   ├── s1_build_corpus/     # Docling parser pool, FineWeb replay stream, MinHash LSH merger
│   ├── s2_pretokenize/      # Corpus shuffler, train/val splitter, and NumPy tokenizer writer
│   ├── s3_dapt/             # Training loop, MemmapDataset, multi-probe evaluators, and gate engine
│   ├── s4_rad_prep/         # Document chunker, BioLinkBERT dense embedder, FAISS/BM25 retriever, router
│   ├── s5_clustering/       # SentenceTransformer embedder, PCA reducer, HDBSCAN clusterer, splitter
│   ├── s6_teacher_benchmarking/ # Teacher response generator, LLM judge, NLI hallucination detector
│   └── utils/               # Centralized configuration loader (config.py), storage adapters, logger
├── logs/                    # Runtime logs, evaluation metrics, and phase completion manifests
├── models/                  # Staged base model weights, PEFT-LoRA adapters, and checkpoint outputs
├── scripts/                 # Maintenance, dataset conversion, and utility scripts
├── tests/                   # Automated Pytest unit and integration test suites
└── ui/                      # Streamlit web UI for interactive probe inspection (online_probes.py)
```

---

## 4. How to Test & Run the Pipeline

### Virtual Environment Setup

Ensure dependencies are installed using `uv` or `pip`:

```bash
# Using uv (recommended)
uv sync

# Or using pip
python -m venv .venv
.venv\Scripts\activate      # Windows
source .venv/bin/activate    # Linux/macOS
pip install -r pyproject.toml
```

### Running Pipeline Steps via CLI (`pipeline.py`)

Execute individual pipeline steps or run the entire pipeline sequentially:

```bash
# Run Step 1: Corpus Construction & Deduplication
python pipeline.py --step s1

# Run Step 2: Offline Pre-tokenization
python pipeline.py --step s2

# Run Step 3: Domain Adaptive Pretraining (DAPT)
python pipeline.py --step s3

# Run Step 4: RAD Prep (sub-modes: index, traces, or full)
python pipeline.py --step s4 --rad-mode full

# Run Step 5: Corpus Engineering & Micro-Clustering
python pipeline.py --step s5

# Run Step 6: Teacher Benchmarking
python pipeline.py --step s6

# Run all steps sequentially (s1 through s6)
python pipeline.py --step all
```

### Interactive Execution via Notebook (`pipeline.ipynb`)

For Google Colab or local Jupyter environments, run `pipeline.ipynb` to customize `cfg` attributes dynamically:

```python
from lib.utils import load_config
from lib.s3_dapt import run_dapt_pipeline

# 1. Load pipeline configuration
cfg = load_config()

# 2. Configure model & parameters
cfg.model.base_model_name = "Qwen/Qwen3-0.6B"
cfg.model.peft_dapt = True
cfg.optimizer.train_batch_size = 8

# 3. Execute step
run_dapt_pipeline(cfg)
```

### Interactive Online Probe UI (`ui/online_probes.py`)

Launch the Streamlit web interface to interactively test and evaluate probe completions in real time:

```bash
streamlit run ui/online_probes.py
```

**Key UI Capabilities:**
- **Dynamic Model Selection**: Select between the base student model (e.g. `Qwen/Qwen3-0.6B`) or any saved training checkpoint (`dapt_eval_*.pt` in `models/checkpoints/`) loaded dynamically into memory with caching.
- **Cloze Probe (Fill-in-the-Blank)**: Test neuroscience terminology recall; displays Top-$k$ completions and formatted few-shot prompt context.
- **QA Probe (Multiple Choice)**: Evaluate multiple-choice questions with log-probability scoring across candidate answer choices (A, B, C, D) and rankings.
- **Concept Probe (Freeform Generation)**: Test open-ended conceptual explanations and definitions with configurable token length.
- **Prompt Inspection**: Inspect the exact formatted prompts and token sequences dispatched to the model in collapsible expanders.

### Running Automated Test Suites (`pytest`)

Verify system integrity across all pipeline modules using `pytest`:

```bash
# Run all test suites across the repository
pytest tests/

# Run specific step test suites
pytest tests/test_corpus_builder.py tests/test_minhash_lsh.py  # Step 1
pytest tests/test_dapt.py                                       # Steps 2 & 3
pytest tests/test_rad_prep.py                                   # Step 4
pytest tests/test_clustering.py                                 # Step 5
pytest tests/test_teacher_benchmarking.py                      # Step 6
```
