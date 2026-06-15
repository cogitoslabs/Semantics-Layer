# Feature Specification: s2_dapt (Domain Adaptive Pretraining)

## Objective
Implement Phase 0.2 (Domain Adaptive Pretraining / Continued Pretraining) for a small base language model (e.g., Qwen3-0.6B) specified in `.env`, including automated evaluations before and after CPT to track improvements in perplexity and neuroscience QA accuracy.

## Background
CPT / DAPT adapts a base LLM to domain-specific terminology and styles by continuing next-token prediction training on the domain corpus. Before proceeding to teacher-distilled instruction tuning, we must verify that the base model's knowledge of the domain has improved.

## Requirements

1. **Configuration**:
   - Retrieve the base model name from `.env` (variable: `BASE_MODEL_NAME`, default: `Qwen/Qwen2.5-0.5B` or `Qwen/Qwen3-0.6B`).
   - Retrieve training parameters (learning rate, batch size, number of epochs, device settings) from environment variables or standard defaults.

2. **Datasets**:
   - **Pretraining Corpus**: Use the parsed Markdown JSONL corpus generated in Step 1 (`outputs/domain_dapt_corpus.jsonl`).
   - **Held-out validation set**: Split the pretraining corpus (e.g., 90% train, 10% validation) to calculate perplexity before and after training.
   - **Held-out probe QA set**: Curate a set of 10 high-quality multiple-choice neuroscience questions (`data/dapt/probe_qa.jsonl`) to evaluate domain QA accuracy.

3. **Evaluation Metric**:
   - **Perplexity (PPL)**: $\exp(\text{cross\_entropy\_loss})$ on the held-out validation split.
   - **QA Accuracy**: Percentage of correct options chosen by the model on the multiple-choice probe set, calculated via next-token log-likelihoods of the option letters (A, B, C, D) given the question prompt.

4. **Training Loop**:
   - Use standard PyTorch/HuggingFace libraries to load the causal language model and tokenizer.
   - Run continued pre-training using PyTorch or `transformers.Trainer` with Causal Language Modeling (next-token prediction) loss.
   - Run on GPU if available, fallback to CPU.

## Verification Criteria
1. Running `python pipeline.py --step s2` executes the DAPT step.
2. The pipeline logs validation perplexity and QA probe accuracy *before* training starts.
3. The pipeline trains the model on the corpus.
4. The pipeline logs validation perplexity and QA probe accuracy *after* training completes, displaying the improvement comparison.
