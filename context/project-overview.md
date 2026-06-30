# LLM Distillation Approach — Revised

**Target:** 1B–4B parameter neuroscience specialist SLM deployable on phone/laptop (int4 quantization)  
**Teachers:** DeepSeek, Qwen, Nvidia frontier models (teacher election)  
**Domain:** Neuroscience reasoning, scientific explanation, knowledge retrieval, mechanistic pathway analysis, and research assistance

---

## Core Design Principles

1. Knowledge before reasoning.
2. Retrieval before memorization whenever practical.
3. Distill probabilities before representations.
4. Optimize for deployed scientific competence, not teacher imitation.
5. Separate development, validation, and release evaluation.
6. Prioritize reliability, calibration, and hallucination resistance.

---

## High-Level Pipeline

```
[Neuroscience DAPT]
│
▼
[Retrieval-Augmented Distillation Preparation]
│
▼
[Corpus Engineering & Micro-Clustering]
│
▼
[Teacher Election & Trace Harmonization]
│
▼
[Cold-Start SFT]
│
▼
[Multi-Teacher Logit Distillation]
│
▼
[On-Policy GOLD Refinement]
│
▼
[Quantization-Aware Validation]
│
▼
[Scientific Reliability Evaluation]
```

---

## Cross-Cutting: Artifact Versioning & Checkpoint Protocol

Before entering Phase 0, establish a versioning and artifact tracking protocol. With a seven-phase pipeline that includes multiple rollback points, reproducibility and recovery depend on disciplined checkpointing.

**Minimum artifact log per phase:**

| Phase | Artifacts to Preserve |
|---|---|
| 0 | Deduplicated corpus manifest (doc count, token count, hash), DAPT checkpoint at convergence, probe evaluation results |
| 1 | Cluster assignments (doc-to-cluster mapping), three-way split indices |
| 2 | Per-cluster teacher election scores, harmonized trace corpus, trace volume per cluster |
| 3 | SFT checkpoint at early stopping, per-track training metrics |
| 4 | Distillation checkpoint, per-cluster KL divergence log |
| 5 | Rollout corpus, retokenization threshold distribution (mean, σ), GOLD checkpoint |
| 6 | Pre- and post-QAT JSD drift measurements per cluster, exported quantized binary |
| 7 | Full evaluation results against sealed test set |

**Rollback protocol:** Phase 7 can return to Phase 5; Phase 0 failure can require full corpus rebuild. Each phase transition must record: input artifact hashes, hyperparameters used, and pass/fail status of all gate conditions. Do not overwrite phase checkpoints. Use versioned directories (e.g., `phase0/v1/`, `phase0/v2/`) when remediation reruns a phase.

---

## Phase 0: Neuroscience Domain Adaptive Pretraining (DAPT)

### Objective

Build a neuroscience knowledge substrate before attempting reasoning distillation. A student lacking domain knowledge cannot fully absorb teacher reasoning traces regardless of distillation quality.

### Step 0.1: Corpus Construction

Aggregate:

- PubMed abstracts
- PubMed Central full text
- bioRxiv neuroscience papers
- Neuroscience textbooks
- Review articles
- Allen Brain Atlas descriptions
- NeuroLex ontology
- Neuroscience lecture notes
- Medical neuroscience question banks

**Target:** 20B–50B tokens  
**Minimum:** 10B tokens

> **Deduplication (new):** PubMed, PMC, and bioRxiv share significant abstract-level overlap. Before logging the final token count, apply document-level deduplication using MinHash LSH (Jaccard threshold ≈ 0.85 recommended as a starting point). Without deduplication, the model risks overfitting on repeated abstracts while appearing to meet the corpus size target. Record the pre- and post-deduplication token counts in the phase artifact log.

### Step 0.2: Domain Adaptive Pretraining

Continue next-token prediction on the deduplicated neuroscience corpus.

```
L_DAPT = CE(P_next, P_target)
```

No teacher traces. No chain-of-thought. Pure neuroscience language modeling.

Evaluate every 500M tokens processed. Stop when all primary gate conditions are met:

**Primary gate (both required):**
- Neuroscience QA accuracy ≥ 55% on the held-out probe set
- Held-out perplexity improvement < 2% over two consecutive evaluations

**Secondary indicators (at least one required):**
- Terminology coverage ≥ 80% of a curated 500-term neuroscience vocabulary
- Retrieval precision ≥ 60% on structured anatomical landmark prompts

**Hard cap:** 3 full corpus passes.

If the primary gate is not cleared by the cap, do not assume later phases will compensate. Remediation options in order of preference:

1. Audit and improve corpus quality (remove low-signal boilerplate, add higher-density review articles)
2. Increase base model size to the next tier (e.g., 1B → 3B)
3. Proceed with an explicit risk flag documenting which metrics fell short and by how much

### Step 0.3: Retrieval-Augmented Distillation Preparation

Teach the model to reason from retrieved evidence rather than solely from parametric memory. This is the primary architectural defense against hallucination in a scientific domain.

Build an indexed retrieval corpus from:

- Neuroscience textbooks
- Review articles
- PubMed and PubMed Central
- Allen Brain Atlas
- Curated neuroscience reference sheets

Index with a biomedical-domain dense retriever rather than a generic one. Recommended embedding models:

- **BioLinkBERT-large** (strong on biomedical entity linking)
- **PubMedBERT** (strong on abstract-level retrieval)
- **Hybrid BM25 + dense retriever** (best recall across both terminology-exact and semantic queries)

> **Why this matters:** Generic retrievers (e.g., `all-mpnet-base-v2`) underperform on biomedical queries involving specific receptor subtypes, pathway nomenclature, and anatomical landmarks. A domain-matched retriever substantially reduces the rate of irrelevant context injection, which is a primary source of retrieval-grounded hallucination.

> **Chunking strategy (new):** Chunk size and overlap significantly affect retrieval quality for dense technical prose. Recommended starting point: **512-token chunks with 64-token overlap** for long-form documents (textbooks, review articles); **256-token chunks with 32-token overlap** for abstracts and short reference entries. Validate chunking strategy by measuring retrieval precision on a sample of structured anatomical landmark prompts before indexing the full corpus.

#### Retrieval Pipeline

```
Question
↓
Biomedical Dense Retriever
↓
Top-K Documents (K = 5–10)
↓
Relevance Re-ranker (optional but recommended)
↓
Final Context Window
```

**Handling retrieval failure:** Not every question will return high-quality context. Before passing retrieved documents to the teacher:

- Score each retrieved chunk with a relevance threshold (cosine similarity ≥ 0.65 recommended as a starting point)
- If fewer than 2 chunks pass the threshold, treat the sample as a no-retrieval case and route it to the question-only trace path in Phase 3.1
- Log no-retrieval rates per micro-cluster — persistent high no-retrieval in a cluster indicates an indexing gap that should be filled before proceeding

#### Grounded Teacher Trace Generation

The teacher receives:

```
[SYSTEM]: You are a neuroscientist. Reason step-by-step using the
provided context. Cite specific passages where applicable.
Wrap your final answer inside \boxed{}.

[CONTEXT]: {retrieved_documents}
[QUESTION]: {question}
[GROUND TRUTH]: {answer}  ← teacher-forcing only, not shown at inference
```

This produces training examples of the form:

```
Question + Retrieved Context → Teacher Reasoning + Answer
```

**Consistent inference-time format:** At inference, the model must always receive the same input structure it was trained on. If retrieval is available, provide context. If not, use a `[NO CONTEXT AVAILABLE]` token to explicitly signal the absence rather than silently omitting the field. This prevents the model from confusing a missing context field with a successful empty retrieval.

---

## Phase 1: Corpus Engineering & Micro-Clustering

### Step 1.1: Latent Domain Discovery

Generate embeddings with `all-mpnet-base-v2`. Apply HDBSCAN to produce hundreds of neuroscience micro-clusters. Examples:

- Synaptic Plasticity
- Neuropharmacology
- Neurodegeneration
- Neuroimaging
- Computational Neuroscience
- Optogenetics
- Hippocampal Circuitry

> **Cluster size imbalance (new):** High-document-count clusters (e.g., Neurodegeneration, Synaptic Plasticity) will naturally dominate raw trace counts. Without explicit reweighting, rare but important clusters (e.g., Optogenetics, Computational Neuroscience) will be understaffed. Apply per-cluster trace count caps during trace generation in Phase 2 to enforce a minimum floor for small clusters. A reasonable starting policy: no cluster should contribute fewer than 2% of total traces or more than 15%, with the remainder distributed proportionally. Adjust based on validation set per-cluster performance.

### Step 1.2: Three-Way Data Split

For every cluster:

| Split | Purpose | Size |
|---|---|---|
| Development | Training, reweighting, feature distillation | 70% |
| Validation | Phase-transition decisions, QAT validation | 20% |
| Final Sealed | Release gate only — opened once at Step 7.1 | 10% |

The validation set is used for all phase-transition decisions (DAPT convergence, SFT early stopping, QAT drift measurement). The sealed set is never touched until final release evaluation.

---

## Phase 2: Teacher Election & Trace Harmonization

### Step 2.1: Teacher Benchmarking

Benchmark every teacher on every cluster across:

- Answer accuracy
- Reasoning quality
- Citation accuracy
- Hallucination rate

> **Reasoning quality measurement (new):** Reasoning quality is listed as a benchmark dimension but requires an explicit measurement protocol. At trace volumes of 300K–2M, human evaluation is not scalable. Use an **LLM-as-judge approach**: prompt a capable frontier model (separate from the teachers being evaluated) with a rubric covering step validity, logical coherence, and absence of circular reasoning. Calibrate the LLM judge against a spot-check human evaluation on 200–500 traces per cluster before using it at scale. Log inter-rater agreement between the LLM judge and human labels.

### Step 2.2: Multi-Metric Teacher Election

Do not elect teachers on accuracy alone. Compute a composite score:

```
Score = w1·Accuracy + w2·ReasoningQuality + w3·CitationQuality − w4·HallucinationRate

Teacher(cluster) = argmax_teacher(Score)
```

On weight selection: The weights below are a starting point, not universal defaults. For a scientific assistant where hallucination carries downstream risk, `w4` should be treated as a priority parameter:

| Weight | Default | Conservative (high-stakes) |
|---|---|---|
| w1 — Accuracy | 0.40 | 0.35 |
| w2 — Reasoning quality | 0.30 | 0.25 |
| w3 — Citation quality | 0.20 | 0.15 |
| w4 — Hallucination penalty | 0.10 | 0.25 |

Tune these weights against the validation set. If a teacher elected by the default weights produces traces with hallucination rates >5% on a cluster, override to the conservative profile for that cluster regardless of overall score.

### Step 2.3: Trace Generation

Target: 500K–2M traces. Practical minimum: 300K.

Compute guidance: Returns diminish beyond 500K when per-cluster trace counts exceed 5,000 for all clusters. At that point, additional compute is better invested in extended DAPT tokens or additional GOLD refinement steps. Use trace volume as a floor guarantee, not an optimization target.

Apply per-cluster trace count caps per Step 1.1 guidance to avoid cluster imbalance.

### Step 2.4: Trace Harmonization

Normalize across all teacher outputs:

- Output structure and `\boxed{}` terminal format
- Neuroscience nomenclature (e.g., standardize to "CA1 pyramidal neuron")
- Verbosity: trim traces **below 200 or above 2,500 tokens**

> **Truncation vs. filtering (new):** Do not truncate traces that exceed the 2,500-token ceiling. A trace cut mid-reasoning teaches the student to produce incomplete chains, which is worse than a slightly verbose one. Instead, **filter out** traces that exceed the ceiling. If a cluster has low trace volume and filtering would leave it understaffed, flag those traces for human review — a human editor can trim them to a complete reasoning endpoint rather than a token boundary.

### Step 2.5: Primary Teacher Election

Select the single best-performing teacher by composite score (Step 2.2) averaged across all clusters. This teacher is used exclusively in Phase 5 to maintain gradient signal consistency during on-policy refinement.

---

## Phase 3: Cold-Start SFT

### Step 3.1: Balanced Trace Construction with Consistent Input Format

Split the trace corpus into three tracks:

**Track A — Retrieval-grounded, question-only (40%):**  
Teacher generates forward reasoning from evidence. No ground truth provided.

```
[CONTEXT]: {retrieved_documents}
[QUESTION]: {question}
```

**Track B — Retrieval-grounded, answer-conditioned (40%):**  
Teacher generates factually grounded explanation with retrieval support.

```
[CONTEXT]: {retrieved_documents}
[QUESTION]: {question}
[GROUND TRUTH]: {answer}
```

**Track C — No-retrieval, question-only (20%):**  
Covers questions where retrieval returned insufficient context (from Step 0.3 routing). Trains the model to reason from parametric knowledge when retrieval is unavailable.

```
[NO CONTEXT AVAILABLE]
[QUESTION]: {question}
```

> **Track C allocation (new):** 20% may be insufficient given the target deployment scenario. On a phone or laptop, retrieval latency may cause users to bypass RAG entirely, or the local index may be stale or absent. If parametric-only performance is a first-class requirement for deployment, consider increasing Track C to 30–35% and adding a dedicated parametric-only evaluation track in Phase 7. Determine this based on expected deployment patterns before committing to the split.

**Format consistency:** All three tracks use the same structural template with explicit field markers. The model always sees the same input skeleton — only the field contents vary. This prevents inference-time confusion between a missing context field and a deliberate no-retrieval signal.

### Step 3.2: Supervised Fine-Tuning

```
L_SFT = CE(teacher, student)
```

### Step 3.3: Performance-Based Early Stopping

Evaluate on the validation set every 0.25 epochs. Stop when all primary metrics plateau (less than 1% relative improvement over two consecutive evaluations):

**Primary metrics (all must plateau):**
- Answer accuracy
- Neuroscience benchmark score
- Reasoning-chain correctness

**Secondary metrics (tracked, not gating):**
- Token overlap
- Format compliance
- Hallucination rate on validation set

**Hard cap:** 3 epochs. Log any metric shortfall and proceed.

---

## Phase 4: Multi-Teacher Logit Distillation

### Design Goal

Transfer predictive distributions rather than hidden states. Logit distillation provides the majority of capability gains while avoiding cross-architecture alignment complexity.

### Step 4.1: Teacher Logit Extraction

Obtain `P_T` from the cluster-elected teacher for each training sample.

```
P_T = Softmax(z_T)
```

### Step 4.2: Dynamic Domain Weighting

Calculate cluster difficulty via KL divergence tracking.

### Step 4.3: Cosine Temperature Annealing

```
τ(t) = τ_min + ½(τ_max − τ_min)(1 + cos(πt/T))
```

**Recommended:** τ_max = 3, τ_min = 0.5  

> **Typo correction (revised):** The original document listed `max = 2, min = 3`, which is inverted. The corrected values above follow the standard cosine annealing convention (τ_max > τ_min). Verify these against your specific model pair before training; the appropriate range depends on the teacher-student logit scale alignment.

If a hard cluster is not converging in the final 20% of training, lower τ_min to 0.1.

### Step 4.4: Distillation Loss

```
L_D = α · KL(P_T || P_S)
```

### Step 4.5: Optional Lightweight Feature Distillation

Enable only if ablations on the validation set show >1.5% improvement. If enabled, restrict to:

- Final hidden layer only
- Shallow linear projection

Do not include by default: attention map matching, deep layer alignment, or cross-family correlation engineering.

> **Cross-family alignment risk (new):** If the teacher and student are from different model families (e.g., Qwen teacher, SmolLM2 student), hidden layer dimensionality mismatch means the projection matrix will carry significant noise regardless of quality. The >1.5% ablation threshold is a good guard, but practitioners should be aware that feature distillation is most reliable within the same model family. When enabling cross-family feature distillation, inspect the projection matrix singular value distribution — a flat spectrum is a signal that the projection is not learning meaningful alignment.

---

## Phase 5: On-Policy GOLD Refinement

### Objective

Remove exposure bias by training on student-generated trajectories evaluated against the primary teacher.

### Step 5.1: Student Rollouts

```
┌──► Re-tokenize via Teacher Vocab ──────────┐
│                                             ▼
[Prompt] ──► [SLM Rollout] ──────────────────────────► [GOLD Logprob Merging]
                                                          ──► [JSD Loss Update]
```

Generate trajectories. Run ablations at 1K, 3K, and 5K tokens. Select the longest length at which training loss remains stable (no entropy spikes in the final 20% of sequences).

### Step 5.2: Retokenization Threshold Calibration

Decode student tokens to text and re-tokenize through the primary teacher's native tokenizer. Discard samples exceeding the calibrated anisotropy threshold.

**Calibration procedure:**

1. Generate 2,000–5,000 clean neuroscience completions from the post-Phase 4 student.
2. Compute `teacher_tokens / student_tokens` for each.
3. Plot the distribution — clean samples cluster tightly (typically 0.85–1.20 for related model families).
4. Set threshold at mean + 2.5σ of the clean-sample distribution.
5. Recompute every 10K rollout steps as the student's distribution shifts.

Do not use a fixed universal constant. Legitimate rare neuroscience terminology produces naturally higher ratios than general text and will be incorrectly discarded if the threshold is set too tight.

> **Distribution shift handling (new):** The protocol specifies recomputing the threshold every 10K rollout steps, but does not specify what to do if the distribution shifts substantially between recalculations — e.g., if the mean shifts by more than 1σ. Recommended protocol: if the mean shifts by >1σ relative to the previous calibration, treat it as a signal of significant student distribution change, re-anchor the threshold by regenerating a fresh clean-sample batch (500–1,000 completions is sufficient for re-anchoring), and log the shift event. Persistent upward drift in the mean may indicate the student is diverging and warrants investigation before continuing rollout training.

### Step 5.3: Teacher Evaluation with Top-P Truncation

Apply Top-P (p = p_0) truncation to the teacher's distribution before logprob merging:

```
P_merge(t_n) = P(token_1) · P(token_2 | token_1, ...)
```

### Step 5.4: GOLD Loss

```
L_D = JSD(P_student || P_teacher_merge)
```

### Step 5.5: Regression Monitoring

Track throughout rollout training. Investigate any regression >2% relative from Phase 4 endpoint:

- Answer accuracy
- Reasoning quality
- Hallucination rate
- Citation accuracy

---

## Phase 6: Quantization-Aware Validation

### Step 6.1: Baseline Quantization

Apply AWQ or GPTQ. Target: int4.

### Step 6.2: Drift Measurement

```
JSD(float32, int4)
```

Measured on the validation set (not the sealed final set). Threshold: 0.05 nats average.

- Below threshold → proceed to Step 6.5
- Above threshold → trigger Step 6.3

### Step 6.3: Quantization-Aware Fine-Tuning

Run 2,000–5,000 QAT steps using the validation set. Simulate int4 noise during the forward pass with gradients in high precision.

**Two-round cap:**

- After round 1: re-measure JSD drift. If below threshold, proceed to Step 6.5.
- After round 2: re-measure. If still above threshold, stop QAT and diagnose.

### Step 6.4: Failure Diagnosis

**Cluster-specific drift** (drift concentrated in 1–3 clusters):
- Augment validation set coverage for affected clusters
- Retry QAT once with augmented data

**Uniform drift across all clusters:**  
The base SLM is likely too small for stable int4 compression at target accuracy. Options in order of preference:

1. Increase base model size (e.g., 1B → 3B)
2. Switch to int8 quantization
3. Accept the accuracy trade-off, document it explicitly in release notes, and define a post-deployment monitoring threshold

### Step 6.5: Platform-Specific Export

| Target Platform | Recommended Format | Notes |
|---|---|---|
| Android / iOS | GGUF (int4) via llama.cpp | Validate on 6GB RAM devices |
| Laptop (CPU) | GGUF Q4_K_M | Best quality-to-speed tradeoff |
| Laptop (GPU) | GPTQ int4 via ExLlamaV2 | Maximizes throughput on NVIDIA VRAM |

---

## Phase 7: Scientific Reliability Evaluation

### Objective

Evaluate deployed scientific competence across knowledge accuracy, hallucination resistance, and calibration. This is the formal release gate — not a monitoring exercise.

### Step 7.1: Final Sealed Benchmark

Break the seal on the final test set from Step 1.2. Evaluate the quantized, exported binary only. Float32 checkpoints are not release candidates.

### Step 7.2: Adversarial Neuroscience Testing

Test the model's ability to detect and reject scientific nonsense. The model must refuse to validate or elaborate on the following categories:

**False anatomical pathways:**
> "Dopamine released by CA1 pyramidal neurons directly activates cerebellar Purkinje cells."

**Invented neuromodulators:**
> "Neuromodulator XZ-17 regulates hippocampal theta oscillations."

**Invalid anatomical connections:**
> "The retina directly projects to Broca's area via the optic nerve."

> **Plausible-but-wrong quantitative claims (new):** Add a fourth adversarial category covering structurally plausible claims with incorrect numerical values. Example: *"Action potentials propagate at 500 m/s in unmyelinated axons"* — this is off by more than an order of magnitude but is harder to reject than an anatomically nonsensical claim because it uses correct terminology and a plausible sentence structure. These represent a more realistic hallucination failure mode in a deployed assistant. Include 20–30 such probes in the adversarial test set, covering speed/velocity claims, receptor binding affinities, ion concentrations, oscillation frequencies, and structural dimensions.

**Acceptable model responses:** explicit correction, expression of uncertainty, or refusal to elaborate.  
**Failure mode:** elaborating on false premises as if they were valid.

### Step 7.3: Hallucination Stress Test

Measure across the sealed benchmark:

- Rate of unsupported factual claims
- Rate of fabricated citations
- Rate of invented terminology

### Step 7.4: Confidence Calibration with Acceptance Thresholds

Evaluate using:

**Expected Calibration Error (ECE):**

```
ECE = Σ_{b=1}^{B} (|B_b| / n) · |acc(B_b) − conf(B_b)|
```

**Brier Score:**

```
S = (1/n) Σ_{t=1}^{n} (f_t − o_t)²
```

**Release gate thresholds:**

| Metric | Pass | Conditional | Fail |
|---|---|---|---|
| ECE | < 0.08 | 0.08–0.15 | > 0.15 |
| Brier Score | < 0.20 | 0.20–0.30 | > 0.30 |

> **Per-cluster calibration (new):** A model can pass aggregate ECE while being badly miscalibrated on rare or specialized clusters. Report per-cluster ECE alongside the aggregate score in the release evaluation. Apply a secondary threshold: no individual cluster should have ECE > 0.15 for a full pass. Clusters at 0.10–0.15 trigger a conditional pass with documented limitations; clusters above 0.15 block release regardless of the aggregate score. Flag affected clusters in release notes so downstream users understand where calibrated uncertainty is least reliable.

> **Why calibration is a release gate:** A neuroscience assistant that expresses high confidence on incorrect answers is more dangerous than one that correctly signals uncertainty. ECE and Brier scores ensure the model knows what it does not know.

**Pass/conditional/fail protocol:**

- **Pass:** proceed to release
- **Conditional:** release with documented uncertainty limitations; flag for post-deployment calibration monitoring
- **Fail:** block release; investigate confidence head or return to Phase 5 for calibration-targeted refinement

> **Parametric-only evaluation track (new):** If Track C allocation was increased in Phase 3 to reflect deployment scenarios where retrieval is unavailable, add a dedicated parametric-only evaluation partition to the sealed benchmark. This ensures the release gate measures real-world performance when the model operates without index access, not just retrieval-augmented performance.

### Step 7.5: Full Deployment Metrics

| Metric | Category |
|---|---|
| Factual accuracy | Knowledge |
| Neuroscience recall | Knowledge |
| Terminology precision | Knowledge |
| Reasoning chain correctness | Reasoning |
| Mechanistic explanation quality | Reasoning |
| Hallucination rate | Reliability |
| Adversarial rejection rate | Reliability |
| ECE (aggregate + per-cluster) | Calibration |
| Brier Score | Calibration |
| Parametric-only accuracy (if applicable) | Deployment |
| Latency (tokens/sec) | Deployment |
| Memory footprint | Deployment |
| Energy consumption | Deployment |

Release only if all acceptance criteria are met. Conditional passes must be documented and communicated to downstream users.

---

## Guiding Principle

A deployable neuroscience assistant should not merely imitate expert teachers. It should retrieve evidence, reason correctly, reject scientific nonsense — including plausible-but-wrong quantitative claims — express calibrated uncertainty, survive quantization, and maintain its performance on real-world hardware.
