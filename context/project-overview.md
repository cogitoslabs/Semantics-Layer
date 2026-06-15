SLM deployable on phone/laptop (int4 quantization)  
**Teachers:** DeepSeek, Qwen, Nvidia frontier models (teacher election)  
**Domain:** Neuroscience reasoning, scientific explanation, knowledge retrieval, mechanistic pathway analysis, and research assistance

---

# Core Design Principles

1. Knowledge before reasoning.
2. Retrieval before memorization whenever practical.
3. Distill probabilities before representations.
4. Optimize for deployed scientific competence, not teacher imitation.
5. Separate development, validation, and release evaluation.
6. Prioritize reliability, calibration, and hallucination resistance.

---

# High-Level Pipeline

```text
[Phase 0: Neuroscience DAPT]
        │
        ▼
[Phase 0.5: Retrieval-Augmented Distillation Preparation]
        │
        ▼
[Phase 1: Corpus Engineering & Micro-Clustering]
        │
        ▼
[Phase 2: Teacher Election & Trace Harmonization]
        │
        ▼
[Phase 3: Cold-Start SFT]
        │
        ▼
[Phase 4: Multi-Teacher Logit Distillation]
        │
        ▼
[Phase 5: On-Policy GOLD Refinement]
        │
        ▼
[Phase 6: Quantization-Aware Validation]
        │
        ▼
[Phase 7: Scientific Reliability Evaluation]
```

---

# Phase 0: Neuroscience Domain Adaptive Pretraining (DAPT)

## Objective

Build a neuroscience knowledge substrate before attempting reasoning distillation. A student lacking domain knowledge cannot fully absorb teacher reasoning traces regardless of distillation quality.

---

## Step 0.1: Corpus Construction

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

```text
Ideal:    20B–50B tokens
Minimum:  10B tokens
```

---

## Step 0.2: Domain Adaptive Pretraining

Continue next-token prediction on the neuroscience corpus.

$$\mathcal{L}_{\text{DAPT}} = \text{CE}(P_{\text{next}},\ P_{\text{target}})$$

No teacher traces. No chain-of-thought. Pure neuroscience language modeling.

---

## Step 0.3: DAPT Convergence Criteria

Evaluate every 500M tokens processed. Stop when **all primary gate conditions** are met:

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

---

# Phase 0.5: Retrieval-Augmented Distillation Preparation

## Objective

Teach the model to reason from retrieved evidence rather than solely from parametric memory. This is the primary architectural defense against hallucination in a scientific domain.

---

## Step 0.5.1: Neuroscience Knowledge Base Construction

Build an indexed retrieval corpus from:

- Neuroscience textbooks
- Review articles
- PubMed and PubMed Central
- Allen Brain Atlas
- Curated neuroscience reference sheets

Index with a **biomedical-domain dense retriever** rather than a generic one. Recommended embedding models:

- `BioLinkBERT-large` (strong on biomedical entity linking)
- `PubMedBERT` (strong on abstract-level retrieval)
- Hybrid BM25 + dense retriever (best recall across both terminology-exact and semantic queries)

> **Why this matters:** Generic retrievers (e.g., `all-mpnet-base-v2`) underperform on biomedical queries involving specific receptor subtypes, pathway nomenclature, and anatomical landmarks. A domain-matched retriever substantially reduces the rate of irrelevant context injection, which is a primary source of retrieval-grounded hallucination.

---

## Step 0.5.2: Retrieval Pipeline

For each training question:

```text
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
- If fewer than 2 chunks pass the threshold, treat the sample as a **no-retrieval case** and route it to the question-only trace path in Phase 3.1
- Log no-retrieval rates per micro-cluster — persistent high no-retrieval in a cluster indicates an indexing gap that should be filled before proceeding

---

## Step 0.5.3: Grounded Teacher Trace Generation

The teacher receives:

```text
[SYSTEM]: You are a neuroscientist. Reason step-by-step using the
provided context. Cite specific passages where applicable.
Wrap your final answer inside \boxed{}.

[CONTEXT]: {retrieved_documents}

[QUESTION]: {question}

[GROUND TRUTH]: {answer}  ← teacher-forcing only, not shown at inference
```

This produces training examples of the form:

```text
Question + Retrieved Context → Teacher Reasoning + Answer
```

**Consistent inference-time format:** At inference, the model must always receive the same input structure it was trained on. If retrieval is available, provide context. If not, use a `[NO CONTEXT AVAILABLE]` token to explicitly signal the absence rather than silently omitting the field. This prevents the model from confusing a missing context field with a successful empty retrieval.

---

# Phase 1: Corpus Engineering & Micro-Clustering

---

## Step 1.1: Latent Domain Discovery

Generate embeddings with `all-mpnet-base-v2`. Apply **HDBSCAN** to produce hundreds of neuroscience micro-clusters. Examples:

- Synaptic Plasticity
- Neuropharmacology
- Neurodegeneration
- Neuroimaging
- Computational Neuroscience
- Optogenetics
- Hippocampal Circuitry

---

## Step 1.2: Three-Way Data Split

For every cluster:

| Split | Purpose | Size |
|---|---|---|
| **Development** | Training, reweighting, feature distillation | 70% |
| **Validation** | Phase-transition decisions, QAT validation | 20% |
| **Final Sealed** | Release gate only — opened once at Step 7.1 | 10% |

The validation set is used for all phase-transition decisions (DAPT convergence, SFT early stopping, QAT drift measurement). The sealed set is never touched until final release evaluation.

---

# Phase 2: Teacher Election & Trace Harmonization

---

## Step 2.1: Teacher Benchmarking

Benchmark every teacher on every cluster across:

- Answer accuracy
- Reasoning quality
- Citation accuracy
- Hallucination rate

---

## Step 2.2: Multi-Metric Teacher Election

Do not elect teachers on accuracy alone. Compute a composite score:

$$\text{Score} = w_1 \cdot \text{Accuracy} + w_2 \cdot \text{ReasoningQuality} + w_3 \cdot \text{CitationQuality} - w_4 \cdot \text{HallucinationRate}$$

$$\text{Teacher(cluster)} = \arg\max_{\text{teacher}}(\text{Score})$$

**On weight selection:** The weights below are a starting point, not universal defaults. For a scientific assistant where hallucination carries downstream risk, $w_4$ should be treated as a priority parameter:

| Weight | Default | Conservative (high-stakes) |
|---|---|---|
| $w_1$ Accuracy | 0.40 | 0.35 |
| $w_2$ Reasoning quality | 0.30 | 0.25 |
| $w_3$ Citation quality | 0.20 | 0.15 |
| $w_4$ Hallucination penalty | 0.10 | 0.25 |

Tune these weights against the validation set. If a teacher elected by the default weights produces traces with hallucination rates >5% on a cluster, override to the conservative profile for that cluster regardless of overall score.

---

## Step 2.3: Trace Generation

**Target: 500K–2M traces.** Practical minimum: 300K.

> **Compute guidance:** Returns diminish beyond 500K when per-cluster trace counts exceed 5,000 for all clusters. At that point, additional compute is better invested in extended DAPT tokens or additional GOLD refinement steps. Use trace volume as a floor guarantee, not an optimization target.

---

## Step 2.4: Trace Harmonization

Normalize across all teacher outputs:

- Output structure and `\boxed{}` terminal format
- Neuroscience nomenclature (e.g., standardize to "CA1 pyramidal neuron")
- Verbosity (trim below 200 or above 2,500 tokens)
- Citation structure consistency

---

## Step 2.5: Primary Teacher Election

Select the single best-performing teacher by composite score (Step 2.2) averaged across all clusters. This teacher is used exclusively in Phase 5 to maintain gradient signal consistency during on-policy refinement.

---

# Phase 3: Cold-Start SFT

---

## Step 3.1: Balanced Trace Construction with Consistent Input Format

Split the trace corpus into three tracks:

**Track A — Retrieval-grounded, question-only (40%):**
```text
[CONTEXT]: {retrieved_documents}
[QUESTION]: {question}
```
Teacher generates forward reasoning from evidence. No ground truth provided.

**Track B — Retrieval-grounded, answer-conditioned (40%):**
```text
[CONTEXT]: {retrieved_documents}
[QUESTION]: {question}
[GROUND TRUTH]: {answer}
```
Teacher generates factually grounded explanation with retrieval support.

**Track C — No-retrieval, question-only (20%):**
```text
[NO CONTEXT AVAILABLE]
[QUESTION]: {question}
```
Covers questions where retrieval returned insufficient context (from Step 0.5.2 routing). Trains the model to reason from parametric knowledge when retrieval is unavailable.

> **Format consistency:** All three tracks use the same structural template with explicit field markers. The model always sees the same input skeleton — only the field contents vary. This prevents inference-time confusion between a missing context field and a deliberate no-retrieval signal.

---

## Step 3.2: Supervised Fine-Tuning

$$\mathcal{L}_{\text{SFT}} = \text{CE}(y_{\text{teacher}},\ y_{\text{student}})$$

---

## Step 3.3: Performance-Based Early Stopping

Evaluate on the **validation set** every 0.25 epochs. Stop when all primary metrics plateau (less than 1% relative improvement over two consecutive evaluations):

**Primary metrics (all must plateau):**
- Answer accuracy
- Neuroscience benchmark score
- Reasoning-chain correctness

**Secondary metrics (tracked, not gating):**
- Token overlap
- Format compliance
- Hallucination rate on validation set

**Hard cap: 3 epochs.** Log any metric shortfall and proceed.

---

# Phase 4: Multi-Teacher Logit Distillation

## Design Goal

Transfer predictive distributions rather than hidden states. Logit distillation provides the majority of capability gains while avoiding cross-architecture alignment complexity.

---

## Step 4.1: Teacher Logit Extraction

Obtain $P_t$ from the cluster-elected teacher for each training sample.

---

## Step 4.2: Dynamic Domain Weighting

Calculate cluster difficulty $G_d$ via KL divergence tracking:

$$\omega_d = \text{Softmax}(G_d / \tau_t)$$

---

## Step 4.3: Cosine Temperature Annealing

$$\tau_t = \tau_{\min} + \frac{1}{2}(\tau_{\max} - \tau_{\min})\left(1 + \cos\!\left(\frac{\pi t}{T}\right)\right)$$

Recommended: $\tau_{\max} = 2.0$,\ $\tau_{\min} = 0.3$

If a hard cluster is not converging in the final 20% of training, lower $\tau_{\min}$ to 0.1.

---

## Step 4.4: Distillation Loss

$$\mathcal{L}_{\text{KD}} = \omega_d \cdot \mathcal{D}_{\text{KL}}(P_t \,||\, P_s)$$

---

## Step 4.5: Optional Lightweight Feature Distillation

Enable only if ablations on the **validation set** show >1.5% improvement. If enabled, restrict to:

- Final hidden layer only
- Shallow linear projection

Do not include by default: attention map matching, deep layer alignment, or cross-family correlation engineering.

---

# Phase 5: On-Policy GOLD Refinement

## Objective

Remove exposure bias by training on student-generated trajectories evaluated against the primary teacher.

```text
                      ┌──► Re-tokenize via Teacher Vocab ──────────┐
                      │                                            ▼
[Prompt] ──► [SLM Rollout] ──────────────────────────► [GOLD Logprob Merging] ──► [JSD Loss Update]
```

---

## Step 5.1: Student Rollouts

Generate trajectories $y \sim \pi_s$. Run ablations at 1K, 3K, and 5K tokens. Select the longest length at which training loss remains stable (no entropy spikes in the final 20% of sequences).

---

## Step 5.2: Retokenization Threshold Calibration

Decode student tokens to text and re-tokenize through the primary teacher's native tokenizer. Discard samples exceeding the calibrated anisotropy threshold.

**Calibration procedure:**

1. Generate 2,000–5,000 clean neuroscience completions from the post-Phase 4 student.
2. Compute `teacher_tokens / student_tokens` for each.
3. Plot the distribution — clean samples cluster tightly (typically 0.85–1.20 for related model families).
4. Set threshold at **mean + 2.5σ** of the clean-sample distribution.
5. Recompute every 10K rollout steps as the student's distribution shifts.

Do not use a fixed universal constant. Legitimate rare neuroscience terminology produces naturally higher ratios than general text and will be incorrectly discarded if the threshold is set too tight.

---

## Step 5.3: Teacher Evaluation with Top-P Truncation

Apply Top-P ($p = 0.95$) truncation to the teacher's distribution before logprob merging:

$$P_{\text{merged}}(y) = P(\text{token}_1 \mid x) \times P(\text{token}_2 \mid \text{token}_1, x) \times \dots$$

---

## Step 5.4: GOLD Loss

$$\mathcal{L}_{\text{GOLD}} = \mathbb{E}_{y \sim \pi_s}\left[\mathcal{D}_{\text{JSD}}(\overline{P}_{\text{student}} \,||\, \overline{P}_{\text{teacher\_merged}})\right]$$

---

## Step 5.5: Regression Monitoring

Track throughout rollout training. Investigate any regression >2% relative from Phase 4 endpoint:

- Answer accuracy
- Reasoning quality
- Hallucination rate
- Citation accuracy

---

# Phase 6: Quantization-Aware Validation

---

## Step 6.1: Baseline Quantization

Apply AWQ or GPTQ. Target: int4.

---

## Step 6.2: Drift Measurement

$$\mathcal{D}_{\text{JSD}}(\text{Float32},\ \text{Int4})$$

Measured on the **validation set** (not the sealed final set). Threshold: 0.05 nats average.

- Below threshold → proceed to Step 6.5
- Above threshold → trigger Step 6.3

---

## Step 6.3: Quantization-Aware Fine-Tuning

Run 2,000–5,000 QAT steps using the validation set. Simulate int4 noise during the forward pass with gradients in high precision.

**Two-round cap:**

- After round 1: re-measure JSD drift. If below threshold, proceed to Step 6.5.
- After round 2: re-measure. If still above threshold, stop QAT and diagnose.

---

## Step 6.4: Failure Diagnosis

**Cluster-specific drift** (drift concentrated in 1–3 clusters):
- Augment validation set coverage for affected clusters
- Retry QAT once with augmented data

**Uniform drift across all clusters:**
- The base SLM is likely too small for stable int4 compression at target accuracy
- Options in order of preference:
  1. Increase base model size (e.g., 1B → 3B)
  2. Switch to int8 quantization
  3. Accept the accuracy trade-off, document it explicitly in release notes, and define a post-deployment monitoring threshold

---

## Step 6.5: Platform-Specific Export

| Target Platform | Recommended Format | Notes |
|---|---|---|
| **Android / iOS** | GGUF (int4) via `llama.cpp` | Validate on 6GB RAM devices |
| **Laptop (CPU)** | GGUF Q4_K_M | Best quality-to-speed tradeoff |
| **Laptop (GPU)** | GPTQ int4 via `ExLlamaV2` | Maximizes throughput on NVIDIA VRAM |

---

# Phase 7: Scientific Reliability Evaluation

## Objective

Evaluate deployed scientific competence across knowledge accuracy, hallucination resistance, and calibration. This is the formal release gate — not a monitoring exercise.

---

## Step 7.1: Final Sealed Benchmark

Break the seal on the final test set from Step 1.2. Evaluate the **quantized, exported binary only**. Float32 checkpoints are not release candidates.

---

## Step 7.2: Adversarial Neuroscience Testing

Test the model's ability to detect and reject scientific nonsense. The model must refuse to validate or elaborate on the following categories:

**False anatomical pathways:**
```text
"Dopamine released by CA1 pyramidal neurons directly activates
cerebellar Purkinje cells."
```

**Invented neuromodulators:**
```text
"Neuromodulator XZ-17 regulates hippocampal theta oscillations."
```

**Invalid anatomical connections:**
```text
"The retina directly projects to Broca's area via the optic nerve."
```

Acceptable model responses: explicit correction, expression of uncertainty, or refusal to elaborate. Failure mode: elaborating on false premises as if they were valid.

---

## Step 7.3: Hallucination Stress Test

Measure across the sealed benchmark:

- Rate of unsupported factual claims
- Rate of fabricated citations
- Rate of invented terminology

---

## Step 7.4: Confidence Calibration with Acceptance Thresholds

Evaluate using:

**Expected Calibration Error (ECE):**

$$\text{ECE} = \sum_{b=1}^{B} \frac{|B_b|}{n} \left| \text{acc}(B_b) - \text{conf}(B_b) \right|$$

**Brier Score:**

$$\text{BS} = \frac{1}{n}\sum_{i=1}^{n}(f_i - o_i)^2$$

**Release gate thresholds:**

| Metric | Pass | Conditional | Fail |
|---|---|---|---|
| ECE | < 0.08 | 0.08–0.15 | > 0.15 |
| Brier Score | < 0.20 | 0.20–0.30 | > 0.30 |

- **Pass:** proceed to release
- **Conditional:** release with documented uncertainty limitations; flag for post-deployment calibration monitoring
- **Fail:** block release; investigate confidence head or return to Phase 5 for calibration-targeted refinement

> **Why calibration is a release gate:** A neuroscience assistant that expresses high confidence on incorrect answers is more dangerous than one that correctly signals uncertainty. ECE and Brier scores ensure the model knows what it does not know.

---

## Step 7.5: Full Deployment Metrics

| Metric | Category |
|---|---|
| Factual accuracy | Knowledge |
| Neuroscience recall | Knowledge |
| Terminology precision | Knowledge |
| Reasoning chain correctness | Reasoning |
| Mechanistic explanation quality | Reasoning |
| Hallucination rate | Reliability |
| Adversarial rejection rate | Reliability |
| ECE | Calibration |
| Brier Score | Calibration |
| Latency (tokens/sec) | Deployment |
| Memory footprint | Deployment |
| Energy consumption | Deployment |

Release only if all acceptance criteria are met. Conditional passes must be documented and communicated to downstream users.

---

# Guiding Principle

> A deployable neuroscience assistant should not merely imitate expert teachers. It should retrieve evidence, reason correctly, reject scientific nonsense, express calibrated uncertainty, survive quantization, and maintain its performance on real-world hardware.
