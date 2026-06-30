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
