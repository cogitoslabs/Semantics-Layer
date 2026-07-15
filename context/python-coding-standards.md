# Coding Standards for AI Model Development in Python

A practical standards guide for research/production ML code — training pipelines, evaluation harnesses, and model infrastructure.

---

## 1. Project Structure

Organize by responsibility, not by file type dumped in one folder.

---

## 2. Configuration Management

- Never hardcode hyperparameters, paths, or magic numbers in training scripts.
- Use a single typed config object (dataclass or Pydantic model) as the source of truth — avoid scattering `argparse` flags across multiple files.
- Every run should be reproducible from its config alone: log the fully-resolved config (not just CLI overrides) alongside checkpoints.

```python
from dataclasses import dataclass

@dataclass
class DAPTConfig:
    model_name: str
    learning_rate: float
    batch_size: int
    max_seq_len: int
    amp_dtype: str = "bf16"   # explicit, not implied
    seed: int = 42
```

- Config files should be diffable (YAML/JSON), version-controlled, and named after the experiment they produced, not overwritten in place.

---

## 3. Reproducibility

- Seed everything: Python `random`, `numpy`, `torch`, and CUDA RNG state. Log the seed used per run.
- Pin dependency versions (`requirements.txt` with hashes, or a lockfile via `uv`/`poetry`) — floating versions silently change numerics.
- Log environment metadata with every run: git commit hash, dirty-working-tree flag, library versions, GPU model/driver, CUDA version.
- Avoid nondeterministic ops where correctness matters; where you can't avoid them (e.g. some CuDNN kernels), document it rather than pretend determinism.

---

## 4. Data Handling

- Validate shapes, dtypes, and value ranges at data-loading boundaries — fail loudly and immediately, not three layers deep in a matmul.
- Keep data transformations pure and testable — a `transform(sample) -> sample` function you can unit test in isolation, not inline logic buried in a `Dataset.__getitem__`.
- Never silently drop or truncate malformed samples — log a count and a reason.
- Separate corpus/dataset versioning from code versioning (e.g. a manifest file with content hashes), especially when corpus size or composition changes between runs — this alone can invalidate probe comparisons across experiments.

---

## 5. Model & Training Code

- Keep the model definition free of training-loop concerns (no optimizer logic, no logging calls inside `forward()`).
- Make device and dtype placement explicit — avoid `.cuda()` calls scattered through the codebase; centralize device management.
- Be explicit and intentional about mixed precision. If bf16/fp16 AMP is load-bearing for your hardware (e.g. VRAM-constrained GPUs), say so in a comment — don't leave a future reader to rediscover this by hitting OOM.
- Guard against silent fallback behavior. If an eval metric (e.g. BERTScore) fails and falls back to a dummy/default value, that fallback must raise a visible warning or hard-fail — never pass silently into logged metrics. A quiet fallback that plateaus a probe score looks identical to a real plateau.
- Checkpoint both model state and optimizer/scheduler state if resumption is expected to be exact.
- Log gradient norms, loss curves, and learning rate on every step (or every N steps) — cheap insurance against silent divergence.

---

## 6. Evaluation & Probes

- Each probe/metric module should be independently runnable and independently testable — don't require a full training run to sanity-check an eval function.
- Version your evaluation logic separately from training code; a probe change should never be conflated with a model change when comparing runs.
- Watch for tokenizer mismatches between components evaluated jointly (e.g. a domain model and a reference/teacher model) — this is a common silent source of degraded eval numbers that looks like a training regression.
- Define convergence/gating criteria (e.g. multi-gate systems) as code, not tribal knowledge — a `gate_logic.py` module with explicit thresholds beats a comment saying "looks converged."

---

## 7. Resource & Memory Management

- Account for memory from first principles when tuning batch size: parameters, gradients, optimizer states, activations, and KV cache (if applicable) all compete for the same VRAM budget.
- When running multiple models simultaneously (e.g. a domain model + a scoring/teacher model), size batch parameters for *simultaneous* residency, not each model's standalone footprint.
- Be aware of allocator caching behavior (e.g. PyTorch's caching allocator) — `nvidia-smi` reported usage and actual tensor memory can diverge; use `torch.cuda.memory_summary()` when debugging OOMs rather than guessing.
- Release large intermediate tensors explicitly in long-running loops where memory pressure is tight; don't rely solely on garbage collection timing.

---

## 8. Testing

- Unit test data transforms, probe/metric functions, and any pure logic (gating rules, batching logic) — these are cheap to test and where silent bugs hide longest.
- Integration test the full pipeline on a tiny synthetic dataset (seconds to run) so CI catches wiring bugs before a multi-hour training run does.
- Test failure modes explicitly: what happens when a metric library throws, when a batch is empty, when a checkpoint is missing a key.
- Prefer regression tests around known-fixed bugs (e.g. a past tokenizer mismatch or fallback bug) so they can't silently reappear.

---

## 9. Logging & Observability

- Structured logging (JSON or key-value) over bare `print()` — makes logs greppable and parseable by downstream tooling.
- Log at the right level: `DEBUG` for tensor shapes/intermediate values, `INFO` for epoch/step progress, `WARNING` for recoverable anomalies (fallbacks, retries), `ERROR` for anything that should stop a run.
- Every long-running job should periodically log a heartbeat with current step, loss, and elapsed time — makes it obvious from the logs alone whether a job is alive or hung.

---

## 10. Code Style & Hygiene

- Follow PEP 8; enforce with `ruff` or `black` + `isort` in CI, not by convention alone.
- Type-hint function signatures, especially at module boundaries (data loaders, model forward signatures, config objects) — this is where mismatches are most costly and least visible.
- Docstrings should state shape/dtype contracts for tensor-heavy functions:

```python
def compute_perplexity(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """
    Args:
        logits: (batch, seq_len, vocab_size), float32/bf16
        targets: (batch, seq_len), int64, -100 for masked positions
    Returns:
        Mean perplexity over unmasked tokens.
    """
```

- Avoid deep nesting in training loops — extract steps into named functions (`train_step`, `eval_step`) even if each is only called once; it makes profiling and testing far easier.
- No bare `except:` — catch specific exceptions, especially around I/O, checkpoint loading, and third-party library calls that are known to fail silently.

---

## 11. Version Control & Review

- Commit messages describe *why*, not just *what* ("fix batch size calc to account for simultaneous model residency" not "fix bug").
- Keep experiment-tracking metadata (config hash, git commit, key metrics) in the commit or an accompanying run manifest — makes it possible to trace a checkpoint back to the exact code that produced it.
- Large binary artifacts (checkpoints, corpora) belong in object storage or DVC/LFS, never in the main git history.

---

## Quick Checklist Before a Training Run

- [ ] Config fully specified and logged (no implicit defaults you'd forget)
- [ ] Seed set and logged
- [ ] Data validated (shapes, dtypes, no silent truncation)
- [ ] Memory budget accounted for all simultaneously-resident models
- [ ] Eval/probe fallback paths tested — no silent dummy values
- [ ] Tokenizer consistency checked across all models in the pipeline
- [ ] Checkpointing tested on a short dry run
- [ ] Git tree clean or dirty-state explicitly logged
