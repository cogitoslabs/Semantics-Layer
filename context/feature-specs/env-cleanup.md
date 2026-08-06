# Feature Spec: Environment Files Cleanup & Consolidation

## Objective

Simplify environment configuration management by merging `.env.common`, `.env.cpu`, and `.env.gpu` into a single `.env` file (with an updated `.env.example` template) and removing the legacy split configuration files.

---

## Rationale & Background

Previously, configuration parameters were split across three files:
- `.env.common`: Shared non-hardware pipeline settings.
- `.env.cpu`: Hardware-specific batch size and concurrency overrides for CPU environments.
- `.env.gpu`: Hardware-specific batch size and concurrency overrides for GPU environments.

The configuration module `lib/utils/config.py` loaded these files conditionally at runtime. However, `config.py` already includes dynamic system auto-resolution logic (`resolve_auto()`) for hardware parameters (`AVAILABLE_GPUS`, `WORKERS_PER_GPU`, `DOCLING_NUM_THREADS`, `MAX_TASKS_PER_CHILD`). Operating with three separate environment files adds unnecessary complexity and potential confusion.

Consolidating into a single `.env` file streamlines setup for developers and pipeline operations.

---

## Detailed Specifications

### 1. Unified `.env` and `.env.example`

- Combine all keys from `.env.common`, `.env.gpu`, and `.env.cpu` into `.env`.
- Ensure standard default values are specified for batch sizes and hardware settings (leveraging `"AUTO"` or standard defaults where appropriate):
  - `AVAILABLE_GPUS=0`
  - `WORKERS_PER_GPU=2`
  - `DOCLING_NUM_THREADS=2`
  - `MAX_TASKS_PER_CHILD=50`
  - `TRAIN_BATCH_SIZE=8`
  - `EVAL_BATCH_SIZE=64`
  - `RAD_TEACHER_BATCH_SIZE=16`
  - `CLUSTERING_EMBED_BATCH_SIZE=64`
  - `BENCHMARK_TEACHER_BATCH_SIZE=4`
- Synchronize `.env.example` with `.env` as a clean, complete template file tracked in version control.

### 2. Update `lib/utils/config.py`

- Modify `config.py` to remove sequential conditional loading of `.env.gpu`, `.env.cpu`, and `.env.common`.
- Replace with a single call:
  ```python
  load_dotenv(dotenv_path=root_dir / ".env")
  ```
- Retain all fallback defaults in dataclasses (`CorpusBuildConfig`, `OptimizerConfig`, `RADPrepConfig`, `ClusteringConfig`, `TeacherBenchmarkingConfig`, etc.) to guarantee backwards compatibility.

### 3. File Deletions

Delete the legacy split environment files:
- `.env.common`
- `.env.cpu`
- `.env.gpu`

### 4. Code & Documentation Cleanup

- Update `.gitignore`:
  - Retain `.env` ignore rule and keep `!.env.example`.
  - Remove specific mentions of `.env.cpu`, `.env.gpu`, `.env.common`.
- Update references in docstrings and comments:
  - `scripts/online_concept_check.py`
  - `scripts/online_cloze_check.py`
  - `docs/S4_RAD_PREP.md`
  - `README.md`

---

## Verification Plan

### 1. Automated Tests
- Run `pytest tests/test_config.py` to ensure dataclasses and configuration loading function properly.
- Run `pytest` across the test suite to verify pipeline step configurations parse environment variables without error.

### 2. Manual Verification
- Verify that pipeline step scripts (e.g. `pipeline.py --help` or test imports) load correctly with the unified `.env`.
- Confirm `.env.common`, `.env.cpu`, and `.env.gpu` are deleted and git working tree is clean except for expected modifications.
