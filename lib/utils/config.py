"""
utils/config.py — Load and validate all Semantics Layer pipeline parameters from the root .env
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from dotenv import load_dotenv
from typing import Optional, List, Dict
from datetime import datetime


# Load environment configuration dynamically based on GPU availability
root_dir = Path(__file__).resolve().parent.parent.parent


def resolve_local_model_path(model_name: str) -> str:
    """
    Checks if model_name exists locally under 'models/' directory first.
    If found and it is a valid model path (contains config.json if directory),
    returns the absolute path of the local directory as a string.
    Otherwise, returns model_name.
    """
    if not model_name:
        return model_name

    def is_valid_model_path(path: Path) -> bool:
        if not path.exists():
            return False
        if path.is_dir():
            return (path / "config.json").is_file()
        return True

    # If it is already a valid absolute path or relative path, return it
    p = Path(model_name)
    if is_valid_model_path(p):
        return str(p.resolve())

    # We check under models/ relative to the root directory
    last_part = model_name.replace("\\", "/").split("/")[-1]
    base = root_dir / "models"

    if base.is_dir():
        # 1. Check exact match
        exact_path = base / last_part
        if exact_path.is_dir() and (exact_path / "config.json").is_file():
            try:
                from lib.utils.logger import get_logger
                logger = get_logger(__name__)
                logger.info(f"Resolved model '{model_name}' to local path '{exact_path.resolve()}'")
            except Exception:
                print(f"Resolved model '{model_name}' to local path '{exact_path.resolve()}'")
            return str(exact_path.resolve())

        # 2. Check lowercase match (e.g. smollm2-135m)
        lower_path = base / last_part.lower()
        if lower_path.is_dir() and (lower_path / "config.json").is_file():
            try:
                from lib.utils.logger import get_logger
                logger = get_logger(__name__)
                logger.info(f"Resolved model '{model_name}' to local path '{lower_path.resolve()}'")
            except Exception:
                print(f"Resolved model '{model_name}' to local path '{lower_path.resolve()}'")
            return str(lower_path.resolve())

        # 3. Check case-insensitive scanning of the directory
        try:
            for child in base.iterdir():
                if child.is_dir() and child.name.lower() == last_part.lower() and (child / "config.json").is_file():
                    try:
                        from lib.utils.logger import get_logger
                        logger = get_logger(__name__)
                        logger.info(f"Resolved model '{model_name}' to local path '{child.resolve()}'")
                    except Exception:
                        print(f"Resolved model '{model_name}' to local path '{child.resolve()}'")
                    return str(child.resolve())
        except Exception:
            pass

    return model_name



def is_gpu_available() -> bool:
    """Detect if an NVIDIA or Apple Silicon GPU is present on the system."""
    import shutil
    # 1. Quick check for nvidia-smi tool in PATH
    if shutil.which("nvidia-smi") is not None:
        return True
    
    # 2. Check standard CUDA environment variable
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        return True

    # 3. Check for macOS/MPS or CUDA via lazy torch import to prevent startup lag
    try:
        import torch
        if torch.cuda.is_available():
            return True
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return True
    except ImportError:
        pass

    return False

# Priority order (first loaded takes precedence under override=False):
# 1. Local overrides (.env) if present
# 2. Specific hardware settings (.env.gpu or .env.cpu)
# 3. Shared settings (.env.common)
load_dotenv(dotenv_path=root_dir / ".env")

if is_gpu_available():
    load_dotenv(dotenv_path=root_dir / ".env.gpu")
else:
    load_dotenv(dotenv_path=root_dir / ".env.cpu")

load_dotenv(dotenv_path=root_dir / ".env.common")

DOCLING_NUM_THREADS_WAS_AUTO = False

# Resolve DOCLING_NUM_THREADS immediately to prevent downstream pydantic validations on import (e.g. in docling)
if os.environ.get("DOCLING_NUM_THREADS", "").strip().upper() == "AUTO":
    DOCLING_NUM_THREADS_WAS_AUTO = True
    try:
        from lib.utils.system_detection import get_cpu_cores
        cpu_count = get_cpu_cores()
    except Exception:
        cpu_count = 4
    workers_per_gpu_raw = os.environ.get("WORKERS_PER_GPU", "1")
    try:
        if workers_per_gpu_raw.strip().upper() == "AUTO":
            workers = max(1, cpu_count // 2)
        else:
            workers = int(workers_per_gpu_raw)
    except Exception:
        workers = 1
    resolved_threads = max(1, cpu_count // workers)
    os.environ["DOCLING_NUM_THREADS"] = str(resolved_threads)


def get(key: str, default, cast=str):
    raw = os.environ.get(key, None)
    if raw is None:
        if default is None:
            return None
        try:
            if isinstance(default, cast):
                return default
        except TypeError:
            pass
        try:
            return cast(default)
        except Exception:
            return default
    if isinstance(raw, str) and raw.strip().upper() == "AUTO":
        return "AUTO"
    try:
        if cast is bool:
            if isinstance(raw, str):
                return raw.lower() in ("true", "1", "yes", "on")
            return bool(raw)
        return cast(raw)
    except (ValueError, TypeError) as e:
        raise ValueError(f"[config] Cannot parse env var {key}={raw!r} as {cast.__name__}: {e}")


def get_with_fallback(key: str, fallback_key: str, default, cast=str):
    if key in os.environ:
        return get(key, default, cast)
    return get(fallback_key, default, cast)


@dataclass
class CorpusBuildConfig:
    available_gpus: str       = field(default_factory=lambda: get("AVAILABLE_GPUS", "0"))
    workers_per_gpu: int | str = field(default_factory=lambda: get("WORKERS_PER_GPU", 1, int))
    chunk_size: int | str      = field(default_factory=lambda: get("CHUNK_SIZE", 10, int))
    output_path: Path         = field(default_factory=lambda: Path(get("OUTPUT_PATH", "./data/dapt/domain_dapt_corpus.jsonl")))
    extracted_output_path: Path = field(default_factory=lambda: Path(get("EXTRACTED_OUTPUT_PATH", "./data/dapt/in/domain_dapt_corpus_extracted.jsonl")))
    maxtasksperchild: int | str | None = field(default_factory=lambda: get("MAX_TASKS_PER_CHILD", None, lambda x: int(x) if x and str(x).lower() not in ("none", "null", "") else None))
    docling_use_ocr: bool      = field(default_factory=lambda: get("DOCLING_USE_OCR", False, bool))
    docling_use_table_structure: bool = field(default_factory=lambda: get("DOCLING_USE_TABLE_STRUCTURE", False, bool))
    docling_use_code_enrichment: bool = field(default_factory=lambda: get("DOCLING_USE_CODE_ENRICHMENT", False, bool))
    docling_use_formula_enrichment: bool = field(default_factory=lambda: get("DOCLING_USE_FORMULA_ENRICHMENT", False, bool))
    docling_use_picture_classification: bool = field(default_factory=lambda: get("DOCLING_USE_PICTURE_CLASSIFICATION", False, bool))
    docling_use_picture_description: bool = field(default_factory=lambda: get("DOCLING_USE_PICTURE_DESCRIPTION", False, bool))
    docling_num_threads: int | str = field(default_factory=lambda: get("DOCLING_NUM_THREADS", 4, int))
    minhash_enabled: bool      = field(default_factory=lambda: get("MINHASH_ENABLED", True, bool))
    minhash_jaccard_threshold: float = field(default_factory=lambda: get("MINHASH_JACCARD_THRESHOLD", 0.85, float))
    minhash_num_perm: int       = field(default_factory=lambda: get("MINHASH_NUM_PERM", 128, int))
    minhash_ngram_size: int     = field(default_factory=lambda: get("MINHASH_NGRAM_SIZE", 5, int))
    minhash_num_bands: int      = field(default_factory=lambda: get("MINHASH_NUM_BANDS", 16, int))

    # Dynamic fields resolved at runtime
    gpu_ids: List[int] = field(default_factory=list, init=False)
    total_workers: int = field(default=0, init=False)
    resolution_logs: List[str] = field(default_factory=list, init=False)

    def __post_init__(self):
        self.resolve_auto()

    def log_auto(self, msg: str, logger) -> None:
        print(msg)
        self.resolution_logs.append(msg)
        logger.info(msg)

    def resolve_auto(self) -> None:
        """Resolve AUTO configuration parameters dynamically using system info."""
        resolved_available_gpus = self.available_gpus
        resolved_workers_per_gpu = self.workers_per_gpu
        resolved_chunk_size = self.chunk_size
        resolved_maxtasksperchild = self.maxtasksperchild
        resolved_docling_num_threads = self.docling_num_threads

        # Check if we need to auto-detect any parameters
        has_auto = (
            str(resolved_available_gpus).strip().upper() == "AUTO"
            or str(resolved_workers_per_gpu).strip().upper() == "AUTO"
            or str(resolved_chunk_size).strip().upper() == "AUTO"
            or (resolved_maxtasksperchild is not None and str(resolved_maxtasksperchild).strip().upper() == "AUTO")
            or DOCLING_NUM_THREADS_WAS_AUTO
        )

        if not has_auto:
            # Parse directly without executing hardware detection
            self.gpu_ids = [int(g.strip()) for g in str(resolved_available_gpus).split(",") if g.strip()]
            self.workers_per_gpu = int(resolved_workers_per_gpu)
            self.total_workers = len(self.gpu_ids) * self.workers_per_gpu
            self.chunk_size = int(resolved_chunk_size)
            self.maxtasksperchild = resolved_maxtasksperchild
            self.docling_num_threads = int(resolved_docling_num_threads)
            os.environ["DOCLING_NUM_THREADS"] = str(self.docling_num_threads)
            return

        from lib.utils.logger import get_logger
        logger = get_logger(__name__)

        from lib.utils.system_detection import (
            get_cpu_cores,
            get_available_system_ram_gb,
            get_available_gpus_and_vram,
        )

        # 1. System hardware detection (only run when AUTO is requested)
        system_gpus = get_available_gpus_and_vram()
        cpu_count = get_cpu_cores()
        ram_gb = get_available_system_ram_gb()

        # 2. Resolve GPU IDs
        if str(resolved_available_gpus).strip().upper() == "AUTO":
            if system_gpus:
                self.gpu_ids = [gpu[0] for gpu in system_gpus]
                self.log_auto(f"[AUTO] Detected GPUs: {self.gpu_ids}", logger)
            else:
                self.gpu_ids = [-1]  # CPU fallback
                self.log_auto("[AUTO] No GPUs detected. Falling back to CPU mode.", logger)
        else:
            self.gpu_ids = [int(g.strip()) for g in str(resolved_available_gpus).split(",") if g.strip()]

        # 3. Resolve Workers Per GPU & Total Workers
        if str(resolved_workers_per_gpu).strip().upper() == "AUTO":
            # Baseline: 4 GB RAM/VRAM per worker
            if self.gpu_ids == [-1]:
                # CPU fallback worker calculation
                # Limit by CPU cores and RAM
                max_workers_by_cpu = max(1, cpu_count // 2)
                max_workers_by_ram = max(1, int(ram_gb // 4))
                self.workers_per_gpu = min(max_workers_by_cpu, max_workers_by_ram)
                self.log_auto(
                    f"[AUTO] CPU Mode workers resolved: {self.workers_per_gpu} "
                    f"(CPU Cores: {cpu_count}, RAM: {ram_gb:.1f} GB)",
                    logger
                )
            else:
                # GPU worker calculation
                # Detect min VRAM per GPU
                min_vram = min(gpu[1] for gpu in system_gpus) if system_gpus else 8.0
                gpu_workers = max(1, int(min_vram // 4))
                # Check system boundaries (CPU / System RAM)
                max_total_workers = min(max(1, cpu_count - 1), max(1, int(ram_gb // 4)))
                total_gpu_workers = gpu_workers * len(self.gpu_ids)
                if total_gpu_workers > max_total_workers:
                    self.workers_per_gpu = max(1, max_total_workers // len(self.gpu_ids))
                    self.log_auto(
                        f"[AUTO] Scaled down workers_per_gpu to {self.workers_per_gpu} to respect CPU/RAM bounds "
                        f"(Cores: {cpu_count}, RAM: {ram_gb:.1f} GB, Min VRAM: {min_vram:.1f} GB)",
                        logger
                    )
                else:
                    self.workers_per_gpu = gpu_workers
                    self.log_auto(
                        f"[AUTO] GPU workers per device resolved: {self.workers_per_gpu} "
                        f"(Min VRAM: {min_vram:.1f} GB)",
                        logger
                    )
        else:
            self.workers_per_gpu = int(resolved_workers_per_gpu)

        self.total_workers = len(self.gpu_ids) * self.workers_per_gpu

        # 4. Resolve Chunk Size
        if str(resolved_chunk_size).strip().upper() == "AUTO":
            if self.total_workers <= 2:
                self.chunk_size = 16
            elif 3 <= self.total_workers <= 8:
                self.chunk_size = 10
            else:
                self.chunk_size = 6
            self.log_auto(f"[AUTO] Chunk size resolved to {self.chunk_size} pages for {self.total_workers} workers", logger)
        else:
            self.chunk_size = int(resolved_chunk_size)

        # 5. Resolve Max Tasks Per Child
        if resolved_maxtasksperchild is not None and str(resolved_maxtasksperchild).strip().upper() == "AUTO":
            self.maxtasksperchild = max(3, min(20, int(ram_gb / self.total_workers)))
            self.log_auto(f"[AUTO] max_tasks_per_child resolved to {self.maxtasksperchild} based on system RAM and workers", logger)
        else:
            self.maxtasksperchild = resolved_maxtasksperchild

        # 6. Resolve Docling Num Threads
        if DOCLING_NUM_THREADS_WAS_AUTO:
            self.docling_num_threads = max(1, cpu_count // self.total_workers)
            os.environ["DOCLING_NUM_THREADS"] = str(self.docling_num_threads)
            self.log_auto(f"[AUTO] docling_num_threads resolved to {self.docling_num_threads} (CPU Cores: {cpu_count}, Total Workers: {self.total_workers})", logger)
        else:
            self.docling_num_threads = int(resolved_docling_num_threads)
            os.environ["DOCLING_NUM_THREADS"] = str(self.docling_num_threads)



@dataclass
class CorpusConfig:
    total_corpus_tokens: int   = field(default_factory=lambda: get("TOTAL_CORPUS_TOKENS", 30_000_000_000, int))
    max_corpus_passes: int     = field(default_factory=lambda: get("MAX_CORPUS_PASSES", 3, int))
    eval_interval_tokens: int  = field(default_factory=lambda: get("EVAL_INTERVAL_TOKENS", 500_000_000, int))
    slow_eval_interval_tokens: int = field(default_factory=lambda: get("SLOW_EVAL_INTERVAL_TOKENS", 250_000_000, int))

    @property
    def hard_stop_tokens(self) -> int:
        return self.total_corpus_tokens * self.max_corpus_passes


@dataclass
class GateConfig:
    # Primary Gate A
    qa_acc_threshold: float        = field(default_factory=lambda: get("QA_ACC_THRESHOLD", 0.55, float))

    # Primary Gate B
    ppl_improvement_threshold: float = field(default_factory=lambda: get("PPL_IMPROVEMENT_THRESHOLD", 2.0, float))
    ppl_plateau_window: int          = field(default_factory=lambda: get("PPL_PLATEAU_WINDOW", 2, int))

    # Secondary Gate
    cloze_threshold: float         = field(default_factory=lambda: get_with_fallback("CLOZE_THRESHOLD", "TERM_COV_THRESHOLD", 0.30, float))
    concept_threshold: float       = field(default_factory=lambda: get_with_fallback("CONCEPT_THRESHOLD", "RET_PREC_THRESHOLD", 0.50, float))

    # Remediation routing
    qa_low_threshold: float        = field(default_factory=lambda: get("QA_LOW_THRESHOLD", 0.40, float))


@dataclass
class ProbeConfig:
    # Probe Activation Toggles
    run_perplexity: bool           = field(default_factory=lambda: get("RUN_PERPLEXITY_PROBE", True, bool))
    run_qa: bool                  = field(default_factory=lambda: get("RUN_QA_PROBE", True, bool))
    run_cloze: bool               = field(default_factory=lambda: get_with_fallback("RUN_CLOZE_PROBE", "RUN_TERMINOLOGY_PROBE", True, bool))
    run_concept: bool             = field(default_factory=lambda: get_with_fallback("RUN_CONCEPT_PROBE", "RUN_RETRIEVAL_PROBE", True, bool))

    # Perplexity probe
    perplexity_max_seq_len: int | None = field(default_factory=lambda: get("PERPLEXITY_MAX_SEQ_LEN", None, lambda x: int(x) if x and str(x).lower() not in ("none", "null", "") else None))
    perplexity_batch_size: int | None  = field(default_factory=lambda: get("PERPLEXITY_BATCH_SIZE", None, lambda x: int(x) if x and str(x).lower() not in ("none", "null", "") else None))

    # QA probe
    qa_max_seq_len: int                   = field(default_factory=lambda: get("QA_MAX_SEQ_LEN", 512, int))
    qa_batch_size: int | None             = field(default_factory=lambda: get("QA_BATCH_SIZE", None, lambda x: int(x) if x and str(x).lower() not in ("none", "null", "") else None))

    # Terminology cloze
    cloze_top_k: int            = field(default_factory=lambda: get_with_fallback("CLOZE_TOP_K", "TERM_COV_TOP_K", 5, int))
    cloze_max_new_tokens: int   = field(default_factory=lambda: get_with_fallback("CLOZE_MAX_NEW_TOKENS", "TERM_COV_MAX_NEW_TOKENS", 3, int))
    cloze_gen_batch_size: int   = field(default_factory=lambda: get_with_fallback("CLOZE_GEN_BATCH_SIZE", "TERM_COV_GEN_BATCH_SIZE", 16, int))
    cloze_max_seq_len: int      = field(default_factory=lambda: get_with_fallback("CLOZE_MAX_SEQ_LEN", "TERM_COV_MAX_SEQ_LEN", 256, int))

    # Anatomical retrieval
    bertscore_model: str           = field(default_factory=lambda: get("BERTSCORE_MODEL", "allenai/scibert_scivocab_uncased", str))
    concept_max_new_tokens: int   = field(default_factory=lambda: get_with_fallback("CONCEPT_MAX_NEW_TOKENS", "RET_PREC_MAX_NEW_TOKENS", 100, int))
    concept_gen_batch_size: int   = field(default_factory=lambda: get_with_fallback("CONCEPT_GEN_BATCH_SIZE", "RET_PREC_GEN_BATCH_SIZE", 16, int))
    concept_max_seq_len: int      = field(default_factory=lambda: get_with_fallback("CONCEPT_MAX_SEQ_LEN", "RET_PREC_MAX_SEQ_LEN", 256, int))
    concept_bertscore_batch_size: int = field(default_factory=lambda: get_with_fallback("CONCEPT_BERTSCORE_BATCH_SIZE", "RET_PREC_BERTSCORE_BATCH_SIZE", 32, int))

    # PPL eval corpus size
    perplexity_eval_tokens: int    = field(default_factory=lambda: get("PERPLEXITY_EVAL_TOKENS", 10_000_000, int))

    def __post_init__(self):
        self.bertscore_model = resolve_local_model_path(self.bertscore_model)


@dataclass
class DataConfig:
    qa_probe_path: Path            = field(default_factory=lambda: Path(
        os.environ.get("QA_PROBE_PATH", os.environ.get("PROBE_QA_PATH", "evals/dapt/probe_qa.jsonl"))
    ))
    ppl_corpus_path: Path          = field(default_factory=lambda: Path(
        os.environ.get("PPL_CORPUS_PATH", os.environ.get("PPL_HELD_OUT_PATH", "data/dapt/ppl_validation_tokens.npy"))
    ))
    cloze_set_path: Path           = field(default_factory=lambda: Path(
        os.environ.get("CLOZE_SET_PATH", os.environ.get("VOCAB_CLOZE_PATH", "evals/dapt/vocab_cloze_set.json"))
    ))
    concept_prompts_path: Path     = field(default_factory=lambda: Path(
        os.environ.get("CONCEPT_PROMPTS_PATH", os.environ.get("RETRIEVAL_PROMPTS_PATH", "evals/dapt/retrieval_prompts.json"))
    ))
    concept_references_path: Path  = field(default_factory=lambda: Path(
        os.environ.get("CONCEPT_REFERENCES_PATH", os.environ.get("RETRIEVAL_REFERENCES_PATH", "evals/dapt/retrieval_references.json"))
    ))
    pretokenized_bin_path: Path    = field(default_factory=lambda: Path(
        os.environ.get("PRETOKENIZED_BIN_PATH", "data/dapt/train_tokens.npy")
    ))
    dapt_in_dir: Path              = field(default_factory=lambda: Path(
        os.environ.get("DAPT_IN_DIR", "data/dapt/in")
    ))


@dataclass
class ModelConfig:
    base_model_name: str           = field(default_factory=lambda: get("BASE_MODEL_NAME", "HuggingFaceTB/SmolLM2-135M"))
    model_dtype: str               = field(default_factory=lambda: get("MODEL_DTYPE", "bfloat16"))
    max_seq_len: int               = field(default_factory=lambda: get("MAX_SEQ_LEN", 512, int))

    checkpoint_dir: Path           = field(default_factory=lambda: Path(get("CHECKPOINT_DIR", "models/checkpoints")))
    best_checkpoint_manifest: Path = field(default_factory=lambda: Path(get("BEST_CHECKPOINT_MANIFEST", "logs/best_checkpoint.json")))
    checkpoint_keep_last: int      = field(default_factory=lambda: get("CHECKPOINT_KEEP_LAST", 5, int))
    save_optimizer_state: bool     = field(default_factory=lambda: get("SAVE_OPTIMIZER_STATE", False, bool))
    restart_from_checkpoint: bool  = field(default_factory=lambda: get("RESTART_TRAINING_FROM_CHECKPOINT", False, bool))
    torch_compile: bool            = field(default_factory=lambda: get("TORCH_COMPILE", True, bool))

    # PEFT-DAPT Configuration
    peft_dapt: bool                = field(default_factory=lambda: get("PEFT_DAPT", False, bool))
    lora_r: int                    = field(default_factory=lambda: get("LORA_R", 16, int))
    lora_alpha: int                = field(default_factory=lambda: get("LORA_ALPHA", 32, int))
    lora_dropout: float            = field(default_factory=lambda: get("LORA_DROPOUT", 0.05, float))
    lora_target_modules: List[str] = field(default_factory=lambda: [m.strip() for m in get("LORA_TARGET_MODULES", "q_proj,v_proj,k_proj,o_proj").split(",") if m.strip()])
    gradient_checkpointing: bool   = field(default_factory=lambda: get("GRADIENT_CHECKPOINTING", False, bool))

    def __post_init__(self):
        self.base_model_name = resolve_local_model_path(self.base_model_name)


@dataclass
class OptimizerConfig:
    learning_rate: float           = field(default_factory=lambda: get("DAPT_LR", 5e-5, float))
    weight_decay: float            = field(default_factory=lambda: get("WEIGHT_DECAY", 0.01, float))
    warmup_steps: int              = field(default_factory=lambda: get("WARMUP_STEPS", 1000, int))
    max_grad_norm: float           = field(default_factory=lambda: get("MAX_GRAD_NORM", 1.0, float))
    train_batch_size: int          = field(default_factory=lambda: get("TRAIN_BATCH_SIZE", 2, int))
    eval_batch_size: int           = field(default_factory=lambda: get("EVAL_BATCH_SIZE", 4, int))
    gradient_accumulation_steps: int = field(default_factory=lambda: get("GRADIENT_ACCUMULATION_STEPS", 1, int))


@dataclass
class StorageConfig:
    storage_target: str       = field(default_factory=lambda: get("STORAGE_TARGET", "local"))
    local_directory_path: str = field(default_factory=lambda: get("LOCAL_DIRECTORY_PATH", "."))
    aws_bucket_name: Optional[str] = field(default_factory=lambda: get("AWS_BUCKET_NAME", None))
    aws_prefix: str           = field(default_factory=lambda: get("AWS_PREFIX", ""))


@dataclass
class LoggingConfig:
    log_dir: Path                  = field(default_factory=lambda: Path(get("LOG_DIR", "logs")))
    metrics_log_file: Path         = field(default_factory=lambda: Path(get("METRICS_LOG_FILE", "logs/dapt_eval_metrics.jsonl")))
    eval_traces_file: Path         = field(default_factory=lambda: Path(get("EVAL_TRACES_FILE", "logs/dapt_eval_traces.csv")))
    model_tracing: bool            = field(default_factory=lambda: get("MODEL_TRACING", False, bool))
    model_trace_file: Path         = field(default_factory=lambda: Path(get("MODEL_TRACE_FILE", "logs/dapt_model_traces.csv")))
    risk_report_path: Path         = field(default_factory=lambda: Path(get("RISK_REPORT_PATH", "logs/dapt_hard_cap_risk_report.json")))
    log_level: str                 = field(default_factory=lambda: get("LOG_LEVEL", "INFO"))
    log_file: str                  = field(default_factory=lambda: get("LOG_FILE", "pipeline.log"))


@dataclass
class MiscConfig:
    seed: int                      = field(default_factory=lambda: get("SEED", 42, int))


@dataclass
class WandbConfig:
    enabled: bool                  = field(default_factory=lambda: get("WANDB_ENABLED", False, bool))
    mode: str                      = field(default_factory=lambda: get("WANDB_MODE", "online", str))
    api_key: Optional[str]         = field(default_factory=lambda: get("WANDB_API_KEY", None, str))
    project: str                   = field(default_factory=lambda: get("WANDB_PROJECT", "semantics-dapt", str))
    entity: Optional[str]          = field(default_factory=lambda: get("WANDB_ENTITY", None, str))
    run_name: Optional[str]        = field(default_factory=lambda: get("WANDB_RUN_NAME", None, str))
    log_interval_steps: int        = field(default_factory=lambda: get("WANDB_LOG_INTERVAL_STEPS", 10, int))

    def __post_init__(self):
        if not self.run_name:
            timestamp = datetime.now().strftime("%y%m%d%H%M%S")
            self.run_name = f"{self.project}_{timestamp}"
        if self.mode:
            os.environ["WANDB_MODE"] = self.mode



@dataclass
class RADPrepConfig:
    # Corpus paths
    retrieval_corpus_path: Path    = field(default_factory=lambda: Path(get("RAD_CORPUS_PATH", "data/rad_prep/retrieval_corpus.jsonl")))
    chunks_path: Path              = field(default_factory=lambda: Path(get("RAD_CHUNKS_PATH", "data/rad_prep/chunks.jsonl")))
    index_dir: Path                = field(default_factory=lambda: Path(get("RAD_INDEX_DIR", "data/rad_prep/index")))
    traces_dir: Path               = field(default_factory=lambda: Path(get("RAD_TRACES_DIR", "data/rad_prep/traces")))
    qa_samples_path: Path          = field(default_factory=lambda: Path(get("RAD_QA_SAMPLES_PATH", "evals/dapt/probe_qa.jsonl")))

    # Retrieval settings
    embedding_model: str           = field(default_factory=lambda: get("RAD_EMBEDDING_MODEL", "biolinkbert"))
    retrieval_mode: str            = field(default_factory=lambda: get("RAD_RETRIEVAL_MODE", "hybrid"))  # dense|sparse|hybrid
    top_k: int                     = field(default_factory=lambda: get("RAD_TOP_K", 7, int))
    relevance_threshold: float     = field(default_factory=lambda: get("RAD_RELEVANCE_THRESHOLD", 0.65, float))
    embed_batch_size: int          = field(default_factory=lambda: get("RAD_EMBED_BATCH_SIZE", 256, int))

    # Chunking
    long_form_chunk_tokens: int    = field(default_factory=lambda: get("RAD_LONG_FORM_CHUNK_TOKENS", 512, int))
    long_form_overlap_tokens: int  = field(default_factory=lambda: get("RAD_LONG_FORM_OVERLAP_TOKENS", 64, int))
    abstract_chunk_tokens: int     = field(default_factory=lambda: get("RAD_ABSTRACT_CHUNK_TOKENS", 256, int))
    abstract_overlap_tokens: int   = field(default_factory=lambda: get("RAD_ABSTRACT_OVERLAP_TOKENS", 32, int))

    # Prompt prep gating
    min_traces: int                = field(default_factory=lambda: get("RAD_MIN_TRACES", 1000, int))




@dataclass
class ClusteringConfig:
    # Input
    corpus_path: Path              = field(default_factory=lambda: Path(get("CLUSTERING_CORPUS_PATH", "data/dapt/domain_dapt_corpus.jsonl")))

    # Embedding
    embedding_model: str           = field(default_factory=lambda: get("CLUSTERING_EMBEDDING_MODEL", "all-mpnet-base-v2"))
    embed_batch_size: int          = field(default_factory=lambda: get("CLUSTERING_EMBED_BATCH_SIZE", 64, int))
    embeddings_cache_path: Path    = field(default_factory=lambda: Path(get("CLUSTERING_EMBEDDINGS_CACHE", "data/clustering/embeddings.npy")))
    doc_ids_cache_path: Path       = field(default_factory=lambda: Path(get("CLUSTERING_DOC_IDS_CACHE", "data/clustering/doc_ids.json")))

    # HDBSCAN
    hdbscan_min_cluster_size: int  = field(default_factory=lambda: get("HDBSCAN_MIN_CLUSTER_SIZE", 6, int))
    hdbscan_min_samples: int       = field(default_factory=lambda: get("HDBSCAN_MIN_SAMPLES", 2, int))
    hdbscan_metric: str            = field(default_factory=lambda: get("HDBSCAN_METRIC", "cosine"))
    min_clusters: int              = field(default_factory=lambda: get("CLUSTERING_MIN_CLUSTERS", 10, int))
    use_pca: bool                  = field(default_factory=lambda: get("CLUSTERING_USE_PCA", True, bool))
    pca_components: int            = field(default_factory=lambda: get("CLUSTERING_PCA_COMPONENTS", 50, int))

    # Noise handling
    noise_assignment: str          = field(default_factory=lambda: get("CLUSTERING_NOISE_ASSIGNMENT", "nearest"))

    # Imbalance reweighting
    cluster_min_fraction: float    = field(default_factory=lambda: get("CLUSTER_MIN_FRACTION", 0.02, float))
    cluster_max_fraction: float    = field(default_factory=lambda: get("CLUSTER_MAX_FRACTION", 0.15, float))

    # Split ratios
    split_dev_ratio: float         = field(default_factory=lambda: get("SPLIT_DEV_RATIO", 0.70, float))
    split_val_ratio: float         = field(default_factory=lambda: get("SPLIT_VAL_RATIO", 0.20, float))
    split_sealed_ratio: float      = field(default_factory=lambda: get("SPLIT_SEALED_RATIO", 0.10, float))

    # Output paths
    output_dir: Path               = field(default_factory=lambda: Path(get("CLUSTERING_OUTPUT_DIR", "data/clustering")))
    assignments_path: Path         = field(default_factory=lambda: Path(get("CLUSTERING_ASSIGNMENTS_PATH", "data/clustering/cluster_assignments.jsonl")))
    splits_path: Path              = field(default_factory=lambda: Path(get("CLUSTERING_SPLITS_PATH", "data/clustering/splits.json")))
    cluster_manifest_path: Path    = field(default_factory=lambda: Path(get("CLUSTERING_MANIFEST_PATH", "data/clustering/cluster_manifest.json")))
    cluster_report_path: Path      = field(default_factory=lambda: Path(get("CLUSTERING_REPORT_PATH", "logs/clustering/cluster_report.json")))


@dataclass
class TeacherBenchmarkingConfig:
    candidate_teachers: List[str]       = field(default_factory=lambda: [t.strip() for t in get("BENCHMARK_TEACHERS", "Qwen/Qwen3-1.7B").split(",") if t.strip()])
    judge_backend: str                  = field(default_factory=lambda: get("BENCHMARK_JUDGE_BACKEND", "api"))
    judge_model_name: str               = field(default_factory=lambda: get("BENCHMARK_JUDGE_MODEL", ""))
    judge_api_url: Optional[str]        = field(default_factory=lambda: get("BENCHMARK_JUDGE_API_URL", None))
    judge_api_key: Optional[str]        = field(default_factory=lambda: get("BENCHMARK_JUDGE_API_KEY", None))
    judge_max_new_tokens: int           = field(default_factory=lambda: get("BENCHMARK_JUDGE_MAX_NEW_TOKENS", 256, int))
    
    # Teacher generation backend & settings
    teacher_backend: str                = field(default_factory=lambda: get_with_fallback("BENCHMARK_TEACHER_BACKEND", "RAD_TEACHER_BACKEND", "hf_local"))
    teacher_model_name: str             = field(default_factory=lambda: get_with_fallback("BENCHMARK_TEACHER_MODEL_NAME", "RAD_TEACHER_MODEL_NAME", "Qwen/Qwen3-1.7B"))
    teacher_api_url: Optional[str]      = field(default_factory=lambda: get_with_fallback("BENCHMARK_TEACHER_API_URL", "RAD_TEACHER_API_URL", None))
    teacher_api_key: Optional[str]      = field(default_factory=lambda: get_with_fallback("BENCHMARK_TEACHER_API_KEY", "RAD_TEACHER_API_KEY", None))
    teacher_max_new_tokens: int         = field(default_factory=lambda: get_with_fallback("BENCHMARK_TEACHER_MAX_NEW_TOKENS", "RAD_TEACHER_MAX_NEW_TOKENS", 1024, int))
    teacher_batch_size: int             = field(default_factory=lambda: get_with_fallback("BENCHMARK_TEACHER_BATCH_SIZE", "RAD_TEACHER_BATCH_SIZE", 16, int))

    
    eval_sample_size: int               = field(default_factory=lambda: get("BENCHMARK_EVAL_SAMPLE_SIZE", 10, int))
    min_eval_samples: int               = field(default_factory=lambda: get("BENCHMARK_MIN_EVAL_SAMPLES", 2, int))
    
    enable_calibration: bool            = field(default_factory=lambda: get("BENCHMARK_ENABLE_CALIBRATION", False, bool))
    human_calibration_size: int         = field(default_factory=lambda: get("BENCHMARK_CALIBRATION_SIZE", 200, int))
    human_labels_path: Optional[Path]   = field(default_factory=lambda: Path(get("BENCHMARK_HUMAN_LABELS_PATH", "")) if get("BENCHMARK_HUMAN_LABELS_PATH", "") else None)
    
    hallucination_nli_threshold: float  = field(default_factory=lambda: get("BENCHMARK_HALLUCINATION_NLI_THRESHOLD", 0.5, float))
    citation_min_overlap: float         = field(default_factory=lambda: get("BENCHMARK_CITATION_MIN_OVERLAP", 0.30, float))
    nli_model: str                      = field(default_factory=lambda: get("BENCHMARK_NLI_MODEL", "cross-encoder/nli-deberta-v3-small"))
    
    output_dir: Path                    = field(default_factory=lambda: Path(get("BENCHMARKING_OUTPUT_DIR", "data/benchmarking")))
    scores_path: Path                   = field(default_factory=lambda: Path(get("BENCHMARKING_SCORES_PATH", "data/benchmarking/scores.jsonl")))
    manifest_path: Path                 = field(default_factory=lambda: Path(get("BENCHMARKING_MANIFEST_PATH", "data/benchmarking/benchmark_manifest.json")))
    calibration_log_path: Path          = field(default_factory=lambda: Path(get("BENCHMARKING_CALIBRATION_LOG", "logs/benchmarking/judge_calibration.jsonl")))
    inter_rater_log_path: Path          = field(default_factory=lambda: Path(get("BENCHMARKING_INTER_RATER_LOG", "logs/benchmarking/inter_rater_agreement.json")))


@dataclass
class PipelineConfig:
    """Top-level config for the Semantics Layer pipeline."""
    build: CorpusBuildConfig  = field(default_factory=CorpusBuildConfig)
    corpus: CorpusConfig      = field(default_factory=CorpusConfig)
    gates: GateConfig         = field(default_factory=GateConfig)
    probes: ProbeConfig       = field(default_factory=ProbeConfig)
    data: DataConfig          = field(default_factory=DataConfig)
    model: ModelConfig        = field(default_factory=ModelConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    storage: StorageConfig    = field(default_factory=StorageConfig)
    logging: LoggingConfig    = field(default_factory=LoggingConfig)
    misc: MiscConfig          = field(default_factory=MiscConfig)
    wandb: WandbConfig        = field(default_factory=WandbConfig)
    rad: RADPrepConfig        = field(default_factory=RADPrepConfig)
    clustering: ClusteringConfig = field(default_factory=ClusteringConfig)
    benchmarking: TeacherBenchmarkingConfig = field(default_factory=TeacherBenchmarkingConfig)

    def validate(self):
        """Sanity-check values that must satisfy domain constraints."""
        errors = []

        if not (0.0 <= self.gates.qa_acc_threshold <= 1.0):
            errors.append(f"QA_ACC_THRESHOLD must be in [0,1], got {self.gates.qa_acc_threshold}")
        if not (0.0 <= self.gates.cloze_threshold <= 1.0):
            errors.append(f"CLOZE_THRESHOLD must be in [0,1], got {self.gates.cloze_threshold}")
        if not (0.0 <= self.gates.concept_threshold <= 1.0):
            errors.append(f"CONCEPT_THRESHOLD must be in [0,1], got {self.gates.concept_threshold}")
        if self.gates.ppl_plateau_window < 2:
            errors.append(f"PPL_PLATEAU_WINDOW must be >= 2, got {self.gates.ppl_plateau_window}")
        if self.corpus.max_corpus_passes < 1:
            errors.append(f"MAX_CORPUS_PASSES must be >= 1, got {self.corpus.max_corpus_passes}")
        if self.corpus.eval_interval_tokens <= 0:
            errors.append(f"EVAL_INTERVAL_TOKENS must be > 0, got {self.corpus.eval_interval_tokens}")
        if self.corpus.slow_eval_interval_tokens <= 0:
            errors.append(f"SLOW_EVAL_INTERVAL_TOKENS must be > 0, got {self.corpus.slow_eval_interval_tokens}")
        if self.optimizer.gradient_accumulation_steps < 1:
            errors.append(f"GRADIENT_ACCUMULATION_STEPS must be >= 1, got {self.optimizer.gradient_accumulation_steps}")
        if self.gates.qa_low_threshold > self.gates.qa_acc_threshold:
            errors.append(
                f"QA_LOW_THRESHOLD ({self.gates.qa_low_threshold}) must be <= "
                f"QA_ACC_THRESHOLD ({self.gates.qa_acc_threshold})"
            )
        if self.build.workers_per_gpu != "AUTO" and self.build.workers_per_gpu < 1:
            errors.append(f"WORKERS_PER_GPU must be >= 1, got {self.build.workers_per_gpu}")
        if self.build.chunk_size != "AUTO" and self.build.chunk_size < 2:
            errors.append(f"CHUNK_SIZE must be >= 2, got {self.build.chunk_size}")
        if (
            self.build.maxtasksperchild is not None
            and self.build.maxtasksperchild != "AUTO"
            and self.build.maxtasksperchild < 1
        ):
            errors.append(f"MAX_TASKS_PER_CHILD must be >= 1 or None, got {self.build.maxtasksperchild}")
        if self.build.docling_num_threads != "AUTO" and self.build.docling_num_threads < 1:
            errors.append(f"DOCLING_NUM_THREADS must be >= 1, got {self.build.docling_num_threads}")
        if self.wandb.log_interval_steps < 1:
            errors.append(f"WANDB_LOG_INTERVAL_STEPS must be >= 1, got {self.wandb.log_interval_steps}")

        # PEFT Validation
        if self.model.peft_dapt:
            if self.model.lora_r < 1:
                errors.append(f"LORA_R must be >= 1, got {self.model.lora_r}")
            if self.model.lora_alpha < 1:
                errors.append(f"LORA_ALPHA must be >= 1, got {self.model.lora_alpha}")
            if not (0.0 <= self.model.lora_dropout < 1.0):
                errors.append(f"LORA_DROPOUT must be in [0,1), got {self.model.lora_dropout}")
            if not self.model.lora_target_modules:
                errors.append("LORA_TARGET_MODULES cannot be empty when PEFT_DAPT is enabled")

        # RAD Prep Validation
        if self.rad.teacher_backend not in ("hf_local", "api", "bedrock"):
            errors.append(f"RAD_TEACHER_BACKEND must be 'hf_local', 'api', or 'bedrock', got '{self.rad.teacher_backend}'")
        if not (0.0 <= self.rad.relevance_threshold <= 1.0):
            errors.append(f"RAD_RELEVANCE_THRESHOLD must be in [0,1], got {self.rad.relevance_threshold}")
        if self.rad.top_k < 1:
            errors.append(f"RAD_TOP_K must be >= 1, got {self.rad.top_k}")
        if self.rad.embed_batch_size < 1:
            errors.append(f"RAD_EMBED_BATCH_SIZE must be >= 1, got {self.rad.embed_batch_size}")
        if self.rad.long_form_chunk_tokens <= self.rad.long_form_overlap_tokens:
            errors.append(f"RAD_LONG_FORM_CHUNK_TOKENS ({self.rad.long_form_chunk_tokens}) must be > overlap ({self.rad.long_form_overlap_tokens})")
        if self.rad.abstract_chunk_tokens <= self.rad.abstract_overlap_tokens:
            errors.append(f"RAD_ABSTRACT_CHUNK_TOKENS ({self.rad.abstract_chunk_tokens}) must be > overlap ({self.rad.abstract_overlap_tokens})")

        # Clustering Validation
        if self.clustering.noise_assignment not in ("nearest", "drop"):
            errors.append(f"CLUSTERING_NOISE_ASSIGNMENT must be 'nearest' or 'drop', got '{self.clustering.noise_assignment}'")
        if self.clustering.hdbscan_min_cluster_size < 2:
            errors.append(f"HDBSCAN_MIN_CLUSTER_SIZE must be >= 2, got {self.clustering.hdbscan_min_cluster_size}")
        if self.clustering.hdbscan_min_samples < 1:
            errors.append(f"HDBSCAN_MIN_SAMPLES must be >= 1, got {self.clustering.hdbscan_min_samples}")
        if self.clustering.min_clusters < 1:
            errors.append(f"CLUSTERING_MIN_CLUSTERS must be >= 1, got {self.clustering.min_clusters}")
        if self.clustering.use_pca and self.clustering.pca_components < 2:
            errors.append(f"CLUSTERING_PCA_COMPONENTS must be >= 2, got {self.clustering.pca_components}")
        if not (0.0 <= self.clustering.cluster_min_fraction <= 1.0):
            errors.append(f"CLUSTER_MIN_FRACTION must be in [0,1], got {self.clustering.cluster_min_fraction}")
        if not (0.0 <= self.clustering.cluster_max_fraction <= 1.0):
            errors.append(f"CLUSTER_MAX_FRACTION must be in [0,1], got {self.clustering.cluster_max_fraction}")
        if self.clustering.cluster_min_fraction > self.clustering.cluster_max_fraction:
            errors.append(f"CLUSTER_MIN_FRACTION ({self.clustering.cluster_min_fraction}) must be <= CLUSTER_MAX_FRACTION ({self.clustering.cluster_max_fraction})")
        
        # Split ratios validation
        ratios_sum = self.clustering.split_dev_ratio + self.clustering.split_val_ratio + self.clustering.split_sealed_ratio
        if not (0.99 <= ratios_sum <= 1.01):
            errors.append(f"Split ratios must sum to 1.0, got {ratios_sum}")
        if not (0.0 <= self.clustering.split_dev_ratio <= 1.0):
            errors.append(f"SPLIT_DEV_RATIO must be in [0,1], got {self.clustering.split_dev_ratio}")
        if not (0.0 <= self.clustering.split_val_ratio <= 1.0):
            errors.append(f"SPLIT_VAL_RATIO must be in [0,1], got {self.clustering.split_val_ratio}")
        if not (0.0 <= self.clustering.split_sealed_ratio <= 1.0):
            errors.append(f"SPLIT_SEALED_RATIO must be in [0,1], got {self.clustering.split_sealed_ratio}")

        # Benchmarking Validation
        if self.benchmarking.judge_backend not in ("hf_local", "api", "bedrock"):
            errors.append(f"BENCHMARK_JUDGE_BACKEND must be 'hf_local', 'api', or 'bedrock', got '{self.benchmarking.judge_backend}'")
        if self.benchmarking.teacher_backend and self.benchmarking.teacher_backend not in ("hf_local", "api", "bedrock"):
            errors.append(f"BENCHMARK_TEACHER_BACKEND must be 'hf_local', 'api', or 'bedrock', got '{self.benchmarking.teacher_backend}'")
        if self.benchmarking.teacher_batch_size < 1:
            errors.append(f"BENCHMARK_TEACHER_BATCH_SIZE must be >= 1, got {self.benchmarking.teacher_batch_size}")
        if self.benchmarking.eval_sample_size < 1:
            errors.append(f"BENCHMARK_EVAL_SAMPLE_SIZE must be >= 1, got {self.benchmarking.eval_sample_size}")
        if self.benchmarking.min_eval_samples < 1:
            errors.append(f"BENCHMARK_MIN_EVAL_SAMPLES must be >= 1, got {self.benchmarking.min_eval_samples}")
        if not (0.0 <= self.benchmarking.citation_min_overlap <= 1.0):
            errors.append(f"BENCHMARK_CITATION_MIN_OVERLAP must be in [0,1], got {self.benchmarking.citation_min_overlap}")
        if not (0.0 <= self.benchmarking.hallucination_nli_threshold <= 1.0):
            errors.append(f"BENCHMARK_HALLUCINATION_NLI_THRESHOLD must be in [0,1], got {self.benchmarking.hallucination_nli_threshold}")

        if errors:
            raise ValueError("Config validation failed:\n" + "\n".join(f"  • {e}" for e in errors))

    def ensure_dirs(self):
        """Create output directories if they don't exist."""
        for d in [
            self.model.checkpoint_dir,
            self.logging.log_dir,
            self.rad.index_dir,
            self.rad.traces_dir,
            self.clustering.output_dir,
            self.clustering.cluster_report_path.parent,
            self.benchmarking.output_dir,
            self.benchmarking.manifest_path.parent,
            self.benchmarking.scores_path.parent,
            self.benchmarking.calibration_log_path.parent,
            self.benchmarking.inter_rater_log_path.parent,
        ]:
            d.mkdir(parents=True, exist_ok=True)

    def summary(self) -> str:
        return (
            f"\n{'='*60}\n"
            f"  DAPT Configuration\n"
            f"{'='*60}\n"
            f"  Model           : {self.model.base_model_name}\n"
            f"  PEFT DAPT       : {self.model.peft_dapt}\n"
            f"  Train batch size: {self.optimizer.train_batch_size}\n"
            f"  Eval batch size : {self.optimizer.eval_batch_size}\n"
            f"  Grad checkpoint : {self.model.gradient_checkpointing}\n"
            f"  Corpus tokens   : {self.corpus.total_corpus_tokens/1e3:.1f}K\n"
            f"  Max passes      : {self.corpus.max_corpus_passes}\n"
            f"  Hard stop       : {self.corpus.hard_stop_tokens/1e3:.1f}K tokens\n"
            f"  Eval interval   : {self.corpus.eval_interval_tokens/1e3:.0f}K tokens\n"
            f"  Slow eval int   : {self.corpus.slow_eval_interval_tokens/1e3:.0f}K tokens\n"
            f"  Probe 1 - Perplexity gate: < {self.gates.ppl_improvement_threshold}% for "
            f"{self.gates.ppl_plateau_window} consecutive evals (Enabled: {self.probes.run_perplexity})\n"
            f"  Probe 2 - QA gate        : >= {self.gates.qa_acc_threshold:.0%} (Enabled: {self.probes.run_qa})\n"
            f"  Probe 3 - Cloze gate     : >= {self.gates.cloze_threshold:.0%} (Enabled: {self.probes.run_cloze})\n"
            f"  Probe 4 - Concept gate   : >= {self.gates.concept_threshold:.0%} (Enabled: {self.probes.run_concept})\n"
            f"  RAD Embedding   : {self.rad.embedding_model}\n"
            f"  RAD Mode        : {self.rad.retrieval_mode}\n"
            f"  RAD Top-K       : {self.rad.top_k}\n"
            f"  RAD Threshold   : {self.rad.relevance_threshold}\n"
            f"  Checkpoint dir  : {self.model.checkpoint_dir}\n"
            f"  Log dir         : {self.logging.log_dir}\n"
            f"{'='*60}\n"
        )


# Alias for backwards compatibility with DAPT pretraining scripts
DAPTConfig = PipelineConfig
