"""
utils/config.py — Load and validate all Semantics Layer pipeline parameters from the root .env
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from dotenv import load_dotenv
from typing import Optional
from datetime import datetime


# Load .env from project root
root_dir = Path(__file__).resolve().parent.parent.parent
load_dotenv(dotenv_path=root_dir / ".env")


def _get(key: str, default, cast=str):
    raw = os.environ.get(key, None)
    if raw is None:
        return cast(default) if not isinstance(default, cast) else default
    try:
        if cast is bool:
            if isinstance(raw, str):
                return raw.lower() in ("true", "1", "yes", "on")
            return bool(raw)
        return cast(raw)
    except (ValueError, TypeError) as e:
        raise ValueError(f"[config] Cannot parse env var {key}={raw!r} as {cast.__name__}: {e}")


@dataclass
class CorpusBuildConfig:
    storage_target: str       = field(default_factory=lambda: _get("STORAGE_TARGET", "local"))
    local_directory_path: str = field(default_factory=lambda: _get("LOCAL_DIRECTORY_PATH", "."))
    aws_bucket_name: Optional[str] = field(default_factory=lambda: _get("AWS_BUCKET_NAME", None))
    aws_prefix: str           = field(default_factory=lambda: _get("AWS_PREFIX", ""))
    gdrive_folder_id: Optional[str] = field(default_factory=lambda: _get("GDRIVE_FOLDER_ID", None))
    available_gpus: str       = field(default_factory=lambda: _get("AVAILABLE_GPUS", "0"))
    workers_per_gpu: int      = field(default_factory=lambda: _get("WORKERS_PER_GPU", 1, int))
    chunk_size: int           = field(default_factory=lambda: _get("CHUNK_SIZE", 10, int))
    output_path: Path         = field(default_factory=lambda: Path(_get("OUTPUT_PATH", "./data/dapt/domain_dapt_corpus.jsonl")))


@dataclass
class CorpusConfig:
    total_corpus_tokens: int   = field(default_factory=lambda: _get("TOTAL_CORPUS_TOKENS", 30_000_000_000, int))
    max_corpus_passes: int     = field(default_factory=lambda: _get("MAX_CORPUS_PASSES", 3, int))
    eval_interval_tokens: int  = field(default_factory=lambda: _get("EVAL_INTERVAL_TOKENS", 500_000_000, int))

    @property
    def hard_stop_tokens(self) -> int:
        return self.total_corpus_tokens * self.max_corpus_passes


@dataclass
class GateConfig:
    # Primary Gate A
    qa_acc_threshold: float        = field(default_factory=lambda: _get("QA_ACC_THRESHOLD", 0.55, float))

    # Primary Gate B
    ppl_improvement_threshold: float = field(default_factory=lambda: _get("PPL_IMPROVEMENT_THRESHOLD", 2.0, float))
    ppl_plateau_window: int          = field(default_factory=lambda: _get("PPL_PLATEAU_WINDOW", 2, int))

    # Secondary Gate
    term_cov_threshold: float      = field(default_factory=lambda: _get("TERM_COV_THRESHOLD", 0.80, float))
    ret_prec_threshold: float      = field(default_factory=lambda: _get("RET_PREC_THRESHOLD", 0.60, float))

    # Remediation routing
    qa_low_threshold: float        = field(default_factory=lambda: _get("QA_LOW_THRESHOLD", 0.40, float))


@dataclass
class ProbeConfig:
    # Terminology cloze
    term_cov_top_k: int            = field(default_factory=lambda: _get("TERM_COV_TOP_K", 5, int))
    term_cov_max_new_tokens: int   = field(default_factory=lambda: _get("TERM_COV_MAX_NEW_TOKENS", 3, int))

    # Anatomical retrieval
    bertscore_model: str           = field(default_factory=lambda: _get("BERTSCORE_MODEL", "allenai/scibert_scivocab_uncased", str))
    ret_prec_max_new_tokens: int   = field(default_factory=lambda: _get("RET_PREC_MAX_NEW_TOKENS", 100, int))

    # PPL eval corpus size
    perplexity_eval_tokens: int    = field(default_factory=lambda: _get("PERPLEXITY_EVAL_TOKENS", 10_000_000, int))


@dataclass
class DataConfig:
    qa_probe_path: Path            = field(default_factory=lambda: Path(
        os.environ.get("PROBE_QA_PATH", os.environ.get("QA_PROBE_PATH", "evals/dapt/probe_qa.jsonl"))
    ))
    ppl_corpus_path: Path          = field(default_factory=lambda: Path(
        os.environ.get("PPL_CORPUS_PATH", os.environ.get("PPL_HELD_OUT_PATH", "evals/dapt/ppl_held_out.txt"))
    ))
    vocab_cloze_path: Path         = field(default_factory=lambda: Path(
        os.environ.get("VOCAB_CLOZE_PATH", "evals/dapt/vocab_cloze_set.json")
    ))
    anatomical_prompts_path: Path  = field(default_factory=lambda: Path(
        os.environ.get("ANATOMICAL_PROMPTS_PATH", "evals/dapt/anatomical_prompts.json")
    ))
    anatomical_references_path: Path = field(default_factory=lambda: Path(
        os.environ.get("ANATOMICAL_REFERENCES_PATH", "evals/dapt/anatomical_references.json")
    ))


@dataclass
class ModelConfig:
    base_model_name: str           = field(default_factory=lambda: _get("BASE_MODEL_NAME", "HuggingFaceTB/SmolLM2-135M"))
    model_dtype: str               = field(default_factory=lambda: _get("MODEL_DTYPE", "bfloat16"))
    max_seq_len: int               = field(default_factory=lambda: _get("MAX_SEQ_LEN", 512, int))


@dataclass
class OptimizerConfig:
    learning_rate: float           = field(default_factory=lambda: _get("DAPT_LR", 5e-5, float))
    weight_decay: float            = field(default_factory=lambda: _get("WEIGHT_DECAY", 0.01, float))
    warmup_steps: int              = field(default_factory=lambda: _get("WARMUP_STEPS", 1000, int))
    max_grad_norm: float           = field(default_factory=lambda: _get("MAX_GRAD_NORM", 1.0, float))
    train_batch_size: int          = field(default_factory=lambda: _get("DAPT_BATCH_SIZE", 2, int))
    eval_batch_size: int           = field(default_factory=lambda: _get("EVAL_BATCH_SIZE", 4, int))


@dataclass
class StorageConfig:
    checkpoint_dir: Path           = field(default_factory=lambda: Path(_get("CHECKPOINT_DIR", "models/checkpoints")))
    log_dir: Path                  = field(default_factory=lambda: Path(_get("LOG_DIR", "logs")))
    metrics_log_file: Path         = field(default_factory=lambda: Path(_get("METRICS_LOG_FILE", "logs/dapt_eval_metrics.jsonl")))
    best_checkpoint_manifest: Path = field(default_factory=lambda: Path(_get("BEST_CHECKPOINT_MANIFEST", "logs/best_checkpoint.json")))
    risk_report_path: Path         = field(default_factory=lambda: Path(_get("RISK_REPORT_PATH", "logs/dapt_hard_cap_risk_report.json")))
    checkpoint_keep_last: int      = field(default_factory=lambda: _get("CHECKPOINT_KEEP_LAST", 5, int))


@dataclass
class MiscConfig:
    seed: int                      = field(default_factory=lambda: _get("SEED", 42, int))
    log_level: str                 = field(default_factory=lambda: _get("LOG_LEVEL", "INFO"))


@dataclass
class WandbConfig:
    enabled: bool                  = field(default_factory=lambda: _get("WANDB_ENABLED", False, bool))
    api_key: Optional[str]         = field(default_factory=lambda: _get("WANDB_API_KEY", None, str))
    project: str                   = field(default_factory=lambda: _get("WANDB_PROJECT", "semantics-dapt", str))
    entity: Optional[str]          = field(default_factory=lambda: _get("WANDB_ENTITY", None, str))
    run_name: Optional[str]        = field(default_factory=lambda: _get("WANDB_RUN_NAME", None, str))
    log_interval_steps: int        = field(default_factory=lambda: _get("WANDB_LOG_INTERVAL_STEPS", 10, int))

    def __post_init__(self):
        if not self.run_name:
            timestamp = datetime.now().strftime("%y%m%d%H%M%S")
            self.run_name = f"{self.project}_{timestamp}"



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
    misc: MiscConfig          = field(default_factory=MiscConfig)
    wandb: WandbConfig        = field(default_factory=WandbConfig)

    def validate(self):
        """Sanity-check values that must satisfy domain constraints."""
        errors = []

        if not (0.0 <= self.gates.qa_acc_threshold <= 1.0):
            errors.append(f"QA_ACC_THRESHOLD must be in [0,1], got {self.gates.qa_acc_threshold}")
        if not (0.0 <= self.gates.term_cov_threshold <= 1.0):
            errors.append(f"TERM_COV_THRESHOLD must be in [0,1], got {self.gates.term_cov_threshold}")
        if not (0.0 <= self.gates.ret_prec_threshold <= 1.0):
            errors.append(f"RET_PREC_THRESHOLD must be in [0,1], got {self.gates.ret_prec_threshold}")
        if self.gates.ppl_plateau_window < 2:
            errors.append(f"PPL_PLATEAU_WINDOW must be >= 2, got {self.gates.ppl_plateau_window}")
        if self.corpus.max_corpus_passes < 1:
            errors.append(f"MAX_CORPUS_PASSES must be >= 1, got {self.corpus.max_corpus_passes}")
        if self.corpus.eval_interval_tokens <= 0:
            errors.append(f"EVAL_INTERVAL_TOKENS must be > 0, got {self.corpus.eval_interval_tokens}")
        if self.gates.qa_low_threshold > self.gates.qa_acc_threshold:
            errors.append(
                f"QA_LOW_THRESHOLD ({self.gates.qa_low_threshold}) must be <= "
                f"QA_ACC_THRESHOLD ({self.gates.qa_acc_threshold})"
            )
        if self.build.workers_per_gpu < 1:
            errors.append(f"WORKERS_PER_GPU must be >= 1, got {self.build.workers_per_gpu}")
        if self.build.chunk_size < 2:
            errors.append(f"CHUNK_SIZE must be >= 2, got {self.build.chunk_size}")
        if self.wandb.log_interval_steps < 1:
            errors.append(f"WANDB_LOG_INTERVAL_STEPS must be >= 1, got {self.wandb.log_interval_steps}")

        if errors:
            raise ValueError("Config validation failed:\n" + "\n".join(f"  • {e}" for e in errors))

    def ensure_dirs(self):
        """Create output directories if they don't exist."""
        for d in [
            self.storage.checkpoint_dir,
            self.storage.log_dir,
        ]:
            d.mkdir(parents=True, exist_ok=True)

    def summary(self) -> str:
        return (
            f"\n{'='*60}\n"
            f"  DAPT Step 0.3 Configuration\n"
            f"{'='*60}\n"
            f"  Model           : {self.model.base_model_name}\n"
            f"  Corpus tokens   : {self.corpus.total_corpus_tokens/1e3:.1f}K\n"
            f"  Max passes      : {self.corpus.max_corpus_passes}\n"
            f"  Hard stop       : {self.corpus.hard_stop_tokens/1e3:.1f}K tokens\n"
            f"  Eval interval   : {self.corpus.eval_interval_tokens/1e3:.0f}K tokens\n"
            f"  QA gate         : >= {self.gates.qa_acc_threshold:.0%}\n"
            f"  PPL gate        : < {self.gates.ppl_improvement_threshold}% for "
            f"{self.gates.ppl_plateau_window} consecutive evals\n"
            f"  Term cov gate   : >= {self.gates.term_cov_threshold:.0%}\n"
            f"  Ret prec gate   : >= {self.gates.ret_prec_threshold:.0%}\n"
            f"  Checkpoint dir  : {self.storage.checkpoint_dir}\n"
            f"  Log dir         : {self.storage.log_dir}\n"
            f"{'='*60}\n"
        )


# Alias for backwards compatibility with DAPT pretraining scripts
DAPTConfig = PipelineConfig
