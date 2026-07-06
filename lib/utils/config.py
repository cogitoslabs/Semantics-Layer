"""
utils/config.py — Load and validate all Semantics Layer pipeline parameters from the root .env
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from dotenv import load_dotenv
from typing import Optional, List, Dict
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
    slow_eval_interval_tokens: int = field(default_factory=lambda: _get("SLOW_EVAL_INTERVAL_TOKENS", 250_000_000, int))

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
    # Probe Activation Toggles
    run_perplexity: bool           = field(default_factory=lambda: _get("RUN_PERPLEXITY_PROBE", True, bool))
    run_qa: bool                  = field(default_factory=lambda: _get("RUN_QA_PROBE", True, bool))
    run_terminology: bool         = field(default_factory=lambda: _get("RUN_TERMINOLOGY_PROBE", True, bool))
    run_retrieval: bool           = field(default_factory=lambda: _get("RUN_RETRIEVAL_PROBE", True, bool))

    # Terminology cloze
    term_cov_top_k: int            = field(default_factory=lambda: _get("TERM_COV_TOP_K", 5, int))
    term_cov_max_new_tokens: int   = field(default_factory=lambda: _get("TERM_COV_MAX_NEW_TOKENS", 3, int))
    term_cov_gen_batch_size: int   = field(default_factory=lambda: _get("TERM_COV_GEN_BATCH_SIZE", 16, int))

    # Anatomical retrieval
    bertscore_model: str           = field(default_factory=lambda: _get("BERTSCORE_MODEL", "allenai/scibert_scivocab_uncased", str))
    ret_prec_max_new_tokens: int   = field(default_factory=lambda: _get("RET_PREC_MAX_NEW_TOKENS", 100, int))
    ret_prec_gen_batch_size: int   = field(default_factory=lambda: _get("RET_PREC_GEN_BATCH_SIZE", 16, int))

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
    retrieval_prompts_path: Path  = field(default_factory=lambda: Path(
        os.environ.get("RETRIEVAL_PROMPTS_PATH", "evals/dapt/retrieval_prompts.json")
    ))
    retrieval_references_path: Path = field(default_factory=lambda: Path(
        os.environ.get("RETRIEVAL_REFERENCES_PATH", "evals/dapt/retrieval_references.json")
    ))
    pretokenized_bin_path: Path    = field(default_factory=lambda: Path(
        os.environ.get("PRETOKENIZED_BIN_PATH", "data/dapt/train_tokens.npy")
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
    train_batch_size: int          = field(default_factory=lambda: _get("TRAIN_BATCH_SIZE", 2, int))
    eval_batch_size: int           = field(default_factory=lambda: _get("EVAL_BATCH_SIZE", 4, int))
    gradient_accumulation_steps: int = field(default_factory=lambda: _get("GRADIENT_ACCUMULATION_STEPS", 1, int))


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
    mode: str                      = field(default_factory=lambda: _get("WANDB_MODE", "online", str))
    api_key: Optional[str]         = field(default_factory=lambda: _get("WANDB_API_KEY", None, str))
    project: str                   = field(default_factory=lambda: _get("WANDB_PROJECT", "semantics-dapt", str))
    entity: Optional[str]          = field(default_factory=lambda: _get("WANDB_ENTITY", None, str))
    run_name: Optional[str]        = field(default_factory=lambda: _get("WANDB_RUN_NAME", None, str))
    log_interval_steps: int        = field(default_factory=lambda: _get("WANDB_LOG_INTERVAL_STEPS", 10, int))

    def __post_init__(self):
        if not self.run_name:
            timestamp = datetime.now().strftime("%y%m%d%H%M%S")
            self.run_name = f"{self.project}_{timestamp}"
        if self.mode:
            os.environ["WANDB_MODE"] = self.mode



@dataclass
class RADPrepConfig:
    # Corpus paths
    retrieval_corpus_path: Path    = field(default_factory=lambda: Path(_get("RAD_CORPUS_PATH", "data/rad_prep/retrieval_corpus.jsonl")))
    chunks_path: Path              = field(default_factory=lambda: Path(_get("RAD_CHUNKS_PATH", "data/rad_prep/chunks.jsonl")))
    index_dir: Path                = field(default_factory=lambda: Path(_get("RAD_INDEX_DIR", "data/rad_prep/index")))
    traces_dir: Path               = field(default_factory=lambda: Path(_get("RAD_TRACES_DIR", "data/rad_prep/traces")))
    qa_samples_path: Path          = field(default_factory=lambda: Path(_get("RAD_QA_SAMPLES_PATH", "evals/dapt/probe_qa.jsonl")))

    # Retrieval settings
    embedding_model: str           = field(default_factory=lambda: _get("RAD_EMBEDDING_MODEL", "biolinkbert"))
    retrieval_mode: str            = field(default_factory=lambda: _get("RAD_RETRIEVAL_MODE", "hybrid"))  # dense|sparse|hybrid
    top_k: int                     = field(default_factory=lambda: _get("RAD_TOP_K", 7, int))
    relevance_threshold: float     = field(default_factory=lambda: _get("RAD_RELEVANCE_THRESHOLD", 0.65, float))
    embed_batch_size: int          = field(default_factory=lambda: _get("RAD_EMBED_BATCH_SIZE", 64, int))

    # Chunking
    long_form_chunk_tokens: int    = field(default_factory=lambda: _get("RAD_LONG_FORM_CHUNK_TOKENS", 512, int))
    long_form_overlap_tokens: int  = field(default_factory=lambda: _get("RAD_LONG_FORM_OVERLAP_TOKENS", 64, int))
    abstract_chunk_tokens: int     = field(default_factory=lambda: _get("RAD_ABSTRACT_CHUNK_TOKENS", 256, int))
    abstract_overlap_tokens: int   = field(default_factory=lambda: _get("RAD_ABSTRACT_OVERLAP_TOKENS", 32, int))

    # Teacher
    teacher_backend: str           = field(default_factory=lambda: _get("RAD_TEACHER_BACKEND", "hf_local"))
    teacher_model_name: str        = field(default_factory=lambda: _get("RAD_TEACHER_MODEL_NAME", "Qwen/Qwen3-1.7B"))
    teacher_api_url: Optional[str] = field(default_factory=lambda: _get("RAD_TEACHER_API_URL", None))
    teacher_api_key: Optional[str] = field(default_factory=lambda: _get("RAD_TEACHER_API_KEY", None))
    teacher_max_new_tokens: int    = field(default_factory=lambda: _get("RAD_TEACHER_MAX_NEW_TOKENS", 1024, int))
    teacher_batch_size: int        = field(default_factory=lambda: _get("RAD_TEACHER_BATCH_SIZE", 4, int))

    # Trace filtering
    trace_min_tokens: int          = field(default_factory=lambda: _get("RAD_TRACE_MIN_TOKENS", 200, int))
    trace_max_tokens: int          = field(default_factory=lambda: _get("RAD_TRACE_MAX_TOKENS", 2500, int))
    min_traces: int                = field(default_factory=lambda: _get("RAD_MIN_TRACES", 1000, int))


@dataclass
class ClusteringConfig:
    # Input
    corpus_path: Path              = field(default_factory=lambda: Path(_get("CLUSTERING_CORPUS_PATH", "data/dapt/domain_dapt_corpus.jsonl")))

    # Embedding
    embedding_model: str           = field(default_factory=lambda: _get("CLUSTERING_EMBEDDING_MODEL", "all-mpnet-base-v2"))
    embed_batch_size: int          = field(default_factory=lambda: _get("CLUSTERING_EMBED_BATCH_SIZE", 64, int))
    embeddings_cache_path: Path    = field(default_factory=lambda: Path(_get("CLUSTERING_EMBEDDINGS_CACHE", "data/clustering/embeddings.npy")))
    doc_ids_cache_path: Path       = field(default_factory=lambda: Path(_get("CLUSTERING_DOC_IDS_CACHE", "data/clustering/doc_ids.json")))

    # HDBSCAN
    hdbscan_min_cluster_size: int  = field(default_factory=lambda: _get("HDBSCAN_MIN_CLUSTER_SIZE", 10, int))
    hdbscan_min_samples: int       = field(default_factory=lambda: _get("HDBSCAN_MIN_SAMPLES", 5, int))
    hdbscan_metric: str            = field(default_factory=lambda: _get("HDBSCAN_METRIC", "cosine"))
    min_clusters: int              = field(default_factory=lambda: _get("CLUSTERING_MIN_CLUSTERS", 10, int))
    use_pca: bool                  = field(default_factory=lambda: _get("CLUSTERING_USE_PCA", True, bool))
    pca_components: int            = field(default_factory=lambda: _get("CLUSTERING_PCA_COMPONENTS", 10, int))

    # Noise handling
    noise_assignment: str          = field(default_factory=lambda: _get("CLUSTERING_NOISE_ASSIGNMENT", "nearest"))

    # Imbalance reweighting
    cluster_min_fraction: float    = field(default_factory=lambda: _get("CLUSTER_MIN_FRACTION", 0.02, float))
    cluster_max_fraction: float    = field(default_factory=lambda: _get("CLUSTER_MAX_FRACTION", 0.15, float))

    # Split ratios
    split_dev_ratio: float         = field(default_factory=lambda: _get("SPLIT_DEV_RATIO", 0.70, float))
    split_val_ratio: float         = field(default_factory=lambda: _get("SPLIT_VAL_RATIO", 0.20, float))
    split_sealed_ratio: float      = field(default_factory=lambda: _get("SPLIT_SEALED_RATIO", 0.10, float))

    # Output paths
    output_dir: Path               = field(default_factory=lambda: Path(_get("CLUSTERING_OUTPUT_DIR", "data/clustering")))
    assignments_path: Path         = field(default_factory=lambda: Path(_get("CLUSTERING_ASSIGNMENTS_PATH", "data/clustering/cluster_assignments.jsonl")))
    splits_path: Path              = field(default_factory=lambda: Path(_get("CLUSTERING_SPLITS_PATH", "data/clustering/splits.json")))
    cluster_manifest_path: Path    = field(default_factory=lambda: Path(_get("CLUSTERING_MANIFEST_PATH", "data/clustering/cluster_manifest.json")))
    cluster_report_path: Path      = field(default_factory=lambda: Path(_get("CLUSTERING_REPORT_PATH", "logs/clustering/cluster_report.json")))


@dataclass
class TeacherBenchmarkingConfig:
    candidate_teachers: List[str]       = field(default_factory=lambda: [t.strip() for t in _get("BENCHMARK_TEACHERS", "Qwen/Qwen3-1.7B").split(",") if t.strip()])
    judge_backend: str                  = field(default_factory=lambda: _get("BENCHMARK_JUDGE_BACKEND", "api"))
    judge_model_name: str               = field(default_factory=lambda: _get("BENCHMARK_JUDGE_MODEL", ""))
    judge_api_url: Optional[str]        = field(default_factory=lambda: _get("BENCHMARK_JUDGE_API_URL", None))
    judge_api_key: Optional[str]        = field(default_factory=lambda: _get("BENCHMARK_JUDGE_API_KEY", None))
    judge_max_new_tokens: int           = field(default_factory=lambda: _get("BENCHMARK_JUDGE_MAX_NEW_TOKENS", 256, int))
    
    teacher_backend: str                = field(default_factory=lambda: _get("BENCHMARK_TEACHER_BACKEND", ""))
    teacher_batch_size: int             = field(default_factory=lambda: _get("BENCHMARK_TEACHER_BATCH_SIZE", 4, int))
    
    eval_sample_size: int               = field(default_factory=lambda: _get("BENCHMARK_EVAL_SAMPLE_SIZE", 200, int))
    min_eval_samples: int               = field(default_factory=lambda: _get("BENCHMARK_MIN_EVAL_SAMPLES", 10, int))
    
    enable_calibration: bool            = field(default_factory=lambda: _get("BENCHMARK_ENABLE_CALIBRATION", False, bool))
    human_calibration_size: int         = field(default_factory=lambda: _get("BENCHMARK_CALIBRATION_SIZE", 200, int))
    human_labels_path: Optional[Path]   = field(default_factory=lambda: Path(_get("BENCHMARK_HUMAN_LABELS_PATH", "")) if _get("BENCHMARK_HUMAN_LABELS_PATH", "") else None)
    
    hallucination_nli_threshold: float  = field(default_factory=lambda: _get("BENCHMARK_HALLUCINATION_NLI_THRESHOLD", 0.5, float))
    citation_min_overlap: float         = field(default_factory=lambda: _get("BENCHMARK_CITATION_MIN_OVERLAP", 0.30, float))
    nli_model: str                      = field(default_factory=lambda: _get("BENCHMARK_NLI_MODEL", "cross-encoder/nli-deberta-v3-small"))
    
    output_dir: Path                    = field(default_factory=lambda: Path(_get("BENCHMARKING_OUTPUT_DIR", "data/benchmarking")))
    scores_path: Path                   = field(default_factory=lambda: Path(_get("BENCHMARKING_SCORES_PATH", "data/benchmarking/scores.jsonl")))
    manifest_path: Path                 = field(default_factory=lambda: Path(_get("BENCHMARKING_MANIFEST_PATH", "data/benchmarking/benchmark_manifest.json")))
    calibration_log_path: Path          = field(default_factory=lambda: Path(_get("BENCHMARKING_CALIBRATION_LOG", "logs/benchmarking/judge_calibration.jsonl")))
    inter_rater_log_path: Path          = field(default_factory=lambda: Path(_get("BENCHMARKING_INTER_RATER_LOG", "logs/benchmarking/inter_rater_agreement.json")))


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
    rad: RADPrepConfig        = field(default_factory=RADPrepConfig)
    clustering: ClusteringConfig = field(default_factory=ClusteringConfig)
    benchmarking: TeacherBenchmarkingConfig = field(default_factory=TeacherBenchmarkingConfig)

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
        if self.corpus.slow_eval_interval_tokens <= 0:
            errors.append(f"SLOW_EVAL_INTERVAL_TOKENS must be > 0, got {self.corpus.slow_eval_interval_tokens}")
        if self.optimizer.gradient_accumulation_steps < 1:
            errors.append(f"GRADIENT_ACCUMULATION_STEPS must be >= 1, got {self.optimizer.gradient_accumulation_steps}")
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
            self.storage.checkpoint_dir,
            self.storage.log_dir,
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
            f"  DAPT Step 0.3 Configuration\n"
            f"{'='*60}\n"
            f"  Model           : {self.model.base_model_name}\n"
            f"  Corpus tokens   : {self.corpus.total_corpus_tokens/1e3:.1f}K\n"
            f"  Max passes      : {self.corpus.max_corpus_passes}\n"
            f"  Hard stop       : {self.corpus.hard_stop_tokens/1e3:.1f}K tokens\n"
            f"  Eval interval   : {self.corpus.eval_interval_tokens/1e3:.0f}K tokens\n"
            f"  Slow eval int   : {self.corpus.slow_eval_interval_tokens/1e3:.0f}K tokens\n"
            f"  QA gate         : >= {self.gates.qa_acc_threshold:.0%} (Enabled: {self.probes.run_qa})\n"
            f"  PPL gate        : < {self.gates.ppl_improvement_threshold}% for "
            f"{self.gates.ppl_plateau_window} consecutive evals (Enabled: {self.probes.run_perplexity})\n"
            f"  Term cov gate   : >= {self.gates.term_cov_threshold:.0%} (Enabled: {self.probes.run_terminology})\n"
            f"  Ret prec gate   : >= {self.gates.ret_prec_threshold:.0%} (Enabled: {self.probes.run_retrieval})\n"
            f"  RAD Embedding   : {self.rad.embedding_model}\n"
            f"  RAD Mode        : {self.rad.retrieval_mode}\n"
            f"  RAD Top-K       : {self.rad.top_k}\n"
            f"  RAD Threshold   : {self.rad.relevance_threshold}\n"
            f"  Checkpoint dir  : {self.storage.checkpoint_dir}\n"
            f"  Log dir         : {self.storage.log_dir}\n"
            f"{'='*60}\n"
        )


# Alias for backwards compatibility with DAPT pretraining scripts
DAPTConfig = PipelineConfig
