"""
utils/logger.py — Shared structured logging and JSONL metrics writer
"""

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


from .config import LoggingConfig


class AutoFlushingFileHandler(logging.FileHandler):
    """FileHandler that automatically flushes log entries to disk immediately after each write."""
    def emit(self, record: logging.LogRecord) -> None:
        super().emit(record)
        self.flush()


def setup_logger(
    name: str,
    cfg: LoggingConfig,
) -> logging.Logger:
    """
    Set up a logger that writes to both stdout and a rotating log file.
    All modules should call this once and share the logger by name.
    """
    log_dir = cfg.log_dir
    level = cfg.log_level
    log_filename = cfg.log_file

    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / log_filename

    numeric_level = getattr(logging, level.upper(), logging.INFO)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    # Automatically set up parent loggers "dapt" and "lib" to catch all sub-logs
    for parent_name in ["dapt", "lib"]:
        parent_logger = logging.getLogger(parent_name)
        if not parent_logger.handlers:
            parent_logger.setLevel(numeric_level)
            parent_logger.propagate = False

            ch_p = logging.StreamHandler(sys.stdout)
            ch_p.setLevel(numeric_level)
            ch_p.setFormatter(formatter)
            parent_logger.addHandler(ch_p)

            fh_p = AutoFlushingFileHandler(log_file, mode="a", encoding="utf-8")
            fh_p.setLevel(numeric_level)
            fh_p.setFormatter(formatter)
            parent_logger.addHandler(fh_p)

    logger = logging.getLogger(name)
    logger.setLevel(numeric_level)

    # Enable propagation for sub-loggers of dapt or lib so they route to the parent handlers
    if name.startswith("dapt.") or name.startswith("lib."):
        logger.propagate = True
    else:
        logger.propagate = False

    # Avoid duplicate handlers on re-import
    if logger.handlers:
        return logger

    # If it is not a sub-logger of dapt or lib, attach handlers directly
    if not (name.startswith("dapt.") or name.startswith("lib.")):
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(numeric_level)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

        fh = AutoFlushingFileHandler(log_file, mode="a", encoding="utf-8")
        fh.setLevel(numeric_level)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger


def get_logger(name: str = "dapt.convergence") -> logging.Logger:
    """Retrieve an already-configured logger by name."""
    return logging.getLogger(name)


def flush_loggers() -> None:
    """Flush all file handlers attached to 'dapt', 'lib', 'pipeline' and root loggers."""
    for logger_name in ["dapt", "lib", "pipeline", ""]:
        l = logging.getLogger(logger_name)
        for h in l.handlers:
            try:
                h.flush()
            except Exception:
                pass


def close_loggers() -> None:
    """Close and remove all file handlers attached to 'dapt', 'lib', 'pipeline' and root loggers."""
    flush_loggers()
    for logger_name in ["dapt", "lib", "pipeline", ""]:
        l = logging.getLogger(logger_name)
        handlers_to_remove = []
        for h in l.handlers:
            if isinstance(h, logging.FileHandler):
                h.close()
                handlers_to_remove.append(h)
        for h in handlers_to_remove:
            l.removeHandler(h)


class MetricsWriter:
    """
    Appends one JSON record per evaluation to a JSONL file.
    Each line is a self-contained evaluation snapshot.
    """

    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, record: Dict[str, Any]) -> None:
        record["timestamp"] = datetime.now(timezone.utc).isoformat()
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=str) + "\n")

    def read_all(self) -> list:
        if not self.path.exists():
            return []
        records = []
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
        return records


def save_json(data: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)


def load_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def format_gate_status(
    eval_id: int,
    tokens_processed: int,
    total_corpus_tokens: int,
    qa_acc: float,
    ppl_history: list,
    ppl_improvements: list,
    cloze_cov: float,
    concept_prec: float,
    qa_gate: bool,
    ppl_gate: bool,
    secondary_gate: bool,
    qa_threshold: float,
    ppl_threshold: float,
    ppl_window: int,
    cloze_threshold: float,
    concept_threshold: float,
    decision: str,
    run_qa: bool = True,
    run_perplexity: bool = True,
    run_cloze: bool = True,
    run_concept: bool = True,
    qa_history: list = None,
    cloze_history: list = None,
    concept_history: list = None,
) -> str:
    """Produce a human-readable gate status block for logging."""
    tick = lambda b, active=True: "[PASS]" if b else ("[SKIP]" if not active else "[FAIL]")
    ppl_str = ", ".join(f"{p:.2f}%" for p in ppl_improvements) if ppl_improvements else "n/a (need >=2 evals)"
    ppl_vals = ", ".join(f"{p:.3f}" for p in ppl_history)
    qa_vals = ", ".join(f"{q:.4f}" for q in qa_history) if qa_history else "n/a"
    cloze_vals = ", ".join(f"{t:.4f}" for t in cloze_history) if cloze_history else "n/a"
    concept_vals = ", ".join(f"{r:.4f}" for r in concept_history) if concept_history else "n/a"

    return (
        f"\n{'-'*62}\n"
        f"  Gate Status @ Eval #{eval_id:>3}\n"
        f"{'-'*62}\n"
        f"  Tokens processed : {tokens_processed/1e3:.2f}K\n"
        f"  Corpus pass      : {tokens_processed/total_corpus_tokens:.3f}x\n"
        f"\n"
        f"  PRIMARY GATES (both required for convergence)\n"
        f"  {tick(qa_gate, run_qa)} QA Accuracy   : {qa_acc:.4f}  (threshold >= {qa_threshold})\n"
        f"  QA history       : [{qa_vals}]\n"
        f"  {tick(ppl_gate, run_perplexity)} PPL Plateau   : improvements = [{ppl_str}]  (need <{ppl_threshold}% for {ppl_window} consecutive)\n"
        f"  PPL history      : [{ppl_vals}]\n"
        f"\n"
        f"  SECONDARY GATES (at least one required)\n"
        f"  {tick(cloze_cov >= cloze_threshold, run_cloze)} Cloze Coverage: {cloze_cov:.4f}  (threshold >= {cloze_threshold})\n"
        f"  Cloze history    : [{cloze_vals}]\n"
        f"  {tick(concept_prec >= concept_threshold, run_concept)} Concept Precision : {concept_prec:.4f}  (threshold >= {concept_threshold})\n"
        f"  Concept history  : [{concept_vals}]\n"
        f"\n"
        f"  DECISION -> {decision}\n"
        f"{'-'*62}\n"
    )
