"""
utils/logger.py — Shared structured logging and JSONL metrics writer
"""

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


def setup_logger(
    name: str,
    log_dir: Path,
    level: str = "INFO",
    log_filename: Optional[str] = None,
) -> logging.Logger:
    """
    Set up a logger that writes to both stdout and a rotating log file.
    All modules should call this once and share the logger by name.
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    
    if not log_filename:
        log_filename = "dapt_convergence.log" if "dapt" in name.lower() else "corpus_building.log"
        
    log_file = log_dir / log_filename

    numeric_level = getattr(logging, level.upper(), logging.INFO)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    logger = logging.getLogger(name)
    logger.setLevel(numeric_level)
    logger.propagate = False

    # Avoid duplicate handlers on re-import
    if logger.handlers:
        return logger

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(numeric_level)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # File handler
    fh = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    fh.setLevel(numeric_level)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger


def get_logger(name: str = "dapt.convergence") -> logging.Logger:
    """Retrieve an already-configured logger by name."""
    return logging.getLogger(name)


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
    term_cov: float,
    ret_prec: float,
    qa_gate: bool,
    ppl_gate: bool,
    secondary_gate: bool,
    qa_threshold: float,
    ppl_threshold: float,
    ppl_window: int,
    term_threshold: float,
    ret_threshold: float,
    decision: str,
    run_qa: bool = True,
    run_perplexity: bool = True,
    run_terminology: bool = True,
    run_retrieval: bool = True,
    qa_history: list = None,
    term_history: list = None,
    ret_history: list = None,
) -> str:
    """Produce a human-readable gate status block for logging."""
    tick = lambda b, active=True: "[PASS]" if b else ("[SKIP]" if not active else "[FAIL]")
    ppl_str = ", ".join(f"{p:.2f}%" for p in ppl_improvements) if ppl_improvements else "n/a (need >=2 evals)"
    ppl_vals = ", ".join(f"{p:.3f}" for p in ppl_history)
    qa_vals = ", ".join(f"{q:.4f}" for q in qa_history) if qa_history else "n/a"
    term_vals = ", ".join(f"{t:.4f}" for t in term_history) if term_history else "n/a"
    ret_vals = ", ".join(f"{r:.4f}" for r in ret_history) if ret_history else "n/a"

    return (
        f"\n{'-'*62}\n"
        f"  Gate Status @ Eval #{eval_id:>3}\n"
        f"{'-'*62}\n"
        f"  Tokens processed : {tokens_processed/1e9:.2f}B\n"
        f"  Corpus pass      : {tokens_processed/total_corpus_tokens:.3f}x\n"
        f"\n"
        f"  PRIMARY GATES (both required for convergence)\n"
        f"  {tick(qa_gate, run_qa)} QA Accuracy   : {qa_acc:.4f}  (threshold >= {qa_threshold})\n"
        f"  QA history       : [{qa_vals}]\n"
        f"  {tick(ppl_gate, run_perplexity)} PPL Plateau   : improvements = [{ppl_str}]  (need <{ppl_threshold}% for {ppl_window} consecutive)\n"
        f"  PPL history      : [{ppl_vals}]\n"
        f"\n"
        f"  SECONDARY GATES (at least one required)\n"
        f"  {tick(term_cov >= term_threshold, run_terminology)} Term Coverage : {term_cov:.4f}  (threshold >= {term_threshold})\n"
        f"  Term history     : [{term_vals}]\n"
        f"  {tick(ret_prec >= ret_threshold, run_retrieval)} Ret Precision : {ret_prec:.4f}  (threshold >= {ret_threshold})\n"
        f"  Ret history      : [{ret_vals}]\n"
        f"\n"
        f"  DECISION -> {decision}\n"
        f"{'-'*62}\n"
    )
