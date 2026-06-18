"""
evaluation/gate_logic.py — Step 0.3 Convergence Gate Logic

Implements the three-gate convergence check (Primary A, Primary B, Secondary),
the hard-cap handler, and the remediation routing logic from v4.1.
"""

from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from lib.utils.logger import get_logger, save_json, format_gate_status
from lib.s2_dapt.probes.perplexity_probe import compute_ppl_improvement, check_ppl_plateau

logger = get_logger("dapt.gates")


# ── Decision enum ─────────────────────────────────────────────────────────────

class DAPTDecision(str, Enum):
    CONVERGED = "CONVERGED"
    CONTINUE  = "CONTINUE"
    HARD_CAP  = "HARD_CAP"


# ── Gate evaluation ───────────────────────────────────────────────────────────

def check_convergence_gates(
    state: Dict[str, Any],
    qa_acc_threshold: float,
    ppl_improvement_threshold: float,
    ppl_plateau_window: int,
    term_cov_threshold: float,
    ret_prec_threshold: float,
    hard_stop_tokens: int,
    total_corpus_tokens: int,
) -> Tuple[DAPTDecision, Dict[str, Any]]:
    """
    Evaluate all convergence gates and return the decision plus a gate detail dict.

    Returns
    -------
    decision : DAPTDecision
    gate_details : dict with individual gate pass/fail and supporting metrics
    """
    ppl_history = state["perplexity_history"]
    qa_acc      = state["qa_acc_history"][-1]  if state["qa_acc_history"]  else 0.0
    term_cov    = state["term_cov_history"][-1] if state["term_cov_history"] else 0.0
    ret_prec    = state["ret_prec_history"][-1] if state["ret_prec_history"] else 0.0

    # ── Primary Gate A: QA Accuracy ──────────────────────────────────────────
    qa_gate = qa_acc >= qa_acc_threshold

    # ── Primary Gate B: PPL Plateau ──────────────────────────────────────────
    ppl_plateau, ppl_improvements = check_ppl_plateau(
        ppl_history=ppl_history,
        improvement_threshold=ppl_improvement_threshold,
        window=ppl_plateau_window,
    )
    ppl_gate = ppl_plateau

    # ── Secondary Gate (at least one) ────────────────────────────────────────
    term_gate = term_cov >= term_cov_threshold
    ret_gate  = ret_prec >= ret_prec_threshold
    secondary_gate = term_gate or ret_gate

    all_primary   = qa_gate and ppl_gate
    all_converged = all_primary and secondary_gate

    gate_details = {
        "qa_gate"          : qa_gate,
        "qa_acc"           : qa_acc,
        "ppl_gate"         : ppl_gate,
        "ppl_improvements" : ppl_improvements,
        "term_gate"        : term_gate,
        "term_cov"         : term_cov,
        "ret_gate"         : ret_gate,
        "ret_prec"         : ret_prec,
        "secondary_gate"   : secondary_gate,
        "all_converged"    : all_converged,
    }

    # ── Hard Cap Check ────────────────────────────────────────────────────────
    if state["tokens_processed"] >= hard_stop_tokens:
        if all_converged:
            return DAPTDecision.CONVERGED, gate_details
        else:
            return DAPTDecision.HARD_CAP, gate_details

    # ── Normal Convergence ────────────────────────────────────────────────────
    if all_converged:
        return DAPTDecision.CONVERGED, gate_details

    return DAPTDecision.CONTINUE, gate_details


# ── Gate status logging ───────────────────────────────────────────────────────

def log_gate_status(
    state: Dict[str, Any],
    gate_details: Dict[str, Any],
    decision: DAPTDecision,
    qa_threshold: float,
    ppl_threshold: float,
    ppl_window: int,
    term_threshold: float,
    ret_threshold: float,
    total_corpus_tokens: int,
) -> None:
    msg = format_gate_status(
        eval_id               = state["eval_count"],
        tokens_processed      = state["tokens_processed"],
        total_corpus_tokens   = total_corpus_tokens,
        qa_acc                = gate_details["qa_acc"],
        ppl_history           = state["perplexity_history"],
        ppl_improvements      = gate_details["ppl_improvements"],
        term_cov              = gate_details["term_cov"],
        ret_prec              = gate_details["ret_prec"],
        qa_gate               = gate_details["qa_gate"],
        ppl_gate              = gate_details["ppl_gate"],
        secondary_gate        = gate_details["secondary_gate"],
        qa_threshold          = qa_threshold,
        ppl_threshold         = ppl_threshold,
        ppl_window            = ppl_window,
        term_threshold        = term_threshold,
        ret_threshold         = ret_threshold,
        decision              = decision.value,
    )
    logger.info(msg)


# ── Remediation handler ───────────────────────────────────────────────────────

def handle_hard_cap(
    state: Dict[str, Any],
    gate_details: Dict[str, Any],
    last_checkpoint_path: Path,
    risk_report_path: Path,
    qa_acc_threshold: float,
    qa_low_threshold: float,
    ppl_improvement_threshold: float,
    term_cov_threshold: float,
    ret_prec_threshold: float,
    total_corpus_tokens: int,
) -> Tuple[Path, Dict[str, Any]]:
    """
    Called when the hard cap fires and gates are not all met.

    Determines the most likely root cause and recommends the prioritized
    remediation action from the v4.1 spec:
      1. Corpus quality audit
      2. Increase base model size
      3. Proceed with risk flag

    Writes a JSON risk report to disk.
    Returns (last_checkpoint_path, risk_report).
    """
    qa_acc   = gate_details["qa_acc"]
    term_cov = gate_details["term_cov"]
    ret_prec = gate_details["ret_prec"]
    ppl_hist = state["perplexity_history"]

    # ── Identify failed gates ─────────────────────────────────────────────────
    failed_gates: List[str] = []

    if not gate_details["qa_gate"]:
        gap = qa_acc_threshold - qa_acc
        failed_gates.append(
            f"QA accuracy {qa_acc:.4f} < threshold {qa_acc_threshold} (gap: {gap:.4f})"
        )

    if not gate_details["ppl_gate"]:
        if gate_details["ppl_improvements"]:
            last_imp = gate_details["ppl_improvements"][-1]
            failed_gates.append(
                f"PPL still improving at {last_imp:.2f}% (threshold: <{ppl_improvement_threshold}%)"
            )
        else:
            failed_gates.append("PPL plateau not reached (insufficient eval history)")

    if not gate_details["secondary_gate"]:
        failed_gates.append(
            f"Neither secondary gate met: "
            f"term_cov={term_cov:.4f} (need >={term_cov_threshold}), "
            f"ret_prec={ret_prec:.4f} (need >={ret_prec_threshold})"
        )

    # ── Remediation routing ───────────────────────────────────────────────────
    if qa_acc < qa_low_threshold:
        priority   = 1
        action     = "CORPUS_QUALITY_AUDIT"
        rationale  = (
            f"QA accuracy is very low ({qa_acc:.4f} < {qa_low_threshold}) after "
            f"{state['tokens_processed']/total_corpus_tokens:.1f} corpus passes. "
            "This strongly suggests insufficient high-signal content. "
            "Actions: (a) remove low-density boilerplate from corpus, "
            "(b) add more review articles and textbooks, "
            "(c) prefer PubMed Central full-text over abstracts-only."
        )
    elif qa_low_threshold <= qa_acc < qa_acc_threshold:
        priority   = 2
        action     = "INCREASE_MODEL_SIZE"
        rationale  = (
            f"QA accuracy ({qa_acc:.4f}) is borderline after full corpus exposure. "
            "The model has absorbed some domain signal but is capacity-limited. "
            "Action: move to the next model tier (e.g., 0.5B → 1B → 3B)."
        )
    else:
        priority   = 3
        action     = "PROCEED_WITH_RISK_FLAG"
        rationale  = (
            "QA accuracy is near threshold but perplexity plateau or secondary "
            "gates may not be fully cleared. Downstream phases may compensate. "
            "Action: proceed but document all metric gaps in the risk report "
            "and monitor Phase 1+ validation metrics closely."
        )

    risk_report = {
        "dapt_outcome"      : "HARD_CAP_NOT_CONVERGED",
        "last_checkpoint"   : str(last_checkpoint_path),
        "tokens_processed"  : state["tokens_processed"],
        "corpus_passes"     : state["tokens_processed"] / total_corpus_tokens,
        "final_metrics"     : {
            "qa_accuracy"       : qa_acc,
            "perplexity"        : ppl_hist[-1] if ppl_hist else None,
            "term_coverage"     : term_cov,
            "retrieval_precision": ret_prec,
        },
        "failed_gates"      : failed_gates,
        "remediation"       : {
            "priority"  : priority,
            "action"    : action,
            "rationale" : rationale,
        },
    }

    save_json(risk_report, risk_report_path)

    logger.warning(
        f"\n{'!'*60}\n"
        f"  HARD CAP reached — gates NOT fully met\n"
        f"  Failed gates: {failed_gates}\n"
        f"  Recommended action (Priority {priority}): {action}\n"
        f"  Rationale: {rationale}\n"
        f"  Risk report: {risk_report_path}\n"
        f"{'!'*60}\n"
    )

    return last_checkpoint_path, risk_report
