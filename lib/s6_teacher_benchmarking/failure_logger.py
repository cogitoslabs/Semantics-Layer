"""
failure_logger.py — Dedicated logging module for Step 6 benchmarking metric failures.
Logs failing evaluation samples to metric-specific JSON files in logs/benchmarking/failures/<metricname>.json.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional

from lib.utils import PipelineConfig

logger = logging.getLogger(__name__)


class BenchmarkFailureLogger:
    """
    Logs failure cases in Step 6 Teacher Benchmarking for detailed inspection.
    Saves outputs into logs/benchmarking/failures/<metricname>.json.
    """

    def __init__(self, cfg: PipelineConfig, reset: bool = True):
        self.cfg = cfg
        self.log_dir = Path(cfg.logging.log_dir) / "benchmarking" / "failures"
        self.log_dir.mkdir(parents=True, exist_ok=True)

        if reset:
            self._reset_failure_logs()

    def _reset_failure_logs(self) -> None:
        """Removes existing failure log JSON files at the beginning of a run."""
        for metric in ["answer_accuracy", "reasoning_quality", "citation_accuracy", "hallucination_rate"]:
            f_path = self.log_dir / f"{metric}.json"
            if f_path.exists():
                try:
                    f_path.unlink()
                except Exception as e:
                    logger.warning(f"Could not remove old failure log file {f_path}: {e}")

    def log_failure(self, metric_name: str, record: Dict[str, Any]) -> None:
        """Appends a failure record to logs/benchmarking/failures/<metricname>.json as a formatted JSON list."""
        out_path = self.log_dir / f"{metric_name}.json"

        records = []
        if out_path.exists():
            try:
                with open(out_path, "r", encoding="utf-8") as f:
                    records = json.load(f)
                    if not isinstance(records, list):
                        records = []
            except Exception as e:
                logger.warning(f"Error reading existing failure log {out_path}: {e}")
                records = []

        records.append(record)

        try:
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(records, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Error writing failure record to {out_path}: {e}")

    def check_and_log_failures(
        self,
        seq_num: int,
        teacher: str,
        sample_id: str,
        cluster_label: str,
        question: str,
        ground_truth: str,
        retrieved_context: str,
        prompt: str,
        response: str,
        answer_accuracy: float,
        reasoning_quality: Optional[float],
        citation_accuracy: Optional[float],
        hallucination_rate: float,
        no_retrieval: bool,
        judge_prompt: Optional[str] = None,
        judge_response: Optional[str] = None,
        explanations: Optional[Dict[str, str]] = None,
    ) -> None:
        """
        Evaluates metric scores against failure criteria and logs failure records to respective metric JSON files.
        """
        base_record = {
            "seq #": seq_num,
            "seq_num": seq_num,
            "teacher": teacher,
            "sample_id": sample_id,
            "cluster_label": cluster_label,
            "question": question,
            "ground_truth": ground_truth,
            "retrieved_context": retrieved_context if retrieved_context else "[NO CONTEXT AVAILABLE]",
            "prompt": prompt,
            "response": response,
            "no_retrieval": no_retrieval,
        }
        explanations = explanations or {}

        # 1. Answer Accuracy Failure
        if answer_accuracy < 1.0:
            rec = dict(base_record)
            rec["metric_name"] = "answer_accuracy"
            rec["metric_score"] = answer_accuracy
            rec["explanation"] = explanations.get("answer_explanation", "")
            rec["failure_reason"] = f"Answer accuracy score {answer_accuracy:.4f} is below 1.0"
            self.log_failure("answer_accuracy", rec)

        # 2. Reasoning Quality Failure (Judge parse failure or score < 0.60)
        if reasoning_quality is None or reasoning_quality < 0.60:
            rec = dict(base_record)
            rec["metric_name"] = "reasoning_quality"
            rec["metric_score"] = reasoning_quality
            rec["judge_prompt"] = judge_prompt if judge_prompt else ""
            rec["judge_response"] = judge_response if judge_response else ""
            rec["explanation"] = explanations.get("reasoning_explanation", "")
            if reasoning_quality is None:
                rec["failure_reason"] = "Judge failed to return valid JSON response after retry"
            else:
                rec["failure_reason"] = f"Reasoning quality score {reasoning_quality:.4f} is below threshold 0.60"
            self.log_failure("reasoning_quality", rec)

        # 3. Citation Accuracy Failure (for grounded traces with citation accuracy < 0.30)
        if not no_retrieval and (citation_accuracy is None or citation_accuracy < 0.30):
            rec = dict(base_record)
            rec["metric_name"] = "citation_accuracy"
            rec["metric_score"] = citation_accuracy
            rec["explanation"] = explanations.get("citation_explanation", "")
            rec["failure_reason"] = (
                "Citation accuracy is None" if citation_accuracy is None
                else f"Citation accuracy score {citation_accuracy:.4f} is below threshold 0.30"
            )
            self.log_failure("citation_accuracy", rec)

        # 4. Hallucination Rate Failure (exceeds threshold e.g. 0.25)
        hallucination_threshold = self.cfg.benchmarking.hallucination_threshold
        if hallucination_rate > hallucination_threshold:
            rec = dict(base_record)
            rec["metric_name"] = "hallucination_rate"
            rec["metric_score"] = hallucination_rate
            rec["threshold"] = hallucination_threshold
            rec["explanation"] = explanations.get("hallucination_explanation", "")
            rec["failure_reason"] = (
                f"Hallucination rate {hallucination_rate:.4f} exceeds threshold {hallucination_threshold:.2f}"
            )
            self.log_failure("hallucination_rate", rec)
