import os
import re
import json
import logging
import copy
from typing import Optional, Dict, Any, List, Tuple
from pathlib import Path

from lib.utils import PipelineConfig
from lib.utils.teacher_backend import LocalHFBackend, APIBackend, BedrockBackend

logger = logging.getLogger(__name__)


def make_judge_backend(cfg: PipelineConfig):
    """Instantiate the teacher backend using the independently configured judge parameters."""
    
    # Create a deep copy of configuration to avoid modifying original rad config
    judge_cfg = copy.deepcopy(cfg)
    
    # Override rad teacher config with benchmarking judge config parameters
    judge_cfg.rad.teacher_model_name = cfg.benchmarking.judge_model_name
    judge_cfg.rad.teacher_backend = cfg.benchmarking.judge_backend
    
    if cfg.benchmarking.judge_api_url:
        judge_cfg.rad.teacher_api_url = cfg.benchmarking.judge_api_url
    if cfg.benchmarking.judge_api_key:
        judge_cfg.rad.teacher_api_key = cfg.benchmarking.judge_api_key
        
    judge_cfg.rad.teacher_max_new_tokens = cfg.benchmarking.judge_max_new_tokens
    
    backend_type = cfg.benchmarking.judge_backend
    if backend_type == "hf_local":
        return LocalHFBackend(judge_cfg)
    elif backend_type == "api":
        return APIBackend(judge_cfg)
    elif backend_type == "bedrock":
        return BedrockBackend(judge_cfg)
    else:
        raise ValueError(f"Unknown judge backend: {backend_type}")


class ReasoningJudge:
    def __init__(self, cfg: PipelineConfig, backend: Optional[Any] = None):
        self.cfg = cfg
        self.backend = backend or make_judge_backend(cfg)
        
        # Track calibration counts per cluster
        self.calibration_counts: Dict[str, int] = {}
        
    def _parse_response(self, response: str) -> Optional[Dict[str, float]]:
        """Extract and parse the expected 3-key JSON dictionary from judge response."""
        try:
            start_idx = response.find("{")
            end_idx = response.rfind("}")
            if start_idx != -1 and end_idx != -1:
                json_str = response[start_idx:end_idx+1]
                data = json.loads(json_str)
                keys = ["step_validity", "logical_coherence", "absence_of_circular_reasoning"]
                if all(k in data for k in keys):
                    return {k: float(data[k]) for k in keys}
        except Exception as e:
            logger.debug(f"JSON parsing error: {e} on response: {response}")
        return None

    def build_judge_prompt(self, question: str, retrieved_context: str, trace: str) -> str:
        """Formulate the rubric prompt for reasoning quality."""
        context_str = retrieved_context if retrieved_context.strip() else "[NO CONTEXT AVAILABLE]"
        return (
            "[SYSTEM]: You are an expert neuroscience reviewer evaluating reasoning quality.\n"
            "Rate the following reasoning trace on three dimensions, each from 1 to 5.\n\n"
            f"[QUESTION]: {question}\n"
            f"[CONTEXT]: {context_str}\n"
            f"[TRACE]: {trace}\n\n"
            "Scoring rubric:\n"
            "- step_validity (1-5): Are individual reasoning steps factually and logically sound?\n"
            "- logical_coherence (1-5): Do steps follow from each other without gaps or leaps?\n"
            "- absence_of_circular_reasoning (1-5): Is circular reasoning absent? (5=fully absent, 1=pervasive)\n\n"
            'Respond ONLY with valid JSON: {"step_validity": N, "logical_coherence": N, "absence_of_circular_reasoning": N}'
        )

    def score_reasoning_quality(
        self, 
        sample_id: str,
        cluster_label: str,
        question: str, 
        retrieved_context: str, 
        trace: str
    ) -> Optional[Tuple[float, Dict[str, float]]]:
        """
        Calls the judge backend, parses and normalizes the score.
        Supports single retry on parse failure.
        """
        prompt = self.build_judge_prompt(question, retrieved_context, trace)
        
        # First attempt
        res_list = self.backend.generate_batch([prompt])
        response = res_list[0] if res_list else ""
        scores = self._parse_response(response)
        
        # Second attempt (retry) if parsing failed
        if scores is None:
            logger.warning(f"Judge parse failed on first attempt for sample {sample_id}. Retrying...")
            res_list = self.backend.generate_batch([prompt])
            response = res_list[0] if res_list else ""
            scores = self._parse_response(response)
            
        if scores is None:
            logger.error(f"Judge failed to return valid JSON after retry for sample {sample_id}.")
            return None
            
        # Normalize: average of three scores, then scale from [1, 5] to [0, 1]
        # Wait, the spec says: "Average the three scores and normalize to [0, 1] by dividing by 5."
        # Wait, if scores are in [1, 5], dividing by 5 gives [0.2, 1.0].
        # Let's check spec formula: "reasoning_quality = mean(step_validity, logical_coherence, absence_of_circular_reasoning) / 5"
        # We will follow the spec formula exactly!
        avg_score = sum(scores.values()) / len(scores)
        normalized_score = avg_score / 5.0
        
        # Handle calibration logging if enabled
        if self.cfg.benchmarking.enable_calibration:
            cal_size = self.cfg.benchmarking.human_calibration_size
            current_count = self.calibration_counts.get(cluster_label, 0)
            if current_count < cal_size:
                self.calibration_counts[cluster_label] = current_count + 1
                self.log_calibration_record(
                    sample_id=sample_id,
                    cluster_label=cluster_label,
                    question=question,
                    retrieved_context=retrieved_context,
                    trace=trace,
                    scores=scores
                )
                
        return normalized_score, scores

    def log_calibration_record(
        self,
        sample_id: str,
        cluster_label: str,
        question: str,
        retrieved_context: str,
        trace: str,
        scores: Dict[str, float]
    ) -> None:
        """Appends a calibration trace record to the calibration log file."""
        log_path = Path(self.cfg.benchmarking.calibration_log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        record = {
            "sample_id": sample_id,
            "cluster_label": cluster_label,
            "question": question,
            "retrieved_context": retrieved_context,
            "teacher_trace": trace,
            "step_validity": scores["step_validity"],
            "logical_coherence": scores["logical_coherence"],
            "absence_of_circular_reasoning": scores["absence_of_circular_reasoning"]
        }
        
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")


def run_cohen_kappa_evaluation(cfg: PipelineConfig) -> None:
    """Computes Cohen's kappa if human labels are provided and writes to log file."""
    labels_path = cfg.benchmarking.human_labels_path
    if not labels_path or not Path(labels_path).exists():
        logger.info("Human labels path not set or file does not exist. Skipping Cohen's kappa evaluation.")
        return
        
    cal_log_path = Path(cfg.benchmarking.calibration_log_path)
    if not cal_log_path.exists():
        logger.warning(f"Calibration log file {cal_log_path} not found. Cannot compute Cohen's kappa.")
        return
        
    # Load LLM ratings
    llm_ratings: Dict[str, Dict[str, int]] = {}
    with open(cal_log_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    record = json.loads(line)
                    s_id = record.get("sample_id")
                    if s_id:
                        llm_ratings[s_id] = {
                            "step_validity": int(round(record["step_validity"])),
                            "logical_coherence": int(round(record["logical_coherence"])),
                            "absence_of_circular_reasoning": int(round(record["absence_of_circular_reasoning"]))
                        }
                except Exception as e:
                    logger.error(f"Error parsing calibration log record: {e}")
                    
    # Load Human ratings (expecting list of objects or line-delimited JSON)
    human_ratings: Dict[str, Dict[str, int]] = {}
    try:
        with open(labels_path, "r", encoding="utf-8") as f:
            content = f.read().strip()
            if content.startswith("["):
                # Array of JSON objects
                records = json.loads(content)
                for record in records:
                    s_id = record.get("sample_id")
                    if s_id:
                        human_ratings[s_id] = {
                            "step_validity": int(round(record["step_validity"])),
                            "logical_coherence": int(round(record["logical_coherence"])),
                            "absence_of_circular_reasoning": int(round(record["absence_of_circular_reasoning"]))
                        }
            else:
                # JSONL
                for line in content.split("\n"):
                    if line.strip():
                        record = json.loads(line)
                        s_id = record.get("sample_id")
                        if s_id:
                            human_ratings[s_id] = {
                                "step_validity": int(round(record["step_validity"])),
                                "logical_coherence": int(round(record["logical_coherence"])),
                                "absence_of_circular_reasoning": int(round(record["absence_of_circular_reasoning"]))
                            }
    except Exception as e:
        logger.error(f"Error loading human labels from {labels_path}: {e}")
        return
        
    # Match ratings
    common_ids = set(llm_ratings.keys()) & set(human_ratings.keys())
    if not common_ids:
        logger.warning("No matching sample IDs between human labels and calibration log.")
        return
        
    from sklearn.metrics import cohen_kappa_score
    
    y_llm = {k: [] for k in ["step_validity", "logical_coherence", "absence_of_circular_reasoning"]}
    y_human = {k: [] for k in ["step_validity", "logical_coherence", "absence_of_circular_reasoning"]}
    
    for s_id in sorted(common_ids):
        for k in y_llm.keys():
            y_llm[k].append(llm_ratings[s_id][k])
            y_human[k].append(human_ratings[s_id][k])
            
    kappa_results = {}
    for k in y_llm.keys():
        try:
            kappa = cohen_kappa_score(y_llm[k], y_human[k])
            kappa_results[f"cohen_kappa_{k}"] = float(kappa)
        except Exception as e:
            logger.error(f"Error computing Cohen's kappa for {k}: {e}")
            kappa_results[f"cohen_kappa_{k}"] = None
            
    kappa_results["sample_count"] = len(common_ids)
    
    out_path = Path(cfg.benchmarking.inter_rater_log_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(kappa_results, f, indent=2)
    logger.info(f"Cohen's kappa results written to {out_path}.")
