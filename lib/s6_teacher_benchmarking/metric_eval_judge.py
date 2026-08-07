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
    judge_cfg = copy.deepcopy(cfg)
    
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


class MetricEvalJudge:
    """
    LLM-as-a-Judge evaluation engine.
    Evaluates candidate Teacher LLM response traces across 4 core metrics:
    1. Answer Accuracy
    2. Reasoning Quality
    3. Citation Accuracy
    4. Hallucination Detection

    Provides quantitative numerical scores and qualitative explanations for every metric.
    """
    def __init__(self, cfg: PipelineConfig, backend: Optional[Any] = None):
        self.cfg = cfg
        self.backend = backend or make_judge_backend(cfg)
        
        self.calibration_counts: Dict[str, int] = {}
        self.last_prompt: Optional[str] = None
        self.last_response: Optional[str] = None

    def build_judge_prompt(
        self,
        question: str,
        ground_truth: str,
        retrieved_context: str,
        trace: str,
        no_retrieval: bool
    ) -> str:
        """Formulate the comprehensive multi-metric evaluation rubric prompt."""
        context_str = (
            "[NO CONTEXT AVAILABLE]"
            if (no_retrieval or not retrieved_context.strip())
            else retrieved_context.strip()
        )
        gt_str = ground_truth.strip() if ground_truth else "[NOT PROVIDED]"

        return (
            "[SYSTEM]: You are an expert scientific and neuroscience reviewer evaluating an AI model response trace.\n"
            "Evaluate the candidate model response across four evaluation dimensions based on the provided Question, Ground Truth, and Context Passages (if available).\n\n"
            f"[QUESTION]: {question}\n"
            f"[GROUND TRUTH]: {gt_str}\n"
            f"[CONTEXT PASSAGES]:\n{context_str}\n\n"
            f"[MODEL RESPONSE TRACE]:\n{trace}\n\n"
            "EVALUATION RUBRIC & INSTRUCTIONS:\n"
            "1. answer_accuracy:\n"
            "   - score (float 0.0 to 1.0): 1.0 if the final answer is completely correct and matches ground truth; 0.0 if completely incorrect; partial credit (e.g. 0.5) for partially correct responses.\n"
            "   - explanation (string): Concise brief explanation of answer correctness.\n\n"
            "2. reasoning_quality:\n"
            "   - step_validity (int 1 to 5): Factual and logical soundness of individual reasoning steps.\n"
            "   - logical_coherence (int 1 to 5): Smooth progression without gaps or leaps in logic.\n"
            "   - absence_of_circular_reasoning (int 1 to 5): Freedom from circular logic (5=fully absent, 1=pervasive).\n"
            "   - explanation (string): Concise brief explanation of reasoning quality.\n\n"
            "3. citation_accuracy:\n"
            "   - precision (float 0.0 to 1.0): Proportion of citations/references in trace that accurately reflect context.\n"
            "   - recall (float 0.0 to 1.0): Proportion of key context facts used that are properly cited.\n"
            "   - accuracy (float 0.0 to 1.0): Overall citation accuracy (harmonic mean F1 of precision & recall).\n"
            "   - explanation (string): Concise brief explanation of citation accuracy. (For NO CONTEXT items, set precision, recall, accuracy to 1.0 and explanation to 'No context provided').\n\n"
            "4. hallucination:\n"
            "   - rate (float 0.0 to 1.0): Proportion/severity of hallucinated or ungrounded claims (0.0=no hallucinations, 1.0=completely hallucinated).\n"
            "   - explanation (string): Concise brief explanation of detected hallucinations or confirmation of factual accuracy.\n\n"
            "CRITICAL: Do NOT output any thinking tags (<think>), reasoning text, or markdown formatting.\n"
            "Output ONLY raw valid JSON starting with { and ending with } using the exact schema below:\n"
            "{\n"
            '  "answer_accuracy": {"score": 1.0, "explanation": "..."},\n'
            '  "reasoning_quality": {"step_validity": 5, "logical_coherence": 5, "absence_of_circular_reasoning": 5, "explanation": "..."},\n'
            '  "citation_accuracy": {"precision": 1.0, "recall": 1.0, "accuracy": 1.0, "explanation": "..."},\n'
            '  "hallucination": {"rate": 0.0, "explanation": "..."}\n'
            "}"
        )

    def _parse_response(self, response: str) -> Optional[Dict[str, Any]]:
        """Extract and parse the multi-metric JSON dictionary from judge response."""
        if not response or not response.strip():
            return None

        # 1. Strip thinking tags (e.g. <think>...</think>)
        cleaned = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL)

        # 2. Strip markdown code block wrappers
        cleaned = re.sub(r"```(?:json)?", "", cleaned).strip()

        # 3. Find JSON candidates
        json_candidates = re.findall(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", cleaned, flags=re.DOTALL)

        if not json_candidates:
            start_idx = cleaned.find("{")
            end_idx = cleaned.rfind("}")
            if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                json_candidates = [cleaned[start_idx:end_idx+1]]

        for candidate in json_candidates:
            try:
                data = json.loads(candidate)
                if isinstance(data, dict):
                    # Check for metric keys
                    if any(k in data for k in ["answer_accuracy", "reasoning_quality", "citation_accuracy", "hallucination"]):
                        return data
            except (json.JSONDecodeError, ValueError, TypeError) as e:
                logger.debug(f"JSON candidate parsing failed: {e}")
                continue

        logger.warning(f"Judge JSON parse failed. Could not extract valid multi-metric JSON object from response: {response!r}")
        return None

    def evaluate_trace(
        self,
        sample_id: str,
        cluster_label: str,
        question: str,
        ground_truth: str,
        retrieved_context: str,
        trace: str,
        no_retrieval: bool
    ) -> Optional[Dict[str, Any]]:
        """
        Calls judge backend, parses response, and derives metric scores + explanations.
        Supports single retry on parse failure.
        """
        prompt = self.build_judge_prompt(question, ground_truth, retrieved_context, trace, no_retrieval)
        self.last_prompt = prompt

        # Attempt 1
        res_list = self.backend.generate_batch([prompt])
        response = res_list[0] if res_list else ""
        self.last_response = response
        parsed = self._parse_response(response)

        # Attempt 2 (retry on parse failure)
        if parsed is None:
            logger.warning(f"Judge parse failed on attempt 1 for sample {sample_id}. Retrying...")
            res_list = self.backend.generate_batch([prompt])
            response = res_list[0] if res_list else ""
            self.last_response = response
            parsed = self._parse_response(response)

        if parsed is None:
            logger.error(f"Judge failed to return valid JSON after retry for sample {sample_id}. Response: {response!r}")
            return None

        # Extract metric sub-dicts safely
        ans_data = parsed.get("answer_accuracy", {})
        if isinstance(ans_data, (int, float)):
            ans_data = {"score": float(ans_data), "explanation": "Parsed numeric score."}
            
        rq_data = parsed.get("reasoning_quality", {})
        if isinstance(rq_data, (int, float)):
            rq_data = {"step_validity": float(rq_data)*5, "logical_coherence": float(rq_data)*5, "absence_of_circular_reasoning": float(rq_data)*5, "explanation": "Parsed numeric score."}

        cit_data = parsed.get("citation_accuracy", {})
        if isinstance(cit_data, (int, float)):
            cit_data = {"precision": float(cit_data), "recall": float(cit_data), "accuracy": float(cit_data), "explanation": "Parsed numeric score."}

        hal_data = parsed.get("hallucination", {})
        if isinstance(hal_data, (int, float)):
            hal_data = {"rate": float(hal_data), "explanation": "Parsed numeric score."}

        # Derive answer accuracy score
        answer_accuracy = float(ans_data.get("score", 0.0))
        answer_explanation = str(ans_data.get("explanation", ""))

        # Derive reasoning quality score (average of 3 scores / 5.0)
        step_val = float(rq_data.get("step_validity", 3.0))
        log_coh = float(rq_data.get("logical_coherence", 3.0))
        abs_circ = float(rq_data.get("absence_of_circular_reasoning", 3.0))
        reasoning_quality = (step_val + log_coh + abs_circ) / 15.0
        reasoning_explanation = str(rq_data.get("explanation", ""))

        # Derive citation accuracy scores
        if no_retrieval:
            cit_prec = None
            cit_rec = None
            cit_acc = None
            citation_explanation = "No context provided."
        else:
            cit_prec = float(cit_data.get("precision", 1.0))
            cit_rec = float(cit_data.get("recall", 1.0))
            cit_acc = float(cit_data.get("accuracy", (cit_prec + cit_rec) / 2.0 if cit_prec is not None and cit_rec is not None else 1.0))
            citation_explanation = str(cit_data.get("explanation", ""))

        # Derive hallucination rate
        hallucination_rate = float(hal_data.get("rate", 0.0))
        hallucination_explanation = str(hal_data.get("explanation", ""))

        # Calibration logging if enabled
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
                    step_validity=step_val,
                    logical_coherence=log_coh,
                    absence_of_circular_reasoning=abs_circ
                )

        return {
            "answer_accuracy": answer_accuracy,
            "answer_explanation": answer_explanation,
            "reasoning_quality": reasoning_quality,
            "reasoning_explanation": reasoning_explanation,
            "step_validity": step_val,
            "logical_coherence": log_coh,
            "absence_of_circular_reasoning": abs_circ,
            "citation_precision": cit_prec,
            "citation_recall": cit_rec,
            "citation_accuracy": cit_acc,
            "citation_explanation": citation_explanation,
            "hallucination_rate": hallucination_rate,
            "hallucination_explanation": hallucination_explanation,
            "judge_prompt": prompt,
            "judge_response": response,
        }

    def log_calibration_record(
        self,
        sample_id: str,
        cluster_label: str,
        question: str,
        retrieved_context: str,
        trace: str,
        step_validity: float,
        logical_coherence: float,
        absence_of_circular_reasoning: float
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
            "step_validity": step_validity,
            "logical_coherence": logical_coherence,
            "absence_of_circular_reasoning": absence_of_circular_reasoning
        }
        
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")


# Alias for backward compatibility
ReasoningJudge = MetricEvalJudge


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
                    
    human_ratings: Dict[str, Dict[str, int]] = {}
    try:
        with open(labels_path, "r", encoding="utf-8") as f:
            content = f.read().strip()
            if content.startswith("["):
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
