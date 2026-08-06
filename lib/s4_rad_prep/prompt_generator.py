import json
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from lib.utils import PipelineConfig
from lib.s4_rad_prep.chunker import Chunk

logger = logging.getLogger(__name__)


@dataclass
class PromptRecord:
    sample_id: str
    cluster_id: Optional[str]
    question: str
    answer: str
    retrieved_context: str
    no_retrieval: bool
    embedding_model: str


def format_prompt(question: str, answer: str, context_chunks: List[Chunk], no_retrieval: bool) -> str:
    """Format prompts with retrieved context structure."""
    if no_retrieval:
        return (
            "[SYSTEM]: You are a neuroscientist. Reason step-by-step using only your knowledge. "
            "Wrap your final answer inside \\boxed{}.\n\n"
            "[NO CONTEXT AVAILABLE]\n"
            f"[QUESTION]: {question}\n"
            f"[GROUND TRUTH]: {answer}"
        )
    else:
        context_text = "\n\n".join(chunk.text for chunk in context_chunks)
        return (
            "[SYSTEM]: You are a neuroscientist. Reason step-by-step using the provided context. "
            "Annotate key statements with bracketed passage citations matching the provided context (e.g. [Context 1] or [Passage 1]). "
            "Wrap your final answer inside \\boxed{}.\n\n"
            f"[CONTEXT]: {context_text}\n"
            f"[QUESTION]: {question}\n"
            f"[GROUND TRUTH]: {answer}"
        )


class PromptGenerator:
    """Generates grounded QA prompt records (with retrieved context) for downstream teacher benchmarking."""
    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg
        self.traces_dir = Path(cfg.rad.traces_dir)
        self.traces_dir.mkdir(parents=True, exist_ok=True)

    def generate_prompts(self, samples: List[Dict[str, Any]], retrieved_results: List[Any], no_retrieval_router: Any) -> Dict[str, int]:
        """Prepare grounded prompt records in batches, writing incrementally."""
        batch_size = self.cfg.rad.teacher_batch_size
        grounded_path = self.traces_dir / "grounded_traces.jsonl"
        no_retrieval_path = self.traces_dir / "no_retrieval_traces.jsonl"

        grounded_count = 0
        no_retrieval_count = 0

        logger.info(f"Preparing grounded QA prompt records for {len(samples)} samples in batches of {batch_size}")

        with open(grounded_path, "a", encoding="utf-8") as f_grounded, \
             open(no_retrieval_path, "a", encoding="utf-8") as f_no_ret:

            for i in range(0, len(samples), batch_size):
                batch_samples = samples[i:i + batch_size]
                batch_retrieved = retrieved_results[i:i + batch_size]

                for idx_in_batch, (sample, ret_res) in enumerate(zip(batch_samples, batch_retrieved)):
                    global_idx = i + idx_in_batch
                    question = sample["question"]

                    if "choices" in sample and "answer_idx" in sample:
                        choices = sample["choices"]
                        ans_idx = sample["answer_idx"]
                        answer = choices[ans_idx]
                    else:
                        answer = sample.get("answer", "")

                    sample_id = sample.get("sample_id", f"sample_{global_idx}")
                    cluster_id = sample.get("cluster", sample.get("cluster_id"))

                    decision = no_retrieval_router.route_sample(sample_id, cluster_id, ret_res.passed_threshold)

                    record_dict = {
                        "sample_id": sample_id,
                        "cluster_id": cluster_id,
                        "question": question,
                        "answer": answer,
                        "retrieved_context": "\n\n".join(chunk.text for chunk in ret_res.chunks),
                        "no_retrieval": decision.no_retrieval,
                        "embedding_model": self.cfg.rad.embedding_model
                    }

                    if decision.no_retrieval:
                        f_no_ret.write(json.dumps(record_dict) + "\n")
                        no_retrieval_count += 1
                    else:
                        f_grounded.write(json.dumps(record_dict) + "\n")
                        grounded_count += 1

                no_retrieval_router.flush_batch()

        return {
            "grounded_count": grounded_count,
            "no_retrieval_count": no_retrieval_count
        }
