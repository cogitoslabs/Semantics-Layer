import json
import logging
import os
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor
import torch
import requests
from transformers import AutoTokenizer, AutoModelForCausalLM
from lib.utils import PipelineConfig
from lib.s3_rad_prep.chunker import Chunk

logger = logging.getLogger(__name__)

@dataclass
class TraceRecord:
    sample_id: str
    cluster_id: Optional[str]
    question: str
    answer: str
    retrieved_context: str
    no_retrieval: bool
    teacher_trace: str
    token_count: int
    teacher_model: str
    embedding_model: str


class TeacherModelBackend:
    def generate_batch(self, prompts: List[str]) -> List[str]:
        raise NotImplementedError


class LocalHFBackend(TeacherModelBackend):
    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg
        self.model_name = cfg.rad.teacher_model_name
        self.max_new_tokens = cfg.rad.teacher_max_new_tokens

        logger.info(f"Loading local teacher tokenizer and model: {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda":
            # Load in bfloat16/float16 if GPU is available to speed up and reduce footprint
            torch_dtype = torch.bfloat16 if cfg.model.model_dtype == "bfloat16" else torch.float16
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch_dtype,
                device_map="auto"
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
        self.model.eval()

    def generate_batch(self, prompts: List[str]) -> List[str]:
        inputs = self.tokenizer(prompts, padding=True, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,  # greedy decoding
                pad_token_id=self.tokenizer.pad_token_id
            )

        generated_texts = []
        for inp, out in zip(inputs["input_ids"], outputs):
            # Extract only the newly generated tokens
            gen_tokens = out[len(inp):]
            text = self.tokenizer.decode(gen_tokens, skip_special_tokens=True)
            generated_texts.append(text)
        return generated_texts


class APIBackend(TeacherModelBackend):
    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg
        self.model_name = cfg.rad.teacher_model_name
        self.api_url = cfg.rad.teacher_api_url
        self.api_key = cfg.rad.teacher_api_key or os.environ.get("RAD_TEACHER_API_KEY", "")
        self.max_new_tokens = cfg.rad.teacher_max_new_tokens

        if not self.api_url:
            raise ValueError("RAD_TEACHER_API_URL must be provided when using api teacher backend.")

        # Ensure correct completion endpoint format
        if not (self.api_url.endswith("/chat/completions") or self.api_url.endswith("/completions")):
            if self.api_url.endswith("/"):
                self.api_url += "chat/completions"
            else:
                self.api_url += "/v1/chat/completions"

        logger.info(f"API Backend configured for endpoint: {self.api_url} using model: {self.model_name}")

    def generate_batch(self, prompts: List[str]) -> List[str]:
        headers = {
            "Content-Type": "application/json"
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        def send_request(prompt: str) -> str:
            payload = {
                "model": self.model_name,
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": self.max_new_tokens,
                "temperature": 0.0,
            }
            try:
                response = requests.post(self.api_url, headers=headers, json=payload, timeout=60)
                response.raise_for_status()
                data = response.json()
                return data["choices"][0]["message"]["content"]
            except Exception as e:
                logger.error(f"Error calling teacher API endpoint: {e}")
                # Return empty string on error so we don't break the batch alignment, but it will be filtered out due to length
                return ""

        max_workers = min(len(prompts), 16)
        logger.info(f"Sending {len(prompts)} concurrent API requests to teacher backend with {max_workers} workers")
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(send_request, prompts))
        return results


class BedrockBackend(TeacherModelBackend):
    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg
        self.model_name = cfg.rad.teacher_model_name
        self.region_name = os.environ.get("AWS_DEFAULT_REGION", os.environ.get("AWS_REGION", "us-east-1"))
        self.max_new_tokens = cfg.rad.teacher_max_new_tokens

        import boto3
        logger.info(f"Initializing AWS Bedrock Runtime client in region {self.region_name}")
        self.client = boto3.client("bedrock-runtime", region_name=self.region_name)

    def generate_batch(self, prompts: List[str]) -> List[str]:
        def send_request(prompt: str) -> str:
            try:
                response = self.client.converse(
                    modelId=self.model_name,
                    messages=[
                        {
                            "role": "user",
                            "content": [{"text": prompt}]
                        }
                    ],
                    inferenceConfig={
                        "maxTokens": self.max_new_tokens,
                        "temperature": 0.0
                    }
                )
                return response["output"]["message"]["content"][0]["text"]
            except Exception as e:
                logger.error(f"Error calling AWS Bedrock model {self.model_name}: {e}")
                # Return empty string on error so we don't break the batch alignment, but it will be filtered out due to length
                return ""

        max_workers = min(len(prompts), 16)
        logger.info(f"Sending {len(prompts)} concurrent Bedrock requests with {max_workers} workers")
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(send_request, prompts))
        return results


def format_prompt(question: str, answer: str, context_chunks: List[Chunk], no_retrieval: bool) -> str:
    """Format prompts exactly as specified by the feature spec."""
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
            "Cite specific passages where applicable. Wrap your final answer inside \\boxed{}.\n\n"
            f"[CONTEXT]: {context_text}\n"
            f"[QUESTION]: {question}\n"
            f"[GROUND TRUTH]: {answer}"
        )


class TraceGenerator:
    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg
        self.student_tokenizer = AutoTokenizer.from_pretrained(cfg.model.base_model_name)

        backend_type = cfg.rad.teacher_backend
        if backend_type == "hf_local":
            self.backend = LocalHFBackend(cfg)
        elif backend_type == "api":
            self.backend = APIBackend(cfg)
        elif backend_type == "bedrock":
            self.backend = BedrockBackend(cfg)
        else:
            raise ValueError(f"Unknown teacher backend: {backend_type}")

        self.traces_dir = Path(cfg.rad.traces_dir)
        self.traces_dir.mkdir(parents=True, exist_ok=True)
        self.discarded_log = Path(cfg.storage.log_dir) / "rad_prep" / "discarded_traces.jsonl"
        self.discarded_log.parent.mkdir(parents=True, exist_ok=True)

    def generate_traces(self, samples: List[Dict[str, Any]], retrieved_results: List[Any], no_retrieval_router: Any) -> None:
        """Generate teacher traces in batches for the parsed samples."""
        batch_size = self.cfg.rad.teacher_batch_size
        grounded_path = self.traces_dir / "grounded_traces.jsonl"
        no_retrieval_path = self.traces_dir / "no_retrieval_traces.jsonl"

        # Prepare all prompts and routing decisions
        prompts = []
        decisions = []
        resolved_answers = []

        for idx, (sample, ret_res) in enumerate(zip(samples, retrieved_results)):
            question = sample["question"]

            # Resolve answer text from direct "answer" string or choices + answer_idx
            if "choices" in sample and "answer_idx" in sample:
                choices = sample["choices"]
                ans_idx = sample["answer_idx"]
                answer = choices[ans_idx]
            else:
                answer = sample.get("answer", "")

            resolved_answers.append(answer)

            sample_id = sample.get("sample_id", f"sample_{idx}")
            cluster_id = sample.get("cluster", sample.get("cluster_id"))

            decision = no_retrieval_router.route_sample(sample_id, cluster_id, ret_res.passed_threshold)
            decisions.append(decision)

            prompt = format_prompt(question, answer, ret_res.chunks, decision.no_retrieval)
            prompts.append(prompt)

        # Generate traces in batches
        all_traces = []
        logger.info(f"Generating teacher traces for {len(prompts)} prompts in batches of {batch_size}")
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i + batch_size]
            batch_traces = self.backend.generate_batch(batch_prompts)
            all_traces.extend(batch_traces)

        # Process and write traces
        logger.info("Writing and filtering generated traces")
        with open(grounded_path, "w", encoding="utf-8") as f_grounded, \
             open(no_retrieval_path, "w", encoding="utf-8") as f_no_ret, \
             open(self.discarded_log, "w", encoding="utf-8") as f_discard:

            for idx, (sample, answer, decision, ret_res, trace) in enumerate(zip(samples, resolved_answers, decisions, retrieved_results, all_traces)):
                sample_id = decision.sample_id
                cluster_id = decision.cluster_id
                question = sample["question"]

                # Token count validation
                token_count = len(self.student_tokenizer.encode(trace, add_special_tokens=False))

                # Check validation boundaries
                min_tok = self.cfg.rad.trace_min_tokens
                max_tok = self.cfg.rad.trace_max_tokens

                record = TraceRecord(
                    sample_id=sample_id,
                    cluster_id=cluster_id,
                    question=question,
                    answer=answer,
                    retrieved_context="\n\n".join(chunk.text for chunk in ret_res.chunks),
                    no_retrieval=decision.no_retrieval,
                    teacher_trace=trace,
                    token_count=token_count,
                    teacher_model=self.cfg.rad.teacher_model_name,
                    embedding_model=self.cfg.rad.embedding_model
                )

                record_dict = {
                    "sample_id": record.sample_id,
                    "cluster_id": record.cluster_id,
                    "question": record.question,
                    "answer": record.answer,
                    "retrieved_context": record.retrieved_context,
                    "no_retrieval": record.no_retrieval,
                    "teacher_trace": record.teacher_trace,
                    "token_count": record.token_count,
                    "teacher_model": record.teacher_model,
                    "embedding_model": record.embedding_model
                }

                if token_count < min_tok:
                    reason = f"Trace token count ({token_count}) is below minimum limit ({min_tok})"
                    f_discard.write(json.dumps({"sample_id": sample_id, "reason": reason, "trace": trace}) + "\n")
                    logger.debug(f"Discarded trace for sample {sample_id}: {reason}")
                elif token_count > max_tok:
                    reason = f"Trace token count ({token_count}) exceeds maximum limit ({max_tok})"
                    f_discard.write(json.dumps({"sample_id": sample_id, "reason": reason, "trace": trace}) + "\n")
                    logger.debug(f"Discarded trace for sample {sample_id}: {reason}")
                else:
                    if record.no_retrieval:
                        f_no_ret.write(json.dumps(record_dict) + "\n")
                    else:
                        f_grounded.write(json.dumps(record_dict) + "\n")
