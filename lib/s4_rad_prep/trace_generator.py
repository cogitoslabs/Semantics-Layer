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
from lib.s4_rad_prep.chunker import Chunk

logger = logging.getLogger(__name__)

@dataclass
class TraceRecord:
    sample_id: str
    cluster_id: Optional[str]
    question: str
    answer: str
    retrieved_context: str
    no_retrieval: bool
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
            dtype = torch.bfloat16 if cfg.model.model_dtype == "bfloat16" else torch.float16
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                dtype=dtype,
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
        from botocore.config import Config

        aws_access_key_id = os.environ.get("AWS_ACCESS_KEY_ID") or os.environ.get("AWS_ACCESS_KEY")
        aws_secret_access_key = os.environ.get("AWS_SECRET_ACCESS_KEY") or os.environ.get("AWS_SECRET_KEY")
        aws_session_token = os.environ.get("AWS_SESSION_TOKEN")

        if not aws_access_key_id or not aws_secret_access_key:
            err_msg = (
                "SEVERE ERROR: Missing AWS credentials in environment variables! "
                "RAD_TEACHER_BACKEND requires AWS_ACCESS_KEY_ID (or AWS_ACCESS_KEY) and "
                "AWS_SECRET_ACCESS_KEY (or AWS_SECRET_KEY) to be explicitly set."
            )
            logger.critical(err_msg)
            raise RuntimeError(err_msg)

        boto_config = Config(max_pool_connections=32)
        client_kwargs = {
            "region_name": self.region_name,
            "aws_access_key_id": aws_access_key_id,
            "aws_secret_access_key": aws_secret_access_key,
            "config": boto_config
        }
        if aws_session_token:
            client_kwargs["aws_session_token"] = aws_session_token

        logger.info(f"Initializing AWS Bedrock Runtime client in region {self.region_name} using explicit credentials")
        try:
            self.client = boto3.client("bedrock-runtime", **client_kwargs)
        except Exception as e:
            err_msg = f"SEVERE ERROR: Failed to initialize AWS Bedrock client: {e}"
            logger.critical(err_msg)
            raise RuntimeError(err_msg) from e

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
                err_msg = f"SEVERE ERROR: AWS Bedrock API call failed for model {self.model_name}: {e}"
                logger.critical(err_msg)
                raise RuntimeError(err_msg) from e

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
            "Annotate key statements with bracketed passage citations matching the provided context (e.g. [Context 1] or [Passage 1]). "
            "Wrap your final answer inside \\boxed{}.\n\n"
            f"[CONTEXT]: {context_text}\n"
            f"[QUESTION]: {question}\n"
            f"[GROUND TRUTH]: {answer}"
        )


class TraceGenerator:
    """Generates grounded QA prompt records (with retrieved context) for downstream teacher benchmarking and trace generation."""
    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg
        self.traces_dir = Path(cfg.rad.traces_dir)
        self.traces_dir.mkdir(parents=True, exist_ok=True)

    def generate_traces(self, samples: List[Dict[str, Any]], retrieved_results: List[Any], no_retrieval_router: Any) -> Dict[str, int]:
        """Prepare grounded prompt records in batches, writing incrementally without LLM inference."""
        batch_size = self.cfg.rad.teacher_batch_size
        grounded_path = self.traces_dir / "grounded_traces.jsonl"
        no_retrieval_path = self.traces_dir / "no_retrieval_traces.jsonl"

        grounded_count = 0
        no_retrieval_count = 0
        discarded_count = 0

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
            "no_retrieval_count": no_retrieval_count,
            "discarded_count": discarded_count
        }

