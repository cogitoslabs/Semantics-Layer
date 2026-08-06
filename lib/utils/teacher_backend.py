import os
import logging
from typing import List
from concurrent.futures import ThreadPoolExecutor
import torch
import requests
from transformers import AutoTokenizer, AutoModelForCausalLM
from lib.utils.config import PipelineConfig

logger = logging.getLogger(__name__)


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
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id
            )

        generated_texts = []
        for inp, out in zip(inputs["input_ids"], outputs):
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
