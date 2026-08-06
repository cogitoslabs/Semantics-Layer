import csv
import os
import json
from pathlib import Path
from unittest.mock import patch
import pytest

from lib.utils.model_tracer import model_trace


@model_trace
def sample_single_gen(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 10,
    device: str = "cpu",
    eval_num: str = "1",
    eval_category: str = "Concept",
    eval_seq_num: int = 1,
) -> str:
    return f"generated_{prompt}"


@model_trace
def sample_batch_gen(
    model,
    tokenizer,
    prompts: list[str],
    max_new_tokens: int = 10,
    device: str = "cpu",
    eval_num: str = "1",
    eval_category: str = "Cloze",
    eval_seq_start: int = 1,
) -> list[str]:
    return [f"generated_{p}" for p in prompts]


def test_model_trace_disabled_by_default(tmp_path):
    trace_file = tmp_path / "model_traces.csv"
    with patch.dict(os.environ, {"MODEL_TRACING": "False", "MODEL_TRACE_FILE": str(trace_file)}):
        res = sample_single_gen(None, None, prompt="Hello world", max_new_tokens=15)
        assert res == "generated_Hello world"
        assert not trace_file.exists()


def test_model_trace_enabled_single(tmp_path):
    trace_file = tmp_path / "model_traces.csv"
    with patch.dict(os.environ, {"MODEL_TRACING": "True", "MODEL_TRACE_FILE": str(trace_file)}):
        res = sample_single_gen(None, None, prompt="Explain sulcus", max_new_tokens=20, device="cpu", eval_num="1", eval_category="Concept", eval_seq_num=5)
        assert res == "generated_Explain sulcus"
        assert trace_file.exists()

        with open(trace_file, "r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))

        assert len(rows) == 1
        assert rows[0]["Eval #"] == "1"
        assert rows[0]["Eval Category"] == "Concept"
        assert rows[0]["Eval Seq #"] == "5"
        assert rows[0]["Function"] == "sample_single_gen"
        assert "Explain sulcus" in rows[0]["Prompt"]
        assert "generated_Explain sulcus" in rows[0]["Output"]

        params = json.loads(rows[0]["Parameters"])
        assert params.get("max_new_tokens") == 20
        assert params.get("device") == "cpu"


def test_model_trace_enabled_batch(tmp_path):
    trace_file = tmp_path / "model_traces.csv"
    with patch.dict(os.environ, {"MODEL_TRACING": "True", "MODEL_TRACE_FILE": str(trace_file)}):
        prompts = ["Prompt A", "Prompt B"]
        res = sample_batch_gen(None, None, prompts=prompts, max_new_tokens=30, eval_num="2", eval_category="Cloze", eval_seq_start=10)
        assert len(res) == 2
        assert trace_file.exists()

        with open(trace_file, "r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))

        assert len(rows) == 2
        assert rows[0]["Eval #"] == "2"
        assert rows[0]["Eval Category"] == "Cloze"
        assert rows[0]["Eval Seq #"] == "10"
        assert rows[0]["Function"] == "sample_batch_gen"
        assert rows[0]["Prompt"] == "Prompt A"
        assert rows[0]["Output"] == "generated_Prompt A"

        assert rows[1]["Eval #"] == "2"
        assert rows[1]["Eval Category"] == "Cloze"
        assert rows[1]["Eval Seq #"] == "11"
        assert rows[1]["Prompt"] == "Prompt B"
        assert rows[1]["Output"] == "generated_Prompt B"

        params = json.loads(rows[0]["Parameters"])
        assert params.get("max_new_tokens") == 30
