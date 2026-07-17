import json
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest
import torch
import numpy as np

import math
from typing import Any, Dict, List

from lib.utils import PipelineConfig
from lib.s3_dapt.dapt import run_dapt_pipeline
from lib.s2_pretokenize import run_pretokenization


def evaluate_perplexity(
    model: Any,
    tokenizer: Any,
    dataset: List[Dict[str, Any]],
    block_size: int = 512
) -> float:
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    orig_max_len = getattr(tokenizer, "model_max_length", None)
    tokenizer.model_max_length = 100_000_000

    try:
        with torch.inference_mode():
            for item in dataset:
                text = item.get("text", "")
                if not text.strip():
                    continue

                tokens = tokenizer.encode(
                    text,
                    add_special_tokens=False
                )

                if tokenizer.eos_token_id is not None:
                    tokens.append(tokenizer.eos_token_id)

                for start in range(0, len(tokens), block_size):
                    chunk = tokens[start:start + block_size]

                    if len(chunk) < 2:
                        continue

                    input_ids = torch.tensor(
                        [chunk],
                        dtype=torch.long,
                        device=model.device
                    )

                    attention_mask = torch.ones_like(input_ids)

                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=input_ids
                    )

                    num_tokens = len(chunk) - 1

                    total_loss += outputs.loss.item() * num_tokens
                    total_tokens += num_tokens
    finally:
        if orig_max_len is not None:
            tokenizer.model_max_length = orig_max_len

    if total_tokens == 0:
        return float("inf")

    avg_nll = total_loss / total_tokens
    return math.exp(avg_nll)


def evaluate_qa_accuracy(model: Any, tokenizer: Any, probe_questions: List[Dict[str, Any]]) -> float:
    model.eval()
    correct = 0
    total = 0
    options = ["A", "B", "C", "D"]
    
    with torch.inference_mode():
        for q in probe_questions:
            prompt = f"Question: {q['question']}\nAnswer:"
            inputs = tokenizer(prompt, return_tensors="pt")
            input_ids = inputs["input_ids"].to(model.device)
            attention_mask = inputs.get("attention_mask", torch.ones_like(input_ids)).to(model.device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            next_token_logits = outputs.logits[0, -1, :]
            next_token_probs = torch.softmax(next_token_logits, dim=-1)
            
            option_probs = {}
            for opt in options:
                opt_token_ids = tokenizer.encode(" " + opt, add_special_tokens=False)
                if len(opt_token_ids) > 0:
                    opt_token_id = opt_token_ids[-1]
                    option_probs[opt] = next_token_probs[opt_token_id].item()
                else:
                    option_probs[opt] = 0.0
            
            best_option = max(option_probs, key=option_probs.get)
            if best_option == q.get("answer"):
                correct += 1
            total += 1
            
    return (correct / total) * 100 if total > 0 else 0.0


@pytest.fixture
def mock_tokenizer():
    tokenizer = MagicMock()
    tokenizer.pad_token = None
    tokenizer.eos_token = "<|im_end|>"
    tokenizer.eos_token_id = 2
    tokenizer.pad_token_id = 3
    
    # Mock tokenization outputs
    def mock_tokenize_fn(text, *args, **kwargs):
        tokens = [1, 2, 3, 4]
        return {
            "input_ids": torch.tensor([tokens]),
            "attention_mask": torch.tensor([[1, 1, 1, 1]])
        }
        
    tokenizer.side_effect = mock_tokenize_fn
    tokenizer.encode.return_value = [5]  # Token ID for option letters
    return tokenizer


@pytest.fixture
def mock_model():
    model = MagicMock()
    model.device = torch.device("cpu")
    
    def mock_forward(*args, **kwargs):
        input_ids = kwargs.get("input_ids")
        if input_ids is None and len(args) > 0:
            input_ids = args[0]
        
        batch_size = 1
        seq_len = 5
        if input_ids is not None:
            batch_size = input_ids.shape[0]
            seq_len = input_ids.shape[1]
            
        logits = torch.zeros((batch_size, seq_len, 10))
        logits[:, -1, 5] = 10.0
        
        out = MagicMock()
        out.loss = torch.tensor(1.0, requires_grad=True)
        out.logits = logits
        return out
        
    model.side_effect = mock_forward
    model.parameters.return_value = [torch.nn.Parameter(torch.zeros(2, 2))]
    return model


def test_evaluate_perplexity(mock_model, mock_tokenizer):
    dataset = [{"text": "Hello, world!"}, {"text": "Test pretraining."}]
    ppl = evaluate_perplexity(mock_model, mock_tokenizer, dataset)
    # loss = 1.0, ppl = exp(1.0) = 2.718
    assert ppl == pytest.approx(2.71828, rel=1e-3)


def test_evaluate_qa_accuracy(mock_model, mock_tokenizer):
    probe_questions = [
        {"question": "Question 1", "answer": "A"},
        {"question": "Question 2", "answer": "B"}
    ]
    
    # Mock tokenizer.encode to return token ID 5 for A
    def mock_encode(text, *args, **kwargs):
        if text.strip() == "A":
            return [5]
        return [6]
        
    mock_tokenizer.encode.side_effect = mock_encode
    
    acc = evaluate_qa_accuracy(mock_model, mock_tokenizer, probe_questions)
    # The model predicts option A with high prob. So first question is correct, second is incorrect.
    # Accuracy should be 50%
    assert acc == 50.0


def test_run_dapt_pipeline(mock_model, mock_tokenizer):
    with tempfile.TemporaryDirectory() as tmpdir:
        corpus_path = os.path.join(tmpdir, "corpus.jsonl")
        probe_qa_path = os.path.join(tmpdir, "probe_qa.jsonl")
        output_dir = os.path.join(tmpdir, "out_model")
        ppl_corpus_path = os.path.join(tmpdir, "ppl_validation_tokens.npy")
        vocab_cloze_path = os.path.join(tmpdir, "vocab_cloze_set.json")
        retrieval_prompts_path = os.path.join(tmpdir, "retrieval_prompts.json")
        retrieval_references_path = os.path.join(tmpdir, "retrieval_references.json")
        
        # Write dummy files
        with open(corpus_path, "w") as f:
            f.write(json.dumps({"text": "document 1"}) + "\n")
            f.write(json.dumps({"text": "document 2"}) + "\n")
            f.write(json.dumps({"text": "document 3"}) + "\n")
            
        with open(probe_qa_path, "w") as f:
            f.write(json.dumps({"question": "Q1", "choices": ["A", "B"], "answer_idx": 0}) + "\n")
 
        np.save(ppl_corpus_path, np.array([1, 2, 3, 4, 5], dtype=np.int32))
        with open(vocab_cloze_path, "w") as f:
            json.dump([{"prompt": "dummy prompt", "target_term": "dummy term", "category": "dummy"}], f)
        with open(retrieval_prompts_path, "w") as f:
            json.dump(["dummy prompt"], f)
        with open(retrieval_references_path, "w") as f:
            json.dump(["dummy reference"], f)
            
        pretokenized_bin_path = os.path.join(tmpdir, "train_tokens.npy")
        np.save(pretokenized_bin_path, np.arange(1000, dtype=np.int32))
            
        # Patch transformers from_pretrained calls and run_all_probes
        with patch("lib.s3_dapt.model_utils.AutoTokenizer.from_pretrained") as mock_from_token, \
             patch("lib.s3_dapt.model_utils.AutoModelForCausalLM.from_pretrained") as mock_from_model, \
             patch("lib.s3_dapt.dapt.run_all_probes") as mock_run_probes, \
             patch("lib.s3_dapt.training_helpers.run_all_probes") as mock_run_probes_th, \
             patch.dict(os.environ, {
                 "PPL_CORPUS_PATH": ppl_corpus_path,
                 "VOCAB_CLOZE_PATH": vocab_cloze_path,
                 "RETRIEVAL_PROMPTS_PATH": retrieval_prompts_path,
                 "RETRIEVAL_REFERENCES_PATH": retrieval_references_path,
                 "PRETOKENIZED_BIN_PATH": pretokenized_bin_path,
                 "WANDB_ENABLED": "False",
                 "PEFT_DAPT": "False",
             }):
                mock_from_token.return_value = mock_tokenizer
                mock_from_model.return_value = mock_model
                mock_run_probes.return_value = {
                    "eval_id": 1,
                    "tokens_processed": 0,
                    "corpus_pass": 0.0,
                    "metrics": {
                        "perplexity": 10.0,
                        "avg_nll_nats": 2.3,
                        "qa_accuracy": 50.0,
                        "qa_correct": 1,
                        "qa_total": 2,
                        "cloze_coverage": 0.5,
                        "concept_precision": 0.5,
                    }
                }
                mock_run_probes_th.return_value = mock_run_probes.return_value
                
                cfg = PipelineConfig()
                cfg.model.base_model_name = "dummy-model"
                cfg.model.max_seq_len = 512
                cfg.build.output_path = Path(corpus_path)
                cfg.data.qa_probe_path = Path(probe_qa_path)
                cfg.corpus.max_corpus_passes = 1
                cfg.optimizer.learning_rate = 1e-5
                cfg.optimizer.train_batch_size = 1
                cfg.model.checkpoint_dir = Path(output_dir)
                
                run_dapt_pipeline(cfg)
                
                # Check model saving and baseline run
                assert mock_model.save_pretrained.called
                assert mock_tokenizer.save_pretrained.called
                assert mock_run_probes.called


def test_run_dapt_pipeline_with_wandb(mock_model, mock_tokenizer):
    with tempfile.TemporaryDirectory() as tmpdir:
        corpus_path = os.path.join(tmpdir, "corpus.jsonl")
        probe_qa_path = os.path.join(tmpdir, "probe_qa.jsonl")
        output_dir = os.path.join(tmpdir, "out_model")
        ppl_corpus_path = os.path.join(tmpdir, "ppl_validation_tokens.npy")
        vocab_cloze_path = os.path.join(tmpdir, "vocab_cloze_set.json")
        retrieval_prompts_path = os.path.join(tmpdir, "retrieval_prompts.json")
        retrieval_references_path = os.path.join(tmpdir, "retrieval_references.json")
        
        # Write dummy files
        with open(corpus_path, "w") as f:
            f.write(json.dumps({"text": "document 1"}) + "\n")
            f.write(json.dumps({"text": "document 2"}) + "\n")
            f.write(json.dumps({"text": "document 3"}) + "\n")
            
        with open(probe_qa_path, "w") as f:
            f.write(json.dumps({"question": "Q1", "choices": ["A", "B"], "answer_idx": 0}) + "\n")
 
        np.save(ppl_corpus_path, np.array([1, 2, 3, 4, 5], dtype=np.int32))
        with open(vocab_cloze_path, "w") as f:
            json.dump([{"prompt": "dummy prompt", "target_term": "dummy term", "category": "dummy"}], f)
        with open(retrieval_prompts_path, "w") as f:
            json.dump(["dummy prompt"], f)
        with open(retrieval_references_path, "w") as f:
            json.dump(["dummy reference"], f)
            
        pretokenized_bin_path = os.path.join(tmpdir, "train_tokens.npy")
        np.save(pretokenized_bin_path, np.arange(1000, dtype=np.int32))
            
        # Patch transformers from_pretrained calls, wandb methods, and run_all_probes
        with patch("lib.s3_dapt.model_utils.AutoTokenizer.from_pretrained") as mock_from_token, \
             patch("lib.s3_dapt.model_utils.AutoModelForCausalLM.from_pretrained") as mock_from_model, \
             patch("lib.s3_dapt.dapt.run_all_probes") as mock_run_probes, \
             patch("lib.s3_dapt.training_helpers.run_all_probes") as mock_run_probes_th, \
             patch("wandb.init") as mock_wandb_init, \
             patch("wandb.login") as mock_wandb_login, \
             patch("wandb.log") as mock_wandb_log, \
             patch("wandb.finish") as mock_wandb_finish, \
             patch.dict(os.environ, {
                 "PPL_CORPUS_PATH": ppl_corpus_path,
                 "VOCAB_CLOZE_PATH": vocab_cloze_path,
                 "RETRIEVAL_PROMPTS_PATH": retrieval_prompts_path,
                 "RETRIEVAL_REFERENCES_PATH": retrieval_references_path,
                 "PRETOKENIZED_BIN_PATH": pretokenized_bin_path,
                 "WANDB_ENABLED": "True",
                 "WANDB_API_KEY": "test-key-12345",
                 "WANDB_PROJECT": "test-project",
                 "WANDB_LOG_INTERVAL_STEPS": "1",
                 "PEFT_DAPT": "False",
             }):
             
            mock_from_token.return_value = mock_tokenizer
            mock_from_model.return_value = mock_model
            mock_run_probes.return_value = {
                "eval_id": 1,
                "tokens_processed": 0,
                "corpus_pass": 0.0,
                "metrics": {
                    "perplexity": 10.0,
                    "avg_nll_nats": 2.3,
                    "qa_accuracy": 50.0,
                    "qa_correct": 1,
                    "qa_total": 2,
                    "cloze_coverage": 0.5,
                    "concept_precision": 0.5,
                }
            }
            mock_run_probes_th.return_value = mock_run_probes.return_value
            
            cfg = PipelineConfig()
            cfg.model.base_model_name = "dummy-model"
            cfg.model.max_seq_len = 512
            cfg.build.output_path = Path(corpus_path)
            cfg.data.qa_probe_path = Path(probe_qa_path)
            cfg.corpus.max_corpus_passes = 1
            cfg.optimizer.learning_rate = 1e-5
            cfg.optimizer.train_batch_size = 1
            cfg.model.checkpoint_dir = Path(output_dir)
            
            run_dapt_pipeline(cfg)
            
            # Verify wandb was initialized, logged metrics, and baseline ran
            assert mock_wandb_login.called
            assert mock_wandb_init.called
            assert mock_wandb_log.called
            assert mock_wandb_finish.called
            assert mock_run_probes.called


def test_run_dapt_pipeline_missing_files_raises_error(mock_model, mock_tokenizer):
    with tempfile.TemporaryDirectory() as tmpdir:
        corpus_path = os.path.join(tmpdir, "corpus.jsonl")
        probe_qa_path = os.path.join(tmpdir, "probe_qa.jsonl")
        output_dir = os.path.join(tmpdir, "out_model")
        
        # Write only corpus and QA probe
        with open(corpus_path, "w") as f:
            f.write(json.dumps({"text": "document 1"}) + "\n")
            f.write(json.dumps({"text": "document 2"}) + "\n")
            f.write(json.dumps({"text": "document 3"}) + "\n")
            
        with open(probe_qa_path, "w") as f:
            f.write(json.dumps({"question": "Q1", "choices": ["A", "B"], "answer_idx": 0}) + "\n")
            
        pretokenized_bin_path = os.path.join(tmpdir, "train_tokens.npy")
        np.save(pretokenized_bin_path, np.arange(1000, dtype=np.int32))
            
        # Point environment variables to non-existent files inside tmpdir
        with patch.dict(os.environ, {
            "PPL_CORPUS_PATH": os.path.join(tmpdir, "missing_ppl.txt"),
            "VOCAB_CLOZE_PATH": os.path.join(tmpdir, "missing_vocab.json"),
            "RETRIEVAL_PROMPTS_PATH": os.path.join(tmpdir, "missing_prompts.json"),
            "RETRIEVAL_REFERENCES_PATH": os.path.join(tmpdir, "missing_references.json"),
            "PRETOKENIZED_BIN_PATH": pretokenized_bin_path,
            "WANDB_ENABLED": "False",
            "PEFT_DAPT": "False",
        }):
            with patch("lib.s3_dapt.model_utils.AutoTokenizer.from_pretrained") as mock_from_token, \
                 patch("lib.s3_dapt.model_utils.AutoModelForCausalLM.from_pretrained") as mock_from_model:
                mock_from_token.return_value = mock_tokenizer
                mock_from_model.return_value = mock_model
                
                with pytest.raises(FileNotFoundError) as exc_info:
                    cfg = PipelineConfig()
                    cfg.model.base_model_name = "dummy-model"
                    cfg.model.max_seq_len = 512
                    cfg.build.output_path = Path(corpus_path)
                    cfg.data.qa_probe_path = Path(probe_qa_path)
                    cfg.corpus.max_corpus_passes = 1
                    cfg.optimizer.learning_rate = 1e-5
                    cfg.optimizer.train_batch_size = 1
                    cfg.model.checkpoint_dir = Path(output_dir)
                    
                    run_dapt_pipeline(cfg)
                assert "Required evaluation files are missing" in str(exc_info.value)


def test_run_pretokenization(mock_tokenizer):
    with tempfile.TemporaryDirectory() as tmpdir:
        corpus_path = os.path.join(tmpdir, "corpus.jsonl")
        pretokenized_bin_path = os.path.join(tmpdir, "train_tokens.npy")
        ppl_corpus_path = os.path.join(tmpdir, "ppl_validation_tokens.npy")
        
        # Write a dummy corpus
        with open(corpus_path, "w") as f:
            f.write(json.dumps({"text": "document 1"}) + "\n")
            f.write(json.dumps({"text": "document 2"}) + "\n")
            f.write(json.dumps({"text": "document 3"}) + "\n")
            
        with patch("lib.s2_pretokenize.pretokenize.AutoTokenizer.from_pretrained") as mock_from_token:
            mock_from_token.return_value = mock_tokenizer
            
            cfg = PipelineConfig()
            cfg.build.output_path = Path(corpus_path)
            cfg.model.base_model_name = "dummy-model"
            cfg.data.pretokenized_bin_path = Path(pretokenized_bin_path)
            cfg.data.ppl_corpus_path = Path(ppl_corpus_path)
            
            run_pretokenization(cfg, val_ratio=0.33)
            
            # Check validation corpus exists
            assert os.path.exists(ppl_corpus_path)
            # Check binary pre-tokenized file exists
            assert os.path.exists(pretokenized_bin_path)
            
            # Load tokens
            tokens = np.load(pretokenized_bin_path)
            assert len(tokens) > 0
            
            val_tokens = np.load(ppl_corpus_path)
            assert len(val_tokens) > 0


def test_eval_qa_accuracy_new_format(mock_model, mock_tokenizer):
    from lib.s3_dapt.probes.qa_probe import eval_qa_accuracy
    from pathlib import Path
    
    probe_items = [
        {
            "question": "Which neurotransmitter is primarily associated with reward?",
            "choices": ["Serotonin", "Dopamine", "GABA", "Acetylcholine"],
            "answer_idx": 1,
            "cluster": "neurotransmitters"
        }
    ]
    
    # Custom mock tokenizer to handle prompt and choice input shapes
    def mock_tokenize_fn(text, *args, **kwargs):
        if "Answer:" in text:
            if any(text.endswith(c) for c in ["Serotonin", "Dopamine", "GABA", "Acetylcholine"]):
                # Prompt + choice: 6 tokens (BOS + prompt + choice)
                tokens = [0, 1, 2, 3, 4, 5]
            else:
                # Prompt only: 4 tokens (BOS + prompt)
                tokens = [0, 1, 2, 3]
        else:
            # Choice only: 3 tokens (BOS + choice)
            tokens = [0, 4, 5]
        return {
            "input_ids": torch.tensor([tokens]),
            "attention_mask": torch.tensor([[1] * len(tokens)])
        }
    
    mock_tokenizer.side_effect = mock_tokenize_fn
    mock_tokenizer.encode.return_value = [1, 2, 3]
    
    def mock_model_forward(*args, **kwargs):
        input_ids = kwargs.get("input_ids")
        if input_ids is None and len(args) > 0:
            input_ids = args[0]
        batch_size = input_ids.shape[0]
        seq_len = input_ids.shape[1]

        logits = torch.zeros((batch_size, seq_len, 10))
        if seq_len == 6:
            logits[:, 3, 4] = 1.0
            logits[:, 4, 5] = 1.0
            if batch_size > 1:
                # Dopamine (choice index 1) is correct conditionally
                logits[1, 3, 4] = 10.0
                logits[1, 4, 5] = 10.0
        else:
            logits[:, 0, 4] = 1.0
            logits[:, 1, 5] = 1.0

        mock_output = MagicMock()
        mock_output.logits = logits
        return mock_output
        
    mock_model.side_effect = mock_model_forward
    
    with tempfile.TemporaryDirectory() as tmpdir:
        probe_qa_path = Path(tmpdir) / "probe_qa.jsonl"
        with open(probe_qa_path, "w", encoding="utf-8") as f:
            for item in probe_items:
                f.write(json.dumps(item) + "\n")
                
        result = eval_qa_accuracy(
            model=mock_model,
            tokenizer=mock_tokenizer,
            qa_probe_path=probe_qa_path,
            device="cpu"
        )
        
        assert result["total"] == 1
        assert result["correct"] == 1
        assert result["accuracy"] == 1.0
        assert result["per_cluster_accuracy"]["neurotransmitters"] == 1.0


def test_eval_cloze_coverage_empty_prompt(mock_model, mock_tokenizer):
    from lib.s3_dapt.probes.cloze_probe import eval_cloze_coverage
    from pathlib import Path
    
    # Custom mock tokenizer that returns empty input_ids when prompt is empty
    def mock_tokenize_fn(text, *args, **kwargs):
        if not text.strip() or text == " ":
            input_ids = torch.empty((1, 0), dtype=torch.long)
            attention_mask = torch.empty((1, 0), dtype=torch.long)
        else:
            input_ids = torch.tensor([[1, 2, 3]])
            attention_mask = torch.tensor([[1, 1, 1]])
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask
        }
        
    mock_tokenizer.side_effect = mock_tokenize_fn
    mock_tokenizer.bos_token = None
    mock_tokenizer.eos_token = "<|im_end|>"
    mock_tokenizer.bos_token_id = None
    mock_tokenizer.eos_token_id = 2
    
    # Mock model.generate to return outputs
    def mock_model_generate(*args, **kwargs):
        # Must return at least prompt_len + some token
        # input_ids will be padded/fixed to have length 1
        return torch.tensor([[2, 5, 6]])
        
    mock_model.generate = mock_model_generate
    
    with tempfile.TemporaryDirectory() as tmpdir:
        vocab_cloze_path = Path(tmpdir) / "vocab_cloze_set.json"
        # Item 1 has prompt starting with ___ which results in empty prefix
        # Item 2 has prompt that is completely ___ which results in empty prefix
        cloze_items = [
            {"prompt": "___ is a neurotransmitter.", "target_term": "Dopamine", "category": "cat1"},
            {"prompt": "___", "target_term": "blank", "category": "cat2"}
        ]
        with open(vocab_cloze_path, "w", encoding="utf-8") as f:
            json.dump(cloze_items, f)
            
        result = eval_cloze_coverage(
            model=mock_model,
            tokenizer=mock_tokenizer,
            vocab_cloze_path=vocab_cloze_path,
            top_k=1,
            max_new_tokens=5,
            device="cpu"
        )
        
        assert result["total"] == 2


def test_eval_concept_precision_empty_prompt(mock_model, mock_tokenizer):
    from lib.s3_dapt.probes.concept_probe import eval_concept_precision
    from pathlib import Path
    
    def mock_tokenize_fn(text, *args, **kwargs):
        if not text.strip() or text == " ":
            input_ids = torch.empty((1, 0), dtype=torch.long)
            attention_mask = torch.empty((1, 0), dtype=torch.long)
        else:
            input_ids = torch.tensor([[1, 2, 3]])
            attention_mask = torch.tensor([[1, 1, 1]])
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask
        }
        
    mock_tokenizer.side_effect = mock_tokenize_fn
    mock_tokenizer.bos_token = None
    mock_tokenizer.eos_token = "<|im_end|>"
    mock_tokenizer.bos_token_id = None
    mock_tokenizer.eos_token_id = 2
    mock_tokenizer.encode.return_value = [5, 6]
    mock_tokenizer.decode.return_value = "generated response"
    
    def mock_model_generate(*args, **kwargs):
        return torch.tensor([[2, 5, 6]])
        
    mock_model.generate = mock_model_generate
    
    with tempfile.TemporaryDirectory() as tmpdir:
        prompts_path = Path(tmpdir) / "retrieval_prompts.json"
        references_path = Path(tmpdir) / "retrieval_references.json"
        
        # Item with empty prompt
        with open(prompts_path, "w", encoding="utf-8") as f:
            json.dump(["", "normal prompt"], f)
        with open(references_path, "w", encoding="utf-8") as f:
            json.dump(["ref1", "ref2"], f)
            
        result = eval_concept_precision(
            model=mock_model,
            tokenizer=mock_tokenizer,
            retrieval_prompts_path=prompts_path,
            retrieval_references_path=references_path,
            bertscore_model="dummy_bertscore",
            max_new_tokens=5,
            device="cpu",
            use_bertscore=False  # use fast lexical overlap F1 to avoid calling BERTScore in tests
        )
        
        assert result["num_samples"] == 2


def test_run_dapt_pipeline_disabled_probes_bypasses_missing_files(mock_model, mock_tokenizer):
    with tempfile.TemporaryDirectory() as tmpdir:
        corpus_path = os.path.join(tmpdir, "corpus.jsonl")
        probe_qa_path = os.path.join(tmpdir, "probe_qa.jsonl")
        output_dir = os.path.join(tmpdir, "out_model")
        
        # Write only corpus and QA probe
        with open(corpus_path, "w") as f:
            f.write(json.dumps({"text": "document 1"}) + "\n")
            
        with open(probe_qa_path, "w") as f:
            f.write(json.dumps({"question": "Q1", "choices": ["A", "B"], "answer_idx": 0}) + "\n")
            
        pretokenized_bin_path = os.path.join(tmpdir, "train_tokens.npy")
        np.save(pretokenized_bin_path, np.arange(1000, dtype=np.int32))
            
        # Point environment variables to non-existent files inside tmpdir,
        # but disable the probes associated with them.
        with patch.dict(os.environ, {
            "RUN_PERPLEXITY_PROBE": "False",
            "RUN_CLOZE_PROBE": "False",
            "RUN_CONCEPT_PROBE": "False",
            "WANDB_ENABLED": "False",
            "PEFT_DAPT": "False",
            "PPL_CORPUS_PATH": os.path.join(tmpdir, "missing_ppl.txt"),
            "CLOZE_SET_PATH": os.path.join(tmpdir, "missing_vocab.json"),
            "CONCEPT_PROMPTS_PATH": os.path.join(tmpdir, "missing_prompts.json"),
            "CONCEPT_REFERENCES_PATH": os.path.join(tmpdir, "missing_references.json"),
            "PRETOKENIZED_BIN_PATH": pretokenized_bin_path,
        }):
            with patch("lib.s3_dapt.model_utils.AutoTokenizer.from_pretrained") as mock_from_token, \
                 patch("lib.s3_dapt.model_utils.AutoModelForCausalLM.from_pretrained") as mock_from_model, \
                 patch("lib.s3_dapt.dapt.run_all_probes") as mock_run_probes, \
                 patch("lib.s3_dapt.training_helpers.run_all_probes") as mock_run_probes_th:
                mock_from_token.return_value = mock_tokenizer
                mock_from_model.return_value = mock_model
                mock_run_probes.return_value = {
                    "eval_id": 1,
                    "tokens_processed": 0,
                    "corpus_pass": 0.0,
                    "metrics": {
                        "perplexity": 0.0,
                        "avg_nll_nats": 0.0,
                        "qa_accuracy": 50.0,
                        "qa_correct": 1,
                        "qa_total": 2,
                        "cloze_coverage": 0.0,
                        "concept_precision": 0.0,
                    }
                }
                mock_run_probes_th.return_value = mock_run_probes.return_value
                
                cfg = PipelineConfig()
                cfg.model.base_model_name = "dummy-model"
                cfg.model.max_seq_len = 512
                cfg.build.output_path = Path(corpus_path)
                cfg.data.qa_probe_path = Path(probe_qa_path)
                cfg.corpus.max_corpus_passes = 1
                cfg.optimizer.learning_rate = 1e-5
                cfg.optimizer.train_batch_size = 1
                cfg.model.checkpoint_dir = Path(output_dir)
                
                # Should not raise FileNotFoundError because disabled files are bypassed
                run_dapt_pipeline(cfg)
                assert mock_model.save_pretrained.called


def test_check_convergence_gates_disabled_probes():
    from lib.s3_dapt.evaluation.gate_logic import check_convergence_gates, DAPTDecision
    
    state = {
        "tokens_processed": 100,
        "perplexity_history": [],
        "qa_acc_history": [0.0],
        "cloze_cov_history": [0.0],
        "concept_prec_history": [0.0],
    }
    
    # QA and PPL are disabled, secondary are enabled but have 0.0 metrics.
    # Since secondary requires at least one, and both are enabled but fail (0.0 < threshold),
    # convergence is not met yet (returns CONTINUE).
    decision, gate_details = check_convergence_gates(
        state=state,
        qa_acc_threshold=0.55,
        ppl_improvement_threshold=2.0,
        ppl_plateau_window=2,
        cloze_threshold=0.80,
        concept_threshold=0.60,
        hard_stop_tokens=1000,
        total_corpus_tokens=500,
        run_qa=False,
        run_perplexity=False,
        run_cloze=True,
        run_concept=True,
    )
    
    assert decision == DAPTDecision.CONTINUE
    assert gate_details["qa_gate"] is True
    assert gate_details["ppl_gate"] is True
    assert gate_details["secondary_gate"] is False

    # Now if we disable terminology and retrieval as well, all gates are satisfied
    decision, gate_details = check_convergence_gates(
        state=state,
        qa_acc_threshold=0.55,
        ppl_improvement_threshold=2.0,
        ppl_plateau_window=2,
        cloze_threshold=0.80,
        concept_threshold=0.60,
        hard_stop_tokens=1000,
        total_corpus_tokens=500,
        run_qa=False,
        run_perplexity=False,
        run_cloze=False,
        run_concept=False,
    )
    assert decision == DAPTDecision.CONVERGED
    assert gate_details["all_converged"] is True


def test_select_best_checkpoint_fallback_metrics():
    from lib.utils.checkpoint import select_best_checkpoint
    
    eval_history = [
        {
            "eval_id": 1,
            "metrics": {
                "qa_accuracy": 0.0,
                "cloze_coverage": 0.1,
                "concept_precision": 0.0,
                "perplexity": 10.0,
            }
        },
        {
            "eval_id": 2,
            "metrics": {
                "qa_accuracy": 0.0,
                "cloze_coverage": 0.9,
                "concept_precision": 0.0,
                "perplexity": 8.0,
            }
        }
    ]
    
    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_dir = Path(tmpdir)
        manifest_path = checkpoint_dir / "best_manifest.json"
        
        # Write mock checkpoint files to disk
        (checkpoint_dir / "dapt_eval_0001.pt").write_text("dummy")
        (checkpoint_dir / "dapt_eval_0002.pt").write_text("dummy")
        
        # QA is disabled, so we fall back to Terminology coverage (run_cloze=True)
        best_ckpt = select_best_checkpoint(
            eval_history=eval_history,
            checkpoint_dir=checkpoint_dir,
            ppl_improvement_threshold=2.0,
            manifest_path=manifest_path,
            run_qa=False,
            run_perplexity=False,
            run_cloze=True,
            run_concept=False,
        )
        
        # Should select eval 2 because cloze_coverage is 0.9 > 0.1
        assert best_ckpt.name == "dapt_eval_0002.pt"


def test_run_inference_and_log_failures(mock_model, mock_tokenizer):
    from lib.s3_dapt.evaluation.eval_runner import run_inference_and_log_failures
    import tempfile
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create mock checkpoint files/folders
        model_dir = Path(tmpdir) / "checkpoint"
        model_dir.mkdir()
        # Save a dummy file so exists() returns true for directory
        (model_dir / "config.json").write_text("{}")
        
        # Write dummy evaluation files
        qa_path = Path(tmpdir) / "probe_qa.jsonl"
        with open(qa_path, "w") as f:
            f.write(json.dumps({"question": "Q1", "choices": ["A", "B"], "answer_idx": 0, "cluster": "cat1"}) + "\n")
            f.write(json.dumps({"question": "Q2", "choices": ["A", "B"], "answer_idx": 1, "cluster": "cat2"}) + "\n")
            
        vocab_path = Path(tmpdir) / "vocab_cloze_set.json"
        with open(vocab_path, "w") as f:
            json.dump([{"prompt": "___ is cool.", "target_term": "Term1", "category": "cat1"}], f)
            
        retrieval_prompts_path = Path(tmpdir) / "retrieval_prompts.json"
        with open(retrieval_prompts_path, "w") as f:
            json.dump(["Prompt1"], f)
            
        retrieval_references_path = Path(tmpdir) / "retrieval_references.json"
        with open(retrieval_references_path, "w") as f:
            json.dump(["Ref1"], f)
            
        # Create pipeline config and run mock inference
        with patch.dict(os.environ, {"PEFT_DAPT": "False", "WANDB_ENABLED": "False"}):
            cfg = PipelineConfig()
            cfg.model.checkpoint_dir = model_dir
            cfg.logging.log_dir = Path(tmpdir) / "logs"
            cfg.data.qa_probe_path = qa_path
            cfg.data.cloze_set_path = vocab_path
            cfg.data.concept_prompts_path = retrieval_prompts_path
            cfg.data.concept_references_path = retrieval_references_path
            cfg.probes.run_qa = True
            cfg.probes.run_cloze = True
            cfg.probes.run_concept = True

            with patch("transformers.AutoModelForCausalLM.from_pretrained") as mock_model_load, \
                 patch("transformers.AutoTokenizer.from_pretrained") as mock_tokenizer_load, \
                 patch("lib.s3_dapt.probes.qa_probe.get_failed_qa_samples") as mock_get_failed_qa, \
                 patch("lib.s3_dapt.probes.cloze_probe.get_failed_cloze_samples") as mock_get_failed_term, \
                 patch("lib.s3_dapt.probes.concept_probe.get_failed_concept_samples") as mock_get_failed_ret:
                 
                mock_model_load.return_value = mock_model
                mock_tokenizer_load.return_value = mock_tokenizer
                
                # Setup dummy failures
                mock_get_failed_qa.return_value = [{"question": "Q2", "expected_idx": 1, "expected_text": "B", "predicted_idx": 0, "predicted_text": "A", "cluster": "cat2"}]
                mock_get_failed_term.return_value = [{"prompt": "___ is cool.", "target_term": "Term1", "generated_completions": ["completions"], "category": "cat1"}]
                mock_get_failed_ret.return_value = [{"prompt": "Prompt1", "reference": "Ref1", "generated": "Gen1", "score": 0.35}]
                
                run_inference_and_log_failures(cfg)
                
                # Verify failures file was written
                failed_evals_file = cfg.logging.log_dir / "failed_evals.json"
                assert failed_evals_file.exists()
                
                with open(failed_evals_file, "r") as f:
                    data = json.load(f)
                    
                assert "qa" in data
                assert len(data["qa"]) == 1
                assert data["qa"][0]["question"] == "Q2"
            
            assert "cloze" in data
            assert len(data["cloze"]) == 1
            assert data["cloze"][0]["target_term"] == "Term1"
            
            assert "concept" in data
            assert len(data["concept"]) == 1
            assert data["concept"][0]["score"] == 0.35


def test_concept_probe_lexical_f1_and_delegation(mock_model, mock_tokenizer):
    from lib.s3_dapt.probes.concept_probe import (
        compute_lexical_f1_batch,
        _patched_tokenizer_context,
        get_failed_concept_samples
    )
    import transformers
    
    # 1. Test lexical F1 cleaning (punctuation stripping)
    hyps = ["amygdala."]
    refs = ["amygdala"]
    scores = compute_lexical_f1_batch(hyps, refs)
    assert scores[0] == 1.0
    
    # 2. Test scoped monkeypatch (it gets cleaned up)
    orig_fn = transformers.AutoTokenizer.from_pretrained
    with _patched_tokenizer_context():
        assert transformers.AutoTokenizer.from_pretrained != orig_fn
    assert transformers.AutoTokenizer.from_pretrained == orig_fn

    # 3. Test get_failed_concept_samples delegation
    mock_tokenizer.eos_token_id = 2
    mock_tokenizer.encode.return_value = [5, 6]
    mock_tokenizer.decode.return_value = "amygdala"
    
    def mock_model_generate(*args, **kwargs):
        return torch.tensor([[2, 5, 6]])
    mock_model.generate = mock_model_generate
    
    with tempfile.TemporaryDirectory() as tmpdir:
        prompts_path = Path(tmpdir) / "retrieval_prompts.json"
        references_path = Path(tmpdir) / "retrieval_references.json"
        
        with open(prompts_path, "w", encoding="utf-8") as f:
            json.dump(["Give me term"], f)
        with open(references_path, "w", encoding="utf-8") as f:
            json.dump(["hippocampus"], f) # different from "amygdala" => fails
            
        failures = get_failed_concept_samples(
            model=mock_model,
            tokenizer=mock_tokenizer,
            retrieval_prompts_path=prompts_path,
            retrieval_references_path=references_path,
            bertscore_model="dummy_bertscore",
            max_new_tokens=5,
            device="cpu",
            use_bertscore=False,
            failure_threshold=0.8
        )
        assert len(failures) == 1
        assert failures[0]["prompt"] == "Give me term"
        assert failures[0]["reference"] == "hippocampus"


def test_patched_tokenizer_context_monkeypatches_special_tokens():
    from lib.s3_dapt.probes.concept_probe import _patched_tokenizer_context
    import transformers
    
    class DummyTokenizerWithoutSpecialTokens:
        def __init__(self):
            self.model_max_length = 500
            self.cls_token_id = 999
            self.sep_token_id = 888

    def mock_from_pretrained(*args, **kwargs):
        return DummyTokenizerWithoutSpecialTokens()

    with patch("transformers.AutoTokenizer.from_pretrained", side_effect=mock_from_pretrained):
        with _patched_tokenizer_context():
            tokenizer = transformers.AutoTokenizer.from_pretrained("dummy")
            assert hasattr(tokenizer, "build_inputs_with_special_tokens")
            special_tokens = tokenizer.build_inputs_with_special_tokens([1, 2, 3])
            assert special_tokens == [999, 1, 2, 3, 888]


def test_compute_bertscore_batch_handles_empty_candidates():
    from lib.s3_dapt.probes.concept_probe import compute_bertscore_batch
    
    with patch("lib.s3_dapt.probes.concept_probe.get_bertscorer") as mock_get_scorer:
        mock_scorer = MagicMock()
        mock_scorer.score.return_value = (None, None, torch.tensor([0.9, 0.85]))
        mock_get_scorer.return_value = mock_scorer
        
        hypotheses = ["", "non-empty candidate 1", "  ", "non-empty candidate 2"]
        references = ["ref1", "ref2", "ref3", "ref4"]
        
        scores = compute_bertscore_batch(
            hypotheses=hypotheses,
            references=references,
            model_type="dummy",
            device="cpu",
            batch_size=2
        )
        
        mock_scorer.score.assert_called_once_with(
            cands=["non-empty candidate 1", "non-empty candidate 2"],
            refs=["ref2", "ref4"],
            batch_size=2,
            verbose=False
        )
        import pytest
        assert scores == [0.0, pytest.approx(0.9), 0.0, pytest.approx(0.85)]


def test_load_model_and_tokenizer_peft(mock_model, mock_tokenizer):
    from lib.s3_dapt.model_utils import load_model_and_tokenizer
    
    cfg = PipelineConfig()
    cfg.model.base_model_name = "dummy-model"
    cfg.model.peft_dapt = True
    cfg.model.lora_r = 16
    cfg.model.lora_alpha = 32
    cfg.model.lora_dropout = 0.05
    cfg.model.lora_target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]
    
    with patch("lib.s3_dapt.model_utils.AutoTokenizer.from_pretrained") as mock_from_token, \
         patch("lib.s3_dapt.model_utils.AutoModelForCausalLM.from_pretrained") as mock_from_model, \
         patch("peft.get_peft_model") as mock_get_peft_model:
         
        mock_from_token.return_value = mock_tokenizer
        mock_from_model.return_value = mock_model
        
        # We want mock_get_peft_model to return a mock wrapped model
        mock_peft_model = MagicMock()
        mock_peft_model.get_nb_trainable_parameters.return_value = (100, 1000)
        mock_get_peft_model.return_value = mock_peft_model
        
        model, tokenizer = load_model_and_tokenizer(cfg, torch.device("cpu"))
        
        assert model == mock_peft_model
        assert tokenizer == mock_tokenizer
        assert mock_get_peft_model.called
        
        # Verify LoraConfig arguments passed to get_peft_model
        call_args = mock_get_peft_model.call_args[0]
        # First arg is the base model
        assert call_args[0] == mock_model
        
        # Second arg is the LoraConfig
        lora_cfg = call_args[1]
        assert lora_cfg.r == 16
        assert lora_cfg.lora_alpha == 32
        assert lora_cfg.lora_dropout == 0.05
        assert set(lora_cfg.target_modules) == set(["q_proj", "v_proj", "k_proj", "o_proj"])
        assert lora_cfg.bias == "none"


def test_run_inference_and_log_failures_peft(mock_model, mock_tokenizer):
    from lib.s3_dapt.evaluation.eval_runner import run_inference_and_log_failures
    
    with tempfile.TemporaryDirectory() as tmpdir:
        log_dir = Path(tmpdir) / "logs"
        log_dir.mkdir()
        checkpoint_dir = Path(tmpdir) / "checkpoints"
        checkpoint_dir.mkdir()
        
        cfg = PipelineConfig()
        cfg.model.base_model_name = "dummy-model"
        cfg.model.checkpoint_dir = checkpoint_dir
        cfg.logging.log_dir = log_dir
        cfg.model.peft_dapt = True
        
        # Setup mock files
        qa_probe_path = Path(tmpdir) / "probe_qa.jsonl"
        with open(qa_probe_path, "w") as f:
            f.write(json.dumps({"question": "Q1", "choices": ["A", "B"], "answer_idx": 0}) + "\n")
        cfg.data.qa_probe_path = qa_probe_path
        
        cfg.probes.run_qa = True
        cfg.probes.run_cloze = False
        cfg.probes.run_concept = False
        
        with patch("transformers.AutoModelForCausalLM.from_pretrained") as mock_base_load, \
             patch("transformers.AutoTokenizer.from_pretrained") as mock_tokenizer_load, \
             patch("peft.PeftModel.from_pretrained") as mock_peft_load, \
             patch("lib.s3_dapt.probes.qa_probe.get_failed_qa_samples") as mock_get_failed_qa:
             
            mock_base_load.return_value = mock_model
            mock_tokenizer_load.return_value = mock_tokenizer
            
            # Setup mock PeftModel
            mock_peft_model = MagicMock()
            mock_peft_load.return_value = mock_peft_model
            
            mock_get_failed_qa.return_value = []
            
            run_inference_and_log_failures(cfg)
            
            # Verify base model load was called with the base model name
            mock_base_load.assert_called_once_with(
                "dummy-model",
                dtype=torch.float32,
                attn_implementation="eager"
            )
            
            # Verify PeftModel.from_pretrained was called with base model and checkpoint dir
            mock_peft_load.assert_called_once_with(mock_model, str(checkpoint_dir))







