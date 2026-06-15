import json
import os
import tempfile
from unittest.mock import MagicMock, patch
import pytest
import torch

from lib.s2_dapt.dapt import evaluate_perplexity, evaluate_qa_accuracy, run_dapt_pipeline


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
    
    # Mock forward pass output
    mock_output = MagicMock()
    # Loss for perplexity calculation
    mock_output.loss = torch.tensor(1.0, requires_grad=True)
    
    # Logits for option prediction: shape [batch_size, seq_len, vocab_size]
    # We set logits of token 5 (our option letter) to be high or low
    mock_logits = torch.zeros((1, 5, 10))
    mock_logits[0, -1, 5] = 10.0  # Make token ID 5 highly probable
    mock_output.logits = mock_logits
    
    model.return_value = mock_output
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
        
        # Write dummy files
        with open(corpus_path, "w") as f:
            f.write(json.dumps({"text": "document 1"}) + "\n")
            f.write(json.dumps({"text": "document 2"}) + "\n")
            f.write(json.dumps({"text": "document 3"}) + "\n")
            
        with open(probe_qa_path, "w") as f:
            f.write(json.dumps({"question": "Q1\nA)\nB)", "answer": "A"}) + "\n")
            
        # Patch transformers from_pretrained calls
        with patch("lib.s2_dapt.dapt.AutoTokenizer.from_pretrained") as mock_from_token:
            with patch("lib.s2_dapt.dapt.AutoModelForCausalLM.from_pretrained") as mock_from_model:
                mock_from_token.return_value = mock_tokenizer
                mock_from_model.return_value = mock_model
                
                run_dapt_pipeline(
                    model_name="dummy-model",
                    corpus_path=corpus_path,
                    probe_qa_path=probe_qa_path,
                    epochs=1,
                    lr=1e-5,
                    batch_size=1,
                    output_dir=output_dir
                )
                
                # Check model saving
                assert mock_model.save_pretrained.called
                assert mock_tokenizer.save_pretrained.called
