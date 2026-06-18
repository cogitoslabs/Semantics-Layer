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
        ppl_corpus_path = os.path.join(tmpdir, "ppl_held_out.txt")
        vocab_cloze_path = os.path.join(tmpdir, "vocab_cloze_set.json")
        anatomical_prompts_path = os.path.join(tmpdir, "anatomical_prompts.json")
        anatomical_references_path = os.path.join(tmpdir, "anatomical_references.json")
        
        # Write dummy files
        with open(corpus_path, "w") as f:
            f.write(json.dumps({"text": "document 1"}) + "\n")
            f.write(json.dumps({"text": "document 2"}) + "\n")
            f.write(json.dumps({"text": "document 3"}) + "\n")
            
        with open(probe_qa_path, "w") as f:
            f.write(json.dumps({"question": "Q1\nA)\nB)", "answer": "A"}) + "\n")

        with open(ppl_corpus_path, "w") as f:
            f.write("dummy ppl text")
        with open(vocab_cloze_path, "w") as f:
            json.dump([{"prompt": "dummy prompt", "target_term": "dummy term", "category": "dummy"}], f)
        with open(anatomical_prompts_path, "w") as f:
            json.dump(["dummy prompt"], f)
        with open(anatomical_references_path, "w") as f:
            json.dump(["dummy reference"], f)
            
        # Patch transformers from_pretrained calls and run_all_probes
        with patch("lib.s2_dapt.dapt.AutoTokenizer.from_pretrained") as mock_from_token, \
             patch("lib.s2_dapt.dapt.AutoModelForCausalLM.from_pretrained") as mock_from_model, \
             patch("lib.s2_dapt.dapt.run_all_probes") as mock_run_probes, \
             patch.dict(os.environ, {
                 "PPL_CORPUS_PATH": ppl_corpus_path,
                 "VOCAB_CLOZE_PATH": vocab_cloze_path,
                 "ANATOMICAL_PROMPTS_PATH": anatomical_prompts_path,
                 "ANATOMICAL_REFERENCES_PATH": anatomical_references_path,
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
                        "term_coverage": 0.5,
                        "retrieval_precision": 0.5,
                    }
                }
                
                run_dapt_pipeline(
                    model_name="dummy-model",
                    corpus_path=corpus_path,
                    probe_qa_path=probe_qa_path,
                    epochs=1,
                    lr=1e-5,
                    batch_size=1,
                    output_dir=output_dir
                )
                
                # Check model saving and baseline run
                assert mock_model.save_pretrained.called
                assert mock_tokenizer.save_pretrained.called
                assert mock_run_probes.called


def test_run_dapt_pipeline_with_wandb(mock_model, mock_tokenizer):
    with tempfile.TemporaryDirectory() as tmpdir:
        corpus_path = os.path.join(tmpdir, "corpus.jsonl")
        probe_qa_path = os.path.join(tmpdir, "probe_qa.jsonl")
        output_dir = os.path.join(tmpdir, "out_model")
        ppl_corpus_path = os.path.join(tmpdir, "ppl_held_out.txt")
        vocab_cloze_path = os.path.join(tmpdir, "vocab_cloze_set.json")
        anatomical_prompts_path = os.path.join(tmpdir, "anatomical_prompts.json")
        anatomical_references_path = os.path.join(tmpdir, "anatomical_references.json")
        
        # Write dummy files
        with open(corpus_path, "w") as f:
            f.write(json.dumps({"text": "document 1"}) + "\n")
            f.write(json.dumps({"text": "document 2"}) + "\n")
            f.write(json.dumps({"text": "document 3"}) + "\n")
            
        with open(probe_qa_path, "w") as f:
            f.write(json.dumps({"question": "Q1\nA)\nB)", "answer": "A"}) + "\n")

        with open(ppl_corpus_path, "w") as f:
            f.write("dummy ppl text")
        with open(vocab_cloze_path, "w") as f:
            json.dump([{"prompt": "dummy prompt", "target_term": "dummy term", "category": "dummy"}], f)
        with open(anatomical_prompts_path, "w") as f:
            json.dump(["dummy prompt"], f)
        with open(anatomical_references_path, "w") as f:
            json.dump(["dummy reference"], f)
            
        # Patch transformers from_pretrained calls, wandb methods, and run_all_probes
        with patch("lib.s2_dapt.dapt.AutoTokenizer.from_pretrained") as mock_from_token, \
             patch("lib.s2_dapt.dapt.AutoModelForCausalLM.from_pretrained") as mock_from_model, \
             patch("lib.s2_dapt.dapt.run_all_probes") as mock_run_probes, \
             patch("wandb.init") as mock_wandb_init, \
             patch("wandb.login") as mock_wandb_login, \
             patch("wandb.log") as mock_wandb_log, \
             patch("wandb.finish") as mock_wandb_finish, \
             patch.dict(os.environ, {
                 "PPL_CORPUS_PATH": ppl_corpus_path,
                 "VOCAB_CLOZE_PATH": vocab_cloze_path,
                 "ANATOMICAL_PROMPTS_PATH": anatomical_prompts_path,
                 "ANATOMICAL_REFERENCES_PATH": anatomical_references_path,
                 "WANDB_ENABLED": "True",
                 "WANDB_API_KEY": "test-key-12345",
                 "WANDB_PROJECT": "test-project",
                 "WANDB_LOG_INTERVAL_STEPS": "1",
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
                    "term_coverage": 0.5,
                    "retrieval_precision": 0.5,
                }
            }
            
            run_dapt_pipeline(
                model_name="dummy-model",
                corpus_path=corpus_path,
                probe_qa_path=probe_qa_path,
                epochs=1,
                lr=1e-5,
                batch_size=1,
                output_dir=output_dir
            )
            
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
            f.write(json.dumps({"question": "Q1\nA)\nB)", "answer": "A"}) + "\n")
            
        # Point environment variables to non-existent files inside tmpdir
        with patch.dict(os.environ, {
            "PPL_CORPUS_PATH": os.path.join(tmpdir, "missing_ppl.txt"),
            "VOCAB_CLOZE_PATH": os.path.join(tmpdir, "missing_vocab.json"),
            "ANATOMICAL_PROMPTS_PATH": os.path.join(tmpdir, "missing_prompts.json"),
            "ANATOMICAL_REFERENCES_PATH": os.path.join(tmpdir, "missing_references.json"),
        }):
            with patch("lib.s2_dapt.dapt.AutoTokenizer.from_pretrained") as mock_from_token, \
                 patch("lib.s2_dapt.dapt.AutoModelForCausalLM.from_pretrained") as mock_from_model:
                mock_from_token.return_value = mock_tokenizer
                mock_from_model.return_value = mock_model
                
                with pytest.raises(FileNotFoundError) as exc_info:
                    run_dapt_pipeline(
                        model_name="dummy-model",
                        corpus_path=corpus_path,
                        probe_qa_path=probe_qa_path,
                        epochs=1,
                        lr=1e-5,
                        batch_size=1,
                        output_dir=output_dir
                    )
                assert "Required evaluation files are missing" in str(exc_info.value)

