import json
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest
import torch
import numpy as np

from lib.utils import PipelineConfig
from lib.s2_dapt.dapt import evaluate_perplexity, evaluate_qa_accuracy, run_dapt_pipeline
from lib.s1_5_pretokenize import run_pretokenization


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
        retrieval_prompts_path = os.path.join(tmpdir, "retrieval_prompts.json")
        retrieval_references_path = os.path.join(tmpdir, "retrieval_references.json")
        
        # Write dummy files
        with open(corpus_path, "w") as f:
            f.write(json.dumps({"text": "document 1"}) + "\n")
            f.write(json.dumps({"text": "document 2"}) + "\n")
            f.write(json.dumps({"text": "document 3"}) + "\n")
            
        with open(probe_qa_path, "w") as f:
            f.write(json.dumps({"question": "Q1", "choices": ["A", "B"], "answer_idx": 0}) + "\n")
 
        with open(ppl_corpus_path, "w") as f:
            f.write("dummy ppl text")
        with open(vocab_cloze_path, "w") as f:
            json.dump([{"prompt": "dummy prompt", "target_term": "dummy term", "category": "dummy"}], f)
        with open(retrieval_prompts_path, "w") as f:
            json.dump(["dummy prompt"], f)
        with open(retrieval_references_path, "w") as f:
            json.dump(["dummy reference"], f)
            
        pretokenized_bin_path = os.path.join(tmpdir, "train_tokens.npy")
        np.save(pretokenized_bin_path, np.arange(1000, dtype=np.int32))
            
        # Patch transformers from_pretrained calls and run_all_probes
        with patch("lib.s2_dapt.model_utils.AutoTokenizer.from_pretrained") as mock_from_token, \
             patch("lib.s2_dapt.model_utils.AutoModelForCausalLM.from_pretrained") as mock_from_model, \
             patch("lib.s2_dapt.dapt.run_all_probes") as mock_run_probes, \
             patch.dict(os.environ, {
                 "PPL_CORPUS_PATH": ppl_corpus_path,
                 "VOCAB_CLOZE_PATH": vocab_cloze_path,
                 "RETRIEVAL_PROMPTS_PATH": retrieval_prompts_path,
                 "RETRIEVAL_REFERENCES_PATH": retrieval_references_path,
                 "PRETOKENIZED_BIN_PATH": pretokenized_bin_path,
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
                
                cfg = PipelineConfig()
                cfg.model.base_model_name = "dummy-model"
                cfg.model.max_seq_len = 512
                cfg.build.output_path = Path(corpus_path)
                cfg.data.qa_probe_path = Path(probe_qa_path)
                cfg.corpus.max_corpus_passes = 1
                cfg.optimizer.learning_rate = 1e-5
                cfg.optimizer.train_batch_size = 1
                cfg.storage.checkpoint_dir = Path(output_dir)
                
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
        ppl_corpus_path = os.path.join(tmpdir, "ppl_held_out.txt")
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
 
        with open(ppl_corpus_path, "w") as f:
            f.write("dummy ppl text")
        with open(vocab_cloze_path, "w") as f:
            json.dump([{"prompt": "dummy prompt", "target_term": "dummy term", "category": "dummy"}], f)
        with open(retrieval_prompts_path, "w") as f:
            json.dump(["dummy prompt"], f)
        with open(retrieval_references_path, "w") as f:
            json.dump(["dummy reference"], f)
            
        pretokenized_bin_path = os.path.join(tmpdir, "train_tokens.npy")
        np.save(pretokenized_bin_path, np.arange(1000, dtype=np.int32))
            
        # Patch transformers from_pretrained calls, wandb methods, and run_all_probes
        with patch("lib.s2_dapt.model_utils.AutoTokenizer.from_pretrained") as mock_from_token, \
             patch("lib.s2_dapt.model_utils.AutoModelForCausalLM.from_pretrained") as mock_from_model, \
             patch("lib.s2_dapt.dapt.run_all_probes") as mock_run_probes, \
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
            
            cfg = PipelineConfig()
            cfg.model.base_model_name = "dummy-model"
            cfg.model.max_seq_len = 512
            cfg.build.output_path = Path(corpus_path)
            cfg.data.qa_probe_path = Path(probe_qa_path)
            cfg.corpus.max_corpus_passes = 1
            cfg.optimizer.learning_rate = 1e-5
            cfg.optimizer.train_batch_size = 1
            cfg.storage.checkpoint_dir = Path(output_dir)
            
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
        }):
            with patch("lib.s2_dapt.model_utils.AutoTokenizer.from_pretrained") as mock_from_token, \
                 patch("lib.s2_dapt.model_utils.AutoModelForCausalLM.from_pretrained") as mock_from_model:
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
                    cfg.storage.checkpoint_dir = Path(output_dir)
                    
                    run_dapt_pipeline(cfg)
                assert "Required evaluation files are missing" in str(exc_info.value)


def test_run_pretokenization(mock_tokenizer):
    with tempfile.TemporaryDirectory() as tmpdir:
        corpus_path = os.path.join(tmpdir, "corpus.jsonl")
        pretokenized_bin_path = os.path.join(tmpdir, "train_tokens.npy")
        ppl_corpus_path = os.path.join(tmpdir, "ppl_held_out.txt")
        
        # Write a dummy corpus
        with open(corpus_path, "w") as f:
            f.write(json.dumps({"text": "document 1"}) + "\n")
            f.write(json.dumps({"text": "document 2"}) + "\n")
            f.write(json.dumps({"text": "document 3"}) + "\n")
            
        with patch("lib.s1_5_pretokenize.pretokenize.AutoTokenizer.from_pretrained") as mock_from_token:
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


def test_eval_qa_accuracy_new_format(mock_model, mock_tokenizer):
    from lib.s2_dapt.probes.qa_probe import eval_qa_accuracy
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
        # Determine sequence length based on prompt or choice addition
        if "Answer:" in text and not any(text.endswith(c) for c in ["Serotonin", "Dopamine", "GABA", "Acetylcholine"]):
            # Prompt only: 3 tokens
            tokens = [1, 2, 3]
        else:
            # Prompt + choice: 5 tokens
            tokens = [1, 2, 3, 4, 5]
        return {
            "input_ids": torch.tensor([tokens]),
            "attention_mask": torch.tensor([[1] * len(tokens)])
        }
    
    mock_tokenizer.side_effect = mock_tokenize_fn
    mock_tokenizer.encode.return_value = [1, 2, 3]
    
    # Let's count how many times model is called to differentiate the choices
    call_count = 0
    
    def mock_model_forward(*args, **kwargs):
        nonlocal call_count
        logits = torch.zeros((1, 5, 10))
        if call_count == 1:
            logits[0, 2, 4] = 10.0
            logits[0, 3, 5] = 10.0
        else:
            logits[0, 2, 4] = 1.0
            logits[0, 3, 5] = 1.0
        
        call_count += 1
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


def test_eval_terminology_coverage_empty_prompt(mock_model, mock_tokenizer):
    from lib.s2_dapt.probes.terminology_probe import eval_terminology_coverage
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
            
        result = eval_terminology_coverage(
            model=mock_model,
            tokenizer=mock_tokenizer,
            vocab_cloze_path=vocab_cloze_path,
            top_k=1,
            max_new_tokens=5,
            device="cpu"
        )
        
        assert result["total"] == 2


def test_eval_retrieval_precision_empty_prompt(mock_model, mock_tokenizer):
    from lib.s2_dapt.probes.retrieval_probe import eval_retrieval_precision
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
            
        result = eval_retrieval_precision(
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
            "RUN_TERMINOLOGY_PROBE": "False",
            "RUN_RETRIEVAL_PROBE": "False",
            "PPL_CORPUS_PATH": os.path.join(tmpdir, "missing_ppl.txt"),
            "VOCAB_CLOZE_PATH": os.path.join(tmpdir, "missing_vocab.json"),
            "RETRIEVAL_PROMPTS_PATH": os.path.join(tmpdir, "missing_prompts.json"),
            "RETRIEVAL_REFERENCES_PATH": os.path.join(tmpdir, "missing_references.json"),
            "PRETOKENIZED_BIN_PATH": pretokenized_bin_path,
        }):
            with patch("lib.s2_dapt.model_utils.AutoTokenizer.from_pretrained") as mock_from_token, \
                 patch("lib.s2_dapt.model_utils.AutoModelForCausalLM.from_pretrained") as mock_from_model, \
                 patch("lib.s2_dapt.dapt.run_all_probes") as mock_run_probes:
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
                        "term_coverage": 0.0,
                        "retrieval_precision": 0.0,
                    }
                }
                
                cfg = PipelineConfig()
                cfg.model.base_model_name = "dummy-model"
                cfg.model.max_seq_len = 512
                cfg.build.output_path = Path(corpus_path)
                cfg.data.qa_probe_path = Path(probe_qa_path)
                cfg.corpus.max_corpus_passes = 1
                cfg.optimizer.learning_rate = 1e-5
                cfg.optimizer.train_batch_size = 1
                cfg.storage.checkpoint_dir = Path(output_dir)
                
                # Should not raise FileNotFoundError because disabled files are bypassed
                run_dapt_pipeline(cfg)
                assert mock_model.save_pretrained.called


def test_check_convergence_gates_disabled_probes():
    from lib.s2_dapt.evaluation.gate_logic import check_convergence_gates, DAPTDecision
    
    state = {
        "tokens_processed": 100,
        "perplexity_history": [],
        "qa_acc_history": [0.0],
        "term_cov_history": [0.0],
        "ret_prec_history": [0.0],
    }
    
    # QA and PPL are disabled, secondary are enabled but have 0.0 metrics.
    # Since secondary requires at least one, and both are enabled but fail (0.0 < threshold),
    # convergence is not met yet (returns CONTINUE).
    decision, gate_details = check_convergence_gates(
        state=state,
        qa_acc_threshold=0.55,
        ppl_improvement_threshold=2.0,
        ppl_plateau_window=2,
        term_cov_threshold=0.80,
        ret_prec_threshold=0.60,
        hard_stop_tokens=1000,
        total_corpus_tokens=500,
        run_qa=False,
        run_perplexity=False,
        run_terminology=True,
        run_retrieval=True,
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
        term_cov_threshold=0.80,
        ret_prec_threshold=0.60,
        hard_stop_tokens=1000,
        total_corpus_tokens=500,
        run_qa=False,
        run_perplexity=False,
        run_terminology=False,
        run_retrieval=False,
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
                "term_coverage": 0.1,
                "retrieval_precision": 0.0,
                "perplexity": 10.0,
            }
        },
        {
            "eval_id": 2,
            "metrics": {
                "qa_accuracy": 0.0,
                "term_coverage": 0.9,
                "retrieval_precision": 0.0,
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
        
        # QA is disabled, so we fall back to Terminology coverage (run_terminology=True)
        best_ckpt = select_best_checkpoint(
            eval_history=eval_history,
            checkpoint_dir=checkpoint_dir,
            ppl_improvement_threshold=2.0,
            manifest_path=manifest_path,
            run_qa=False,
            run_perplexity=False,
            run_terminology=True,
            run_retrieval=False,
        )
        
        # Should select eval 2 because term_coverage is 0.9 > 0.1
        assert best_ckpt.name == "dapt_eval_0002.pt"




