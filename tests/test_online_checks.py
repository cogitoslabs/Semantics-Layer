import sys
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from unittest.mock import MagicMock, patch
import pytest

from lib.utils.config import PipelineConfig
from scripts.online_cloze_check import run_cloze_check, main as main_cloze
from scripts.online_concept_check import run_concept_check, main as main_concept



@pytest.fixture
def mock_model_and_tokenizer():
    mock_tokenizer = MagicMock()
    mock_tokenizer.eos_token_id = 2
    mock_tokenizer.bos_token_id = 1
    mock_tokenizer.pad_token_id = 0
    mock_tokenizer.encode.return_value = [10]
    
    # Setup tokenizer return value for input_ids
    mock_input_ids = MagicMock()
    mock_input_ids.shape = [1, 5]
    mock_tokenizer.return_value = {
        "input_ids": mock_input_ids,
        "attention_mask": MagicMock()
    }
    mock_tokenizer.decode.return_value = "dopamine"

    mock_model = MagicMock()
    mock_model.generate.return_value = MagicMock()

    return mock_model, mock_tokenizer


def test_run_cloze_check(mock_model_and_tokenizer, capsys):
    mock_model, mock_tokenizer = mock_model_and_tokenizer
    cfg = PipelineConfig()
    device = MagicMock()
    device.type = "cpu"

    run_cloze_check("A neurotransmitter involved in reward is ___", mock_model, mock_tokenizer, cfg, device)

    captured = capsys.readouterr()
    assert "Input Prompt" in captured.out
    assert "Model Top-" in captured.out
    assert "dopamine" in captured.out
    mock_model.generate.assert_called_once()


def test_run_concept_check(mock_model_and_tokenizer, capsys):
    mock_model, mock_tokenizer = mock_model_and_tokenizer
    cfg = PipelineConfig()
    device = MagicMock()
    device.type = "cpu"

    run_concept_check("Explain synaptic plasticity", mock_model, mock_tokenizer, cfg, device)

    captured = capsys.readouterr()
    assert "Input Prompt: Explain synaptic plasticity" in captured.out
    assert "Model Response:" in captured.out
    assert "dopamine" in captured.out
    mock_model.generate.assert_called_once()


def test_cloze_check_cli(mock_model_and_tokenizer):
    mock_model, mock_tokenizer = mock_model_and_tokenizer

    with patch("sys.argv", ["online_cloze_check.py", "--prompt", "Test cloze prompt ___"]), \
         patch("scripts.online_cloze_check.load_model_and_tokenizer", return_value=(mock_model, mock_tokenizer)):
        main_cloze()

    mock_model.generate.assert_called_once()


def test_concept_check_cli(mock_model_and_tokenizer):
    mock_model, mock_tokenizer = mock_model_and_tokenizer

    with patch("sys.argv", ["online_concept_check.py", "-p", "Test concept prompt"]), \
         patch("scripts.online_concept_check.load_model_and_tokenizer", return_value=(mock_model, mock_tokenizer)):
        main_concept()

    mock_model.generate.assert_called_once()
