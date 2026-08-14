import sys
import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from lib.utils.config import PipelineConfig
from ui.online_probes import run_probe_logic, list_available_checkpoints, load_cached_model, render_probe_output


@pytest.fixture
def mock_model_and_tokenizer():
    mock_tokenizer = MagicMock()
    mock_tokenizer.eos_token_id = 2
    mock_tokenizer.bos_token_id = 1
    mock_tokenizer.pad_token_id = 0
    mock_tokenizer.encode.return_value = [10]

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


def test_ui_run_cloze_probe_logic(mock_model_and_tokenizer):
    mock_model, mock_tokenizer = mock_model_and_tokenizer
    cfg = PipelineConfig()
    device = MagicMock()
    device.type = "cpu"

    result = run_probe_logic("Cloze Probe", "Test ___", mock_model, mock_tokenizer, cfg, device)
    assert result["type"] == "cloze"
    assert "completions" in result
    assert result["top_k"] == cfg.probes.cloze_top_k
    mock_model.generate.assert_called_once()


def test_ui_run_concept_probe_logic(mock_model_and_tokenizer):
    mock_model, mock_tokenizer = mock_model_and_tokenizer
    cfg = PipelineConfig()
    device = MagicMock()
    device.type = "cpu"

    result = run_probe_logic("Concept Probe", "Explain synapses", mock_model, mock_tokenizer, cfg, device)
    assert result["type"] == "concept"
    assert result["response"] == "dopamine"
    mock_model.generate.assert_called_once()


def test_list_available_checkpoints():
    test_dir = root_dir / "tests" / ".tmp_ckpt_test"
    if test_dir.exists():
        shutil.rmtree(test_dir, ignore_errors=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    try:
        assert list_available_checkpoints(test_dir / "non_existent") == []
        
        ckpt_dir = test_dir / "checkpoints"
        ckpt_dir.mkdir()
        assert list_available_checkpoints(ckpt_dir) == []
        
        f1 = ckpt_dir / "dapt_eval_0001.pt"
        f2 = ckpt_dir / "dapt_eval_0002.pt"
        f1.write_text("mock1")
        f2.write_text("mock2")
        
        ckpts = list_available_checkpoints(ckpt_dir)
        assert len(ckpts) == 2
        assert {c.name for c in ckpts} == {"dapt_eval_0001.pt", "dapt_eval_0002.pt"}
    finally:
        shutil.rmtree(test_dir, ignore_errors=True)


def test_ui_run_qa_probe_logic(mock_model_and_tokenizer):
    mock_model, mock_tokenizer = mock_model_and_tokenizer
    cfg = PipelineConfig()
    device = MagicMock()
    device.type = "cpu"

    with patch("ui.online_probes.score_choices_by_logprob") as mock_score:
        mock_score.return_value = 0
        choices = ["Hippocampus", "Cerebellum", "Occipital lobe", "Medulla"]
        result = run_probe_logic("QA Probe", "What forms memory?", mock_model, mock_tokenizer, cfg, device, choices=choices)
        
        assert result["type"] == "qa"
        assert result["predicted_idx"] == 0
        assert result["predicted_choice"] == "Hippocampus"
        mock_score.assert_called_once()


def test_ui_run_qa_probe_logic_empty_choices(mock_model_and_tokenizer):
    mock_model, mock_tokenizer = mock_model_and_tokenizer
    cfg = PipelineConfig()
    device = MagicMock()
    device.type = "cpu"

    with pytest.raises(ValueError, match="No valid choices provided"):
        run_probe_logic("QA Probe", "Question?", mock_model, mock_tokenizer, cfg, device, choices=[])


def test_render_probe_output_cloze():
    result = {
        "type": "cloze",
        "formatted_prompt": "Prompt: Test\nAnswer:",
        "completions": ["acetylcholine", "dopamine"],
        "top_k": 2,
    }
    with patch("streamlit.markdown") as mock_md, patch("streamlit.info") as mock_info, patch("streamlit.expander"):
        render_probe_output(result)
        mock_md.assert_called()
        assert mock_info.call_count == 2


def test_render_probe_output_qa():
    result = {
        "type": "qa",
        "question": "Sample Q",
        "formatted_prompt": "Question: Sample Q\nAnswer:",
        "choices": ["Choice 1", "Choice 2"],
        "predicted_idx": 0,
        "predicted_choice": "Choice 1",
    }
    with patch("streamlit.success") as mock_success, patch("streamlit.markdown") as mock_md, patch("streamlit.expander"):
        render_probe_output(result)
        mock_success.assert_called_once()
        mock_md.assert_called()


def test_render_probe_output_concept():
    result = {
        "type": "concept",
        "response": "Detailed explanation...",
        "max_new_tokens": 100,
    }
    with patch("streamlit.success") as mock_success, patch("streamlit.markdown") as mock_md:
        render_probe_output(result)
        mock_success.assert_called_once()
        mock_md.assert_called()
