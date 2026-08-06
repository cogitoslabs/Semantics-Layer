import os
import tempfile
from pathlib import Path
import pytest

from lib.utils.config import resolve_local_model_path, ModelConfig, ProbeConfig, PipelineConfig


def test_resolve_local_model_path_not_found():
    # If the model does not exist locally, it should fall back to the original model name.
    assert resolve_local_model_path("non_existent_model/test-123") == "non_existent_model/test-123"
    assert resolve_local_model_path("") == ""


def test_resolve_local_model_path_exact_and_case_insensitive():
    # Use a temporary directory as the mock root_dir
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        models_dir = tmp_path / "models"
        models_dir.mkdir()

        # Create some mock local model directories
        scibert_dir = models_dir / "scibert_scivocab_uncased"
        scibert_dir.mkdir()
        (scibert_dir / "config.json").write_text("{}")
        
        smollm_dir = models_dir / "smollm2-135m"
        smollm_dir.mkdir()
        (smollm_dir / "config.json").write_text("{}")

        # Create an incomplete directory (missing config.json)
        incomplete_dir = models_dir / "incomplete_model"
        incomplete_dir.mkdir()

        # Patch the root_dir in config module to point to our temp directory
        import lib.utils.config as config_mod
        orig_root_dir = config_mod.root_dir
        config_mod.root_dir = tmp_path

        try:
            # 1. Test exact match resolution
            resolved = resolve_local_model_path("allenai/scibert_scivocab_uncased")
            assert resolved == str(scibert_dir.resolve())

            # 2. Test lowercase/case-insensitive resolution
            resolved_smollm = resolve_local_model_path("HuggingFaceTB/SmolLM2-135M")
            assert resolved_smollm == str(smollm_dir.resolve())

            # 3. Test exact match resolution directly (already an existing path)
            assert resolve_local_model_path(str(scibert_dir)) == str(scibert_dir.resolve())

            # 4. Test incomplete folder resolution fallback (must NOT resolve because config.json is missing)
            assert resolve_local_model_path("some_user/incomplete_model") == "some_user/incomplete_model"

        finally:
            # Restore the original root_dir
            config_mod.root_dir = orig_root_dir


def test_config_dataclass_post_init_resolution():
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        models_dir = tmp_path / "models"
        models_dir.mkdir()

        # Create mock directories
        scibert_dir = models_dir / "scibert_scivocab_uncased"
        scibert_dir.mkdir()
        (scibert_dir / "config.json").write_text("{}")
        
        smollm_dir = models_dir / "smollm2-135m"
        smollm_dir.mkdir()
        (smollm_dir / "config.json").write_text("{}")


        import lib.utils.config as config_mod
        orig_root_dir = config_mod.root_dir
        config_mod.root_dir = tmp_path

        try:
            # When ModelConfig/ProbeConfig are instantiated, they should automatically resolve
            model_cfg = ModelConfig(base_model_name="HuggingFaceTB/SmolLM2-135M")
            assert model_cfg.base_model_name == str(smollm_dir.resolve())

            probe_cfg = ProbeConfig(bertscore_model="allenai/scibert_scivocab_uncased")
            assert probe_cfg.bertscore_model == str(scibert_dir.resolve())

            # When PipelineConfig is instantiated, it should compose resolved configs
            # Mock environment variables for config loading
            os.environ["BASE_MODEL_NAME"] = "HuggingFaceTB/SmolLM2-135M"
            os.environ["BERTSCORE_MODEL"] = "allenai/scibert_scivocab_uncased"
            
            pipeline_cfg = PipelineConfig()
            
            # Since the sub-configs are instantiated during PipelineConfig instantiation,
            # they should automatically call resolve_local_model_path
            assert pipeline_cfg.model.base_model_name == str(smollm_dir.resolve())
            assert pipeline_cfg.probes.bertscore_model == str(scibert_dir.resolve())
            
        finally:
            config_mod.root_dir = orig_root_dir
            if "BASE_MODEL_NAME" in os.environ:
                del os.environ["BASE_MODEL_NAME"]
            if "BERTSCORE_MODEL" in os.environ:
                del os.environ["BERTSCORE_MODEL"]


def test_gradient_checkpointing_config():
    # 1. Test default value is False
    model_cfg = ModelConfig()
    assert model_cfg.gradient_checkpointing is False

    # 2. Test environment variable override
    os.environ["GRADIENT_CHECKPOINTING"] = "True"
    try:
        model_cfg_override = ModelConfig()
        assert model_cfg_override.gradient_checkpointing is True
    finally:
        del os.environ["GRADIENT_CHECKPOINTING"]


def test_pipeline_config_summary():
    cfg = PipelineConfig()
    summary_text = cfg.summary()
    assert "PEFT DAPT" in summary_text
    assert "Train batch size" in summary_text
    assert "Eval batch size" in summary_text
    assert "Grad checkpoint" in summary_text


def test_rad_reranker_config_validation():
    cfg = PipelineConfig()
    cfg.rad.top_k = 10
    cfg.rad.rerank_candidate_k = 5  # invalid: candidate_k < top_k
    with pytest.raises(ValueError, match="RAD_RERANK_CANDIDATE_K"):
        cfg.validate()

    cfg.rad.rerank_candidate_k = 20
    cfg.rad.reranker_batch_size = 0  # invalid: batch_size < 1
    with pytest.raises(ValueError, match="RAD_RERANKER_BATCH_SIZE"):
        cfg.validate()


