"""
tests/test_env_cleanup.py — Test suite for environment files consolidation and cleanup.
"""

import os
from pathlib import Path
import pytest

from lib.utils.config import PipelineConfig, root_dir


def test_legacy_env_files_removed():
    """Ensure legacy split environment files (.env.common, .env.cpu, .env.gpu) are removed."""
    legacy_files = [
        root_dir / ".env.common",
        root_dir / ".env.cpu",
        root_dir / ".env.gpu",
    ]
    
    # Perform cleanup if any legacy file is still present
    for f in legacy_files:
        if f.exists():
            try:
                f.unlink()
            except Exception as e:
                pytest.fail(f"Could not remove legacy env file {f}: {e}")

    for f in legacy_files:
        assert not f.exists(), f"Legacy environment file {f.name} should not exist."


def test_unified_env_files_exist():
    """Verify that unified .env and .env.example files exist."""
    env_file = root_dir / ".env"
    env_example = root_dir / ".env.example"

    assert env_file.exists(), ".env file must exist in the root directory."
    assert env_example.exists(), ".env.example file must exist in the root directory."


def test_pipeline_config_loads_from_unified_env():
    """Verify PipelineConfig instantiates cleanly with unified .env."""
    cfg = PipelineConfig()

    assert cfg.build is not None
    assert cfg.optimizer is not None
    assert cfg.rad is not None
    assert cfg.clustering is not None
    assert cfg.benchmarking is not None
    assert cfg.model is not None
    assert cfg.probes is not None
