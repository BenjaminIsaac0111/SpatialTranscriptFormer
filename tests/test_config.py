import os
import pytest
from spatial_transcript_former.config import ProjectConfig, get_config


def test_config_loading():
    """Verify that config.yaml is loaded and values are accessible."""
    # Check if we can get a default value
    data_dirs = get_config("data_dirs")
    assert isinstance(data_dirs, list)
    assert len(data_dirs) > 0

    # Check dot notation
    lr = get_config("training.learning_rate")
    assert isinstance(lr, float)
    assert lr == 0.0001


def test_config_fallback():
    """Verify fallback behavior for missing keys."""
    val = get_config("missing.key", default="fallback")
    assert val == "fallback"


def test_config_singleton():
    """Verify that ProjectConfig maintains state."""
    # Force load
    ProjectConfig.load()
    assert ProjectConfig._loaded

    # Change a value manually (for testing purposes)
    ProjectConfig._config["test_val"] = 123
    assert get_config("test_val") == 123
