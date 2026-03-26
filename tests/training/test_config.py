"""
Merged tests: test_config.py, test_warnings.py
"""

import os
import warnings

import pytest

from spatial_transcript_former.config import ProjectConfig, get_config

# --- From test_config.py ---


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


# --- From test_warnings.py ---


def function_that_warns():
    warnings.warn("This is a deprecated feature", DeprecationWarning)
    return True


def function_that_warns_user():
    warnings.warn("FigureCanvasAgg is non-interactive", UserWarning)
    return True


def test_demonstrate_warning_assertion():
    """
    Demonstrates how to assert that a specific warning is raised.
    This is useful for ensuring your code warns users correctly.
    """
    with pytest.warns(DeprecationWarning, match="deprecated feature"):
        result = function_that_warns()
        assert result is True


def test_global_filter_demonstration():
    """
    This test will pass without showing warnings in the output
    because we added filters to pyproject.toml.

    Specifically 'FigureCanvasAgg is non-interactive' is filtered.
    """
    result = function_that_warns_user()
    assert result is True


def test_how_to_catch_and_ignore_locally():
    """
    If you want to ignore a warning locally in a specific test
    without adding it to the global pyproject.toml.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = function_that_warns()
        assert result is True
