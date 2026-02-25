import pytest
import warnings


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
