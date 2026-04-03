"""
Tests for torch.compile compatibility and environment validation.
"""

import time
import torch
import pytest
from spatial_transcript_former.models.interaction import SpatialTranscriptFormer


def test_torch_compile_feature_available():
    """Verify that the current environment supports torch.compile."""
    if not hasattr(torch, "compile"):
        pytest.skip("torch.compile not found in this PyTorch version.")


def test_generic_function_compilation():
    """Verify that a simple function can be compiled and executed."""
    if not hasattr(torch, "compile"):
        pytest.skip("torch.compile not found.")

    def simple_op(x, y):
        return torch.sin(x) + torch.cos(y)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    x = torch.randn(100, 100).to(device)
    y = torch.randn(100, 100).to(device)

    try:
        opt_op = torch.compile(simple_op)
        # Warmup
        _ = opt_op(x, y)
    except RuntimeError as e:
        if "not supported on Python 3.14" in str(e):
            pytest.skip(f"torch.compile not yet supported on this Python version: {e}")
        raise e

    # Execute
    out = opt_op(x, y)
    assert out.shape == (100, 100)


def test_model_compilation_sanity():
    """
    Verify that the SpatialTranscriptFormer model can be compiled.
    This is an important environment check for production/training performance.
    """
    if not hasattr(torch, "compile"):
        pytest.skip("torch.compile not found.")

    # Using a small model for fast compilation check
    model = SpatialTranscriptFormer(
        token_dim=64, n_heads=2, n_layers=1, pretrained=False
    ).eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    try:
        compiled_model = torch.compile(model)

        # Mock input
        features = torch.randn(1, 4, 2048).to(device)
        coords = torch.randn(1, 4, 2).to(device)

        # Warmup (this is where most errors occur)
        with torch.no_grad():
            _ = compiled_model(features, rel_coords=coords)

        print("Model compilation successful!")
    except RuntimeError as e:
        if "not supported on Python 3.14" in str(e):
            pytest.skip(f"torch.compile not supported on Python 3.14: {e}")
        pytest.fail(
            f"SpatialTranscriptFormer compilation failed with RuntimeError: {e}"
        )
    except Exception as e:
        pytest.fail(f"SpatialTranscriptFormer compilation failed: {e}")


@pytest.mark.parametrize("backend", ["eager", "inductor"])
def test_compile_backends(backend):
    """Test specific compilation backends for availability."""
    if not hasattr(torch, "compile"):
        pytest.skip("torch.compile not found.")

    def simple_op(x):
        return x * 2

    device = "cuda" if torch.cuda.is_available() else "cpu"
    x = torch.randn(10).to(device)

    try:
        opt_op = torch.compile(simple_op, backend=backend)
        _ = opt_op(x)
    except Exception as e:
        # Some backends might not be supported on all OS/HW combinations
        pytest.skip(f"Backend '{backend}' not supported: {e}")
