"""
Merged tests: test_compile.py, test_compile_backends.py
"""

import time

import torch
import pytest

# --- From test_compile.py ---


def foo(x, y):
    a = torch.sin(x)
    b = torch.cos(y)
    return a + b


def test_compile():
    if not hasattr(torch, "compile"):
        print("torch.compile not found!")
        return

    print(f"PyTorch version: {torch.__version__}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Testing compilation on {device}...")

    x = torch.randn(10000, 10000).to(device)
    y = torch.randn(10000, 10000).to(device)

    # Eager execution
    start = time.time()
    for _ in range(5):
        _ = foo(x, y)
    torch.cuda.synchronize() if device == "cuda" else None
    print(f"Eager time: {time.time() - start:.4f}s")

    # Compile
    print("Compiling...")
    try:
        opt_foo = torch.compile(foo)

        # Warmup
        _ = opt_foo(x, y)

        # Compiled execution
        start = time.time()
        for _ in range(5):
            _ = opt_foo(x, y)
        torch.cuda.synchronize() if device == "cuda" else None
        print(f"Compiled time: {time.time() - start:.4f}s")
        print("Success!")
    except Exception as e:
        print(f"Compilation failed: {e}")


# --- From test_compile_backends.py ---


def foo(x, y):
    a = torch.sin(x)
    b = torch.cos(y)
    return a + b


@pytest.mark.parametrize("backend", ["cudagraphs", "eager"])
def test_compile_backend(backend):
    if not hasattr(torch, "compile"):
        print("torch.compile not found!")
        return

    print(f"\nTesting compilation with backend='{backend}'...")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    x = torch.randn(10000, 10000).to(device)
    y = torch.randn(10000, 10000).to(device)

    try:
        opt_foo = torch.compile(foo, backend=backend)

        # Warmup
        _ = opt_foo(x, y)

        # Compiled execution
        start = time.time()
        for _ in range(5):
            _ = opt_foo(x, y)
        torch.cuda.synchronize() if device == "cuda" else None
        print(f"Backend '{backend}' time: {time.time() - start:.4f}s")
        print("Success!")
    except Exception as e:
        print(f"Backend '{backend}' failed: {e}")
