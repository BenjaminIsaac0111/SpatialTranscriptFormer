import torch
import time
import pytest

def foo(x, y):
    a = torch.sin(x)
    b = torch.cos(y)
    return a + b

@pytest.mark.parametrize("backend", ["cudagraphs", "eager"])
def test_compile_backend(backend):
    if not hasattr(torch, 'compile'):
        print("torch.compile not found!")
        return

    print(f"\nTesting compilation with backend='{backend}'...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
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
        torch.cuda.synchronize() if device == 'cuda' else None
        print(f"Backend '{backend}' time: {time.time() - start:.4f}s")
        print("Success!")
    except Exception as e:
        print(f"Backend '{backend}' failed: {e}")

if __name__ == "__main__":
    # Test valid backends on Windows
    # 'inductor' failed (needs triton)
    # 'cudagraphs' might work
    # 'eager' is just pass-through
    for backend in ['cudagraphs', 'eager']:
        test_compile_backend(backend)
