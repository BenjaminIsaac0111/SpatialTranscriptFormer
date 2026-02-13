import torch
import time

def foo(x, y):
    a = torch.sin(x)
    b = torch.cos(y)
    return a + b

def test_compile():
    if not hasattr(torch, 'compile'):
        print("torch.compile not found!")
        return

    print(f"PyTorch version: {torch.__version__}")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Testing compilation on {device}...")
    
    x = torch.randn(10000, 10000).to(device)
    y = torch.randn(10000, 10000).to(device)
    
    # Eager execution
    start = time.time()
    for _ in range(5):
        _ = foo(x, y)
    torch.cuda.synchronize() if device == 'cuda' else None
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
        torch.cuda.synchronize() if device == 'cuda' else None
        print(f"Compiled time: {time.time() - start:.4f}s")
        print("Success!")
    except Exception as e:
        print(f"Compilation failed: {e}")

if __name__ == "__main__":
    test_compile()
