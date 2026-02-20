import torch
import torch.nn as nn
from spatial_transcript_former.models.interaction import SpatialTranscriptFormer
import time
import psutil
import os

def print_memory_usage():
    process = psutil.Process(os.getpid())
    print(f"RAM Usage: {process.memory_info().rss / 1024**2:.2f} MB")
    if torch.cuda.is_available():
        print(f"GPU Memory: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
        print(f"GPU Max Memory: {torch.cuda.max_memory_allocated() / 1024**2:.2f} MB")

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Testing on {device}")
    
    # Model config closely matching training
    model = SpatialTranscriptFormer(
        num_genes=1000,
        num_pathways=50,
        backbone_name='ctranspath',
        pretrained=False, # Speed up initialization
        token_dim=512,
        n_heads=8,
        n_layers=2,
        use_nystrom=True
    ).to(device)
    
    print("Model initialized.")
    print_memory_usage()
    
    # Simulate WSI input
    # B=1, N=4000 patches, D=768 (CTransPath dim)
    N = 4000
    D = 768
    print(f"Creating dummy input: (1, {N}, {D})")
    x = torch.randn(1, N, D).to(device)
    
    print_memory_usage()
    
    print("Running forward_dense...")
    start_time = time.time()
    
    with torch.no_grad():
        with torch.amp.autocast('cuda'):
             output = model.forward_dense(x)
             
    end_time = time.time()
    print(f"Forward pass completed in {end_time - start_time:.4f} seconds.")
    print(f"Output shape: {output.shape}")
    print_memory_usage()
    
    # Check if Nystrom is actually being used
    # This is implicit if it doesn't OOM with N=10000+ but 4000 is a safe start
    
    # Try larger N
    del x
    del output
    torch.cuda.empty_cache()
    
    N_large = 10000
    print(f"\nTesting larger input: (1, {N_large}, {D})")
    try:
        x = torch.randn(1, N_large, D).to(device)
        with torch.no_grad():
             with torch.amp.autocast('cuda'):
                 output = model.forward_dense(x)
        print(f"Large forward pass successful! Output shape: {output.shape}")
    except RuntimeError as e:
        print(f"OOM or Error: {e}")
    
    print_memory_usage()

if __name__ == "__main__":
    main()
