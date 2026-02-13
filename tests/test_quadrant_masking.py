import torch
import torch.nn as nn
import sys
import os

# Add src to path
sys.path.append(os.path.abspath('src'))

from spatial_transcript_former.models.interaction import MultimodalFusion

def test_quadrant_masking_interactions():
    dim = 64
    n_heads = 4
    n_layers = 1
    Np = 5
    Nh = 10
    B = 2
    
    # Mode 1: Full Attention
    fusion_full = MultimodalFusion(dim, n_heads, n_layers)
    
    # Mode 2: Mask H2H
    fusion_masked = MultimodalFusion(dim, n_heads, n_layers, masked_quadrants=['H2H'])
    
    p_tokens = torch.randn(B, Np, dim, requires_grad=True)
    h_tokens = torch.randn(B, Nh, dim, requires_grad=True)
    
    # Forward full
    out_full = fusion_full(p_tokens, h_tokens)
    loss_full = out_full.sum()
    loss_full.backward()
    
    h_grad_full = h_tokens.grad.abs().sum()
    print(f"H tokens grad (Full): {h_grad_full.item()}")
    
    # Reset grad
    h_tokens.grad.zero_()
    p_tokens.grad.zero_()
    
    # Forward masked H2H
    # NOTE: Masking H2H in self-attention technically means H tokens don't attend to H tokens.
    # But H tokens can still attend to P tokens (H2P) unless we mask that too.
    # If we mask H2H, H tokens are purely transformed by their interaction with P tokens.
    out_masked = fusion_masked(p_tokens, h_tokens)
    loss_masked = out_masked.sum()
    loss_masked.backward()
    
    h_grad_masked = h_tokens.grad.abs().sum()
    print(f"H tokens grad (Masked H2H): {h_grad_masked.item()}")
    
    # Forward masked H2H AND H2P (H tokens isolated)
    h_tokens.grad.zero_()
    fusion_isolated = MultimodalFusion(dim, n_heads, n_layers, masked_quadrants=['H2H', 'H2P'])
    out_iso = fusion_isolated(p_tokens, h_tokens)
    out_iso.sum().backward()
    
    h_grad_iso = h_tokens.grad.abs().sum()
    print(f"H tokens grad (H2H + H2P Masked): {h_grad_iso.item()}")
    
    # In H2H + H2P masked mode, output depends on P tokens attending to H and P.
    # H tokens shouldn't receive any gradients from the output if they are isolated 
    # and the output only contains the transformed P tokens.
    # WAIT: P tokens STILL attend to H tokens (P2H). So H tokens WILL get gradients.
    
    # Let's verify P2H masking
    h_tokens.grad.zero_()
    fusion_no_hist = MultimodalFusion(dim, n_heads, n_layers, masked_quadrants=['P2H'])
    out_no_hist = fusion_no_hist(p_tokens, h_tokens)
    out_no_hist.sum().backward()
    h_grad_no_hist = h_tokens.grad.abs().sum()
    print(f"H tokens grad (P2H Masked): {h_grad_no_hist.item()}")
    
    assert h_grad_no_hist < 1e-5, "H tokens should not receive gradients if P2H is masked"
    print("Quadrant masking verification passed!")

if __name__ == "__main__":
    test_quadrant_masking_interactions()
