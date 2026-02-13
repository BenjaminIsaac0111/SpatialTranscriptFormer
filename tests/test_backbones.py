import torch
import torch.nn as nn
import time
import traceback
from spatial_transcript_former.models.backbones import get_backbone

def test_ctranspath():
    print("\nTesting CTransPath...")
    try:
        model, dim = get_backbone('ctranspath', pretrained=False)
        print(f"Success! Dim: {dim}")
        
        x = torch.randn(1, 3, 224, 224)
        out = model(x)
        print(f"Output shape: {out.shape}")
        
    except Exception as e:
        traceback.print_exc()
        print(f"Failed: {e}")

def test_hibou():
    print("\nTesting Hibou-B...")
    try:
        model, dim = get_backbone('hibou-b', pretrained=False)
        print(f"Success! Dim: {dim}")
        
        x = torch.randn(1, 3, 224, 224)
        out = model(x)
        print(f"Output shape: {out.shape}")
        
    except Exception as e:
        traceback.print_exc()
        print(f"Failed: {e}")

def test_phikon():
    print("\nTesting Phikon (with pretrained weights)...")
    try:
        model, dim = get_backbone('phikon', pretrained=True)
        print(f"Success! Dim: {dim}")
        
        x = torch.randn(1, 3, 224, 224)
        out = model(x)
        print(f"Output shape: {out.shape}")
        
    except Exception as e:
        traceback.print_exc()
        print(f"Failed: {e}")

def test_plip():
    print("\nTesting PLIP...")
    try:
        model, dim = get_backbone('plip', pretrained=False)
        print(f"Success! Dim: {dim}")
        
        x = torch.randn(1, 3, 224, 224)
        out = model(x)
        print(f"Output shape: {out.shape}")
        
    except Exception as e:
        traceback.print_exc()
        print(f"Failed: {e}")

if __name__ == "__main__":
    test_ctranspath()
    test_hibou()
    test_phikon()
    test_plip()
