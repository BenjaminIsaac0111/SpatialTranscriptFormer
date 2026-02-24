import torch
import numpy as np
import sys
import os

# Add src to path
sys.path.append(os.path.abspath("src"))

from spatial_transcript_former.models import SpatialTranscriptFormer
from spatial_transcript_former.data.dataset import HEST_Dataset


def test_neighborhood_model_forward():
    B = 2
    S = 9  # 1 center + 8 neighbors
    C, H, W = 3, 224, 224
    G = 1000

    model = SpatialTranscriptFormer(num_genes=G)

    # Input sequence
    x = torch.randn(B, S, C, H, W)
    rel_coords = torch.randn(B, S, 2)

    # Forward pass
    out = model(x, rel_coords=rel_coords)

    print(f"Neighborhood Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")

    assert out.shape == (B, G), f"Expected (B, G), got {out.shape}"
    print("Neighborhood forward pass test passed!")


if __name__ == "__main__":
    test_neighborhood_model_forward()
