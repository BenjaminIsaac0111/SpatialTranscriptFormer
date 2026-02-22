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


def test_spatial_masking():
    B = 1
    S = 4
    G = 100
    # Center at (0,0), neighbors at (10,0), (100,0), (1000,0)
    rel_coords = torch.tensor(
        [[[0, 0], [10, 0], [100, 0], [1000, 0]]], dtype=torch.float32
    )

    # Set mask radius to 50 -> neighbors 2 and 3 should be masked
    model = SpatialTranscriptFormer(num_genes=G, mask_radius=50)

    mask = model._generate_spatial_mask(rel_coords)
    # Expected mask: [False, False, True, True] (True means ignore)

    print(f"Rel Coords: {rel_coords}")
    print(f"Generated Mask: {mask}")

    assert mask[0, 0] == False
    assert mask[0, 1] == False
    assert mask[0, 2] == True
    assert mask[0, 3] == True

    print("Spatial masking logic test passed!")


if __name__ == "__main__":
    test_neighborhood_model_forward()
    test_spatial_masking()
