import pytest
import torch
from spatial_transcript_former.models import SpatialTranscriptFormer


def test_spatial_mixing_with_large_coordinates():
    """
    Verifies that the model correctly normalizes large pixel coordinates (e.g. 256px steps)
    so that local mixing can occur between adjacent patches.
    """
    # ResNet50 output dim is 2048, so we must use token_dim=2048 or project it.
    # The model has an image_proj layer: Linear(image_feature_dim, token_dim).
    # Wait, the error was 2x64 and 2048x64?
    # Ah, the error "mat1 and mat2 shapes cannot be multiplied (2x64 and 2048x64)"
    # suggests the input x has dim 64, but the first layer (backbone) expects something else?
    # Actually, the model forward() expects images (B, C, H, W) OR features (B, N, D).
    # If passing features, D must match image_feature_dim (2048 for ResNet).

    # Let's verify model source:
    # if x.dim() == 3: features = x
    # memory = self.image_proj(features)
    # image_proj is Linear(2048, token_dim).

    # So input x mus be (B, N, 2048).
    dim = 2048
    token_dim = 64

    model = SpatialTranscriptFormer(
        num_genes=10, token_dim=token_dim, n_layers=1, use_spatial_pe=True
    )

    # Create two patches that are physically adjacent (256px apart) but logically neighbors
    # Shape: (B, N, 2048)
    x = torch.randn(1, 2, dim, requires_grad=True)

    # Raw coordinates: (0,0) and (256,0) representing adjacent tiles in a WSI
    coords = torch.tensor([[[0.0, 0.0], [256.0, 0.0]]])

    # Forward pass
    output = model(x, rel_coords=coords)

    # Check gradient flow from Patch 1 to Patch 0's output
    # If mixing works, output[0] should be influenced by input[1] via the LocalPatchMixer
    # We sum output[0] and check grad w.r.t x[0, 1]
    loss = output[0, 0].sum()
    loss.backward()

    grad_neighbor = x.grad[0, 1].abs().sum().item()

    assert grad_neighbor > 0.0, (
        f"Gradient from neighbor is zero ({grad_neighbor}). "
        "The model failed to mix features between patches separated by 256px. "
        "This indicates the LocalPatchMixer is treating them as distant neighbors."
    )


def test_coordinate_normalization_logic():
    """
    Unit test for the internal normalization logic (once implemented).
    Checks that [0, 256, 512] becomes [0, 1, 2].
    """
    # This test assumes the method _normalize_coords exists or we check the effect
    # For now, we rely on the mixing test as the functional verification.
    pass
