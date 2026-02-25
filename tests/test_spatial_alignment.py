import pytest
import torch
from spatial_transcript_former.models import SpatialTranscriptFormer


def test_spatial_mixing_with_large_coordinates():
    """
    Verifies that the model correctly handles large pixel coordinates (e.g. 256px steps)
    and that gradient flows between patches via the pathway bottleneck.
    """
    dim = 2048
    token_dim = 64

    model = SpatialTranscriptFormer(
        num_genes=10, token_dim=token_dim, n_layers=2, use_spatial_pe=True
    )

    # Create two patches that are physically adjacent (256px apart) but logically neighbors
    # Shape: (B, N, 2048)
    x = torch.randn(1, 2, dim, requires_grad=True)

    # Raw coordinates: (0,0) and (256,0) representing adjacent tiles in a WSI
    coords = torch.tensor([[[0.0, 0.0], [256.0, 0.0]]])

    # Forward pass
    output = model(x, rel_coords=coords)

    # Check gradient flow from Patch 1 to Patch 0's output
    # With the pathway bottleneck, both patches attend to pathway tokens, so
    # indirect gradient flow exists through the shared pathway embeddings.
    loss = output[0, 0].sum()
    loss.backward()

    grad_neighbor = x.grad[0, 1].abs().sum().item()

    assert grad_neighbor > 0.0, (
        f"Gradient from neighbor is zero ({grad_neighbor}). "
        "The model failed to propagate gradients between patches via pathway tokens."
    )


def test_coordinate_normalization_logic():
    """
    Unit test for the internal normalization logic (once implemented).
    Checks that [0, 256, 512] becomes [0, 1, 2].
    """
    # This test assumes the method _normalize_coords exists or we check the effect
    # For now, we rely on the mixing test as the functional verification.
    pass
