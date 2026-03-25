"""
Merged tests: test_spatial_augment.py, test_augmentation_sync.py, test_spatial_alignment.py
"""

import torch
import numpy as np
import pytest

from spatial_transcript_former.recipes.hest.dataset import apply_dihedral_augmentation
from spatial_transcript_former.recipes.hest.dataset import (
    apply_dihedral_augmentation,
    apply_dihedral_to_tensor,
)
from spatial_transcript_former.models import SpatialTranscriptFormer


# --- From test_spatial_augment.py ---


def test_apply_dihedral_augmentation_torch():
    coords = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])

    # Run multiple times to see different ops
    results = []
    for _ in range(100):
        out, _ = apply_dihedral_augmentation(coords)
        results.append(out.numpy().tolist())

    # Check that we get more than 1 unique result (proving it's stochastic)
    unique_results = []
    for r in results:
        if r not in unique_results:
            unique_results.append(r)

    print(f"Unique results found: {len(unique_results)}")
    assert len(unique_results) > 1
    assert len(unique_results) <= 8

    # Check a specific transformation (e.g. Identity should be in there)
    assert coords.tolist() in unique_results


def test_apply_dihedral_augmentation_numpy():
    coords = np.array([[1.0, 0.0], [0.0, 1.0]])
    out, _ = apply_dihedral_augmentation(coords)
    assert isinstance(out, np.ndarray)
    assert out.shape == coords.shape

# --- From test_augmentation_sync.py ---


def test_sync_logic():
    # 1. Create a dummy image with a point at (scale, 0)
    # C, H, W = 1, 10, 10. Center is (5, 5)
    img = torch.zeros((1, 11, 11))
    img[0, 5, 8] = 1.0  # Point at +3 on X from center

    # Coordinates relative to center
    # Center (5, 5) -> (0, 0)
    # Point (5, 8) -> (+3, 0) (Note: HEST uses [x, y], but arrays use [y, x])
    # [3.0, 0.0]
    rel_coords = torch.tensor([[0.0, 0.0], [3.0, 0.0]])

    for op in range(8):
        # Transform Image
        aug_img = apply_dihedral_to_tensor(img, op)

        # Transform Coords
        aug_coords, _ = apply_dihedral_augmentation(rel_coords, op=op)

        # Find where the pixel moved
        # Original (5, 8) -> relative (0, +3) in y-down (indexing [5, 8])
        # After rotation, find the new index of '1.0'
        new_pos = torch.nonzero(aug_img[0])
        if len(new_pos) > 0:
            ny, nx = new_pos[0]  # Pixel indexing
            # Convert to relative to center (5, 5)
            rx, ry = nx.item() - 5, ny.item() - 5

            # Check against augmented coords
            # Note: aug_coords[1] is the moved point
            ax, ay = aug_coords[1].numpy()

            print(
                f"Op {op} ({aug_img.shape}): Found pixel at ({nx}, {ny}) -> Rel=({rx}, {ry}) | Expected Coords=({ax}, {ay})"
            )

            assert rx == pytest.approx(ax), f"X mismatch in op {op}"
            assert ry == pytest.approx(ay), f"Y mismatch in op {op}"
        else:
            # Should not happen as identity/rotations/flips preserve count
            assert False, f"Pixel lost in op {op}"

# --- From test_spatial_alignment.py ---


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
