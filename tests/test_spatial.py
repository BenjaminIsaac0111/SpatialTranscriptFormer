import pytest
import torch
from spatial_transcript_former.models import SpatialTranscriptFormer


def test_neighborhood_forward_pass(mock_neighborhood_batch, mock_rel_coords):
    """
    EDUCATIONAL: This test verifies that the model can process a sequence of
    histology patches (Neighborhood Mode).

    Input shape: (Batch, Sequence Length, Channels, Height, Width)
    Output shape: (Batch, Number of Genes)
    """
    num_genes = 1000
    model = SpatialTranscriptFormer(num_genes=num_genes)

    # Forward pass with neighborhood sequence and relative coordinates
    output = model(mock_neighborhood_batch, rel_coords=mock_rel_coords)

    assert output.shape == (mock_neighborhood_batch.shape[0], num_genes)


def test_distance_based_spatial_masking():
    """
    EDUCATIONAL: This test verifies that the 'mask_radius' correctly suppresses
    interactions with distant patches in the neighborhood.

    We place patches at different distances and verify that the generated
    attention mask ignores those beyond the radius.
    """
    G = 100
    # Relative Coords: Center(0,0), Near(10,0), Far(100,0), Very Far(1000,0)
    rel_coords = torch.tensor(
        [
            [
                [0, 0],  # Index 0 (Center)
                [10, 0],  # Index 1 (Near)
                [100, 0],  # Index 2 (Far)
                [1000, 0],  # Index 3 (Very Far)
            ]
        ],
        dtype=torch.float32,
    )

    # Set radius to 50: Only Center and Near should be visible
    model = SpatialTranscriptFormer(num_genes=G, mask_radius=50)

    # Internal method creates a boolean mask (True = Ignore)
    mask = model._generate_spatial_mask(rel_coords)

    # Expected: [False (Seen), False (Seen), True (Masked), True (Masked)]
    assert mask[0, 0] == False
    assert mask[0, 1] == False
    assert mask[0, 2] == True
    assert mask[0, 3] == True
