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
