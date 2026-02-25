import pytest
import torch
from spatial_transcript_former.models import SpatialTranscriptFormer


def test_interaction_output_shape(mock_image_batch):
    """
    EDUCATIONAL: This test verifies that the SpatialTranscriptFormer correctly
    maps a batch of histology images to a high-dimensional gene expression vector.
    """
    num_genes = 1000
    model = SpatialTranscriptFormer(num_genes=num_genes)

    # Must provide rel_coords since use_spatial_pe defaults to True
    B = mock_image_batch.shape[0]
    if mock_image_batch.dim() == 5:
        S = mock_image_batch.shape[1]
    elif mock_image_batch.dim() == 4:
        S = 1
    else:
        S = mock_image_batch.shape[1]

    rel_coords = torch.randn(B, S, 2)
    output = model(mock_image_batch, rel_coords=rel_coords)

    # Verify shape: (Batch Size, Number of Genes)
    assert output.shape == (B, num_genes)


def test_sparsity_regularization_loss():
    """
    EDUCATIONAL: This test verifies the 'L1 Sparsity' calculation.
    Sparsity forces each pathway token to only contribute to a small, distinct
    set of genes, creating a biologically-interpretable bottleneck.
    """
    num_genes = 100
    num_pathways = 10
    model = SpatialTranscriptFormer(num_genes=num_genes, num_pathways=num_pathways)

    sparsity_loss = model.get_sparsity_loss()

    # Expect a positive scalar (L1 norm of reconstruction weights)
    assert sparsity_loss > 0
    assert sparsity_loss.dim() == 0
