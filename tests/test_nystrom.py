import torch
import pytest
from spatial_transcript_former.models import SpatialTranscriptFormer


def test_nystrom_jaume_mode(mock_image_batch):
    """
    EDUCATIONAL: Verifies that Nystrom Attention works in 'jaume' (early fusion) mode.
    Nystrom provides linear complexity, allowing for multimodal interaction
    without the quadratic memory wall of standard self-attention.
    """
    num_genes = 100
    model = SpatialTranscriptFormer(
        num_genes=num_genes, use_nystrom=True, num_landmarks=32  # Small for testing
    )

    output = model(mock_image_batch)
    assert output.shape == (mock_image_batch.shape[0], num_genes)


def test_nystrom_no_quadrant_mask_support(mock_image_batch):
    """
    EDUCATIONAL: Nystrom Attention (linear complexity) does not support
    standard 2D quadrant masking. This test verifies the model still
    executes without crashing, but we acknowledge the mask is ignored.
    """
    num_genes = 100
    model = SpatialTranscriptFormer(
        num_genes=num_genes,
        use_nystrom=True,
        masked_quadrants=["H2H"],
        num_landmarks=32,
    )

    output = model(mock_image_batch)
    assert output.shape == (mock_image_batch.shape[0], num_genes)


def test_nystrom_scalability():
    """
    EDUCATIONAL: This test verifies that Nystrom mode can handle very large
    histology sequences that would typically crash a standard transformer.
    """
    B, S, D = 1, 1024, 512  # Large neighborhood
    num_genes = 10

    # We'll use mock projection features instead of raw images to save memory in CI
    # token_dim=512 is used in SpatialTranscriptFormer
    features = torch.randn(B, S, 2048)  # Backbone output dim

    model = SpatialTranscriptFormer(
        num_genes=num_genes, use_nystrom=True, num_landmarks=128
    )

    # Forward pass on a sequence of 1024 tokens
    with torch.no_grad():
        output = model(features)

    assert output.shape == (B, num_genes)
