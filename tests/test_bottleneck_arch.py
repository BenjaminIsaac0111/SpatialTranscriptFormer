import torch
import sys
import os

# Add src to path
sys.path.append(os.path.abspath("src"))

from spatial_transcript_former.models import SpatialTranscriptFormer


def test_interaction_output_shape():
    # Model parameters
    num_genes = 1000
    num_pathways = 50
    model = SpatialTranscriptFormer(num_genes=num_genes, num_pathways=num_pathways)

    # Dummy input (Batch, Channel, Height, Width)
    x = torch.randn(4, 3, 224, 224)
    # Single patch => S=1
    rel_coords = torch.randn(4, 1, 2)

    # Forward pass
    output = model(x, rel_coords=rel_coords)

    # Verify shape (Batch, num_genes)
    assert output.shape == (
        4,
        num_genes,
    ), f"Expected shape (4, {num_genes}), got {output.shape}"
    print("Shape test passed!")


def test_interaction_gradient_flow():
    num_genes = 1000
    model = SpatialTranscriptFormer(num_genes=num_genes)
    x = torch.randn(2, 3, 224, 224)
    rel_coords = torch.randn(2, 1, 2)

    output = model(x, rel_coords=rel_coords)
    loss = output.sum()
    loss.backward()

    # Check gradients in key layers
    # 1. Backbone
    for name, param in model.backbone.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"No gradient in backbone param {name}"
            break

    # 2. Gene Reconstructor
    assert (
        model.gene_reconstructor.weight.grad is not None
    ), "No gradient in gene_reconstructor"

    print("Gradient flow test passed!")


if __name__ == "__main__":
    test_interaction_output_shape()
    test_interaction_gradient_flow()
