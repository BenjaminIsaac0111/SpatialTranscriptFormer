import pytest
import torch
from spatial_transcript_former.models.interaction import SpatialTranscriptFormer
from spatial_transcript_former.training.losses import (
    AuxiliaryPathwayLoss,
    MaskedMSELoss,
)


def test_pathway_initialization_stability_and_gradients():
    """
    Verifies that initializing the model with a binary pathway matrix:
    1. Does not cause predictions to exponentially explode (numerical stability).
    2. Allows gradients to flow properly when using AuxiliaryPathwayLoss.
    """
    torch.manual_seed(42)
    num_pathways = 50
    num_genes = 100

    # Create a synthetic MSigDB-style binary matrix
    pathway_matrix = (torch.rand(num_pathways, num_genes) > 0.8).float()
    # Ensure no empty pathways to avoid division by zero
    pathway_matrix[:, 0] = 1.0

    # Initialize model with pathway_init
    model = SpatialTranscriptFormer(
        num_genes=num_genes,
        num_pathways=num_pathways,
        pathway_init=pathway_matrix,
        use_spatial_pe=False,
        output_mode="counts",
        pretrained=False,
    )

    # Dummy inputs
    B, S, D = (
        2,
        10,
        2048,
    )  # Using D=2048 since backbone='resnet50' requires it natively, or provided features
    feats = torch.randn(B, S, D, requires_grad=True)
    coords = torch.randn(B, S, 2)
    target_genes = torch.randn(B, S, num_genes).abs()
    mask = torch.zeros(B, S, dtype=torch.bool)

    # Forward pass
    # return_pathways=True is needed to get the intermediate pathway preds for Auxiliary loss
    gene_preds, pathway_preds = model(
        feats, rel_coords=coords, return_dense=True, return_pathways=True
    )

    # 1. Numerical Stability Check
    # Without L1 normalization and removing temperature, predictions would explode.
    # With the fix, Softplus should keep outputs reasonably small.
    max_pred = gene_preds.max().item()
    print(f"Max prediction value at initialization: {max_pred:.2f}")
    assert (
        max_pred < 100.0
    ), f"Predictions exploded! Max value: {max_pred}. Check L1 normalization."
    assert not torch.isnan(gene_preds).any(), "Found NaNs in initial predictions."

    # 2. Gradient Flow Check (Compatibility with Training)
    loss_fn = AuxiliaryPathwayLoss(pathway_matrix, MaskedMSELoss(), lambda_pathway=1.0)
    loss = loss_fn(gene_preds, target_genes, mask=mask, pathway_preds=pathway_preds)

    assert loss.isfinite(), "Loss is not finite."

    loss.backward()

    # Verify gradients reached the core transformer layers
    target_layer_grad = model.fusion_engine.layers[0].linear1.weight.grad
    assert target_layer_grad is not None, "Gradients did not reach the fusion engine."
    assert target_layer_grad.norm() > 0, "Vanishing gradients in the fusion engine."
    assert torch.isfinite(
        target_layer_grad
    ).all(), "Exploding/NaN gradients in fusion engine."

    # Verify gradients reached the final reconstructor layer
    recon_grad = model.gene_reconstructor.weight.grad
    assert recon_grad is not None, "Gradients did not reach the gene reconstructor."
    assert recon_grad.norm() > 0, "Vanishing gradients in the gene reconstructor."
    assert torch.isfinite(
        recon_grad
    ).all(), "Exploding/NaN gradients in gene reconstructor."

    print("Pathway initialization is fully stable and compatible with NN training.")
