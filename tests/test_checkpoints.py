import pytest
import torch
import torch.nn as nn
import os
import tempfile
from spatial_transcript_former.models import SpatialTranscriptFormer
from spatial_transcript_former.data.pathways import get_pathway_init


def test_model_structure_consistency():
    """
    Verify that the model structure (specifically the gene_reconstructor)
    matches the expected biological initialization.
    """
    num_genes = 100
    num_pathways = 50

    # Mock pathway_init
    pathway_init = torch.randn(num_pathways, num_genes)

    model = SpatialTranscriptFormer(
        num_genes=num_genes, num_pathways=num_pathways, pathway_init=pathway_init
    )

    # Check shape
    # Weight shape should be (num_genes, num_pathways) because it's (out, in)
    assert model.gene_reconstructor.weight.shape == (num_genes, num_pathways)

    # Verify values match (within tolerance)
    # The interaction model now L1-normalizes the pathways for stability
    # shape of pathway_init is (num_pathways, num_genes)
    import torch.nn.functional as F

    # We must normalize the columns of pathway_init.T, which correspond to the rows of pathway_init
    # Adding a small epsilon as done in interaction.py
    normalized_pathway_init = pathway_init / (
        pathway_init.sum(dim=1, keepdim=True) + 1e-6
    )

    assert torch.allclose(
        model.gene_reconstructor.weight, normalized_pathway_init.T, atol=1e-5
    )


def test_checkpoint_save_load():
    """
    Verify that a model can be saved and loaded without losing architectural state.
    """
    num_genes = 64
    model = SpatialTranscriptFormer(num_genes=num_genes)

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = os.path.join(tmpdir, "test_ckpt.pt")

        # Save state
        torch.save(
            {"model_state_dict": model.state_dict(), "num_genes": num_genes}, ckpt_path
        )

        # Load state
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        new_model = SpatialTranscriptFormer(num_genes=ckpt["num_genes"])
        new_model.load_state_dict(ckpt["model_state_dict"])

        # Verify weight equality
        for p1, p2 in zip(model.parameters(), new_model.parameters()):
            assert torch.equal(p1, p2)


def test_sparsity_loss_calculation():
    """
    Verify that the L1 sparsity loss is correctly computed for the reconstruction layer.
    """
    num_genes = 10
    num_pathways = 5
    model = SpatialTranscriptFormer(num_genes=num_genes, num_pathways=num_pathways)

    # Manual L1
    expected_l1 = torch.norm(model.gene_reconstructor.weight, p=1)

    calculated_l1 = model.get_sparsity_loss()

    assert torch.isclose(expected_l1, calculated_l1)
