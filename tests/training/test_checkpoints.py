"""
Merged tests: test_checkpoint.py, test_checkpoints.py
"""

import os
import tempfile

import pytest
import torch
import torch.nn as nn
import torch.optim as optim

from spatial_transcript_former.models import SpatialTranscriptFormer
from spatial_transcript_former.train import save_checkpoint, load_checkpoint
from spatial_transcript_former.data.pathways import get_pathway_init


# --- From test_checkpoint.py ---


@pytest.fixture
def small_model():
    """A small SpatialTranscriptFormer for fast testing."""
    return SpatialTranscriptFormer(
        num_genes=100,
        num_pathways=10,
        token_dim=64,
        n_heads=4,
        n_layers=2,
    )


@pytest.fixture
def checkpoint_dir():
    """Temporary directory for checkpoints."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


# ---------------------------------------------------------------------------
# Save / Load Roundtrip
# ---------------------------------------------------------------------------


class TestCheckpointRoundtrip:
    def test_save_load_preserves_weights(self, small_model, checkpoint_dir):
        """Model weights should be identical after save â†’ load."""
        optimizer = optim.Adam(small_model.parameters(), lr=1e-4)

        # Modify weights slightly (simulate training)
        with torch.no_grad():
            for p in small_model.parameters():
                p.add_(torch.randn_like(p) * 0.01)

        # Save
        save_checkpoint(
            small_model,
            optimizer,
            None,
            None,  # schedulers
            epoch=42,
            best_val_loss=0.123,
            output_dir=checkpoint_dir,
            model_name="interaction",
        )

        # Load into fresh model
        fresh_model = SpatialTranscriptFormer(
            num_genes=100,
            num_pathways=10,
            token_dim=64,
            n_heads=4,
            n_layers=2,
        )
        fresh_optimizer = optim.Adam(fresh_model.parameters(), lr=1e-4)

        start_epoch, best_val, loaded_schedulers = load_checkpoint(
            fresh_model,
            fresh_optimizer,
            None,
            None,
            checkpoint_dir,
            "interaction",
            "cpu",
        )

        # Verify metadata
        assert start_epoch == 43  # epoch + 1
        assert best_val == pytest.approx(0.123)

        # Verify weights match
        for (n1, p1), (n2, p2) in zip(
            small_model.named_parameters(), fresh_model.named_parameters()
        ):
            assert n1 == n2
            assert torch.allclose(p1, p2), f"Mismatch in {n1}"

    def test_save_load_preserves_scaler(self, small_model, checkpoint_dir):
        """AMP scaler state should survive checkpoint roundtrip."""
        optimizer = optim.Adam(small_model.parameters(), lr=1e-4)
        scaler = torch.amp.GradScaler("cuda")

        save_checkpoint(
            small_model,
            optimizer,
            scaler,
            None,  # schedulers
            epoch=10,
            best_val_loss=0.5,
            output_dir=checkpoint_dir,
            model_name="interaction",
        )

        fresh_model = SpatialTranscriptFormer(
            num_genes=100,
            num_pathways=10,
            token_dim=64,
            n_heads=4,
            n_layers=2,
        )
        fresh_optimizer = optim.Adam(fresh_model.parameters(), lr=1e-4)
        fresh_scaler = torch.amp.GradScaler("cuda")

        load_checkpoint(
            fresh_model,
            fresh_optimizer,
            fresh_scaler,
            None,  # schedulers
            checkpoint_dir,
            "interaction",
            "cpu",
        )

        # Scaler state dicts should match
        assert scaler.state_dict() == fresh_scaler.state_dict()

    def test_no_checkpoint_starts_fresh(self, small_model, checkpoint_dir):
        """Missing checkpoint should return epoch 0 and inf loss."""
        optimizer = optim.Adam(small_model.parameters(), lr=1e-4)
        start_epoch, best_val, loaded_schedulers = load_checkpoint(
            small_model, optimizer, None, None, checkpoint_dir, "nonexistent", "cpu"
        )
        assert start_epoch == 0
        assert best_val == float("inf")

# --- From test_checkpoints.py ---


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
