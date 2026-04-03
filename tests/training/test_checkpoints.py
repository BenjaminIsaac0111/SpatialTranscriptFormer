"""
Tests for model checkpointing (save/load) and optimization state.
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

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def small_model():
    """A small SpatialTranscriptFormer for fast testing."""
    return SpatialTranscriptFormer(
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


# ---------------------------------------------------------------------------
# Basic State Dict
# ---------------------------------------------------------------------------

def test_checkpoint_save_load():
    """
    Verify that a model can be saved and loaded without losing architectural state.
    """
    num_pathways = 50
    model = SpatialTranscriptFormer()

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = os.path.join(tmpdir, "test_ckpt.pt")

        # Save state
        torch.save(
            {"model_state_dict": model.state_dict(), "num_pathways": num_pathways}, ckpt_path
        )

        # Load state
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        new_model = SpatialTranscriptFormer()
        new_model.load_state_dict(ckpt["model_state_dict"])

        # Verify weight equality
        for p1, p2 in zip(model.parameters(), new_model.parameters()):
            assert torch.equal(p1, p2)