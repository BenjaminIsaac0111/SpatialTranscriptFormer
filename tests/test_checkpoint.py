"""
Tests for checkpoint save/load and training resumption.

Verifies that model state, optimizer state, and training metadata
are correctly preserved across save/load cycles.
"""

import pytest
import os
import torch
import torch.nn as nn
import torch.optim as optim
import tempfile
from spatial_transcript_former.models import SpatialTranscriptFormer
from spatial_transcript_former.train import save_checkpoint, load_checkpoint

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def small_model():
    """A small SpatialTranscriptFormer for fast testing."""
    return SpatialTranscriptFormer(
        num_genes=100,
        num_pathways=10,
        token_dim=64,
        n_heads=4,
        n_layers=1,
        use_nystrom=False,
    )


@pytest.fixture
def small_nystrom_model():
    """A small SpatialTranscriptFormer with Nystrom attention."""
    return SpatialTranscriptFormer(
        num_genes=100,
        num_pathways=10,
        token_dim=64,
        n_heads=4,
        n_layers=1,
        use_nystrom=True,
        num_landmarks=32,
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
        """Model weights should be identical after save → load."""
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
            n_layers=1,
            use_nystrom=False,
        )
        fresh_optimizer = optim.Adam(fresh_model.parameters(), lr=1e-4)

        start_epoch, best_val = load_checkpoint(
            fresh_model, fresh_optimizer, None, checkpoint_dir, "interaction", "cpu"
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
            n_layers=1,
            use_nystrom=False,
        )
        fresh_optimizer = optim.Adam(fresh_model.parameters(), lr=1e-4)
        fresh_scaler = torch.amp.GradScaler("cuda")

        load_checkpoint(
            fresh_model,
            fresh_optimizer,
            fresh_scaler,
            checkpoint_dir,
            "interaction",
            "cpu",
        )

        # Scaler state dicts should match
        assert scaler.state_dict() == fresh_scaler.state_dict()

    def test_no_checkpoint_starts_fresh(self, small_model, checkpoint_dir):
        """Missing checkpoint should return epoch 0 and inf loss."""
        optimizer = optim.Adam(small_model.parameters(), lr=1e-4)
        start_epoch, best_val = load_checkpoint(
            small_model, optimizer, None, checkpoint_dir, "nonexistent", "cpu"
        )
        assert start_epoch == 0
        assert best_val == float("inf")


# ---------------------------------------------------------------------------
# Architecture Mismatch
# ---------------------------------------------------------------------------


class TestArchitectureMismatch:
    def test_nystrom_checkpoint_fails_on_standard(
        self, small_nystrom_model, checkpoint_dir
    ):
        """Nystrom checkpoint should fail to load into standard model."""
        optimizer = optim.Adam(small_nystrom_model.parameters(), lr=1e-4)
        save_checkpoint(
            small_nystrom_model,
            optimizer,
            None,
            epoch=5,
            best_val_loss=0.3,
            output_dir=checkpoint_dir,
            model_name="interaction",
        )

        # Try loading into non-Nystrom model
        standard_model = SpatialTranscriptFormer(
            num_genes=100,
            num_pathways=10,
            token_dim=64,
            n_heads=4,
            n_layers=1,
            use_nystrom=False,
        )
        standard_optimizer = optim.Adam(standard_model.parameters(), lr=1e-4)

        with pytest.raises(RuntimeError):
            load_checkpoint(
                standard_model,
                standard_optimizer,
                None,
                checkpoint_dir,
                "interaction",
                "cpu",
            )

    def test_matching_architecture_loads(self, small_nystrom_model, checkpoint_dir):
        """Nystrom checkpoint should load into matching Nystrom model."""
        optimizer = optim.Adam(small_nystrom_model.parameters(), lr=1e-4)
        save_checkpoint(
            small_nystrom_model,
            optimizer,
            None,
            epoch=5,
            best_val_loss=0.3,
            output_dir=checkpoint_dir,
            model_name="interaction",
        )

        fresh = SpatialTranscriptFormer(
            num_genes=100,
            num_pathways=10,
            token_dim=64,
            n_heads=4,
            n_layers=1,
            use_nystrom=True,
            num_landmarks=32,
        )
        fresh_opt = optim.Adam(fresh.parameters(), lr=1e-4)

        start_epoch, _ = load_checkpoint(
            fresh, fresh_opt, None, checkpoint_dir, "interaction", "cpu"
        )
        assert start_epoch == 6  # Resumed correctly
