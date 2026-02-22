import torch
import pytest
from unittest.mock import MagicMock
from spatial_transcript_former.models.interaction import (
    SpatialTranscriptFormer,
    LocalPatchMixer,
    GraphPatchMixer,
)
from spatial_transcript_former.training.engine import train_one_epoch, validate


def test_normalize_coords_finds_correct_step():
    """
    Test that _normalize_coords correctly infers the grid spacing (e.g., 224)
    even when absolute coordinates are very large, which breaks the current median
    absolute value heuristic.
    """
    model = SpatialTranscriptFormer(num_genes=10, use_nystrom=False)

    # Simulate a 2x2 grid of patches with patch size 224 in absolute pixel coords
    coords = torch.tensor(
        [[10000.0, 20000.0], [10224.0, 20000.0], [10000.0, 20224.0], [10224.0, 20224.0]]
    ).unsqueeze(
        0
    )  # Output shape: (1, 4, 2)

    normalized = model._normalize_coords(coords)

    # LocalPatchMixer subtracts the minimum coordinate to create a zero-indexed grid.
    min_c = normalized.min(dim=1, keepdim=True)[0]
    grid_coords = normalized - min_c

    expected_grid = torch.tensor(
        [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
    ).unsqueeze(0)

    # The current heuristic fails this because it divides by ~15000 (median)
    assert torch.allclose(
        grid_coords, expected_grid
    ), f"Coordinate normalization failed, returned:\n{grid_coords}"


def test_engine_passes_coords_to_dense_forward():
    """
    Verify that the training engine passes the generated spatial coordinates to
    the model in whole_slide mode.
    """
    model = SpatialTranscriptFormer(num_genes=10)

    # Mock to track if coords are passed
    model.forward_dense = MagicMock(return_value=torch.randn(2, 5, 10))
    model.get_sparsity_loss = MagicMock(return_value=torch.tensor(0.0))

    fake_coords = torch.randn(2, 5, 2)
    fake_mask = torch.zeros(2, 5).bool()

    # Datloader yielding (feats, genes, coords, mask)
    loader = [(torch.randn(2, 5, 512), torch.randn(2, 5, 10), fake_coords, fake_mask)]

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    def dummy_criterion(p, t, mask=None):
        return torch.tensor(1.0, requires_grad=True)

    train_one_epoch(model, loader, dummy_criterion, optimizer, "cpu", whole_slide=True)

    model.forward_dense.assert_called_once()
    kwargs = model.forward_dense.call_args.kwargs

    assert "coords" in kwargs, "Engine did not pass 'coords' kwargs to forward_dense!"
    assert torch.allclose(
        kwargs["coords"], fake_coords
    ), "Engine passed wrong coordinate tensor!"


def test_engine_validate_passes_coords():
    """
    Verify validation loop passes coords.
    """
    model = SpatialTranscriptFormer(num_genes=10)
    model.forward_dense = MagicMock(return_value=torch.randn(2, 5, 10))

    fake_coords = torch.randn(2, 5, 2)
    loader = [
        (
            torch.randn(2, 5, 512),
            torch.randn(2, 5, 10),
            fake_coords,
            torch.zeros(2, 5).bool(),
        )
    ]

    def dummy_criterion(p, t, mask=None):
        return torch.tensor(1.0)

    validate(model, loader, dummy_criterion, "cpu", whole_slide=True)

    model.forward_dense.assert_called_once()
    kwargs = model.forward_dense.call_args.kwargs

    assert (
        "coords" in kwargs
    ), "Validate engine did not pass 'coords' kwargs to forward_dense!"
    assert torch.allclose(
        kwargs["coords"], fake_coords
    ), "Validate engine passed wrong coordinate tensor!"


def test_graph_patch_mixer():
    """Verify that the GraphPatchMixer correctly performs message passing over a k-NN graph."""
    B, N, D = 2, 10, 32
    k = 3
    mixer = GraphPatchMixer(dim=D, k=k, heads=4)

    x = torch.randn(B, N, D)
    coords = torch.randn(B, N, 2)

    # 1. Forward Pass
    out = mixer(x, coords)

    # 2. Shape Verification
    assert out.shape == (B, N, D), f"Expected shape {(B, N, D)}, got {out.shape}"

    # 3. Gradient Flow Verification
    out.sum().backward()
    assert (
        mixer.to_qkv.weight.grad is not None
    ), "Gradients did not flow through the GAT layer."


def test_spatial_transcript_former_with_gnn_refiner():
    """Instantiate the model with GNN refiner and ensure forward() and forward_dense() execute properly."""
    model = SpatialTranscriptFormer(
        num_genes=50,
        token_dim=64,
        n_heads=4,
        n_layers=1,
        early_mixer=None,
        late_refiner="gnn",
    )

    B, N, D = 2, 10, 2048
    features = torch.randn(B, N, D)
    coords = torch.randn(B, N, 2)

    # Check that early_mixer is None and late_refiner is initialized
    assert model.early_mixer is None, "early_mixer should be None"
    assert (
        hasattr(model, "late_refiner") and model.late_refiner is not None
    ), "late_refiner not initialized"

    # 1. Standard Forward Pass
    out = model(features, rel_coords=coords)
    assert out.shape == (
        B,
        50,
    ), f"Expected dense output shape {(B, 50)}, got {out.shape}"

    # Reset grads
    model.zero_grad()

    # 2. Dense Forward Pass (This is the one that uses the Graph Refiner explicitly)
    out_dense = model.forward_dense(features, coords=coords)
    assert out_dense.shape == (
        B,
        N,
        50,
    ), f"Expected dense output shape {(B, N, 50)}, got {out_dense.shape}"

    out_dense.sum().backward()
    assert (
        model.late_refiner.to_qkv.weight.grad is not None
    ), "Gradients did not flow through the late_refiner in dense pass."
