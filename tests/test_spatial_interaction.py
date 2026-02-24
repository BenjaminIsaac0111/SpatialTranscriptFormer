import torch
import pytest
from unittest.mock import MagicMock
from spatial_transcript_former.models.interaction import (
    SpatialTranscriptFormer,
    LearnedSpatialEncoder,
    VALID_INTERACTIONS,
)
from spatial_transcript_former.training.engine import train_one_epoch, validate


def test_n_layers_enforcement():
    """n_layers < 2 with h2h blocked should raise ValueError."""
    with pytest.raises(ValueError, match="n_layers must be >= 2"):
        SpatialTranscriptFormer(
            num_genes=50, n_layers=1, interactions=["p2p", "p2h", "h2p"]
        )


def test_n_layers_ok_with_full_interactions():
    """n_layers=1 is allowed when h2h is enabled (full attention)."""
    model = SpatialTranscriptFormer(
        num_genes=50,
        token_dim=64,
        n_heads=4,
        n_layers=1,
        interactions=["p2p", "p2h", "h2p", "h2h"],
    )
    assert model is not None


def test_invalid_interaction_key():
    """Unknown interaction keys should raise ValueError."""
    with pytest.raises(ValueError, match="Unknown interaction keys"):
        SpatialTranscriptFormer(num_genes=50, interactions=["p2p", "x2y"])


@pytest.mark.parametrize(
    "interactions",
    [
        ["p2p", "p2h", "h2p", "h2h"],  # full
        ["p2p", "p2h", "h2p"],  # bottleneck
        ["p2p", "p2h"],  # pathway-only
    ],
)
def test_interaction_combinations(interactions):
    """Various interaction combos should produce correct output shapes."""
    model = SpatialTranscriptFormer(
        num_genes=50,
        token_dim=64,
        n_heads=4,
        n_layers=2,
        interactions=interactions,
    )
    B, N, D = 2, 5, 2048
    features = torch.randn(B, N, D)
    coords = torch.randn(B, N, 2)
    out = model(features, rel_coords=coords)
    assert out.shape == (B, 50)


def test_full_interactions_returns_no_mask():
    """When all interactions are enabled, _build_interaction_mask returns None."""
    model = SpatialTranscriptFormer(
        num_genes=50,
        token_dim=64,
        n_heads=4,
        n_layers=2,
        interactions=["p2p", "p2h", "h2p", "h2h"],
    )
    mask = model._build_interaction_mask(p=10, s=20, device=torch.device("cpu"))
    assert mask is None


def test_missing_coords_raises():
    """use_spatial_pe=True without coords should raise ValueError."""
    model = SpatialTranscriptFormer(num_genes=50, token_dim=64, n_heads=4, n_layers=2)
    features = torch.randn(2, 10, 2048)
    with pytest.raises(ValueError, match="rel_coords was not provided"):
        model(features)


def test_spatial_transcript_former_dense_forward():
    """Instantiate the model and ensure forward() with return_dense=True executes properly."""
    model = SpatialTranscriptFormer(
        num_genes=50,
        token_dim=64,
        n_heads=4,
        n_layers=2,
    )

    B, N, D = 2, 10, 2048
    features = torch.randn(B, N, D)
    coords = torch.randn(B, N, 2)

    # 1. Standard Forward Pass
    out = model(features, rel_coords=coords)
    assert out.shape == (
        B,
        50,
    ), f"Expected global output shape {(B, 50)}, got {out.shape}"

    # Reset grads
    model.zero_grad()

    # 2. Dense Forward Pass
    out_dense = model.forward(features, rel_coords=coords, return_dense=True)
    assert out_dense.shape == (
        B,
        N,
        50,
    ), f"Expected dense output shape {(B, N, 50)}, got {out_dense.shape}"

    out_dense.sum().backward()
    assert (
        model.fusion_engine.layers[0].self_attn.in_proj_weight.grad is not None
    ), "Gradients did not flow through the transformer in dense pass."


def test_unified_pathway_scoring():
    """Both global and dense modes should produce pathway_scores from dot-products."""
    model = SpatialTranscriptFormer(
        num_genes=50,
        token_dim=64,
        n_heads=4,
        n_layers=2,
        num_pathways=10,
    )

    B, N, D = 2, 5, 2048
    features = torch.randn(B, N, D)
    coords = torch.randn(B, N, 2)

    # Global mode returns (gene_expression, pathway_scores)
    gene_expr, pw_scores = model(features, rel_coords=coords, return_pathways=True)
    assert gene_expr.shape == (B, 50)
    assert pw_scores.shape == (B, 10)  # (B, P) from pooled dot-product

    # Dense mode returns (gene_expression, pathway_scores)
    gene_expr_d, pw_scores_d = model(
        features, rel_coords=coords, return_pathways=True, return_dense=True
    )
    assert gene_expr_d.shape == (B, N, 50)
    assert pw_scores_d.shape == (B, N, 10)  # (B, S, P) from per-patch dot-product


def test_engine_passes_coords_to_forward():
    """
    Verify that the training engine passes spatial coordinates to
    model.forward() in whole_slide mode.
    """
    model = SpatialTranscriptFormer(num_genes=10)

    # Mock forward to track calls
    original_forward = model.forward
    model.forward = MagicMock(return_value=torch.randn(2, 5, 10))
    model.get_sparsity_loss = MagicMock(return_value=torch.tensor(0.0))

    fake_coords = torch.randn(2, 5, 2)
    fake_mask = torch.zeros(2, 5).bool()

    # Dataloader yielding (feats, genes, coords, mask)
    loader = [(torch.randn(2, 5, 512), torch.randn(2, 5, 10), fake_coords, fake_mask)]

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    def dummy_criterion(p, t, mask=None):
        return torch.tensor(1.0, requires_grad=True)

    train_one_epoch(model, loader, dummy_criterion, optimizer, "cpu", whole_slide=True)

    model.forward.assert_called_once()
    kwargs = model.forward.call_args.kwargs

    assert "rel_coords" in kwargs, "Engine did not pass 'rel_coords' kwargs to forward!"
    assert torch.allclose(
        kwargs["rel_coords"], fake_coords
    ), "Engine passed wrong coordinate tensor!"


def test_engine_validate_passes_coords():
    """
    Verify validation loop passes coords.
    """
    model = SpatialTranscriptFormer(num_genes=10)
    model.forward = MagicMock(return_value=torch.randn(2, 5, 10))

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

    model.forward.assert_called_once()
    kwargs = model.forward.call_args.kwargs

    assert (
        "rel_coords" in kwargs
    ), "Validate engine did not pass 'rel_coords' kwargs to forward!"
    assert torch.allclose(
        kwargs["rel_coords"], fake_coords
    ), "Validate engine passed wrong coordinate tensor!"
