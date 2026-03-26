"""
Merged tests: test_interactions.py, test_spatial_interaction.py
"""

import sys
import os
from unittest.mock import MagicMock

import torch
import torch.nn as nn
import numpy as np
import pytest

from spatial_transcript_former.models.interaction import SpatialTranscriptFormer
from spatial_transcript_former.models.interaction import (
    SpatialTranscriptFormer,
    LearnedSpatialEncoder,
    VALID_INTERACTIONS,
)
from spatial_transcript_former.training.engine import train_one_epoch, validate

# --- From test_interactions.py ---


def test_mask_logic():
    print("Testing Mask Logic...")
    model = SpatialTranscriptFormer(
        num_genes=100, num_pathways=10, interactions=["p2p", "p2h"]
    )

    # p=10, s=20
    mask = model._build_interaction_mask(10, 20, "cpu")

    # Check p2p (allowed)
    assert mask[0, 1] == False, "p2p should be allowed"
    # Check p2h (allowed)
    assert mask[0, 15] == False, "p2h should be allowed"
    # Check h2p (blocked)
    assert mask[15, 0] == True, "h2p should be blocked"
    # Check h2h (blocked)
    assert mask[15, 16] == True, "h2h should be blocked"

    print("Mask logic test passed!")


def test_connectivity():
    print("Testing Patch-to-Patch Connectivity (h2h)...")
    # Enable all interactions including h2h
    model = SpatialTranscriptFormer(
        num_genes=100,
        num_pathways=10,
        n_layers=2,
        interactions=["p2p", "p2h", "h2p", "h2h"],
        use_spatial_pe=False,  # Disable PE to verify raw attention connectivity
        pretrained=False,
    )
    model.eval()

    # dummy features (B=1, S=2, D=2048)
    feats = torch.randn(1, 2, 2048, requires_grad=True)
    coords = torch.randn(1, 2, 2)

    # We want to see if the output for patch 0 depends on the input of patch 1
    # We use return_dense=True to get per-patch gene predictions
    output = model(feats, rel_coords=coords, return_dense=True)  # (1, 2, 100)

    # Loss on patch 0 output
    loss = output[0, 0].sum()
    loss.backward()

    # Check if gradient flows to patch 1 input
    grad_patch_1 = feats.grad[0, 1].norm()
    print(f"Gradient at Patch 1 from Patch 0 output: {grad_patch_1.item():.6e}")

    assert (
        grad_patch_1 > 0
    ), "Patch 0 output should depend on Patch 1 input when h2h is enabled"

    # Now try with h2h disabled
    print("Testing Connectivity with h2h disabled...")
    model_no_h2h = SpatialTranscriptFormer(
        num_genes=100,
        num_pathways=10,
        n_layers=2,
        interactions=["p2p", "p2h", "h2p"],
        use_spatial_pe=False,
        pretrained=False,
    )
    model_no_h2h.eval()

    feats_2 = torch.randn(1, 2, 2048, requires_grad=True)
    output_2 = model_no_h2h(feats_2, rel_coords=coords, return_dense=True)

    loss_2 = output_2[0, 0].sum()
    loss_2.backward()

    grad_patch_1_no_h2h = feats_2.grad[0, 1].norm()
    print(
        f"Gradient at Patch 1 from Patch 0 output (no h2h): {grad_patch_1_no_h2h.item():.6e}"
    )

    # It should still be non-zero because patches interact via pathways [Patch 1 -> Pathway -> Patch 0]
    assert (
        grad_patch_1_no_h2h > 0
    ), "Patch 0 should still interact with Patch 1 via pathways even if h2h is disabled"

    # To truly see zero interaction, block pathways too
    print("Testing ZERO Connectivity (only p2p enabled)...")
    model_isolated = SpatialTranscriptFormer(
        num_genes=100,
        num_pathways=10,
        n_layers=2,
        interactions=["p2p"],
        use_spatial_pe=False,
        pretrained=False,
    )
    model_isolated.eval()
    feats_3 = torch.randn(1, 2, 2048, requires_grad=True)
    output_3 = model_isolated(feats_3, rel_coords=coords, return_dense=True)
    loss_3 = output_3[0, 0].sum()
    loss_3.backward()
    grad_patch_1_isolated = feats_3.grad[0, 1].norm()
    print(f"Gradient at Patch 1 (fully isolated): {grad_patch_1_isolated.item():.6e}")
    assert (
        grad_patch_1_isolated < 1e-10
    ), "Patch 0 should NOT depend on Patch 1 when only p2p is enabled"

    print("Connectivity tests passed!")


def test_attention_extraction():
    print("Testing Attention Extraction...")
    p, s = 10, 20
    model = SpatialTranscriptFormer(
        num_genes=100,
        num_pathways=p,
        n_layers=2,
        interactions=["p2p", "p2h"],  # Block h2p, h2h
        pretrained=False,
    )
    model.eval()

    feats = torch.randn(1, s, 2048)
    coords = torch.randn(1, s, 2)

    # Forward with attention
    _, attentions = model(feats, rel_coords=coords, return_attention=True)

    # attentions is list of weights [layers]
    for i, attn in enumerate(attentions):
        print(f"Testing Layer {i}...")
        # attn is (B, H, T, T)
        h = model.fusion_engine.layers[0].self_attn.num_heads
        assert attn.shape == (1, h, p + s, p + s)

        # We expect blocked regions to have 0 attention across all heads
        # h2p_region is (H, s, p)
        h2p_region = attn[0, :, p:, :p]
        h2h_region = attn[0, :, p:, p:]

        # For h2h, we must ignore diagonal within each head
        # We can just check that the entire (H, s, s) block is 0 except the diag
        h2h_zeroed = h2h_region.clone()
        for head_idx in range(h):
            h2h_zeroed[head_idx].fill_diagonal_(0)

        print(f"Layer {i} h2p attention max: {h2p_region.max().item():.2e}")
        print(f"Layer {i} h2h off-diag attention max: {h2h_zeroed.max().item():.2e}")

        assert (
            h2p_region.max() < 1e-10
        ), f"Layer {i} h2p attention should be zero when blocked"
        assert (
            h2h_zeroed.max() < 1e-10
        ), f"Layer {i} h2h attention should be zero when blocked"

        # Check that allowed regions have non-zero attention in at least one head
        p2p_region = attn[0, :, :p, :p]
        p2h_region = attn[0, :, :p, p:]
        print(f"Layer {i} p2p attention max: {p2p_region.max().item():.2e}")
        print(f"Layer {i} p2h attention max: {p2h_region.max().item():.2e}")

        assert p2p_region.max() > 0, f"Layer {i} p2p attention should be non-zero"
        assert p2h_region.max() > 0, f"Layer {i} p2h attention should be non-zero"

    print("Attention extraction test passed!")


# --- From test_spatial_interaction.py ---


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


def test_spatial_encoder_normalization():
    """Verify LearnedSpatialEncoder handles extreme coords and centers them."""
    encoder = LearnedSpatialEncoder(64)
    # Extreme coordinates: very far and very close
    coords = torch.tensor([[[1000.0, 1000.0], [1000.1, 1000.1]]])
    normed = encoder._normalize_coords(coords)

    # Should be centered (mean 0)
    assert torch.allclose(normed.mean(dim=1), torch.zeros(1, 2), atol=1e-5)
    # Should be bounded by [-1, 1]
    assert normed.abs().max() <= 1.0

    # Verify forward doesn't crash
    out = encoder(coords)
    assert out.shape == (1, 2, 64)


def test_interaction_mask_bits():
    """Explicitly verify which bits are blocked in the interaction mask."""
    model = SpatialTranscriptFormer(
        num_genes=50, interactions=["p2h", "h2p", "h2h"]
    )  # No p2p
    p, s = 2, 3
    mask = model._build_interaction_mask(p, s, torch.device("cpu"))

    # mask[i, j] is True if blocked
    # p2p is index [0:p, 0:p]. Should be blocked (True) except diagonal
    assert mask[0, 1] == True, "p2p interaction [0, 1] should be blocked"

    # p2h is index [0:p, p:]. Should be enabled (False)
    assert mask[0, 2] == False, "p2h interaction [0, 2] should be enabled"

    # h2p is index [p:, 0:p]. Should be enabled (False)
    assert mask[2, 0] == False, "h2p interaction [2, 0] should be enabled"

    # h2h is index [p:, p:]. Should be enabled (False)
    assert mask[2, 3] == False, "h2h interaction [2, 3] should be enabled"


def test_return_attention_values():
    """Validate attention weight extraction logic."""
    model = SpatialTranscriptFormer(
        num_genes=10, token_dim=64, n_heads=2, n_layers=2
    ).eval()
    B, S, D = 1, 4, 2048
    features = torch.randn(B, S, D)
    coords = torch.randn(B, S, 2)
    P = model.num_pathways

    # [gene_expr, pw_scores, attentions]
    with torch.no_grad():
        _, _, attentions = model(
            features, rel_coords=coords, return_attention=True, return_pathways=True
        )

    assert len(attentions) == 2  # n_layers
    for layer_attn in attentions:
        # Expected shape: (B, n_heads, Total_T, Total_T) where Total_T = P + S
        expected_shape = (B, 2, P + S, P + S)
        assert layer_attn.shape == expected_shape

        # In eval mode, attention should sum to 1.0 across the last dimension (softmax)
        sums = layer_attn.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-6)
