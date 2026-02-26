import torch
import torch.nn as nn
import sys
import os
import numpy as np

# Add src to path
sys.path.append(os.path.abspath("src"))

from spatial_transcript_former.models.interaction import SpatialTranscriptFormer


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


if __name__ == "__main__":
    test_mask_logic()
    test_connectivity()
    test_attention_extraction()
