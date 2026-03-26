"""
Merged tests: test_models.py, test_model_backbones.py, test_backbones.py, test_bottleneck_arch.py
"""

import time
import traceback
import sys
import os

import pytest
import torch
import torch.nn as nn

from spatial_transcript_former.models import SpatialTranscriptFormer
from spatial_transcript_former.models.regression import get_backbone, HE2RNA, ViT_ST
from spatial_transcript_former.models.interaction import SpatialTranscriptFormer
from spatial_transcript_former.models.mil import AttentionMIL, TransMIL
from spatial_transcript_former.models.backbones import get_backbone

# --- From test_models.py ---


def test_interaction_output_shape(mock_image_batch):
    """
    EDUCATIONAL: This test verifies that the SpatialTranscriptFormer correctly
    maps a batch of histology images to a high-dimensional gene expression vector.
    """
    num_genes = 1000
    model = SpatialTranscriptFormer(num_genes=num_genes)

    # Must provide rel_coords since use_spatial_pe defaults to True
    B = mock_image_batch.shape[0]
    if mock_image_batch.dim() == 5:
        S = mock_image_batch.shape[1]
    elif mock_image_batch.dim() == 4:
        S = 1
    else:
        S = mock_image_batch.shape[1]

    rel_coords = torch.randn(B, S, 2)
    output = model(mock_image_batch, rel_coords=rel_coords)

    # Verify shape: (Batch Size, Number of Genes)
    assert output.shape == (B, num_genes)


def test_sparsity_regularization_loss():
    """
    EDUCATIONAL: This test verifies the 'L1 Sparsity' calculation.
    Sparsity forces each pathway token to only contribute to a small, distinct
    set of genes, creating a biologically-interpretable bottleneck.
    """
    num_genes = 100
    num_pathways = 10
    model = SpatialTranscriptFormer(num_genes=num_genes, num_pathways=num_pathways)

    sparsity_loss = model.get_sparsity_loss()

    # Expect a positive scalar (L1 norm of reconstruction weights)
    assert sparsity_loss > 0
    assert sparsity_loss.dim() == 0


# --- From test_model_backbones.py ---


@pytest.mark.parametrize("backbone_name", ["resnet50"])
def test_get_backbone_basic(backbone_name):
    num_genes = 100
    model, feature_dim = get_backbone(
        backbone_name, pretrained=False, num_classes=num_genes
    )
    assert isinstance(model, nn.Module)
    assert feature_dim > 0

    # Test forward pass
    x = torch.randn(2, 3, 224, 224)
    out = model(x)
    assert out.shape == (2, num_genes)


@pytest.mark.parametrize("backbone_name", ["resnet50"])
def test_get_backbone_no_head(backbone_name):
    model, feature_dim = get_backbone(backbone_name, pretrained=False, num_classes=None)
    assert isinstance(model, nn.Module)

    # Test forward pass returns features
    x = torch.randn(2, 3, 224, 224)
    out = model(x)
    assert out.shape == (2, feature_dim)


def test_interaction_model_backbone():
    num_genes = 100
    model = SpatialTranscriptFormer(
        num_genes=num_genes, backbone_name="resnet50", pretrained=False
    )

    # Test with raw image input (single patch => S=1)
    x = torch.randn(4, 3, 224, 224)
    rel_coords = torch.randn(4, 1, 2)
    out = model(x, rel_coords=rel_coords)
    assert out.shape == (4, num_genes)


def test_attention_mil_backbone():
    num_genes = 100
    # Process 5 patches of size 224x224
    model = AttentionMIL(
        output_dim=num_genes, backbone_name="resnet50", pretrained=False
    )

    x = torch.randn(2, 5, 3, 224, 224)  # (B, N, C, H, W)
    out, attn = model(x, return_attention=True)
    assert out.shape == (2, num_genes)
    assert attn.shape == (2, 5, 1)


def test_trans_mil_backbone():
    num_genes = 100
    model = TransMIL(output_dim=num_genes, backbone_name="resnet50", pretrained=False)

    x = torch.randn(2, 5, 3, 224, 224)
    out = model(x)
    assert out.shape == (2, num_genes)


def test_he2rna_backbone():
    num_genes = 100
    model = HE2RNA(num_genes=num_genes, backbone="resnet50", pretrained=False)
    x = torch.randn(2, 3, 224, 224)
    out = model(x)
    assert out.shape == (2, num_genes)


def test_vit_st_backbone():
    num_genes = 100
    # Use resnet50 name to test fallback if vit_b_16 not available or just use vit_b_16 directly
    model = ViT_ST(num_genes=num_genes, model_name="resnet50", pretrained=False)
    x = torch.randn(2, 3, 224, 224)
    out = model(x)
    assert out.shape == (2, num_genes)


# --- From test_backbones.py ---


def test_ctranspath():
    print("\nTesting CTransPath...")
    try:
        model, dim = get_backbone("ctranspath", pretrained=False)
        print(f"Success! Dim: {dim}")

        x = torch.randn(1, 3, 224, 224)
        out = model(x)
        print(f"Output shape: {out.shape}")

    except Exception as e:
        traceback.print_exc()
        print(f"Failed: {e}")


def test_hibou():
    print("\nTesting Hibou-B...")
    try:
        model, dim = get_backbone("hibou-b", pretrained=False)
        print(f"Success! Dim: {dim}")

        x = torch.randn(1, 3, 224, 224)
        out = model(x)
        print(f"Output shape: {out.shape}")

    except Exception as e:
        traceback.print_exc()
        print(f"Failed: {e}")


def test_phikon():
    print("\nTesting Phikon (with pretrained weights)...")
    try:
        model, dim = get_backbone("phikon", pretrained=True)
        print(f"Success! Dim: {dim}")

        x = torch.randn(1, 3, 224, 224)
        out = model(x)
        print(f"Output shape: {out.shape}")

    except Exception as e:
        traceback.print_exc()
        print(f"Failed: {e}")


def test_plip():
    print("\nTesting PLIP...")
    try:
        model, dim = get_backbone("plip", pretrained=False)
        print(f"Success! Dim: {dim}")

        x = torch.randn(1, 3, 224, 224)
        out = model(x)
        print(f"Output shape: {out.shape}")

    except Exception as e:
        traceback.print_exc()
        print(f"Failed: {e}")


# --- From test_bottleneck_arch.py ---


def test_interaction_output_shape_2():
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
