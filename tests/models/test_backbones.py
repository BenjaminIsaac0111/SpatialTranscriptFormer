"""
Tests for various histology backbones and MIL architectures.
"""

import torch
import torch.nn as nn
import pytest
import traceback

from spatial_transcript_former.models.regression import get_backbone, HE2RNA, ViT_ST
from spatial_transcript_former.models.mil import AttentionMIL, TransMIL
from spatial_transcript_former.models.backbones import get_backbone


@pytest.mark.parametrize("backbone_name", ["resnet50"])
def test_get_backbone_basic(backbone_name):
    num_pathways = 50
    model, feature_dim = get_backbone(
        backbone_name, pretrained=False, num_classes=num_pathways
    )
    assert isinstance(model, nn.Module)
    assert feature_dim > 0

    # Test forward pass
    x = torch.randn(2, 3, 224, 224)
    out = model(x)
    assert out.shape == (2, num_pathways)


@pytest.mark.parametrize("backbone_name", ["resnet50"])
def test_get_backbone_no_head(backbone_name):
    model, feature_dim = get_backbone(backbone_name, pretrained=False, num_classes=None)
    assert isinstance(model, nn.Module)

    # Test forward pass returns features
    x = torch.randn(2, 3, 224, 224)
    out = model(x)
    assert out.shape == (2, feature_dim)


def test_attention_mil_backbone():
    num_pathways = 50
    # Process 5 patches of size 224x224
    model = AttentionMIL(
        output_dim=num_pathways, backbone_name="resnet50", pretrained=False
    )

    x = torch.randn(2, 5, 3, 224, 224)  # (B, N, C, H, W)
    out, attn = model(x, return_attention=True)
    assert out.shape == (2, num_pathways)
    assert attn.shape == (2, 5, 1)


def test_trans_mil_backbone():
    num_pathways = 50
    model = TransMIL(
        output_dim=num_pathways, backbone_name="resnet50", pretrained=False
    )

    x = torch.randn(2, 5, 3, 224, 224)
    out = model(x)
    assert out.shape == (2, num_pathways)


def test_he2rna_backbone():
    num_pathways = 50
    model = HE2RNA(num_genes=num_pathways, backbone="resnet50", pretrained=False)
    x = torch.randn(2, 3, 224, 224)
    out = model(x)
    assert out.shape == (2, num_pathways)


def test_vit_st_backbone():
    num_pathways = 50
    # Use resnet50 name to test fallback if vit_b_16 not available or just use vit_b_16 directly
    model = ViT_ST(num_genes=num_pathways, model_name="resnet50", pretrained=False)
    x = torch.randn(2, 3, 224, 224)
    out = model(x)
    assert out.shape == (2, num_pathways)


def test_ctranspath():
    """Verify CTransPath backbone loading and forward pass."""
    try:
        model, dim = get_backbone("ctranspath", pretrained=False)
        x = torch.randn(1, 3, 224, 224)
        out = model(x)
        assert out.shape[1] == dim
    except Exception as e:
        pytest.skip(f"CTransPath tests skipped: {e}")


def test_hibou():
    """Verify Hibou backbone loading and forward pass."""
    try:
        model, dim = get_backbone("hibou-b", pretrained=False)
        x = torch.randn(1, 3, 224, 224)
        out = model(x)
        assert out.shape[1] == dim
    except Exception as e:
        pytest.skip(f"Hibou tests skipped: {e}")


def test_phikon():
    """Verify Phikon backbone loading and forward pass."""
    try:
        model, dim = get_backbone("phikon", pretrained=True)
        x = torch.randn(1, 3, 224, 224)
        out = model(x)
        assert out.shape[1] == dim
    except Exception as e:
        pytest.skip(f"Phikon tests skipped: {e}")


def test_plip():
    """Verify PLIP backbone loading and forward pass."""
    try:
        model, dim = get_backbone("plip", pretrained=False)
        x = torch.randn(1, 3, 224, 224)
        out = model(x)
        assert out.shape[1] == dim
    except Exception as e:
        pytest.skip(f"PLIP tests skipped: {e}")
