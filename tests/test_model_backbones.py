import torch
import pytest
import torch.nn as nn
from spatial_transcript_former.models.regression import get_backbone, HE2RNA, ViT_ST
from spatial_transcript_former.models.interaction import SpatialTranscriptFormer
from spatial_transcript_former.models.mil import AttentionMIL, TransMIL

@pytest.mark.parametrize("backbone_name", ["resnet50"])
def test_get_backbone_basic(backbone_name):
    num_genes = 100
    model, feature_dim = get_backbone(backbone_name, pretrained=False, num_classes=num_genes)
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
    model = SpatialTranscriptFormer(num_genes=num_genes, backbone_name='resnet50', pretrained=False)
    
    # Test with raw image input
    x = torch.randn(4, 3, 224, 224)
    out = model(x)
    assert out.shape == (4, num_genes)

def test_attention_mil_backbone():
    num_genes = 100
    # Process 5 patches of size 224x224
    model = AttentionMIL(output_dim=num_genes, backbone_name='resnet50', pretrained=False)
    
    x = torch.randn(2, 5, 3, 224, 224) # (B, N, C, H, W)
    out, attn = model(x, return_attention=True)
    assert out.shape == (2, num_genes)
    assert attn.shape == (2, 5, 1)

def test_trans_mil_backbone():
    num_genes = 100
    model = TransMIL(output_dim=num_genes, backbone_name='resnet50', pretrained=False)
    
    x = torch.randn(2, 5, 3, 224, 224)
    out = model(x)
    assert out.shape == (2, num_genes)

def test_he2rna_backbone():
    num_genes = 100
    model = HE2RNA(num_genes=num_genes, backbone='resnet50', pretrained=False)
    x = torch.randn(2, 3, 224, 224)
    out = model(x)
    assert out.shape == (2, num_genes)

def test_vit_st_backbone():
    num_genes = 100
    # Use resnet50 name to test fallback if vit_b_16 not available or just use vit_b_16 directly
    model = ViT_ST(num_genes=num_genes, model_name='resnet50', pretrained=False)
    x = torch.randn(2, 3, 224, 224)
    out = model(x)
    assert out.shape == (2, num_genes)
