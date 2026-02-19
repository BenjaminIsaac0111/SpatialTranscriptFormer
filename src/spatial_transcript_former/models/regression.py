"""
Patch-level regression baselines for spatial transcriptomics.

Each model takes a single histology patch (B, 3, H, W) and regresses a
gene-expression vector of length *num_genes*.  These serve as lightweight
baselines relative to the full SpatialTranscriptFormer.
"""
import torch.nn as nn
from .backbones import get_backbone

class HE2RNA(nn.Module):
    """ResNet-50 baseline that regresses gene expression from a single patch.

    The backbone's classification head is replaced with a linear layer of size
    *num_genes*.  Weights come from the ``get_backbone`` factory, so any
    supported backbone identifier can be supplied via *backbone*.

    Reference:
        Schmauch et al. (2020). "A deep learning model to predict RNA-Seq
        expression of tumours from whole slide images." *Nature Communications*.
    """
    def __init__(self, num_genes, backbone='resnet50', pretrained=True):
        """Initialise HE2RNA with the chosen backbone and output size."""
        super().__init__()
        self.backbone, self.feature_dim = get_backbone(backbone, pretrained=pretrained, num_classes=num_genes)

    def forward(self, x):
        return self.backbone(x)

class ViT_ST(nn.Module):
    """Vision Transformer baseline for spatial transcriptomics regression.

    Adapts a ViT backbone (default ``vit_b_16``) to the ST task by replacing
    its classification head with a linear layer of size *num_genes*.  Any
    backbone name accepted by ``get_backbone`` can be passed via *model_name*.

    Reference:
        Dosovitskiy et al. (2020). "An Image is Worth 16x16 Words:
        Transformers for Image Recognition at Scale." *ICLR*.
    """
    def __init__(self, num_genes, model_name='vit_b_16', pretrained=True):
        """Initialise ViT_ST with the chosen backbone and output size."""
        super().__init__()
        self.backbone, self.feature_dim = get_backbone(model_name, pretrained=pretrained, num_classes=num_genes)

    def forward(self, x):
        return self.backbone(x)
