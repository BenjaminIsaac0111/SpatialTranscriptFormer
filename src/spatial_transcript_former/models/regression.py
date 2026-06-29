"""
Patch-level regression baselines for spatial transcriptomics.

Each model takes a single histology patch (B, 3, H, W) and regresses a
pathway-activity vector of length *num_pathways*.  These serve as lightweight
baselines relative to the full SpatialTranscriptFormer.
"""

import torch.nn as nn
from .backbones import get_backbone


class HE2RNA(nn.Module):
    """ResNet-50 baseline that regresses pathway activity from a single patch.

    The backbone's classification head is replaced with a linear layer of size
    *num_pathways*.  Weights come from the ``get_backbone`` factory, so any
    supported backbone identifier can be supplied via *backbone*.

    Reference:
        Schmauch et al. (2020). "A deep learning model to predict RNA-Seq
        expression of tumours from whole slide images." *Nature Communications*.
    """

    def __init__(self, num_pathways, backbone="resnet50", pretrained=True):
        """Initialise HE2RNA with the chosen backbone and output size."""
        super().__init__()
        self.backbone, self.feature_dim = get_backbone(
            backbone, pretrained=pretrained, num_classes=num_pathways
        )

    def forward(self, x):
        return self.backbone(x)


class ViT_ST(nn.Module):
    """Vision Transformer baseline for spatial transcriptomics regression.

    Adapts a ViT backbone (default ``vit_b_16``) to the ST task by replacing
    its classification head with a linear layer of size *num_pathways*.  Any
    backbone name accepted by ``get_backbone`` can be passed via *model_name*.

    Reference:
        Dosovitskiy et al. (2020). "An Image is Worth 16x16 Words:
        Transformers for Image Recognition at Scale." *ICLR*.
    """

    def __init__(self, num_pathways, model_name="vit_b_16", pretrained=True):
        """Initialise ViT_ST with the chosen backbone and output size."""
        super().__init__()
        self.backbone, self.feature_dim = get_backbone(
            model_name, pretrained=pretrained, num_classes=num_pathways
        )

    def forward(self, x):
        return self.backbone(x)


class LinearProbe(nn.Module):
    """Linear probe baseline on precomputed features."""

    def __init__(self, input_dim=1024, num_pathways=50):
        super().__init__()
        self.classifier = nn.Linear(input_dim, num_pathways)

    def forward(self, x):
        # Supports input shape (B, L) or (B, N, L)
        return self.classifier(x)


class MLPProbe(nn.Module):
    """MLP baseline on precomputed features."""

    def __init__(self, input_dim=1024, num_pathways=50, hidden_dim=512, dropout=0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_pathways),
        )

    def forward(self, x):
        # Supports input shape (B, L) or (B, N, L)
        return self.mlp(x)
