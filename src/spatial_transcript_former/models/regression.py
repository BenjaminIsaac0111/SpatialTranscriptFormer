import torch.nn as nn
from .backbones import get_backbone

class HE2RNA(nn.Module):
    """
    HE2RNA (ResNet-based regression).
    Takes an image patch and predicts gene expression vector.
    """
    def __init__(self, num_genes, backbone='resnet50', pretrained=True):
        super().__init__()
        self.backbone, self.feature_dim = get_backbone(backbone, pretrained=pretrained, num_classes=num_genes)

    def forward(self, x):
        return self.backbone(x)

class ViT_ST(nn.Module):
    """
    Vision Transformer for Spatial Transcriptomics regression.
    """
    def __init__(self, num_genes, model_name='vit_b_16', pretrained=True):
        super().__init__()
        self.backbone, self.feature_dim = get_backbone(model_name, pretrained=pretrained, num_classes=num_genes)

    def forward(self, x):
        return self.backbone(x)
