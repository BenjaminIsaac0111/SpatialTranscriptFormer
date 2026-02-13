import torch
import torch.nn as nn
import torch.nn.functional as F
from nystrom_attention import NystromAttention

class AttentionMIL(nn.Module):
    """
    Gated Attention MIL for regression.
    Aggregates patch features to predict a global slide-level gene expression vector.
    """
    def __init__(self, input_dim=1024, hidden_dim=256, output_dim=1000, dropout=0.25, backbone_name=None, pretrained=True):
        super().__init__()
        
        if backbone_name:
            from spatial_transcript_former.models.regression import get_backbone
            self.backbone, self.L = get_backbone(backbone_name, pretrained=pretrained)
        else:
            self.backbone = None
            self.L = input_dim
            
        self.D = hidden_dim
        self.K = 1 # attention score dimension
        
        self.feature_extractor = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.attention = nn.Sequential(
            nn.Linear(self.D, self.D),
            nn.Tanh(),
            nn.Linear(self.D, self.K)
        )
        
        # Gated Attention (optional improvement)
        self.attention_V = nn.Sequential(
            nn.Linear(self.D, self.D),
            nn.Tanh()
        )
        self.attention_U = nn.Sequential(
            nn.Linear(self.D, self.D),
            nn.Sigmoid()
        )
        self.attention_w = nn.Linear(self.D, self.K)

        self.classifier = nn.Sequential(
            nn.Linear(self.D, output_dim)
        )

    def forward(self, x):
        # x: (B, N, L) where N is number of instances (patches)
        # OR x: (B, N, 3, H, W) if backbone is used
        
        if self.backbone is not None:
             B, N, C, H, W = x.shape
             x = x.view(B * N, C, H, W)
             x = self.backbone(x)
             x = x.view(B, N, -1)
             
        x = self.feature_extractor(x) # (B, N, D)
        
        # Gated Attention Mechanism
        A_V = self.attention_V(x) # (B, N, D)
        A_U = self.attention_U(x) # (B, N, D)
        A = self.attention_w(A_V * A_U) # (B, N, 1)
        A = torch.softmax(A, dim=1) # (B, N, 1)
        
        M = torch.sum(x * A, dim=1) # (B, D)
        
        Y_prob = self.classifier(M) # (B, output_dim)
        
        return Y_prob, A

class TransMIL(nn.Module):
    """
    TransMIL: Transformer based Correlated Multiple Instance Learning for WSI.
    Adapted for Regression (Gene Expression).
    Reference: Shao et al., NeurIPS 2021.
    """
    def __init__(self, input_dim=1024, output_dim=1000, dropout=0.1, backbone_name=None, pretrained=True):
        super(TransMIL, self).__init__()
        
        if backbone_name:
            from spatial_transcript_former.models.regression import get_backbone
            self.backbone, self.L = get_backbone(backbone_name, pretrained=pretrained)
        else:
            self.backbone = None
            self.L = input_dim
            
        self.pos_layer = PPEG(dim=512)
        self._fc1 = nn.Sequential(nn.Linear(self.L, 512), nn.ReLU())
        self.cls_token = nn.Parameter(torch.randn(1, 1, 512))
        self.n_classes = output_dim
        self.layer1 = TransLayer(dim=512)
        self.layer2 = TransLayer(dim=512)
        self.norm = nn.LayerNorm(512)
        self._fc2 = nn.Linear(512, self.n_classes)

    def forward(self, x):
        # x: (B, N, L)
        # OR x: (B, N, 3, H, W) if backbone is used
        
        if self.backbone is not None:
             B, N, C, H, W = x.shape
             x = x.view(B * N, C, H, W)
             x = self.backbone(x)
             x = x.view(B, N, -1)
             
        h = self._fc1(x) # [B, N, 512]
        
        # PPEG (positional encoding)
        h = self.pos_layer(h, 0) # Implicitly handles H/W usually, but here assumes 1D sequence or needs coords? 
        # Only works if we have spatial structure. 
        # Standard TransMIL assumes a squarified grid or just works on sequence.
        
        # Append CLS token
        B = h.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1)
        h = torch.cat((cls_tokens, h), dim=1)

        # Transformer Layers
        h = self.layer1(h)
        h = self.layer2(h)
        
        h = self.norm(h)[:, 0] # Take CLS token
        
        logits = self._fc2(h)
        return logits

class PPEG(nn.Module):
    def __init__(self, dim=512):
        super(PPEG, self).__init__()
        self.proj = nn.Conv2d(dim, dim, 7, 1, 7//2, groups=dim)
        self.proj1 = nn.Conv2d(dim, dim, 5, 1, 5//2, groups=dim)
        self.proj2 = nn.Conv2d(dim, dim, 3, 1, 3//2, groups=dim)

    def forward(self, x, H=None, W=None):
        # x: (B, N, D)
        # Ideally we need H, W to reconstruct grid. 
        # If not provided, skip or approximate (standard TransMIL assumes N is square-ish or reshapes)
        # For this implementation, we'll skip PPEG if H,W unknown to avoid crash, or do a naive 1D conv?
        # Let's do 1D conv approximation if 2D not possible, or Just return x
        
        B, N, C = x.shape
        # Naive: try to find closest square?
        if H is None:
            H = int(N**0.5)
            W = N // H
            # Crop to fit square for PPEG? Or Padding?
            # Standard implementation pads.
            
            # Let's just return X for now to be safe until we handle spatial coords explicitly
            return x
            
        return x

class TransLayer(nn.Module):
    def __init__(self, norm_layer=nn.LayerNorm, dim=512):
        super().__init__()
        self.norm = norm_layer(dim)
        self.attn = NystromAttention(
            dim = dim,
            dim_head = dim // 8,
            heads = 8,
            num_landmarks = 256,    # number of landmarks
            pinv_iterations = 6,    # number of moore-penrose iterations for approximating pinverse. 6 was recommended by the paper
            residual = True,         # whether to do an extra residual with the value or not. supposedly faster convergence if turned on
            dropout=0.1
        )

    def forward(self, x):
        x = x + self.attn(self.norm(x))
        return x
