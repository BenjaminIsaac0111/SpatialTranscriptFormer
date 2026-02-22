"""
Multiple-Instance Learning (MIL) baselines for slide-level gene expression.

Models here aggregate an arbitrary number of patch features into a single
slide-level prediction, trained with bag-level (mean-expression) supervision.

* ``AttentionMIL``  — gated attention pooling (Ilse et al., 2018)
* ``TransMIL``      — Nyström Transformer with PPEG positional encoding
                      (Shao et al., 2021), adapted for regression
* ``PPEG``          — multi-scale depthwise conv positional encoding module
* ``TransLayer``    — single Nyström self-attention block used by TransMIL
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from nystrom_attention import NystromAttention


class AttentionMIL(nn.Module):
    """
    Gated Attention Multiple Instance Learning (MIL) for regression.
    Aggregates patch features to predict a global slide-level gene expression vector.

    Reference:
        Ilse et al. (2018). "Attention-based Deep Multiple Instance Learning." ICML.
    """

    def __init__(
        self,
        input_dim=1024,
        hidden_dim=256,
        output_dim=1000,
        dropout=0.25,
        backbone_name=None,
        pretrained=True,
    ):
        super().__init__()

        if backbone_name:
            from .backbones import get_backbone

            self.backbone, self.L = get_backbone(backbone_name, pretrained=pretrained)
        else:
            self.backbone = None
            self.L = input_dim

        self.D = hidden_dim
        self.K = 1

        self.feature_extractor = nn.Sequential(
            nn.Linear(self.L, self.D), nn.ReLU(), nn.Dropout(dropout)
        )

        self.attention = nn.Sequential(
            nn.Linear(self.D, self.D), nn.Tanh(), nn.Linear(self.D, self.K)
        )

        # Gated Attention (optional improvement)
        self.attention_V = nn.Sequential(nn.Linear(self.D, self.D), nn.Tanh())
        self.attention_U = nn.Sequential(nn.Linear(self.D, self.D), nn.Sigmoid())
        self.attention_w = nn.Linear(self.D, self.K)

        self.classifier = nn.Sequential(nn.Linear(self.D, output_dim))

    def forward(self, x, return_attention=False):
        # x: (B, N, L) where N is number of instances (patches)
        # OR x: (B, N, 3, H, W) if backbone is used

        if self.backbone is not None and x.dim() == 5:
            B, N, C, H, W = x.shape
            x = x.view(B * N, C, H, W)
            x = self.backbone(x)
            x = x.view(B, N, -1)

        # Support case where x is (N, L) instead of (B, N, L)
        if x.dim() == 2:
            x = x.unsqueeze(0)

        x = self.feature_extractor(x)  # (B, N, D)

        # Gated Attention Mechanism
        A_V = self.attention_V(x)  # (B, N, D)
        A_U = self.attention_U(x)  # (B, N, D)
        A = self.attention_w(A_V * A_U)  # (B, N, 1)
        A = torch.softmax(A, dim=1)  # (B, N, 1)

        M = torch.sum(x * A, dim=1)  # (B, D)

        Y_prob = self.classifier(M)  # (B, output_dim)

        if return_attention:
            return Y_prob, A
        return Y_prob


class TransMIL(nn.Module):
    """
    TransMIL: Transformer based Correlated Multiple Instance Learning for WSI.
    Adapted for Regression (Gene Expression).

    Reference:
        Shao et al. (2021). "TransMIL: Transformer based Correlated Multiple
        Instance Learning for Whole Slide Image Classification." NeurIPS.
    """

    def __init__(
        self,
        input_dim=1024,
        output_dim=1000,
        dropout=0.1,
        backbone_name=None,
        pretrained=True,
    ):
        super(TransMIL, self).__init__()

        if backbone_name:
            from .backbones import get_backbone

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

    def forward(self, x, return_attention=False):
        # x: (B, N, L)
        # OR x: (B, N, 3, H, W) if backbone is used

        if self.backbone is not None and x.dim() == 5:
            B, N, C, H, W = x.shape
            x = x.view(B * N, C, H, W)
            x = self.backbone(x)
            x = x.view(B, N, -1)

        # Support case where x is (N, L) instead of (B, N, L)
        if x.dim() == 2:
            x = x.unsqueeze(0)

        h = self._fc1(x)  # [B, N, 512]
        h = self.pos_layer(h)

        # Append CLS token
        B = h.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1)
        h = torch.cat((cls_tokens, h), dim=1)

        # Transformer Layers
        # We can extract attention from TransLayer if we modify it to return it
        h, attn1 = self.layer1(h, return_attn=True)
        h, attn2 = self.layer2(h, return_attn=True)

        h = self.norm(h)[:, 0]  # Take CLS token

        logits = self._fc2(h)

        if return_attention:
            # Aggregate attention maps from both layers
            # Focusing on the CLS token's attention to other patches
            # (B, H, 1, N+1) -> (B, N)
            combined_attn = (attn1 + attn2) / 2
            # Take the attention from CLS (index 0) to patches (indices 1:)
            cls_attn = combined_attn[:, :, 0, 1:].mean(dim=1)  # Average over heads
            return logits, cls_attn
        return logits


class PPEG(nn.Module):
    """Pyramid Position Encoding Generator used in TransMIL.

    Applies three parallel depthwise convolutions with kernel sizes 7, 5, and 3
    and sums their outputs with the input.  When the sequence length *N* is a
    perfect square (as it typically is for a regular patch grid), the tokens are
    temporarily reshaped to a 2-D spatial grid (H×W = sqrt(N)×sqrt(N)) so the
    convolutions encode genuine 2-D proximity.  If *N* is not a perfect square
    the module is a no-op and the input is returned unchanged.
    """

    def __init__(self, dim=512):
        super(PPEG, self).__init__()
        self.proj = nn.Conv2d(dim, dim, 7, 1, 7 // 2, groups=dim)
        self.proj1 = nn.Conv2d(dim, dim, 5, 1, 5 // 2, groups=dim)
        self.proj2 = nn.Conv2d(dim, dim, 3, 1, 3 // 2, groups=dim)

    def forward(self, x, H=None, W=None):
        # x: (B, N, D)
        B, N, C = x.shape

        # Attempt to reconstruct 2D grid for PPEG
        if H is None or W is None:
            H = int(N**0.5)
            W = N // H

        if H * W == N:
            # Reshape to grid
            x = x.transpose(1, 2).view(B, C, H, W)
            x = self.proj(x) + self.proj1(x) + self.proj2(x) + x
            x = x.flatten(2).transpose(1, 2)

        return x


class TransLayer(nn.Module):
    """Single pre-norm Nyström self-attention block used inside TransMIL.

    When *return_attn* is ``False`` this is a standard residual block.  When
    ``True``, a separate scaled dot-product between the CLS token (index 0) and
    all tokens is computed *after* the Nyström forward pass to approximate the
    CLS attention map.  This approximation avoids the O(N²) full attention
    matrix, which would OOM on large whole-slide inputs.
    """

    def __init__(self, norm_layer=nn.LayerNorm, dim=512):
        super().__init__()
        self.norm = norm_layer(dim)
        self.attn = NystromAttention(
            dim=dim,
            dim_head=dim // 8,
            heads=8,
            num_landmarks=256,  # number of landmarks
            pinv_iterations=6,  # number of moore-penrose iterations for approximating pinverse. 6 was recommended by the paper
            residual=True,  # whether to do an extra residual with the value or not. supposedly faster convergence if turned on
            dropout=0.1,
        )

    def forward(self, x, return_attn=False):
        if not return_attn:
            x = x + self.attn(self.norm(x))
            return x
        else:
            # If we need attention, we'd ideally get it from nystrom_attention.
            # However, since it's an external package, we might need to approximate or
            # calculate the dot product if x is not too large.
            # For whole slide, a standard dot product (N^2) might OOM.
            # Let's assume for validation/single case we can afford it or return dummy.

            # Dummy for now to avoid OOM on large slides
            h = self.norm(x)
            out = self.attn(h)
            x = x + out

            # Approximate attention: For the CLS token (index 0)
            # We can compute its similarity to all other tokens
            q = h[:, 0:1, :]  # (B, 1, D)
            k = h  # (B, N+1, D)
            sim = torch.matmul(q, k.transpose(1, 2)) / (h.shape[-1] ** 0.5)
            attn = torch.softmax(sim, dim=-1)  # (B, 1, N+1)
            # Reshape to match head format: (B, 1, 1, N+1)
            return x, attn.unsqueeze(1)
