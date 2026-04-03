"""
Pathway-histology interaction layers for SpatialTranscriptFormer.

This module defines the building blocks that fuse learnable pathway tokens
with histology patch features via self-attention:

* ``LearnedSpatialEncoder``     — 2-D learned positional embeddings
* ``SpatialTranscriptFormer``   — the full model
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .backbones import get_backbone


class LearnedSpatialEncoder(nn.Module):
    """Encodes 2D spatial coordinates via a small learned MLP.

    Unlike sinusoidal PE, this produces smooth, non-periodic embeddings
    that vary gradually across the tissue. Coordinates are normalised to
    [-1, 1] per-batch before encoding for scale invariance.
    """

    def __init__(self, dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )

    def _normalize_coords(self, coords):
        """Normalize coordinates to [-1, 1] range per batch."""
        # Centre at zero
        coords = coords - coords.mean(dim=1, keepdim=True)
        # Scale to [-1, 1]
        scale = coords.abs().amax(dim=(1, 2), keepdim=True).clamp(min=1.0)
        return coords / scale

    def forward(self, rel_coords):
        """
        Args:
            rel_coords (torch.Tensor): (B, S, 2) spatial coordinates.

        Returns:
            torch.Tensor: (B, S, dim) positional embeddings.
        """
        return self.mlp(self._normalize_coords(rel_coords))


VALID_INTERACTIONS = {"p2p", "p2h", "h2p", "h2h"}


class SpatialTranscriptFormer(nn.Module):
    """Transformer for predicting gene expression from histology and spatial context.

    Integrates histology feature extraction with pathway-based bottleneck
    attention to predict gene transcript counts. Follows a standard Vision
    Transformer architecture where pathway tokens act as [CLS]-like bottlenecks.

    Attributes:
        num_pathways (int): Number of pathway bottlenecks.
    """

    def __init__(
        self,
        num_pathways=50,
        backbone_name="resnet50",
        pretrained=True,
        token_dim=256,
        n_heads=4,
        n_layers=2,
        dropout=0.1,
        use_spatial_pe=True,
        interactions=None,
    ):
        """Initializes the SpatialTranscriptFormer.

        Args:
            num_pathways (int): Number of hidden pathway tokens.
            backbone_name (str): Identifier for backbone model.
            pretrained (bool): Load pretrained backbone weights.
            token_dim (int): Common embedding dimension.
            n_heads (int): Number of attention heads.
            n_layers (int): Number of transformer/interaction layers.
            dropout (float): Dropout probability.
            use_spatial_pe (bool): Incorporate relative gradients into attention.
            interactions (list[str], optional): Which attention interactions to
                enable.  Valid keys are ``p2p``, ``p2h``, ``h2p``, ``h2h``.
                Defaults to all four (full self-attention).
        """
        super().__init__()

        if interactions is None:
            interactions = list(VALID_INTERACTIONS)
        unknown = set(interactions) - VALID_INTERACTIONS
        if unknown:
            raise ValueError(
                f"Unknown interaction keys: {unknown}. "
                f"Valid keys are: {VALID_INTERACTIONS}"
            )
        self.interactions = set(interactions)

        # Enforce minimum 2 layers when h2h is blocked.
        # Layer 1 lets pathways gather patch info, Layer 2 lets patches
        # read the now-informed pathway tokens.
        if "h2h" not in self.interactions and n_layers < 2:
            raise ValueError(
                f"n_layers must be >= 2 when h2h is not in interactions. "
                f"Got n_layers={n_layers}. Layer 1 lets pathways gather spatial info, "
                f"Layer 2 lets patches read contextualized pathways."
            )

        self.num_pathways = num_pathways
        self.use_spatial_pe = use_spatial_pe

        # 1. Image Encoder Backbone
        self.backbone, self.image_feature_dim = get_backbone(
            backbone_name, pretrained=pretrained
        )
        self._backbone_name = backbone_name

        # 2. Image Projector
        self.image_proj = nn.Linear(self.image_feature_dim, token_dim)

        # 3. Learnable pathway tokens (one per pathway, shared across batch)
        self.pathway_tokens = nn.Parameter(torch.randn(1, num_pathways, token_dim))

        # 4. Spatial Positional Encoder (optional)
        self.spatial_encoder = None
        if use_spatial_pe:
            self.spatial_encoder = LearnedSpatialEncoder(token_dim)

        # 5. Interaction Engine (Standard Transformer)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=token_dim,
            nhead=n_heads,
            dim_feedforward=token_dim * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )

        self.fusion_engine = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers,
            norm=nn.LayerNorm(token_dim),
            enable_nested_tensor=False,
        )

        # Interaction engine complete. Model outputs pathways directly.

    def _build_interaction_mask(self, p, s, device):
        """Build ``(P+S, P+S)`` boolean attention mask from ``self.interactions``.

        In PyTorch transformer convention, ``True`` means **blocked**.

        Returns:
            torch.Tensor or None: Mask tensor, or ``None`` when all
            interactions are enabled (no masking needed).
        """
        if self.interactions >= VALID_INTERACTIONS:
            return None  # everything enabled, skip masking

        total = p + s
        # Start fully blocked
        mask = torch.ones(total, total, dtype=torch.bool, device=device)

        if "p2p" in self.interactions:
            mask[:p, :p] = False
        if "p2h" in self.interactions:
            mask[:p, p:] = False
        if "h2p" in self.interactions:
            mask[p:, :p] = False
        if "h2h" in self.interactions:
            mask[p:, p:] = False

        # Always allow self-attention (diagonal)
        mask.fill_diagonal_(False)
        return mask

    @classmethod
    def from_pretrained(cls, checkpoint_dir, device="cpu", **kwargs):
        """Load a pretrained SpatialTranscriptFormer from a checkpoint directory.

        The directory should contain ``config.json`` and ``model.pth``
        (created by :func:`~spatial_transcript_former.checkpoint.save_pretrained`).
        An optional ``gene_names.json`` will be loaded as ``model.gene_names``.

        Args:
            checkpoint_dir (str): Path to checkpoint directory.
            device (str): Torch device to load onto.
            **kwargs: Override any config values.

        Returns:
            SpatialTranscriptFormer: Model in eval mode.
        """
        from spatial_transcript_former.checkpoint import load_pretrained

        return load_pretrained(checkpoint_dir, device=device, **kwargs)

    def forward(
        self,
        x,
        rel_coords=None,
        mask=None,
        return_dense=False,
        return_attention=False,
    ):
        """Main inference path.

        Args:
            x (torch.Tensor): Image data or pre-computed features.
                - (B, 3, H, W): Single image patch.
                - (B, S, D): Pre-computed features.
            rel_coords (torch.Tensor, optional): Spatial relative coordinates.
            mask (torch.Tensor, optional): Boolean padding mask for patches (B, S) where True = Padding.
            return_dense (bool): If True, returns per-patch pathway predictions instead of global predictions.
            return_attention (bool): If True, returns attention maps from all layers.

        Returns:
            torch.Tensor: Predicted pathway scores (B, num_pathways) or (B, N, num_pathways) if return_dense.
            (Optional) list[torch.Tensor]: Attention maps [L, B, H, T, T] if return_attention.
        """
        if x.dim() == 4:
            # Single Patch Mode: (B, C, H, W)
            features = self.backbone(x).unsqueeze(1)
            b, s = features.shape[0], 1
        else:
            # Pre-computed features: (B, S, D)
            features = x
            b, s = features.shape[0], features.shape[1]

        # 1. Project features into latent interaction space
        memory = self.image_proj(features)

        # 1b. Inject Spatial Positional Encodings
        if self.use_spatial_pe:
            if rel_coords is None:
                raise ValueError(
                    "use_spatial_pe is True, but rel_coords was not provided. "
                    "Ensure the dataloader passes spatial coordinates."
                )
            pe = self.spatial_encoder(rel_coords)
            memory = memory + pe

        # 2. Retrieve learnable pathway tokens
        pathway_tokens = self.pathway_tokens.expand(b, -1, -1)  # (B, P, D)
        p = pathway_tokens.shape[1]

        # 3. Process Interactions (Standard ViT sequence: [Pathways, Patches])
        sequence = torch.cat([pathway_tokens, memory], dim=1)  # (B, P + S, D)

        # Build attention mask from configured interactions
        interaction_mask = self._build_interaction_mask(p, s, sequence.device)

        # If sparse/padded inputs, mask out padding so it doesn't attend
        pad_mask = None
        if mask is not None:
            # Pad pathway tokens with False (don't ignore)
            pad_mask = torch.cat(
                [torch.zeros(b, p, dtype=torch.bool, device=mask.device), mask], dim=1
            )

        attentions = []
        if return_attention:
            # Manual forward through fusion_engine layers to extract weights
            # Standard nn.TransformerEncoder suppresses weights for performance.
            x_layer = sequence
            for layer in self.fusion_engine.layers:
                # 1. Attention Block
                qkv = layer.norm1(x_layer) if layer.norm_first else x_layer

                # Extract per-head attention weights without a second forward pass.
                # need_weights=True, average_attn_weights=False → (B, H, T, T).
                attn_output, attn_weights = layer.self_attn(
                    qkv,
                    qkv,
                    qkv,
                    attn_mask=interaction_mask,
                    key_padding_mask=pad_mask,
                    need_weights=True,
                    average_attn_weights=False,
                )
                attentions.append(attn_weights)

                # Continue forward pass reusing attn_output (avoids a second
                # self-attention call and keeps dropout masks consistent).
                if layer.norm_first:
                    x_layer = x_layer + layer.dropout1(attn_output)
                    x_layer = x_layer + layer._ff_block(layer.norm2(x_layer))
                else:
                    x_layer = layer.norm1(x_layer + layer.dropout1(attn_output))
                    x_layer = layer.norm2(x_layer + layer._ff_block(x_layer))
            out = x_layer
        else:
            out = self.fusion_engine(
                sequence, mask=interaction_mask, src_key_padding_mask=pad_mask
            )

        # Extract focused pathway tokens
        processed_pathway_tokens = out[:, :p, :]  # (B, P, D)

        # Extract processed patch tokens
        processed_patch_tokens = out[:, p:, :]  # (B, S, D)

        # 5. Compute pathway scores via cosine similarity
        # L2-normalize both sets of tokens to produce cosine similarities in [-1, 1]
        norm_pathway = F.normalize(processed_pathway_tokens, dim=-1)  # (B, P, D)

        if return_dense:
            # Dense prediction: per-patch cosine similarity with pathway tokens
            norm_patch = F.normalize(processed_patch_tokens, dim=-1)  # (B, S, D)
            # (B, S, D) @ (B, D, P) -> (B, S, P)
            pathway_scores = torch.matmul(norm_patch, norm_pathway.transpose(1, 2))
        else:
            # Global prediction: pool patches first, then compute scores
            global_patch_token = processed_patch_tokens.mean(
                dim=1, keepdim=True
            )  # (B, 1, D)
            norm_global = F.normalize(global_patch_token, dim=-1)  # (B, 1, D)
            pathway_scores = torch.matmul(norm_global, norm_pathway.transpose(1, 2))
            pathway_scores = pathway_scores.squeeze(1)  # (B, P)

        results = [pathway_scores]
        if return_attention:
            results.append(attentions)

        if len(results) == 1:
            return results[0]
        return tuple(results)
