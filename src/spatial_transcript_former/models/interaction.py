"""
Pathway-histology interaction layers for SpatialTranscriptFormer.

This module defines the building blocks that fuse learnable pathway tokens
with histology patch features via self- and cross-attention:

* ``PathwayTokenizer``          — learnable pathway embeddings
* ``LocalPatchMixer``           — scatter-gather depthwise conv for neighbourhood mixing
* ``SinusoidalPositionalEncoder`` — 2-D sinusoidal positional embeddings
* ``EarlyFusionBlock``          — concat-attention-slice wrapper (Jaume-mode)
* ``MultimodalFusion``          — standard Transformer early fusion
* ``NystromEncoder``            — Nyström-based early fusion
* ``SpatialTranscriptFormer``   — the full model
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .backbones import get_backbone


class PathwayTokenizer(nn.Module):
    """Projects pathway indices or data into latent embeddings.

    This module maintains a set of learnable embeddings of size dim for each
    of the num_pathways defined.

    Attributes:
        num_pathways (int): Number of pathway tokens to generate.
        dim (int): Dimension of each pathway token.
    """

    def __init__(self, num_pathways, dim):
        """Initializes the PathwayTokenizer.

        Args:
            num_pathways (int): Total number of pathways.
            dim (int): Token embedding dimension.
        """
        super().__init__()
        self.num_pathways = num_pathways
        self.dim = dim
        self.pathway_embeddings = nn.Parameter(torch.randn(1, num_pathways, dim))

    def forward(self, batch_size):
        """Generates pathway tokens for the batch.

        Args:
            batch_size (int): Number of samples in the batch.

        Returns:
            torch.Tensor: Pathway tokens of shape (batch_size, num_pathways, dim).
        """
        return self.pathway_embeddings.expand(batch_size, -1, -1)


class LocalPatchMixer(nn.Module):
    """Mixes each patch's features with its immediate spatial neighbours via a depthwise conv.

    The forward pass performs three steps:

    1. **Scatter** — place each patch feature vector into a 2-D dense grid at its
       grid-coordinate position.  Coordinates are zero-based and must be
       integer-like (grid indices, *not* raw pixel coordinates).
    2. **Depthwise Conv + GELU** — apply a ``kernel_size x kernel_size`` grouped
       convolution over the spatial grid.  Each channel is processed independently
       so the parameter count scales with *dim*, not *dim²*.
    3. **Gather + Residual** — read the convolved values back at the same grid
       positions and add them to the original features (residual connection).

    A safety guard skips mixing if the bounding-box of grid coordinates exceeds
    256×256 cells, which would indicate malformed / non-grid coordinates and
    could allocate prohibitive memory.
    """

    def __init__(self, dim, kernel_size=3):
        super().__init__()
        self.dim = dim
        self.conv = nn.Conv2d(
            dim, dim, kernel_size=kernel_size, padding=kernel_size // 2, groups=dim
        )
        self.act = nn.GELU()

    def forward(self, x, coords):
        """
        Args:
            x: (B, N, D) Patch features
            coords: (B, N, 2) Coordinates (absolute-ish indices)

        Returns:
            (B, N, D) Enriched features
        """
        B, N, D = x.shape
        device = x.device

        # 1. Normalize coords to grid indices
        # We assume coords are already roughly grid-like or pixel coords.
        # But to be safe, we subtract min and assume unit step?
        # Actually, let's assume coords ARE grid indices for now, or scaled pixels.
        # If they are large pixels (e.g. 10000), we need to know the patch size (224).
        # We'll trust the caller to pass grid-indices.

        # Find bounds per batch? Or global max?
        # To batch this efficiently, we find global bounds in the batch
        min_c = coords.min(dim=1, keepdim=True)[0]  # (B, 1, 2)
        grid_coords = coords - min_c  # Zero-based

        max_c = grid_coords.max(dim=1)[0].max(dim=0)[0]  # (2,)
        H, W = int(max_c[1].item()) + 1, int(max_c[0].item()) + 1

        # Cap memory usage if outliers exist
        if H * W > 256 * 256:
            # Fallback: Don't crash on massive outlier. Just skip mix?
            # Or clamp? Let's skip mixing if grid is absurd.
            return x

        # 2. Scatter
        grid = torch.zeros(B, D, H, W, device=device, dtype=x.dtype)

        b_idx = torch.arange(B, device=device).view(B, 1, 1)
        d_idx = torch.arange(D, device=device).view(1, D, 1)

        gx = grid_coords[..., 0].long()  # (B, N)
        gy = grid_coords[..., 1].long()  # (B, N)

        gy_idx = gy.unsqueeze(1)  # (B, 1, N)
        gx_idx = gx.unsqueeze(1)  # (B, 1, N)

        x_T = x.transpose(1, 2)  # (B, D, N)
        grid[b_idx, d_idx, gy_idx, gx_idx] = x_T

        # 3. Conv
        out_grid = self.act(self.conv(grid))

        # 4. Gather & Residual
        out = out_grid[b_idx, d_idx, gy_idx, gx_idx].transpose(
            1, 2
        )  # (B, D, N) -> (B, N, D)

        return x + out


class GraphPatchMixer(nn.Module):
    """Mixes each patch's features with its k-nearest physical neighbors using Graph Attention.

    This acts as a spatial refiner. It constructs a k-NN graph on the fly from the
    physical coordinates and performs message passing.
    """

    def __init__(self, dim, k=8, heads=4):
        super().__init__()
        self.dim = dim
        self.k = k
        self.heads = heads
        self.head_dim = dim // heads

        assert self.head_dim * heads == dim, "dim must be divisible by heads"

        # GAT-style linear projections
        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)
        self.act = nn.GELU()

    def forward(self, x, coords, mask=None):
        """
        Args:
            x: (B, N, D) Patch features
            coords: (B, N, 2) Coordinates (absolute physical positions)
            mask: (B, N) Boolean padding mask (True = padding)

        Returns:
            (B, N, D) Refined features
        """
        B, N, D = x.shape
        device = x.device

        # 1. Build k-NN graph
        # Compute pairwise distances (B, N, N)
        dist = torch.cdist(coords, coords)

        k_actual = min(self.k + 1, N)  # +1 for self-loop, bounded by N
        dist_for_topk = dist.clone()
        if mask is not None:
            dist_for_topk.masked_fill_(mask.unsqueeze(1).expand(B, N, N), float("inf"))

        _, nn_idx = torch.topk(-dist_for_topk, k=k_actual, dim=-1)  # (B, N, K)

        # 2. Extract neighbor features
        batch_indices = torch.arange(B, device=device).view(-1, 1, 1)  # (B, 1, 1)
        x_neighbors = x[batch_indices, nn_idx, :]  # (B, N, K, D)

        # 3. Message Passing (GAT-style)
        # Query comes from the center node, Key/Value come from neighbors
        qkv_center = self.to_qkv(x)  # (B, N, 3D)
        q_center, _, _ = qkv_center.chunk(3, dim=-1)  # (B, N, D)

        qkv_neighbors = self.to_qkv(x_neighbors)
        _, k_neighbors, v_neighbors = qkv_neighbors.chunk(3, dim=-1)  # (B, N, K, D)

        # Reshape for multi-head attention
        q = q_center.view(B, N, self.heads, self.head_dim)  # (B, N, H, d)
        k = k_neighbors.view(
            B, N, k_actual, self.heads, self.head_dim
        )  # (B, N, K, H, d)
        v = v_neighbors.view(
            B, N, k_actual, self.heads, self.head_dim
        )  # (B, N, K, H, d)

        # Attention scores: Q * K^T
        q = q.unsqueeze(3)  # (B, N, H, 1, d)
        k = k.permute(0, 1, 3, 2, 4)  # (B, N, H, K, d)

        # dot product over 'd'
        attn = (q * k).sum(dim=-1) / (self.head_dim**0.5)  # (B, N, H, K)

        if mask is not None:
            batch_indices = torch.arange(B, device=device).view(-1, 1, 1)  # (B, 1, 1)
            neighbor_is_padded = mask[batch_indices, nn_idx]  # (B, N, K)
            neighbor_is_padded = neighbor_is_padded.unsqueeze(2).expand(
                -1, -1, self.heads, -1
            )
            attn = attn.masked_fill(neighbor_is_padded, float("-inf"))

        attn = F.softmax(attn, dim=-1)  # (B, N, H, K)
        if mask is not None:
            attn = torch.nan_to_num(attn, nan=0.0)

        # Weighted sum of values
        v = v.permute(0, 1, 3, 2, 4)  # (B, N, H, K, d)
        out = (attn.unsqueeze(-1) * v).sum(dim=-2)  # (B, N, H, d)

        # Reshape back to D
        out = out.reshape(B, N, D)
        out = self.proj(self.act(out))

        # 4. Residual Connection
        return x + out


class SinusoidalPositionalEncoder(nn.Module):
    """Encodes 2D spatial coordinates into sinusoidal embeddings.


    Based on the "Attention is All You Need" 1D PE, extended to 2D.
    Each spatial dimension (x, y) is encoded separately with dim/2 channels.
    """

    def __init__(self, dim, temperature=10000):
        super().__init__()
        self.dim = dim
        self.temperature = temperature

    def forward(self, rel_coords):
        """
        Args:
            rel_coords (torch.Tensor): (B, S, 2) relative coordinates.

        Returns:
            torch.Tensor: (B, S, dim) positional embeddings.
        """
        x = rel_coords[..., 0]
        y = rel_coords[..., 1]

        # Split dim into two halves for x and y
        dim_x = self.dim // 2
        dim_y = self.dim - dim_x

        # Create geometric progression of frequencies
        # div_term = 1 / (temperature ** (2i / d_model))
        div_term_x = torch.exp(
            torch.arange(0, dim_x, 2, dtype=torch.float32, device=rel_coords.device)
            * -(torch.log(torch.tensor(self.temperature)) / dim_x)
        )
        div_term_y = torch.exp(
            torch.arange(0, dim_y, 2, dtype=torch.float32, device=rel_coords.device)
            * -(torch.log(torch.tensor(self.temperature)) / dim_y)
        )

        pe_x = torch.zeros(*x.shape, dim_x, device=rel_coords.device)
        pe_y = torch.zeros(*y.shape, dim_y, device=rel_coords.device)

        # Sin/Cos pairs for X
        pe_x[..., 0::2] = torch.sin(x.unsqueeze(-1) * div_term_x)
        pe_x[..., 1::2] = torch.cos(x.unsqueeze(-1) * div_term_x)

        # Sin/Cos pairs for Y
        pe_y[..., 0::2] = torch.sin(y.unsqueeze(-1) * div_term_y)
        pe_y[..., 1::2] = torch.cos(y.unsqueeze(-1) * div_term_y)

        # Concatenate X and Y embeddings -> (B, S, dim)
        return torch.cat([pe_x, pe_y], dim=-1)


class EarlyFusionBlock(nn.Module):
    """Generic wrapper for Early Fusion interaction.

    Handles the common logic for "concatenation -> attention -> slicing" used in
    multimodal interaction (Jaume et al. mode).
    """

    def __init__(self, encoder, masked_quadrants=None):
        """Initializes EarlyFusionBlock.

        Args:
            encoder (nn.Module): The attention mechanism (Standard or Nystrom).
            masked_quadrants (list, optional): Quadrants to mask.
        """
        super().__init__()
        self.encoder = encoder
        self.masked_quadrants = masked_quadrants or []

    def _generate_quadrant_mask(self, num_p, num_h, device):
        """Generates Eq 1 quadrant mask to restrict attention between modalities.

        [ P2P  P2H ]
        [ H2P  H2H ]
        """
        n_total = num_p + num_h
        mask = torch.zeros((n_total, n_total), device=device, dtype=torch.bool)

        p_slice = slice(0, num_p)
        h_slice = slice(num_p, n_total)

        if "H2H" in self.masked_quadrants:
            mask[h_slice, h_slice] = True
        if "H2P" in self.masked_quadrants:
            mask[h_slice, p_slice] = True
        if "P2H" in self.masked_quadrants:
            mask[p_slice, h_slice] = True

        return mask

    def forward(self, p_tokens, h_tokens, return_all_tokens=False):
        """Performs multimodal fusion.

        Args:
            p_tokens: Pathway tokens (B, Np, D)
            h_tokens: Histology tokens (B, Nh, D)
            return_all_tokens: If True, returns full concatenated sequence.

        Returns:
            Contextualized tokens.
        """
        np = p_tokens.shape[1]
        nh = h_tokens.shape[1]

        x = torch.cat([p_tokens, h_tokens], dim=1)

        mask = None
        if self.masked_quadrants:
            mask = self._generate_quadrant_mask(np, nh, p_tokens.device)

        # Standard PyTorch MHA and NystromAttention use different mask signatures.
        # We handle the dispatch here to ensure correct mask application.
        out = self.encoder(x, mask=mask)

        if return_all_tokens:
            return out

        return out[:, :np, :]


class MultimodalFusion(EarlyFusionBlock):
    """Standard Transformer-based Early Fusion."""

    def __init__(self, dim, n_heads, n_layers, dropout=0.1, masked_quadrants=None):
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=n_heads,
            dim_feedforward=dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        super().__init__(encoder=transformer, masked_quadrants=masked_quadrants)


class NystromStack(nn.Module):
    """Sequential stack of Nyström attention layers, compatible with ``EarlyFusionBlock``.

    Each element of *layers* is a ``nn.ModuleList`` of
    ``(norm1, NystromAttention, norm2, FeedForward)`` sub-modules following a
    pre-norm residual layout.  The attention mask convention differs from
    standard PyTorch MHA: ``True`` means *keep* (attend), so the mask is
    inverted (``~mask``) before being passed to ``NystromAttention``.
    """

    def __init__(self, layers):
        super().__init__()
        self.layers = nn.ModuleList(layers)

    def forward(self, x, mask=None):
        attn_mask = ~mask if mask is not None else None

        for norm1, attn, norm2, ff in self.layers:
            x = x + attn(norm1(x), mask=attn_mask)
            x = x + ff(norm2(x))
        return x


class NystromEncoder(EarlyFusionBlock):
    """Nystrom-based Early Fusion."""

    def __init__(
        self,
        dim,
        n_heads,
        n_layers,
        dropout=0.1,
        num_landmarks=256,
        masked_quadrants=None,
    ):
        from nystrom_attention import NystromAttention

        layers = []
        for _ in range(n_layers):
            layers.append(
                nn.ModuleList(
                    [
                        nn.LayerNorm(dim),
                        NystromAttention(
                            dim=dim,
                            heads=n_heads,
                            dim_head=dim // n_heads,
                            num_landmarks=num_landmarks,
                            dropout=dropout,
                        ),
                        nn.LayerNorm(dim),
                        nn.Sequential(
                            nn.Linear(dim, dim * 4),
                            nn.GELU(),
                            nn.Dropout(dropout),
                            nn.Linear(dim * 4, dim),
                            nn.Dropout(dropout),
                        ),
                    ]
                )
            )

        stack = NystromStack(layers)
        super().__init__(encoder=stack, masked_quadrants=masked_quadrants)


# Removed NystromDecoderLayer and NystromDecoder as we are focusing solely on the Jaume pipeline.


class SpatialTranscriptFormer(nn.Module):
    """Transformer for predicting gene expression from histology and spatial context.

    Integrates histology feature extraction with pathway-based bottleneck
    attention to predict gene transcript counts. Supports standard decoder-style
    interaction or early fusion multimodal interaction.

    Architectural Inspiration:
        Jaume et al. (2024). "Modeling Dense Multimodal Interactions Between
        Biological Pathways and Histology for Survival Prediction" (SurvPath).

    Benchmark/Framework:
        Jaume et al. (2024). "HEST-1k: A Dataset for Spatial Transcriptomics
        and Histology Image Analysis." NeurIPS (Spotlight).

    Attributes:
        num_pathways (int): Number of pathway bottlenecks.
        use_nystrom (bool): Whether to use efficient Nystrom attention.
    """

    def __init__(
        self,
        num_genes,
        num_pathways=50,
        backbone_name="resnet50",
        pretrained=True,
        token_dim=512,
        n_heads=8,
        n_layers=2,
        dropout=0.1,
        use_nystrom=False,
        mask_radius=None,
        masked_quadrants=None,
        num_landmarks=256,
        pathway_init=None,
        use_spatial_pe=True,
        early_mixer="conv",
        late_refiner=None,
        k_neighbors=8,
    ):
        """Initializes the SpatialTranscriptFormer.

        Args:
            num_genes (int): Total number of output genes.
            num_pathways (int): Number of hidden pathway tokens.
            backbone_name (str): Identifier for backbone model.
            pretrained (bool): Load pretrained backbone weights.
            token_dim (int): Common embedding dimension.
            n_heads (int): Number of attention heads.
            n_layers (int): Number of transformer/interaction layers.
            dropout (float): Dropout probability.
            use_nystrom (bool): Enable linear-complexity attention.
            mask_radius (float, optional): Distance-based masking threshold.
            masked_quadrants (list, optional): Mask configuration for fusion.
            num_landmarks (int): Landmarks for Nystrom attention.
            pathway_init (Tensor, optional): Biological pathway membership
                matrix of shape (P, G) to initialize gene_reconstructor.
            use_spatial_pe (bool): Incorporate relative gradients into attention.
            early_mixer (str, optional): 'conv' or None.
            late_refiner (str, optional): 'gnn' or None.
            k_neighbors (int): k-NN for GNN refiner.
        """
        super().__init__()

        # Override num_pathways if biological init is provided
        if pathway_init is not None:
            num_pathways = pathway_init.shape[0]
            print(f"Pathway init: overriding num_pathways to {num_pathways}")

        self.num_pathways = num_pathways
        self.use_nystrom = use_nystrom
        self.mask_radius = mask_radius
        self.use_spatial_pe = use_spatial_pe

        # 1. Image Encoder Backbone
        self.backbone, self.image_feature_dim = get_backbone(
            backbone_name, pretrained=pretrained
        )

        # 2. Image Projector & Spatial Modules
        self.image_proj = nn.Linear(self.image_feature_dim, token_dim)

        self.early_mixer = None
        if early_mixer == "conv":
            self.early_mixer = LocalPatchMixer(token_dim)

        self.late_refiner = None
        if late_refiner == "gnn":
            self.late_refiner = GraphPatchMixer(
                dim=token_dim, k=k_neighbors, heads=n_heads
            )

        self.pathway_tokenizer = PathwayTokenizer(num_pathways, token_dim)

        # 3b. Spatial Positional Encoder
        self.spatial_encoder = None
        if use_spatial_pe:
            self.spatial_encoder = SinusoidalPositionalEncoder(token_dim)

        # 4. Interaction Engine (Unified Early Fusion)
        if use_nystrom:
            self.fusion_engine = NystromEncoder(
                token_dim,
                n_heads,
                n_layers,
                dropout=dropout,
                num_landmarks=num_landmarks,
                masked_quadrants=masked_quadrants,
            )
        else:
            self.fusion_engine = MultimodalFusion(
                token_dim,
                n_heads,
                n_layers,
                dropout=dropout,
                masked_quadrants=masked_quadrants,
            )

        self.pathway_activator = nn.Linear(token_dim, 1)
        self.gene_reconstructor = nn.Linear(num_pathways, num_genes)

        if pathway_init is not None:
            with torch.no_grad():
                # gene_reconstructor.weight is (num_genes, num_pathways)
                # pathway_init is (num_pathways, num_genes)
                self.gene_reconstructor.weight.copy_(pathway_init.T)
                self.gene_reconstructor.bias.zero_()
            print("Initialized gene_reconstructor with MSigDB Hallmarks")

        # Dense Head (Whole Slide Mode)
        self.gene_head = nn.Linear(token_dim, num_genes)

    def get_sparsity_loss(self):
        """Computes L1 norm of reconstruction weights for sparsity regularization.

        Returns:
            torch.Tensor: L1 loss value.
        """
        return torch.norm(self.gene_reconstructor.weight, p=1)

    def _generate_spatial_mask(self, rel_coords):
        """Generates distance-based masks for neighborhood attention.

        Args:
            rel_coords (torch.Tensor): Relative coordinates (B, S, 2).

        Returns:
            torch.Tensor: Boolean mask (B, S) where True means ignore.
        """
        if self.mask_radius is None:
            return None

        # Calculate Euclidean distances from center (0, 0)
        dists = torch.norm(rel_coords, dim=-1)

        # Mask elements that are beyond the specified radius
        mask = dists > self.mask_radius
        return mask

    def forward(self, x, rel_coords=None, return_pathways=False):
        """Main inference path.

        Args:
            x (torch.Tensor): Image data or pre-computed features.
                - (B, 3, H, W): Single image patch.
                - (B, S, 3, H, W): Image neighborhood.
                - (B, S, D): Pre-computed features.
            rel_coords (torch.Tensor, optional): Spatial relative coordinates.
            return_pathways (bool): Whether to return pathway activations.

        Returns:
            torch.Tensor: Predicted gene counts (B, num_genes).
            (Optional) torch.Tensor: Pathway activations (B, num_pathways).
        """
        if x.dim() == 5:
            # Neighborhood Mode: Extract features for all patches
            b, s, c, h, w = x.shape
            x_flat = x.view(b * s, c, h, w)
            features_flat = self.backbone(x_flat)
            features = features_flat.view(b, s, -1)
        elif x.dim() == 4:
            # Single Patch Mode
            features = self.backbone(x).unsqueeze(1)
            b, s = features.shape[0], 1
        else:
            # Assumed pre-computed or pre-reshaped features (B, S, D)
            features = x
            if features.dim() == 2:
                features = features.unsqueeze(1)
            b, s = features.shape[0], features.shape[1]

        # 1. Project features into latent interaction space
        memory = self.image_proj(features)

        # 1b. Inject Spatial Positional Encodings
        if self.use_spatial_pe and rel_coords is not None:
            # Add spatial PE to visual features
            pe = self.spatial_encoder(rel_coords)
            memory = memory + pe

        # 1c. Local Patch Mixing (Conv)
        if (
            hasattr(self, "early_mixer")
            and self.early_mixer is not None
            and rel_coords is not None
        ):
            # Enforce mixing of histology features before they attend to pathways
            memory = self.early_mixer(memory, rel_coords)

        # 2. Retrieve learnable pathway tokens
        tgt = self.pathway_tokenizer(b)

        # 3. Process Interactions (Unified Early Fusion Path)
        out = self.fusion_engine(p_tokens=tgt, h_tokens=memory)

        # 4. Project focused pathway tokens back to gene space
        # We project each pathway token to a scalar activation, then use these
        # activations to reconstruct gene expression levels (the bottleneck).
        pathway_activations = self.pathway_activator(out).squeeze(
            -1
        )  # (B, num_pathways)
        gene_expression = self.gene_reconstructor(pathway_activations)  # (B, num_genes)

        if return_pathways:
            return gene_expression, pathway_activations
        return gene_expression

    def forward_dense(self, x, mask=None, return_pathways=False, coords=None):
        """Predicts individualized gene counts for every patch in a slide.

        Optimized for dense prediction on whole slide images via global context.

        Args:
            x (torch.Tensor): Precomputed features for N patches (B, N, D).
            mask (torch.Tensor, optional): Boolean padding mask (B, N) where True = Padding.
            return_pathways (bool): Whether to return pathway scores.
            coords (torch.Tensor, optional): Absolute coordinates (B, N, 2) for PE.

        Returns:
            torch.Tensor: Contextualized predictions for each patch (B, N, G).
            (Optional) torch.Tensor: Pathway scores (B, N, P).
        """
        if x.dim() == 2:
            x = x.unsqueeze(0)

        b = x.shape[0]  # Dynamic batch size

        # 1. Project to latent space
        memory = self.image_proj(x)

        # 1b. Inject Spatial Positional Encodings (Global)
        if self.use_spatial_pe and coords is not None:
            pe = self.spatial_encoder(coords)
            memory = memory + pe

        # 1c. Local Patch Mixing (Conv)
        if (
            hasattr(self, "early_mixer")
            and self.early_mixer is not None
            and coords is not None
        ):
            memory = self.early_mixer(memory, coords)

        # 2. Retrieve learnable pathway tokens (global context)
        pathway_tokens = self.pathway_tokenizer(b)  # (B, P, D)

        # 3. Perform dense interaction (Global Patch-to-Patch + Pathway context)
        # The MultimodalFusion/NystromEncoder class handles the concatenation and quadrant masking.
        all_tokens = self.fusion_engine(
            p_tokens=pathway_tokens, h_tokens=memory, return_all_tokens=True
        )

        # Sliced Histology tokens: indices [Np:]
        np = pathway_tokens.shape[1]
        context_features = all_tokens[:, np:, :]

        # 3b. Local GNN Refinement
        if self.late_refiner is not None and coords is not None:
            # Explicit skip connection: Inject raw spatial/visual memory back into contextualized tokens
            context_features = self.late_refiner(
                context_features + memory, coords, mask=mask
            )

        # 4. Dense prediction head (Pathway Bottleneck enforcement)
        # Project updated patch tokens onto pathway embeddings: (B, N, D) @ (B, D, P) -> (B, N, P)
        # We scale by 1/sqrt(D) to maintain reasonable activation variance, as this is effectively attention.
        pathway_scores = torch.matmul(
            context_features, pathway_tokens.transpose(1, 2)
        ) / (context_features.shape[-1] ** 0.5)

        # Reconstruct Genes: (B, N, P) @ (P, G) -> (B, N, G)
        # Reuses the gene_reconstructor for biological consistency.
        if return_pathways:
            return self.gene_reconstructor(pathway_scores), pathway_scores
        return self.gene_reconstructor(pathway_scores)
