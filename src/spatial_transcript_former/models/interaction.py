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


class MultimodalFusion(nn.Module):
    """Early Fusion mechanism for pathway and histology interaction.
    
    Architectural basis:
        Jaume et al. (2024). "Modeling Dense Multimodal Interactions Between 
        Biological Pathways and Histology for Survival Prediction." CVPR.
        (SurvPath - see Equation 1 for quadrant-based attention masking)
        
    Multimodal Framework Context:
        Jaume et al. (2024). "HEST-1k: A Dataset for Spatial Transcriptomics 
        and Histology Image Analysis." NeurIPS (Spotlight).
    Inherits from Jaume et al. (Eq 1), concatenating Pathway and Histology tokens
    for shared attention interaction.

    Attributes:
        masked_quadrants (list): List of quadrants to mask out (e.g., ['H2H']).
        transformer (nn.TransformerEncoder): Transformer for fusion.

    .. warning::
        This module uses standard O(N^2) attention. It is suitable for small neighborhoods
        (e.g., N < 500) but MUST NOT be used for Whole Slide Images (WSI) where N > 10,000.
        For WSI, use NystromEncoder or other linear attention mechanisms.
    """

    def __init__(self, dim, n_heads, n_layers, dropout=0.1, masked_quadrants=None):
        """Initializes MultimodalFusion.

        Args:
            dim (int): Hidden dimension size.
            n_heads (int): Number of attention heads.
            n_layers (int): Number of transformer layers.
            dropout (float): Dropout probability.
            masked_quadrants (list, optional): Quadrants to mask in attention.
        """
        super().__init__()
        self.masked_quadrants = masked_quadrants or []

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=n_heads,
            dim_feedforward=2048,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

    def _generate_quadrant_mask(self, num_p, num_h, device):
        """Generates Eq 1 quadrant mask to restrict attention between modalities.

        The mask is structured as:
        [ P2P  P2H ]
        [ H2P  H2H ]

        Args:
            num_p (int): Number of pathway tokens.
            num_h (int): Number of histology tokens.
            device (torch.device): Device to place the mask on.

        Returns:
            torch.Tensor: Boolean mask where True indicates ignored locations.
        """
        n_total = num_p + num_h
        mask = torch.zeros((n_total, n_total), device=device, dtype=torch.bool)

        p_slice = slice(0, num_p)
        h_slice = slice(num_p, n_total)

        if 'H2H' in self.masked_quadrants:
            mask[h_slice, h_slice] = True
        if 'H2P' in self.masked_quadrants:
            mask[h_slice, p_slice] = True
        if 'P2H' in self.masked_quadrants:
            mask[p_slice, h_slice] = True

        return mask

    def forward(self, p_tokens, h_tokens):
        """Performs multimodal fusion via self-attention.

        Args:
            p_tokens (torch.Tensor): Pathway tokens (B, Np, D).
            h_tokens (torch.Tensor): Histology tokens (B, Nh, D).

        Returns:
            torch.Tensor: Contextualized pathway tokens (B, Np, D).
        """
        nh = h_tokens.shape[1]
        np = p_tokens.shape[1]

        # Concatenate tokens from both modalities
        x = torch.cat([p_tokens, h_tokens], dim=1)

        mask = self._generate_quadrant_mask(np, nh, p_tokens.device)
        out = self.transformer(x, mask=mask)

        # Return only relevant pathway tokens as the bottleneck state
        return out[:, :np, :]


class NystromEncoder(nn.Module):
    """Memory-efficient Transformer Encoder using Nystrom Attention.
    
    Reduces attention complexity from O(N^2) to O(N*m), making it suitable for
    processing large spatial neighborhoods or whole slides.
    
    Reference:
        Xiong et al. (2021). "Nyströmformer: A Nyström-based Algorithm for 
        Approximating Self-Attention." AAAI.
    
    Attributes:
        layers (nn.ModuleList): Sequential Nystrom attention layers.
    """

    def __init__(self, dim, n_heads, n_layers, dropout=0.1, num_landmarks=256):
        """Initializes NystromEncoder.

        Args:
            dim (int): Input dimension.
            n_heads (int): Number of attention heads.
            n_layers (int): Number of attention layers.
            dropout (float): Dropout rate.
            num_landmarks (int): Number of landmark points for Nystrom approximation.
        """
        super().__init__()
        from nystrom_attention import NystromAttention

        self.layers = nn.ModuleList([])
        for _ in range(n_layers):
            self.layers.append(nn.ModuleList([
                nn.LayerNorm(dim),
                NystromAttention(
                    dim=dim,
                    heads=n_heads,
                    dim_head=dim // n_heads,
                    num_landmarks=num_landmarks,
                    dropout=dropout
                ),
                nn.LayerNorm(dim),
                nn.Sequential(
                    nn.Linear(dim, dim * 4),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(dim * 4, dim),
                    nn.Dropout(dropout)
                )
            ]))

    def forward(self, p_tokens, h_tokens, mask=None):
        """Forward pass with Nystrom self-attention.

        Args:
            p_tokens (torch.Tensor): Primary tokens (queries).
            h_tokens (torch.Tensor): Context tokens (keys/values).
            mask (torch.Tensor, optional): Boolean padding mask.

        Returns:
            torch.Tensor: Attended pathway tokens.
        """
        np = p_tokens.shape[1]
        x = torch.cat([p_tokens, h_tokens], dim=1)

        # Nystrom mask expects True to keep, so we invert padding masks
        attn_mask = ~mask if mask is not None else None

        for norm1, attn, norm2, ff in self.layers:
            x = x + attn(norm1(x), mask=attn_mask)
            x = x + ff(norm2(x))

        return x[:, :np, :]


class NystromDecoderLayer(nn.Module):
    """Hybrid layer combining Nystrom Self-Attention and Standard Cross-Attention.

    Optimized for scenarios where queries (Pathways) are few but context (Histology)
    is vast. Self-attention is efficient (Nystrom), while Cross-attention remains
    standard as its complexity is already linear with respect to context size.
    """

    def __init__(self, dim, n_heads, dropout=0.1, num_landmarks=256):
        """Initializes NystromDecoderLayer.

        Args:
            dim (int): Hidden dimension.
            n_heads (int): Heads for multi-head attention.
            dropout (float): Dropout probability.
            num_landmarks (int): Nystrom landmarks.
        """
        super().__init__()
        from nystrom_attention import NystromAttention

        self.norm1 = nn.LayerNorm(dim)
        self.self_attn = NystromAttention(
            dim=dim, heads=n_heads, dim_head=dim // n_heads,
            num_landmarks=num_landmarks, dropout=dropout
        )

        self.norm2 = nn.LayerNorm(dim)
        # Using manual linear layers and SDPA for memory efficiency
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.n_heads = n_heads
        self.dropout_p = dropout

        self.norm3 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout)
        )

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None):
        """Processes tgt through self and cross attention.

        Args:
            tgt (torch.Tensor): Current state of queries (Pathways).
            memory (torch.Tensor): Latent features of histology patches.
            tgt_mask: (Ignored, for API compatibility).
            memory_mask: (Ignored, for API compatibility).
            tgt_key_padding_mask (torch.Tensor): Boolean mask for queries.
            memory_key_padding_mask (torch.Tensor): Boolean mask for context (True=Ignore).

        Returns:
            torch.Tensor: Updated tgt tokens.
        """
        # Self-Attention on Pathways
        sa_mask = ~tgt_key_padding_mask if tgt_key_padding_mask is not None else None
        tgt = tgt + self.self_attn(self.norm1(tgt), mask=sa_mask)

        # Cross-Attention (Pathways query Histology)
        # Use scaled_dot_product_attention for memory efficiency (Flash Attention)
        q = self.q_proj(self.norm2(tgt))
        k = self.k_proj(memory)
        v = self.v_proj(memory)
        
        # Reshape for multi-head SDPA
        b, nq, d = q.shape
        _, nk, _ = k.shape
        h = self.n_heads
        head_dim = d // h
        
        q = q.view(b, nq, h, head_dim).transpose(1, 2)
        k = k.view(b, nk, h, head_dim).transpose(1, 2)
        v = v.view(b, nk, h, head_dim).transpose(1, 2)
        
        # Prepare mask for SDPA (Standard MHA mask: True = ignore)
        # SDPA mask: True = KEEP (if mask is provided) or Float mask
        # If memory_key_padding_mask is provided, we convert it to an additive mask or ignore it if SDPA doesn't support bool masks easily
        # PyTorch 2.0+ SDPA supports attn_mask of shape (B, H, L, S)
        sdpa_mask = None
        if memory_key_padding_mask is not None:
             # memory_key_padding_mask is (B, nk)
             # Expand to (B, 1, 1, nk) for broadcasting
             sdpa_mask = memory_key_padding_mask.view(b, 1, 1, nk)
             # Invert because SDPA mask uses False to ignore
             sdpa_mask = ~sdpa_mask
             
        attn_out = F.scaled_dot_product_attention(
            q, k, v, 
            attn_mask=sdpa_mask,
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=False
        )
        
        # Combine heads and project back
        attn_out = attn_out.transpose(1, 2).contiguous().view(b, nq, d)
        attn_out = self.out_proj(attn_out)
        
        tgt = tgt + attn_out

        # Feed Forward
        tgt = tgt + self.ff(self.norm3(tgt))
        return tgt


class NystromDecoder(nn.Module):
    """Sequential stack of NystromDecoderLayers for pathway refinement."""

    def __init__(self, layers):
        """Initializes the decoder with a list of layers.

        Args:
            layers (list): List of NystromDecoderLayer modules.
        """
        super().__init__()
        self.layers = nn.ModuleList(layers)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None):
        """Passes tgt through all decoder layers.

        Args:
            tgt (torch.Tensor): Target tokens.
            memory (torch.Tensor): Memory context.
            ... (masks for individual attention/padding)

        Returns:
            torch.Tensor: Re-contextualized target tokens.
        """
        out = tgt
        for layer in self.layers:
            out = layer(
                out, memory,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask
            )
        return out


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
        fusion_mode (str): 'decoder' or 'jaume' mode.
    """

    def __init__(self,
                 num_genes,
                 num_pathways=50,
                 backbone_name='resnet50',
                 pretrained=True,
                 token_dim=512,
                 n_heads=8,
                 n_layers=2,
                 dropout=0.1,
                 use_nystrom=False,
                 mask_radius=None,
                 fusion_mode='decoder',
                 masked_quadrants=None,
                 num_landmarks=256):
        """Initializes the SpatialTranscriptFormer.

        Args:
            num_genes (int): Total number of output genes.
            num_pathways (int): Number of hidden pathway tokens.
            backbone_name (str): Identifier for backbone model.
            pretrained (bool): Load pretrained backbone weights.
            token_dim (int): Common embedding dimension.
            n_heads (int): Hits for attention layers.
            n_layers (int): Number of transformer/interaction layers.
            dropout (float): Dropout probability.
            use_nystrom (bool): Enable linear-complexity attention.
            mask_radius (float, optional): Distance-based masking threshold.
            fusion_mode (str): Architecture style ('decoder' or 'jaume').
            masked_quadrants (list, optional): Mask configuration for fusion.
            num_landmarks (int): Landmarks for Nystrom attention.
        """
        super().__init__()

        self.num_pathways = num_pathways
        self.use_nystrom = use_nystrom
        self.mask_radius = mask_radius
        self.fusion_mode = fusion_mode

        # 1. Image Encoder Backbone
        self.backbone, self.image_feature_dim = get_backbone(backbone_name, pretrained=pretrained)

        # 2. Image Projector - Unified space for multimodal interaction
        self.image_proj = nn.Linear(self.image_feature_dim, token_dim)

        # 3. Pathway Tokenizer - Learnable latent representations
        self.pathway_tokenizer = PathwayTokenizer(num_pathways, token_dim)

        # 4. Interaction Module
        if fusion_mode == 'jaume':
            if use_nystrom:
                self.interaction = NystromEncoder(
                    token_dim, n_heads, n_layers, dropout=dropout, num_landmarks=num_landmarks
                )
            else:
                layer = nn.TransformerEncoderLayer(
                    d_model=token_dim, nhead=n_heads, dropout=dropout, batch_first=True
                )
                self.interaction = nn.TransformerEncoder(layer, num_layers=n_layers)
        else:  # Decoder Mode
            if use_nystrom:
                layers = [
                    NystromDecoderLayer(token_dim, n_heads, dropout=dropout,
                                        num_landmarks=num_landmarks)
                    for _ in range(n_layers)
                ]
                self.interaction = NystromDecoder(layers)
            else:
                decoder_layer = nn.TransformerDecoderLayer(
                    d_model=token_dim,
                    nhead=n_heads,
                    dim_feedforward=2048,
                    dropout=dropout,
                    batch_first=True
                )
                self.interaction = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)

        # 5. Prediction Head (Pathway Bottleneck)
        self.pathway_activator = nn.Linear(token_dim, 1)
        self.gene_reconstructor = nn.Linear(num_pathways, num_genes)

        # 6. Dense Head (for Whole Slide Mode/Local Predictions)
        # We want to predict genes from the interaction outputs.
        # If outputs are (B, N, D), we can project to genes.
        # Ideally we should use the same reconstruction logic: D -> Pathways -> Genes?
        # But 'out' is (B, N, D) in dense mode.
        # We can learn D -> Genes directly (current) or D -> Pathways -> Genes.
        # Let's keep it simple for now but use the interaction module properly.
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

    def forward(self, x, rel_coords=None):
        """Main inference path.

        Args:
            x (torch.Tensor): Image data or pre-computed features.
                - (B, 3, H, W): Single image patch.
                - (B, S, 3, H, W): Image neighborhood.
                - (B, S, D): Pre-computed features.
            rel_coords (torch.Tensor, optional): Spatial relative coordinates.

        Returns:
            torch.Tensor: Predicted gene counts (B, num_genes).
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

        # 2. Retrieve learnable pathway tokens
        tgt = self.pathway_tokenizer(b)

        # 3. Process interactions
        if self.fusion_mode == 'jaume':
            out = self.interaction(tgt, memory)
        else:
            # Decoder processes neighborhood with optional spatial gating
            key_mask = None
            if rel_coords is not None and self.mask_radius is not None:
                key_mask = self._generate_spatial_mask(rel_coords)

            out = self.interaction(
                tgt,
                memory,
                memory_key_padding_mask=key_mask
            )

        # 4. Project focused pathway tokens back to gene space
        # We project each pathway token to a scalar activation, then use these
        # activations to reconstruct gene expression levels (the bottleneck).
        pathway_activations = self.pathway_activator(out).squeeze(-1)  # (B, num_pathways)
        gene_expression = self.gene_reconstructor(pathway_activations)  # (B, num_genes)

        return gene_expression

    def forward_dense(self, x):
        """Predicts individualized gene counts for every patch in a slide.

        Optimized for dense prediction on whole slide images via global context.

        Args:
            x (torch.Tensor): Precomputed features for N patches (1, N, D).

        Returns:
            torch.Tensor: Contextualized predictions for each patch (1, N, G).
        """
        if x.dim() == 2:
            x = x.unsqueeze(0)
            
        # 1. Project to latent space
        memory = self.image_proj(x)

        # 2. Retrieve learnable pathway tokens (global context)
        pathway_tokens = self.pathway_tokenizer(1) # (1, P, D)

        # 3. Perform dense interaction
        # We treat 'memory' (patches) as the target because we want to update THEM.
        # We treat 'pathway_tokens' as the context/memory for cross-attention.
        # Structure:
        #   Self-Attention on Patches (Nystrom - Efficient)
        #   Cross-Attention to Pathways (Standard - Efficient since P is small)
        context_features = self.interaction(tgt=memory, memory=pathway_tokens)

        # 4. Dense prediction head (Pathway Bottleneck enforcement)
        # We must project the updated patch tokens onto the pathway embeddings to see
        # "How much of Pathway K is in Patch N?"
        # context_features: (1, N, D)
        # pathway_tokens: (1, P, D)
        
        # Calculate scores: (1, N, D) @ (1, D, P) -> (1, N, P)
        # This acts as a dynamic "Pathway Activation" map for every patch
        pathway_scores = torch.matmul(context_features, pathway_tokens.transpose(1, 2))
        
        # Reconstruct Genes from these scores: (1, N, P) @ (P, G) -> (1, N, G)
        # We reuse the same gene_reconstructor used in the single-patch forward pass
        # This ensures biological consistency.
        return self.gene_reconstructor(pathway_scores)
