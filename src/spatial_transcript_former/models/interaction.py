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
    """Mixes patch features with their spatial neighbors using a Scatter-Gather Conv."""
    
    def __init__(self, dim, kernel_size=3):
        super().__init__()
        self.dim = dim
        self.conv = nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=kernel_size//2, groups=dim)
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
        min_c = coords.min(dim=1, keepdim=True)[0] # (B, 1, 2)
        grid_coords = coords - min_c # Zero-based
        
        max_c = grid_coords.max(dim=1)[0].max(dim=0)[0] # (2,)
        H, W = int(max_c[1].item()) + 1, int(max_c[0].item()) + 1
        
        # Cap memory usage if outliers exist
        if H * W > 256 * 256: 
             # Fallback: Don't crash on massive outlier. Just skip mix? 
             # Or clamp? Let's skip mixing if grid is absurd.
             return x

        # 2. Scatter
        grid = torch.zeros(B, D, H, W, device=device, dtype=x.dtype)
        
        # We need advanced indexing: grid[b, :, y, x] = feat
        # Flatten batch for easier indexing if B>1, but here we loop or use gather scatter
        # Let's loop for clarity and safety with B=1 expected mostly. 
        # Actually, pytorch advanced indexing works fine.
        
        # indices: (B, N, 2) -> we want to assign x[b,n] to grid[b, :, y[b,n], x[b,n]]
        # This is tricky to vectorize perfectly across High B without scatter.
        # But B is usually small (1 for Inference, 8 for Train).
        
        # Let's just Loop over Batch
        for b in range(B):
            gx = grid_coords[b, :, 0].long()
            gy = grid_coords[b, :, 1].long()
            
            # Simple assignment (last one wins if collision, but shouldn't be collisions in ST)
            grid[b, :, gy, gx] = x[b].T
            
        # 3. Conv
        out_grid = self.act(self.conv(grid))
        
        # 4. Gather & Residual
        out = torch.zeros_like(x)
        for b in range(B):
            gx = grid_coords[b, :, 0].long()
            gy = grid_coords[b, :, 1].long()
            
            # Extract columns
            mixed = out_grid[b, :, gy, gx].T # (N, D)
            out[b] = mixed
            
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
        div_term_x = torch.exp(torch.arange(0, dim_x, 2, dtype=torch.float32, device=rel_coords.device) * -(torch.log(torch.tensor(self.temperature)) / dim_x))
        div_term_y = torch.exp(torch.arange(0, dim_y, 2, dtype=torch.float32, device=rel_coords.device) * -(torch.log(torch.tensor(self.temperature)) / dim_y))

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

        if 'H2H' in self.masked_quadrants:
            mask[h_slice, h_slice] = True
        if 'H2P' in self.masked_quadrants:
            mask[h_slice, p_slice] = True
        if 'P2H' in self.masked_quadrants:
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
        
        # Check if encoder supports/needs mask
        # Helper: standard transformer accepts src_mask or mask
        # Nystrom accepts mask
        
        mask = None
        if self.masked_quadrants:
            mask = self._generate_quadrant_mask(np, nh, p_tokens.device)
            
        # Dispatch to encoder
        if isinstance(self.encoder, nn.TransformerEncoder):
            out = self.encoder(x, mask=mask)
        else:
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
            dim_feedforward=2048,
            dropout=dropout,
            batch_first=True
        )
        transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        super().__init__(encoder=transformer, masked_quadrants=masked_quadrants)


class NystromStack(nn.Module):
    """Stack of Nystrom Layers for use inside EarlyFusionBlock."""
    def __init__(self, layers):
        super().__init__()
        self.layers = nn.ModuleList(layers)
        
    def forward(self, x, mask=None):
        # Nystrom attention typically expects an inverted mask (True=Keep) compared to
        # standard PyTorch MHA (True=Ignore). However, implementation specifics vary.
        # Here we follow the convention: attn_mask = ~mask if mask is not None else None.
        
        attn_mask = ~mask if mask is not None else None
        
        for norm1, attn, norm2, ff in self.layers:
            x = x + attn(norm1(x), mask=attn_mask)
            x = x + ff(norm2(x))
        return x


class NystromEncoder(EarlyFusionBlock):
    """Nystrom-based Early Fusion."""
    
    def __init__(self, dim, n_heads, n_layers, dropout=0.1, num_landmarks=256, masked_quadrants=None):
        from nystrom_attention import NystromAttention

        layers = []
        for _ in range(n_layers):
            layers.append(nn.ModuleList([
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
            
        stack = NystromStack(layers)
        super().__init__(encoder=stack, masked_quadrants=masked_quadrants)


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
        
        # Prepare mask for SDPA using Flash Attention compatible logic
        # Typically, memory_key_padding_mask is (B, K) where True=Ignore.
        # SDPA expects a float mask (additive) or bool mask.
        sdpa_mask = None
        if memory_key_padding_mask is not None:
             # Expand to (B, 1, 1, nk) for broadcasting across heads and queries
             sdpa_mask = memory_key_padding_mask.view(b, 1, 1, nk)
             # Invert because SDPA mask uses False to ignore in some versions, 
             # but standard is usually True=Keep for bool masks in attention ops.
             # We align with Nystrom convention: ~mask.
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
                 num_landmarks=256,
                 pathway_init=None,
                 use_spatial_pe=True):
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
            pathway_init (Tensor, optional): Biological pathway membership
                matrix of shape (P, G) to initialize gene_reconstructor.
            use_spatial_pe (bool): Incorporate relative gradients into attention.
        """
        super().__init__()

        # Override num_pathways if biological init is provided
        if pathway_init is not None:
            num_pathways = pathway_init.shape[0]
            print(f"Pathway init: overriding num_pathways to {num_pathways}")

        self.num_pathways = num_pathways
        self.use_nystrom = use_nystrom
        self.mask_radius = mask_radius
        self.fusion_mode = fusion_mode
        self.use_spatial_pe = use_spatial_pe

        # 1. Image Encoder Backbone
        self.backbone, self.image_feature_dim = get_backbone(backbone_name, pretrained=pretrained)

        # 2. Image Projector - Unified space for multimodal interaction
        self.image_proj = nn.Linear(self.image_feature_dim, token_dim)

        # 3. Pathway Tokenizer - Learnable latent representations
        self.pathway_tokenizer = PathwayTokenizer(num_pathways, token_dim)

        # 3b. Spatial Positional Encoder
        self.spatial_encoder = None
        if use_spatial_pe:
            self.spatial_encoder = SinusoidalPositionalEncoder(token_dim)

        # 4. Fusion Engine
        self.fusion_engine = None
        self.decoder_engine = None
        
        if fusion_mode == 'jaume':
            if use_nystrom:
                self.fusion_engine = NystromEncoder(
                    token_dim, n_heads, n_layers, dropout=dropout, num_landmarks=num_landmarks,
                    masked_quadrants=masked_quadrants
                )
            else:
                self.fusion_engine = MultimodalFusion(
                    token_dim, n_heads, n_layers, dropout=dropout,
                    masked_quadrants=masked_quadrants
                )
        else:  # Decoder Mode
            # Local Mixing Layer (Depthwise Conv)
            self.local_mixer = LocalPatchMixer(token_dim)
            print("Enabled LocalPatchMixer (Scatter-Gather Conv)")

            if use_nystrom:
                layers = [
                    NystromDecoderLayer(token_dim, n_heads, dropout=dropout,
                                        num_landmarks=num_landmarks)
                    for _ in range(n_layers)
                ]
                self.decoder_engine = NystromDecoder(layers)
            else:
                decoder_layer = nn.TransformerDecoderLayer(
                    d_model=token_dim,
                    nhead=n_heads,
                    dim_feedforward=2048,
                    dropout=dropout,
                    batch_first=True
                )
                self.decoder_engine = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)

        # 5. Prediction Head (Pathway Bottleneck)
        self.pathway_activator = nn.Linear(token_dim, 1)
        self.gene_reconstructor = nn.Linear(num_pathways, num_genes)
        
        # Initialize gene_reconstructor weights...

        if pathway_init is not None:
            with torch.no_grad():
                # gene_reconstructor.weight is (num_genes, num_pathways)
                # pathway_init is (num_pathways, num_genes)
                self.gene_reconstructor.weight.copy_(pathway_init.T)
                self.gene_reconstructor.bias.zero_()
            print("Initialized gene_reconstructor with MSigDB Hallmarks")

        # 6. Dense Head (Whole Slide Mode)
        # Predicts genes directly from token dimension for local patch predictions.
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

        # 2. Retrieve learnable pathway tokens
        tgt = self.pathway_tokenizer(b)

        # 3. Process interactions
        if self.fusion_mode == 'jaume':
            out = self.fusion_engine(p_tokens=tgt, h_tokens=memory)
        else:
            # Decoder processes neighborhood with optional spatial gating
            key_mask = None
            if rel_coords is not None and self.mask_radius is not None:
                key_mask = self._generate_spatial_mask(rel_coords)

            out = self.decoder_engine(
                tgt,
                memory,
                memory_key_padding_mask=key_mask
            )

        # 4. Project focused pathway tokens back to gene space
        # We project each pathway token to a scalar activation, then use these
        # activations to reconstruct gene expression levels (the bottleneck).
        pathway_activations = self.pathway_activator(out).squeeze(-1)  # (B, num_pathways)
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

        # 2. Retrieve learnable pathway tokens (global context)
        pathway_tokens = self.pathway_tokenizer(b) # (B, P, D)

        # 3. Perform dense interaction
        if self.fusion_mode == 'jaume':
            # Jaume mode: Early fusion via concatenation
            # The MultimodalFusion/NystromEncoder class handles the concatenation and masking
            all_tokens = self.fusion_engine(
                p_tokens=pathway_tokens,
                h_tokens=memory,
                return_all_tokens=True
            )
            # all_tokens is (B, Np+Nh, D)
            # We need the Histology tokens: indices [Np:]
            np = pathway_tokens.shape[1]
            context_features = all_tokens[:, np:, :]
        else:
            # Decoder processes neighborhood with optional spatial gating
            # NOTE: 'tgt_key_padding_mask' handles the self-attention on patches (x)
            # We pass 'mask' (True=Padding) to it.
            context_features = self.decoder_engine(
                tgt=memory, 
                memory=pathway_tokens, 
                tgt_key_padding_mask=mask
            )

        # 4. Dense prediction head (Pathway Bottleneck enforcement)
        # Project updated patch tokens onto pathway embeddings: (B, N, D) @ (B, D, P) -> (B, N, P)
        # This yields a dynamic "Pathway Activation" map for every patch.
        pathway_scores = torch.matmul(context_features, pathway_tokens.transpose(1, 2))
        
        # Reconstruct Genes: (B, N, P) @ (P, G) -> (B, N, G)
        # Reuses the gene_reconstructor for biological consistency.
        if return_pathways:
            return self.gene_reconstructor(pathway_scores), pathway_scores
        return self.gene_reconstructor(pathway_scores)
