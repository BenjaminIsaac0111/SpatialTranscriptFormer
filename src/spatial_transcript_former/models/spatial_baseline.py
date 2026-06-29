"""
Spatial baselines for the pathway-activity prediction task.

These exist to isolate *which* part of :class:`SpatialTranscriptFormer` is
doing the work:

* ``SpatialTransformerRegressor`` — a plain transformer encoder over the patch
  features (optionally with the same spatial positional encoding STF uses)
  followed by a per-spot MLP head.  It performs generic spatial mixing among
  patches but has **no pathway tokens and no pathway↔histology interaction**.
  Comparing it to STF isolates the contribution of the quad-flow interaction
  mechanism rather than spatial context in general.

* ``KNNRetrievalBaseline`` — a non-parametric retrieval predictor (a BLEEP-style
  baseline, Xie et al. 2023).  Each query spot is predicted as the mean pathway
  target of its k nearest training spots in backbone-feature space.  It measures
  how much the frozen foundation-model features already encode pathway activity,
  with no learning at all.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .interaction import LearnedSpatialEncoder


class SpatialTransformerRegressor(nn.Module):
    """Transformer encoder over patch features with a per-spot pathway head.

    Unlike :class:`SpatialTranscriptFormer`, there are no learnable pathway
    tokens: patches attend to one another (generic spatial mixing) and a shared
    MLP head regresses the pathway-activity vector for each spot independently.
    The final ``Softplus`` keeps outputs non-negative, matching the
    mean-log1p-CP10k targets (same output range as the STF scoring head).

    The forward signature mirrors STF (``x, rel_coords, mask, return_dense``) so
    the training engine can drive it through the same whole-slide code path.

    Args:
        input_dim (int): Dimension of the pre-computed backbone features.
        num_pathways (int): Number of pathway-activity outputs.
        token_dim (int): Transformer hidden width.
        n_heads (int): Attention heads.
        n_layers (int): Transformer encoder layers.
        dropout (float): Dropout probability.
        use_spatial_pe (bool): Add slide-stationary spatial positional encoding.
    """

    def __init__(
        self,
        input_dim=768,
        num_pathways=50,
        token_dim=256,
        n_heads=4,
        n_layers=2,
        dropout=0.1,
        use_spatial_pe=True,
    ):
        super().__init__()
        self.num_pathways = num_pathways
        self.use_spatial_pe = use_spatial_pe

        self.input_proj = nn.Linear(input_dim, token_dim)
        self.spatial_encoder = (
            LearnedSpatialEncoder(token_dim) if use_spatial_pe else None
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=token_dim,
            nhead=n_heads,
            dim_feedforward=token_dim * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers,
            norm=nn.LayerNorm(token_dim),
            enable_nested_tensor=False,
        )

        self.head = nn.Sequential(
            nn.Linear(token_dim, token_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(token_dim, num_pathways),
            nn.Softplus(),
        )

    def forward(self, x, rel_coords=None, mask=None, return_dense=False):
        """Predict pathway activity from patch features.

        Args:
            x (torch.Tensor): ``(B, S, D)`` pre-computed features (a leading
                ``(B, D)`` is promoted to ``(B, 1, D)``).
            rel_coords (torch.Tensor, optional): ``(B, S, 2)`` slide-stationary
                coordinates. Required when ``use_spatial_pe`` is True.
            mask (torch.Tensor, optional): ``(B, S)`` boolean padding mask where
                True marks padding positions to ignore.
            return_dense (bool): If True return per-spot ``(B, S, P)`` scores;
                otherwise return a masked-mean global ``(B, P)`` vector.

        Returns:
            torch.Tensor: ``(B, S, P)`` if ``return_dense`` else ``(B, P)``.
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)

        h = self.input_proj(x)
        if self.use_spatial_pe:
            if rel_coords is None:
                raise ValueError(
                    "use_spatial_pe is True, but rel_coords was not provided."
                )
            h = h + self.spatial_encoder(rel_coords)

        out = self.encoder(h, src_key_padding_mask=mask)
        scores = self.head(out)  # (B, S, P)

        if return_dense:
            return scores

        if mask is not None:
            valid = (~mask).unsqueeze(-1).float()
            return (scores * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1.0)
        return scores.mean(dim=1)


class KNNRetrievalBaseline(nn.Module):
    """Non-parametric k-NN retrieval predictor in backbone-feature space.

    Stores a bank of training-spot features and their pathway targets. Each
    query spot is predicted as the (unweighted) mean target of its ``k`` nearest
    bank entries. No parameters are trained; this quantifies how linearly
    decodable the frozen features already are. Conceptually a BLEEP-style
    retrieval baseline (Xie et al., NeurIPS 2023).

    The bank tensors are registered as non-persistent buffers so ``.to(device)``
    moves them but they are not written into ``state_dict``.

    Args:
        feature_bank (torch.Tensor): ``(M, D)`` training-spot features.
        target_bank (torch.Tensor): ``(M, P)`` matching pathway targets.
        k (int): Number of neighbours to average.
        metric (str): ``"cosine"`` or ``"l2"``.
        chunk (int): Query rows processed per similarity batch (memory knob).
    """

    def __init__(self, feature_bank, target_bank, k=16, metric="cosine", chunk=1024):
        super().__init__()
        if metric not in ("cosine", "l2"):
            raise ValueError(f"metric must be 'cosine' or 'l2', got {metric!r}")
        if feature_bank.shape[0] != target_bank.shape[0]:
            raise ValueError(
                f"feature_bank ({feature_bank.shape[0]}) and target_bank "
                f"({target_bank.shape[0]}) must have the same number of rows"
            )
        self.k = min(k, feature_bank.shape[0])
        self.metric = metric
        self.chunk = chunk

        self.register_buffer("feature_bank", feature_bank.float(), persistent=False)
        self.register_buffer("target_bank", target_bank.float(), persistent=False)
        # Pre-normalised bank for cosine similarity (cached once).
        norm_bank = (
            F.normalize(feature_bank.float(), dim=-1)
            if metric == "cosine"
            else feature_bank.float()
        )
        self.register_buffer("_bank_norm", norm_bank, persistent=False)

    @torch.no_grad()
    def forward(self, x, rel_coords=None, mask=None, return_dense=False):
        """Retrieve a pathway vector for every spot in ``x``.

        Accepts the same call shapes as the other models so it can be evaluated
        through the standard engine; ``rel_coords``/``mask``/``return_dense`` are
        ignored (padding rows are simply predicted and masked out downstream).

        Args:
            x (torch.Tensor): ``(..., D)`` query features (any leading shape).

        Returns:
            torch.Tensor: ``(..., P)`` retrieved pathway predictions.
        """
        lead_shape = x.shape[:-1]
        d = x.shape[-1]
        q = x.reshape(-1, d)
        p = self.target_bank.shape[1]
        preds = torch.empty(q.shape[0], p, device=q.device, dtype=torch.float32)

        query = F.normalize(q, dim=-1) if self.metric == "cosine" else q
        for start in range(0, q.shape[0], self.chunk):
            qc = query[start : start + self.chunk]
            if self.metric == "cosine":
                sim = qc @ self._bank_norm.T  # (c, M) — higher is closer
                idx = sim.topk(self.k, dim=1).indices
            else:
                dist = torch.cdist(qc, self.feature_bank)  # (c, M)
                idx = dist.topk(self.k, dim=1, largest=False).indices
            neighbours = self.target_bank[idx]  # (c, k, P)
            preds[start : start + qc.shape[0]] = neighbours.mean(dim=1)

        return preds.reshape(*lead_shape, p)
