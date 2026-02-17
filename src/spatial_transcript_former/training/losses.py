"""
Loss functions for SpatialTranscriptFormer.

Supports MSE, PCC, and composite MSE+PCC objectives.
All losses handle both patch-level (B, G) and dense (B, N, G) inputs,
with optional masking for padded positions in whole-slide mode.
"""

import torch
import torch.nn as nn


class PCCLoss(nn.Module):
    """
    Pearson Correlation Coefficient Loss.

    Computes per-spot PCC across genes, then averages. This measures
    whether the predicted gene profile at each location matches the
    shape of the ground truth profile, regardless of scale.

    PCC is scale-invariant: all genes contribute equally, making
    this objective robust to gene expression imbalance.
    """

    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, preds, target, mask=None):
        """
        Args:
            preds:  (B, G) or (B, N, G)
            target: (B, G) or (B, N, G)
            mask:   (B, N) boolean, True = padded (ignore). Optional.

        Returns:
            Scalar loss = 1 - mean(PCC).
        """
        if preds.dim() == 3:
            B, N, G = preds.shape
            preds = preds.reshape(-1, G)   # (B*N, G)
            target = target.reshape(-1, G)

            if mask is not None:
                valid = ~mask.reshape(-1)   # (B*N,)
                preds = preds[valid]
                target = target[valid]

        # Centre per-spot
        vx = preds - preds.mean(dim=1, keepdim=True)
        vy = target - target.mean(dim=1, keepdim=True)

        # Correlation
        cost = (
            torch.sum(vx * vy, dim=1)
            / (torch.sqrt(torch.sum(vx ** 2, dim=1))
               * torch.sqrt(torch.sum(vy ** 2, dim=1))
               + self.eps)
        )
        return 1 - cost.mean()


class MaskedMSELoss(nn.Module):
    """
    MSE loss with optional masking for padded positions.

    When no mask is provided, behaves identically to nn.MSELoss().
    """

    def forward(self, preds, target, mask=None):
        """
        Args:
            preds:  (B, G) or (B, N, G)
            target: (B, G) or (B, N, G)
            mask:   (B, N) boolean, True = padded (ignore). Optional.

        Returns:
            Scalar MSE loss over valid positions.
        """
        diff = (preds - target) ** 2

        if mask is not None and preds.dim() == 3:
            # Expand mask to gene dimension: (B, N) -> (B, N, G)
            valid = ~mask.unsqueeze(-1).expand_as(diff)
            return (diff * valid.float()).sum() / valid.sum()

        return diff.mean()


class CompositeLoss(nn.Module):
    """
    Combined MSE + PCC loss for spatial gene expression prediction.

    L = MSE + alpha * (1 - PCC)

    MSE focuses on magnitude accuracy; PCC ensures spatial pattern
    coherence across all genes equally (scale-invariant). Together
    they address the gene expression imbalance problem.

    Args:
        alpha: Weight for the PCC term. Default 1.0.
        eps:   Numerical stability for PCC. Default 1e-8.
    """

    def __init__(self, alpha=1.0, eps=1e-8):
        super().__init__()
        self.alpha = alpha
        self.mse = MaskedMSELoss()
        self.pcc = PCCLoss(eps=eps)

    def forward(self, preds, target, mask=None):
        """
        Args:
            preds:  (B, G) or (B, N, G)
            target: (B, G) or (B, N, G)
            mask:   (B, N) boolean, True = padded (ignore). Optional.

        Returns:
            Scalar composite loss.
        """
        mse_val = self.mse(preds, target, mask)
        pcc_val = self.pcc(preds, target, mask)
        return mse_val + self.alpha * pcc_val
