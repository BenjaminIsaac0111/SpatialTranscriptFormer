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

    Computes **gene-wise** spatial correlation, then averages across genes.
    This measures whether the predicted spatial pattern (map) for each gene
    matches the pattern of the ground truth map, regardless of overall intensity.

    PCC is scale-invariant: highly expressed genes and lowly expressed
    genes contribute equally, making this objective robust to typical
    spatial transcriptomics expression imbalances.
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
        if preds.dim() == 2:
            preds = preds.unsqueeze(1)  # (B, 1, G)
            target = target.unsqueeze(1)  # (B, 1, G)
            if mask is not None:
                mask = mask.unsqueeze(1)  # (B, 1)

        B, N, G = preds.shape

        # If N == 1 (e.g., standard patch-wise prediction without context),
        # PCC across a spatial dimension of 1 is undefined (variance is 0).
        # We fallback to batch-wise correlation in this specific edge case.
        if N == 1:
            preds = preds.squeeze(1)  # (B, G)
            target = target.squeeze(1)  # (B, G)
            if mask is not None:
                valid = ~mask.squeeze(1)  # (B)
                preds = preds[valid]
                target = target[valid]

            if preds.shape[0] < 2:
                return torch.tensor(0.0, device=preds.device, requires_grad=True)

            vx = preds - preds.mean(dim=0, keepdim=True)
            vy = target - target.mean(dim=0, keepdim=True)
            cost = torch.sum(vx * vy, dim=0) / (
                torch.sqrt(torch.sum(vx**2, dim=0) + self.eps)
                * torch.sqrt(torch.sum(vy**2, dim=0) + self.eps)
            )
            return 1 - cost.mean()

        # 1. Masking: Zero out padded positions so they don't contribute to sums
        if mask is not None:
            valid = ~mask.unsqueeze(-1)  # (B, N, 1)
            preds = preds * valid.float()
            target = target * valid.float()
            # valid_counts: (B, 1, 1) to enable broadcasting across N (dim 1) and G (dim 2)
            valid_counts = valid.sum(dim=1, keepdim=True).clamp(min=1.0)
        else:
            valid_counts = torch.tensor(N, dtype=torch.float32, device=preds.device)

        # 2. Centre per-slide (over spatial dimension N)
        # Calculate mean using valid counts to avoid skew from zeros
        pred_means = preds.sum(dim=1, keepdim=True) / valid_counts
        target_means = target.sum(dim=1, keepdim=True) / valid_counts

        vx = preds - pred_means
        vy = target - target_means

        if mask is not None:
            vx = vx * valid.float()
            vy = vy * valid.float()

        # 3. Correlation (covariance / (std_x * std_y)) per slide, per gene
        # sum over spatial dimension N -> (B, G)
        cov = torch.sum(vx * vy, dim=1)
        var_x = torch.sum(vx**2, dim=1)
        var_y = torch.sum(vy**2, dim=1)

        cost = cov / (
            torch.sqrt(var_x + self.eps) * torch.sqrt(var_y + self.eps)
        )  # (B, G)

        # Average across genes, then average across batch
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

    def __init__(self, alpha=1.0, eps=1e-8, mse_type="mse"):
        super().__init__()
        self.alpha = alpha
        if mse_type == "huber":
            self.mse = MaskedHuberLoss()
        else:
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
