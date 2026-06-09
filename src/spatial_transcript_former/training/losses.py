"""
Loss functions for SpatialTranscriptFormer.

Supports MSE, PCC, CCC, Huber, CLIP-alignment, and composite objectives.
All losses handle both patch-level (B, G) and dense (B, N, G) inputs,
with optional masking for padded positions in whole-slide mode.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


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


class CCCLoss(PCCLoss):
    """
    Concordance Correlation Coefficient Loss.

    CCC = 2 * cov(pred, target) / (var(pred) + var(target) + (mean(pred) - mean(target))^2)
    Loss = 1 - mean(CCC)

    Strictly more informative than PCC: a prediction that is correctly correlated
    but systematically offset (wrong mean or variance) will have CCC < PCC.
    """

    def forward(self, preds, target, mask=None):
        """
        Args:
            preds:  (B, G) or (B, N, G)
            target: (B, G) or (B, N, G)
            mask:   (B, N) boolean, True = padded (ignore). Optional.

        Returns:
            Scalar loss = 1 - mean(CCC).
        """
        if preds.dim() == 2:
            preds = preds.unsqueeze(1)
            target = target.unsqueeze(1)
            if mask is not None:
                mask = mask.unsqueeze(1)

        B, N, G = preds.shape

        if N == 1:
            preds = preds.squeeze(1)
            target = target.squeeze(1)
            if mask is not None:
                valid = ~mask.squeeze(1)
                preds = preds[valid]
                target = target[valid]

            if preds.shape[0] < 2:
                return torch.tensor(0.0, device=preds.device, requires_grad=True)

            pred_mean = preds.mean(dim=0)
            target_mean = target.mean(dim=0)
            vx = preds - pred_mean
            vy = target - target_mean
            cov = (vx * vy).sum(dim=0)
            var_x = (vx**2).sum(dim=0)
            var_y = (vy**2).sum(dim=0)
            mean_diff_sq = (pred_mean - target_mean) ** 2
            ccc = (2 * cov) / (var_x + var_y + mean_diff_sq + self.eps)
            return 1 - ccc.mean()

        if mask is not None:
            valid = ~mask.unsqueeze(-1)
            preds = preds * valid.float()
            target = target * valid.float()
            valid_counts = valid.sum(dim=1, keepdim=True).clamp(min=1.0)
        else:
            valid_counts = torch.tensor(N, dtype=torch.float32, device=preds.device)

        pred_means = preds.sum(dim=1, keepdim=True) / valid_counts
        target_means = target.sum(dim=1, keepdim=True) / valid_counts

        vx = preds - pred_means
        vy = target - target_means

        if mask is not None:
            vx = vx * valid.float()
            vy = vy * valid.float()

        cov = (vx * vy).sum(dim=1)  # (B, G)
        var_x = (vx**2).sum(dim=1)  # (B, G)
        var_y = (vy**2).sum(dim=1)  # (B, G)

        mean_diff_sq = (pred_means.squeeze(1) - target_means.squeeze(1)) ** 2  # (B, G)
        ccc = (2 * cov) / (var_x + var_y + mean_diff_sq + self.eps)  # (B, G)

        return 1 - ccc.mean()


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
            valid = ~mask.unsqueeze(-1).expand_as(diff)
            return (diff * valid.float()).sum() / valid.sum()

        return diff.mean()


class MaskedHuberLoss(nn.Module):
    """
    Huber (smooth L1) loss with optional masking for padded positions.

    More robust to outlier pathway activity values than MSE: quadratic near
    zero, linear beyond delta.

    Args:
        delta: Threshold between quadratic and linear regimes. Default 1.0.
    """

    def __init__(self, delta=1.0):
        super().__init__()
        self.delta = delta

    def forward(self, preds, target, mask=None):
        """
        Args:
            preds:  (B, G) or (B, N, G)
            target: (B, G) or (B, N, G)
            mask:   (B, N) boolean, True = padded (ignore). Optional.

        Returns:
            Scalar Huber loss over valid positions.
        """
        diff = F.huber_loss(preds, target, reduction="none", delta=self.delta)

        if mask is not None and preds.dim() == 3:
            valid = ~mask.unsqueeze(-1).expand_as(diff)
            return (diff * valid.float()).sum() / valid.sum()

        return diff.mean()


class CLIPAlignmentLoss(nn.Module):
    """
    Batch-discriminative regulariser over predicted vs. target pathway vectors.

    Despite the name, this is **not** cross-modal CLIP. Both inputs already
    live in pathway space: the model's predicted pathway vector and the
    ground-truth pathway vector. We borrow CLIP's loss form — symmetric
    cross-entropy on a B×B cosine-similarity matrix with diagonal labels —
    to enforce "your prediction must rank its own target highest in the
    batch." Acts as anti-collapse pressure on regression.

    Status (2026-04): kept available behind ``CompositeLoss(clip_weight=...)``
    and the ``mse_ccc_clip`` builder option, but **not part of the current
    experimentation track**. Direct-regression baselines (``mse_ccc``) are
    the focus while target construction and validation metrics stabilise.

    Deferred follow-up — true cross-modal CLIP. A separate variant would
    align the model's image embedding (e.g. mean-pooled transformer output)
    with a learned pathway-target encoder in a shared space, giving
    retrieval / zero-shot capability. Not implemented; left as a future
    direction once the regression baseline is solid.

    Args:
        temperature: Softmax temperature τ. Default 0.07 (CLIP default).
    """

    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, preds, target, mask=None, **kwargs):
        """
        Args:
            preds:  (B, G) or (B, N, G)
            target: (B, G) or (B, N, G)
            mask:   (B, N) boolean, True = padded. Used to average dense inputs.

        Returns:
            Scalar symmetric cross-entropy loss.
        """
        if preds.dim() == 3:
            if mask is not None:
                valid = (~mask).unsqueeze(-1).float()
                preds = (preds * valid).sum(1) / valid.sum(1).clamp(min=1)
                target = (target * valid).sum(1) / valid.sum(1).clamp(min=1)
            else:
                preds = preds.mean(1)
                target = target.mean(1)

        B = preds.shape[0]
        if B < 2:
            return torch.tensor(0.0, device=preds.device, requires_grad=True)

        norm_p = F.normalize(preds, dim=-1)  # (B, G)
        norm_t = F.normalize(target, dim=-1)  # (B, G)

        sim = (norm_p @ norm_t.T) / self.temperature  # (B, B)
        labels = torch.arange(B, device=sim.device)

        return 0.5 * (F.cross_entropy(sim, labels) + F.cross_entropy(sim.T, labels))


class CompositeLoss(nn.Module):
    """
    Combined MSE + PCC/CCC loss, with an optional CLIP alignment term.

    L = MSE + alpha * (1 - PCC/CCC) [+ clip_weight * L_CLIP]

    MSE focuses on magnitude accuracy; PCC/CCC ensures spatial pattern
    coherence; CLIP prevents variance collapse across the batch.

    Args:
        alpha:            Weight for the PCC/CCC term. Default 1.0.
        eps:              Numerical stability for correlation losses. Default 1e-8.
        mse_type:         "mse" or "huber". Default "mse".
        pcc_type:         "pcc" or "ccc". Default "pcc".
        clip_weight:      Weight for CLIP alignment term. 0.0 disables it.
        clip_temperature: Temperature τ for CLIP loss. Default 0.07.
    """

    def __init__(
        self,
        alpha=1.0,
        eps=1e-8,
        mse_type="mse",
        pcc_type="pcc",
        clip_weight=0.0,
        clip_temperature=0.07,
    ):
        super().__init__()
        self.alpha = alpha
        self.mse = MaskedHuberLoss() if mse_type == "huber" else MaskedMSELoss()
        self.pcc = CCCLoss(eps=eps) if pcc_type == "ccc" else PCCLoss(eps=eps)
        self.clip = CLIPAlignmentLoss(clip_temperature) if clip_weight > 0 else None
        self.clip_weight = clip_weight

    def forward(self, preds, target, mask=None):
        """
        Args:
            preds:  (B, G) or (B, N, G)
            target: (B, G) or (B, N, G)
            mask:   (B, N) boolean, True = padded (ignore). Optional.

        Returns:
            Scalar composite loss.
        """
        loss = self.mse(preds, target, mask)
        loss = loss + self.alpha * self.pcc(preds, target, mask)
        if self.clip is not None:
            loss = loss + self.clip_weight * self.clip(preds, target, mask)
        return loss
