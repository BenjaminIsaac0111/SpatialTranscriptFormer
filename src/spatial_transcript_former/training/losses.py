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


class MaskedHuberLoss(nn.Module):
    """
    Huber loss (SmoothL1) with optional masking.
    Robust to outliers.
    """

    def __init__(self, delta=1.0):
        super().__init__()
        self.loss = nn.HuberLoss(delta=delta, reduction="none")

    def forward(self, preds, target, mask=None):
        loss = self.loss(preds, target)

        if mask is not None and preds.dim() == 3:
            valid = ~mask.unsqueeze(-1).expand_as(loss)
            return (loss * valid.float()).sum() / valid.sum()

        return loss.mean()


class ZINBLoss(nn.Module):
    """
    Zero-Inflated Negative Binomial (ZINB) Loss.

    Designed for highly dispersed, zero-inflated count data (like raw RNA-seq).
    Requires the model to output three parameters per gene:
      - pi: probability of zero inflation (dropout)
      - mu: mean of the negative binomial distribution
      - theta: inverse dispersion parameter
    """

    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, preds, target, mask=None):
        """
        Args:
            preds: Tuple of (pi, mu, theta), each (B, G) or (B, N, G).
                   pi should be [0, 1] (e.g. via Sigmoid).
                   mu, theta should be > 0 (e.g. via Softplus or Exp).
            target: Raw integer counts (B, G) or (B, N, G).
            mask:   (B, N) boolean, True = padded (ignore). Optional.

        Returns:
            Scalar negative log-likelihood over valid positions.
        """
        pi, mu, theta = preds

        # Add numerical stability to theta and mu
        theta = torch.clamp(theta, min=self.eps, max=1e6)
        mu = torch.clamp(mu, min=self.eps, max=1e6)

        # Ensure target is valid for ZINB (no negatives, we expect counts)
        target = torch.clamp(target, min=0)

        # 1. Negative Binomial Probability
        # NB(y; mu, theta) = Gamma(y+theta)/(Gamma(theta)*y!) * (theta/(theta+mu))^theta * (mu/(theta+mu))^y
        # Log version for stability using lgamma:
        # ln(Gamma(target + theta)) - ln(Gamma(theta)) - ln(Gamma(target + 1))
        # + theta * ln(theta / (theta + mu)) + target * ln(mu / (theta + mu))

        t1 = (
            torch.lgamma(target + theta)
            - torch.lgamma(theta)
            - torch.lgamma(target + 1)
        )
        t2 = theta * (torch.log(theta + self.eps) - torch.log(theta + mu + self.eps))
        t3 = target * (torch.log(mu + self.eps) - torch.log(theta + mu + self.eps))

        nb_log_prob = t1 + t2 + t3

        # 2. Zero Inflation Mask
        is_zero = (target == 0).float()

        # 3. ZINB Log Likelihood
        # 3a. If target == 0: ln(pi + (1-pi) * NB(0; mu, theta))
        # NB(0; mu, theta) = (theta / (theta + mu))^theta
        # Log space for stability: theta * (log(theta) - log(theta + mu))
        nb_zero_log_prob = theta * (
            torch.log(theta + self.eps) - torch.log(theta + mu + self.eps)
        )

        # zero_case_prob = pi + (1 - pi) * exp(nb_zero_log_prob)
        # We need ln(pi + (1-pi)*exp(nb_zero_log_prob)).
        # clamp heavily before taking log to prevent NaNs when pi -> 0 and exp(nb) -> 0
        zero_case_prob = pi + (1 - pi) * torch.exp(nb_zero_log_prob)
        zero_case_prob = torch.clamp(zero_case_prob, min=self.eps, max=1.0)
        zero_case_log_prob = torch.log(zero_case_prob)

        # 3b. If target > 0: ln((1-pi) * NB(target; mu, theta))
        #                  = ln(1-pi) + ln(NB)
        non_zero_case_log_prob = torch.log(1 - pi + self.eps) + nb_log_prob

        # Combine
        log_likelihood = (
            is_zero * zero_case_log_prob + (1 - is_zero) * non_zero_case_log_prob
        )

        if mask is not None and pi.dim() == 3:
            # Expand mask to gene dimension: (B, N) -> (B, N, G)
            valid = ~mask.unsqueeze(-1).expand_as(log_likelihood)
            return -(log_likelihood * valid.float()).sum() / valid.sum()

        return -log_likelihood.mean()


class AuxiliaryPathwayLoss(nn.Module):
    """Wrapper combining gene-level loss with PCC-based pathway auxiliary loss.

    Directly supervises the pathway bottleneck scores using PCC against
    pathway ground truth computed from gene expression via MSigDB membership.
    This provides a direct gradient signal to the pathway tokens, preventing
    bottleneck collapse and ensuring each pathway learns its correct spatial
    activation pattern.

    The total loss is::

        L = L_gene + lambda * (1 - PCC(pathway_scores, target_pathways))

    where ``target_pathways = target_genes @ M.T`` and M is the MSigDB
    membership matrix.
    """

    def __init__(self, pathway_matrix, gene_criterion, lambda_pathway=0.5):
        """
        Args:
            pathway_matrix (torch.Tensor): Binary MSigDB membership matrix
                of shape ``(P, G)`` where ``P`` is the number of pathways and
                ``G`` is the number of genes.
            gene_criterion (nn.Module): Base loss for gene prediction
                (e.g. ``MaskedMSELoss``, ``CompositeLoss``).
            lambda_pathway (float): Weight for the auxiliary pathway loss.
        """
        super().__init__()
        self.register_buffer("pathway_matrix", pathway_matrix.float())
        self.gene_criterion = gene_criterion
        self.lambda_pathway = lambda_pathway
        self.pcc = PCCLoss()

    def forward(self, gene_preds, target_genes, mask=None, pathway_preds=None):
        """
        Args:
            gene_preds: (B, G) or (B, N, G) predicted gene expression.
            target_genes: (B, G) or (B, N, G) ground truth gene expression.
            mask: (B, N) boolean, True = padded (ignore). Optional.
            pathway_preds: (B, P) or (B, N, P) predicted pathway scores.
                If None, only gene loss is computed (graceful fallback).

        Returns:
            Scalar total loss.
        """
        gene_loss = self.gene_criterion(gene_preds, target_genes, mask=mask)

        if pathway_preds is None or self.lambda_pathway == 0:
            return gene_loss

        # Compute pathway ground truth from gene expression
        # target_genes: (B, [N,] G), pathway_matrix: (P, G)
        # result: (B, [N,] P)
        with torch.no_grad():
            target_pathways = torch.matmul(target_genes, self.pathway_matrix.T)

        pathway_loss = self.pcc(pathway_preds, target_pathways, mask=mask)

        return gene_loss + self.lambda_pathway * pathway_loss
