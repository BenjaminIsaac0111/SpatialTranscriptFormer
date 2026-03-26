"""
Merged tests: test_losses.py, test_losses_robust.py
"""

import pytest
import torch
import torch.nn as nn

from spatial_transcript_former.training.losses import (
    MaskedMSELoss,
    PCCLoss,
    CompositeLoss,
    AuxiliaryPathwayLoss,
)
from spatial_transcript_former.training.losses import ZINBLoss, PCCLoss

# --- From test_losses.py ---


@pytest.fixture
def tensors_2d():
    """Patch-level tensors: (B, G)."""
    torch.manual_seed(42)
    return torch.randn(32, 100), torch.randn(32, 100)


@pytest.fixture
def tensors_3d():
    """Dense whole-slide tensors: (B, N, G) with padding mask."""
    torch.manual_seed(42)
    B, N, G = 2, 100, 50
    preds = torch.randn(B, N, G)
    target = torch.randn(B, N, G)
    mask = torch.zeros(B, N, dtype=torch.bool)
    mask[0, 80:] = True  # 20 padded positions in sample 0
    mask[1, 90:] = True  # 10 padded positions in sample 1
    return preds, target, mask


# ---------------------------------------------------------------------------
# MaskedMSELoss
# ---------------------------------------------------------------------------


class TestMaskedMSE:
    def test_no_mask_matches_standard(self, tensors_2d):
        """MaskedMSELoss without mask should equal nn.MSELoss."""
        preds, target = tensors_2d
        expected = nn.MSELoss()(preds, target)
        actual = MaskedMSELoss()(preds, target)
        assert torch.allclose(expected, actual, atol=1e-6)

    def test_mask_ignores_padding(self, tensors_3d):
        """Padded positions should not contribute to loss."""
        preds, target, mask = tensors_3d

        # Compute masked loss
        masked_loss = MaskedMSELoss()(preds, target, mask=mask)

        # Manually compute: zero out padded then average over valid only
        valid = ~mask.unsqueeze(-1).expand_as(preds)
        diff_sq = (preds - target) ** 2
        expected = (diff_sq * valid.float()).sum() / valid.sum()

        assert torch.allclose(masked_loss, expected, atol=1e-6)

    def test_mask_changes_result(self, tensors_3d):
        """Loss with mask should differ from loss without mask."""
        preds, target, mask = tensors_3d
        loss_no_mask = MaskedMSELoss()(preds, target)
        loss_masked = MaskedMSELoss()(preds, target, mask=mask)
        assert not torch.allclose(loss_no_mask, loss_masked)

    def test_gradient_flow(self, tensors_3d):
        """Gradients should flow through masked MSE."""
        preds, target, mask = tensors_3d
        preds = preds.clone().requires_grad_(True)
        loss = MaskedMSELoss()(preds, target, mask=mask)
        loss.backward()
        assert preds.grad is not None
        assert preds.grad.shape == preds.shape


# ---------------------------------------------------------------------------
# PCCLoss
# ---------------------------------------------------------------------------


class TestPCC:
    def test_perfect_correlation(self):
        """Identical inputs should give loss = 0."""
        x = torch.randn(50, 100)
        loss = PCCLoss()(x, x)
        assert loss.item() == pytest.approx(0.0, abs=1e-5)

    def test_anticorrelation(self):
        """Negated inputs should give loss â‰ˆ 2.0."""
        x = torch.randn(50, 100)
        loss = PCCLoss()(x, -x)
        assert loss.item() == pytest.approx(2.0, abs=1e-5)

    def test_scale_invariance(self):
        """Scaling predictions should not change PCC."""
        x = torch.randn(50, 100)
        y = torch.randn(50, 100)
        loss_original = PCCLoss()(x, y)
        loss_scaled = PCCLoss()(x * 100, y)
        assert loss_original.item() == pytest.approx(loss_scaled.item(), abs=1e-4)

    def test_shift_invariance(self):
        """Adding a constant offset should not change PCC."""
        x = torch.randn(50, 100)
        y = torch.randn(50, 100)
        loss_original = PCCLoss()(x, y)
        loss_shifted = PCCLoss()(x + 1000, y)
        assert loss_original.item() == pytest.approx(loss_shifted.item(), abs=1e-4)

    def test_3d_with_mask(self, tensors_3d):
        """PCC should handle 3D inputs with mask."""
        preds, target, mask = tensors_3d
        loss = PCCLoss()(preds, target, mask=mask)
        assert loss.isfinite()
        assert 0.0 <= loss.item() <= 2.0

    def test_gradient_flow_2(self, tensors_2d):
        """Gradients should flow through PCC loss."""
        preds, target = tensors_2d
        preds = preds.clone().requires_grad_(True)
        loss = PCCLoss()(preds, target)
        loss.backward()
        assert preds.grad is not None

    def test_pcc_fallback_n1(self):
        """Verify the N=1 fallback (batch-wise correlation) is robust."""
        # preds/target: (B, 1, G). With B=2, N=1
        preds = torch.tensor([[[1.0, 2.0]], [[2.0, 3.0]]])
        target = torch.tensor([[[1.0, 2.0]], [[2.0, 3.0]]])

        # Perfect correlation => loss 0
        loss = PCCLoss()(preds, target)
        assert loss.item() == pytest.approx(0.0, abs=1e-5)

        # Anti-correlation => loss 2
        loss_anti = PCCLoss()(preds, -target)
        assert loss_anti.item() == pytest.approx(2.0, abs=1e-5)


# ---------------------------------------------------------------------------
# CompositeLoss
# ---------------------------------------------------------------------------


class TestCompositeLoss:
    def test_equals_sum_of_parts(self, tensors_2d):
        """CompositeLoss should equal MSE + alpha * PCC."""
        preds, target = tensors_2d
        alpha = 1.5

        mse_val = MaskedMSELoss()(preds, target)
        pcc_val = PCCLoss()(preds, target)
        expected = mse_val + alpha * pcc_val

        actual = CompositeLoss(alpha=alpha)(preds, target)
        assert torch.allclose(expected, actual, atol=1e-5)

    def test_alpha_zero_is_mse(self, tensors_2d):
        """With alpha=0, CompositeLoss should equal MSE only."""
        preds, target = tensors_2d
        mse_val = MaskedMSELoss()(preds, target)
        comp_val = CompositeLoss(alpha=0.0)(preds, target)
        assert torch.allclose(mse_val, comp_val, atol=1e-6)

    def test_default_alpha(self, tensors_2d):
        """Ensure the default alpha is 1.0."""
        preds, target = tensors_2d
        loss_default = CompositeLoss()(preds, target)
        loss_explicit = CompositeLoss(alpha=1.0)(preds, target)
        assert torch.allclose(loss_default, loss_explicit)

        # Ensure it's NOT 0.0
        loss_mse = MaskedMSELoss()(preds, target)
        assert not torch.allclose(loss_default, loss_mse)

    def test_mask_support(self, tensors_3d):
        """CompositeLoss should handle masks in 3D mode."""
        preds, target, mask = tensors_3d
        loss = CompositeLoss(alpha=1.0)(preds, target, mask=mask)
        assert loss.isfinite()

    def test_gradient_flow_both_terms(self, tensors_3d):
        """Both MSE and PCC terms should contribute gradients."""
        preds, target, mask = tensors_3d
        preds = preds.clone().requires_grad_(True)

        loss = CompositeLoss(alpha=1.0)(preds, target, mask=mask)
        loss.backward()

        assert preds.grad is not None
        # Gradients at padded positions should be zero
        padded_grad = preds.grad[0, 80:, :]  # sample 0, padded region
        assert padded_grad.abs().sum() == 0.0

    def test_different_alphas(self, tensors_2d):
        """Higher alpha should increase the PCC contribution."""
        preds, target = tensors_2d
        loss_low = CompositeLoss(alpha=0.1)(preds, target)
        loss_high = CompositeLoss(alpha=10.0)(preds, target)
        # They should differ since PCC != 0
        assert loss_low.item() != pytest.approx(loss_high.item(), abs=0.01)


class TestMaskedHuber:
    def test_3d_mask_impact(self, tensors_3d):
        """Verify that padding mask works correctly for Huber 3D."""
        from spatial_transcript_former.training.losses import MaskedHuberLoss

        preds, target, mask = tensors_3d

        loss_fn = MaskedHuberLoss()
        loss_masked = loss_fn(preds, target, mask=mask)
        loss_unmasked = loss_fn(preds, target)

        assert not torch.allclose(loss_masked, loss_unmasked)
        assert loss_masked.isfinite()


# ---------------------------------------------------------------------------
# ZINBLoss
# ---------------------------------------------------------------------------


@pytest.fixture
def zinb_tensors():
    """Tensors for ZINB testing: pi, mu, theta, and counts."""
    torch.manual_seed(42)
    B, G = 16, 50
    pi = torch.rand(B, G)  # [0, 1]
    mu = torch.rand(B, G) * 10 + 0.1  # > 0
    theta = torch.rand(B, G) * 5 + 0.1  # > 0

    # Simulate some zero-inflated counts
    counts = torch.poisson(mu)
    dropout_mask = torch.rand(B, G) < 0.3
    counts[dropout_mask] = 0.0

    return (pi, mu, theta), counts


class TestZINBLoss:
    def test_basic_computation(self, zinb_tensors):
        """ZINBLoss should compute a finite scalar loss."""
        from spatial_transcript_former.training.losses import ZINBLoss

        preds, target = zinb_tensors
        loss_fn = ZINBLoss()
        loss = loss_fn(preds, target)

        assert loss.isfinite()
        assert loss.item() > 0  # NLL should be positive

    def test_gradient_flow_3(self, zinb_tensors):
        """Gradients should flow through all three parameters."""
        from spatial_transcript_former.training.losses import ZINBLoss

        pi, mu, theta = zinb_tensors[0]
        pi = pi.clone().requires_grad_(True)
        mu = mu.clone().requires_grad_(True)
        theta = theta.clone().requires_grad_(True)

        target = zinb_tensors[1]
        loss = ZINBLoss()((pi, mu, theta), target)
        loss.backward()

        assert pi.grad is not None
        assert mu.grad is not None
        assert theta.grad is not None
        assert not torch.isnan(pi.grad).any()
        assert not torch.isnan(mu.grad).any()
        assert not torch.isnan(theta.grad).any()

    def test_mask_support_2(self):
        """ZINBLoss should handle masks in 3D mode correctly."""
        from spatial_transcript_former.training.losses import ZINBLoss

        torch.manual_seed(42)
        B, N, G = 2, 10, 5
        pi = torch.rand(B, N, G)
        mu = torch.rand(B, N, G) * 5 + 0.1
        theta = torch.rand(B, N, G) * 5 + 0.1
        target = torch.randint(0, 10, (B, N, G)).float()

        mask = torch.zeros(B, N, dtype=torch.bool)
        mask[0, 5:] = True  # sample 0 half padded

        pi.requires_grad_(True)

        loss_fn = ZINBLoss()
        loss = loss_fn((pi, mu, theta), target, mask=mask)
        loss.backward()

        assert loss.isfinite()

        # Gradients in padded regions should be zero
        padded_grad = pi.grad[0, 5:, :]
        assert padded_grad.abs().sum() == 0.0

    def test_zinb_zero_vs_nonzero(self):
        """Verify that ZINB treats 0 and non-zero targets differently (branch coverage)."""
        from spatial_transcript_former.training.losses import ZINBLoss

        loss_fn = ZINBLoss()

        # B=1, G=2. G0=0 (zero-inflation branch), G1=10 (NB branch)
        target = torch.tensor([[0.0, 10.0]])
        # Fix params for predictable results: pi=0.5, mu=1.0, theta=1.0
        pi = torch.tensor([[0.5, 0.5]])
        mu = torch.tensor([[1.0, 1.0]])
        theta = torch.tensor([[1.0, 1.0]])

        # If we change target[0, 1] from 10 to 0, the loss should change significantly
        loss1 = loss_fn((pi, mu, theta), target)
        target2 = torch.tensor([[0.0, 0.0]])
        loss2 = loss_fn((pi, mu, theta), target2)
        assert not torch.allclose(loss1, loss2)

    def test_zinb_extreme_stability(self):
        """Verify stability with very large or small parameters (clamping logic)."""
        from spatial_transcript_former.training.losses import ZINBLoss

        loss_fn = ZINBLoss()
        target = torch.tensor([[0.0, 10.0]])

        # mu and theta at extremes
        pi = torch.tensor([[0.1, 0.1]])
        mu = torch.tensor([[1e-12, 1e12]])
        theta = torch.tensor([[1e12, 1e-12]])

        loss = loss_fn((pi, mu, theta), target)
        print(f"DEBUG: ZINB extreme loss value: {loss}")
        assert torch.isfinite(loss).item(), f"Loss is not finite: {loss}"
        assert not torch.isnan(loss).item(), f"Loss is NaN: {loss}"


# ---------------------------------------------------------------------------
# AuxiliaryPathwayLoss
# ---------------------------------------------------------------------------


@pytest.fixture
def pathway_tensors():
    """Tensors for AuxiliaryPathwayLoss testing."""
    torch.manual_seed(42)
    B, N, G, P = 2, 100, 50, 10
    gene_preds = torch.randn(B, N, G)
    target_genes = torch.randn(B, N, G).abs()  # Positive counts
    pathway_preds = torch.randn(B, N, P)
    # Binary MSigDB-like matrix: (P, G)
    pathway_matrix = (torch.rand(P, G) > 0.8).float()
    mask = torch.zeros(B, N, dtype=torch.bool)
    mask[0, 80:] = True
    return gene_preds, target_genes, pathway_preds, pathway_matrix, mask


class TestAuxiliaryPathwayLoss:
    def test_basic_computation_2(self, pathway_tensors):
        """AuxiliaryPathwayLoss should produce a finite scalar."""
        gene_preds, targets, pw_preds, pw_matrix, mask = pathway_tensors
        loss_fn = AuxiliaryPathwayLoss(pw_matrix, MaskedMSELoss(), lambda_pathway=0.5)
        loss = loss_fn(gene_preds, targets, mask=mask, pathway_preds=pw_preds)
        assert loss.isfinite()

    def test_includes_gene_loss(self, pathway_tensors):
        """Total loss should be >= gene loss alone."""
        gene_preds, targets, pw_preds, pw_matrix, mask = pathway_tensors
        base = MaskedMSELoss()
        aux = AuxiliaryPathwayLoss(pw_matrix, base, lambda_pathway=0.5)
        gene_only = base(gene_preds, targets, mask=mask)
        total = aux(gene_preds, targets, mask=mask, pathway_preds=pw_preds)
        # Pathway PCC loss is >= 0, so total >= gene_only
        assert total.item() >= gene_only.item() - 1e-5

    def test_gradient_flows_to_both(self, pathway_tensors):
        """Gradients should flow to both gene_preds and pathway_preds."""
        gene_preds, targets, pw_preds, pw_matrix, mask = pathway_tensors
        gene_preds = gene_preds.clone().requires_grad_(True)
        pw_preds = pw_preds.clone().requires_grad_(True)
        loss_fn = AuxiliaryPathwayLoss(pw_matrix, MaskedMSELoss(), lambda_pathway=0.5)
        loss = loss_fn(gene_preds, targets, mask=mask, pathway_preds=pw_preds)
        loss.backward()
        assert gene_preds.grad is not None
        assert pw_preds.grad is not None
        assert gene_preds.grad.abs().sum() > 0
        assert pw_preds.grad.abs().sum() > 0

    def test_fallback_without_pathways(self, pathway_tensors):
        """When pathway_preds is None, should fall back to gene loss only."""
        gene_preds, targets, _, pw_matrix, mask = pathway_tensors
        base = MaskedMSELoss()
        aux = AuxiliaryPathwayLoss(pw_matrix, base, lambda_pathway=0.5)
        gene_only = base(gene_preds, targets, mask=mask)
        fallback = aux(gene_preds, targets, mask=mask, pathway_preds=None)
        assert torch.allclose(gene_only, fallback)

    def test_lambda_zero_disables(self, pathway_tensors):
        """lambda_pathway=0 should produce the same result as gene loss."""
        gene_preds, targets, pw_preds, pw_matrix, mask = pathway_tensors
        base = MaskedMSELoss()
        aux = AuxiliaryPathwayLoss(pw_matrix, base, lambda_pathway=0.0)
        gene_only = base(gene_preds, targets, mask=mask)
        total = aux(gene_preds, targets, mask=mask, pathway_preds=pw_preds)
        assert torch.allclose(gene_only, total)

    def test_zinb_integration(self):
        """AuxiliaryPathwayLoss should work with ZINB (pi, mu, theta) output."""
        from spatial_transcript_former.training.losses import (
            ZINBLoss,
            AuxiliaryPathwayLoss,
            MaskedMSELoss,
        )

        torch.manual_seed(42)
        B, N, G, P = 2, 10, 50, 10
        pi = torch.rand(B, N, G)
        mu = torch.rand(B, N, G) * 5 + 1.0
        theta = torch.rand(B, N, G) * 5 + 1.0
        zinb_preds = (pi, mu, theta)

        targets = torch.randint(0, 10, (B, N, G)).float()
        pw_preds = torch.randn(B, N, P)
        pw_matrix = (torch.rand(P, G) > 0.8).float()
        mask = torch.zeros(B, N, dtype=torch.bool)
        mask[0, 5:] = True

        loss_fn = AuxiliaryPathwayLoss(pw_matrix, ZINBLoss(), lambda_pathway=1.0)
        loss = loss_fn(zinb_preds, targets, mask=mask, pathway_preds=pw_preds)

        assert loss.isfinite()
        assert loss.item() > 0

    def test_perfect_match_zero_aux(self, pathway_tensors):
        """When pathway preds perfectly correlate with truth, aux loss contribution is 0."""
        gene_preds, targets, _, pw_matrix, mask = pathway_tensors
        base = MaskedMSELoss()
        aux = AuxiliaryPathwayLoss(pw_matrix, base, lambda_pathway=1.0)

        # Compute ground truth pathways matching the new AuxiliaryPathwayLoss logic
        with torch.no_grad():
            if targets.dim() == 2:
                # Patch level: (B, G). Normalize across the batch dimension
                means = targets.mean(dim=0, keepdim=True)
                stds = targets.std(dim=0, keepdim=True).clamp(min=1e-6)
                norm_genes = (targets - means) / stds
            else:
                # Whole slide: (B, N, G). Normalize across valid spatial positions N
                valid_mask = (
                    ~mask.unsqueeze(-1)
                    if mask is not None
                    else torch.ones_like(targets, dtype=torch.bool)
                )
                valid_counts = valid_mask.sum(dim=1, keepdim=True).clamp(min=1.0)
                means = (targets * valid_mask.float()).sum(
                    dim=1, keepdim=True
                ) / valid_counts
                diffs = (targets - means) * valid_mask.float()
                vars = (diffs**2).sum(dim=1, keepdim=True) / (valid_counts - 1).clamp(
                    min=1.0
                )
                stds = torch.sqrt(vars).clamp(min=1e-6)
                norm_genes = (diffs / stds) * valid_mask.float()

            target_pathways = torch.matmul(norm_genes, pw_matrix.T)
            member_counts = pw_matrix.sum(dim=1, keepdim=True).T.clamp(min=1.0)
            target_pathways = target_pathways / member_counts

        gene_loss = base(gene_preds, targets, mask=mask)
        # Use target_pathways as pathway_preds
        total_loss = aux(gene_preds, targets, mask=mask, pathway_preds=target_pathways)

        # Total loss should roughly equal gene loss (since 1-PCC should be 0)
        assert total_loss.item() == pytest.approx(gene_loss.item(), abs=1e-5)

    def test_numerical_stability_empty_spots(self, pathway_tensors):
        """Test stability when targets for some pathways are all zero."""
        gene_preds, targets, pw_preds, pw_matrix, mask = pathway_tensors

        # Zero out targets for the first pathway
        # Matrix is (P, G), so genes involved in pathway 0
        genes_in_p0 = pw_matrix[0].bool()
        targets[..., genes_in_p0] = 0.0

        loss_fn = AuxiliaryPathwayLoss(pw_matrix, MaskedMSELoss(), lambda_pathway=1.0)
        loss = loss_fn(gene_preds, targets, mask=mask, pathway_preds=pw_preds)

        assert loss.isfinite()
        assert not torch.isnan(loss)

    def test_lambda_scaling(self, pathway_tensors):
        """Doubling lambda_pathway should increase the aux contribution accordingly."""
        gene_preds, targets, pw_preds, pw_matrix, mask = pathway_tensors
        base = MaskedMSELoss()

        aux1 = AuxiliaryPathwayLoss(pw_matrix, base, lambda_pathway=0.5)
        aux2 = AuxiliaryPathwayLoss(pw_matrix, base, lambda_pathway=1.0)

        loss1 = aux1(gene_preds, targets, mask=mask, pathway_preds=pw_preds)
        loss2 = aux2(gene_preds, targets, mask=mask, pathway_preds=pw_preds)
        gene_loss = base(gene_preds, targets, mask=mask)

        term1 = loss1 - gene_loss
        term2 = loss2 - gene_loss

        # term2 should be approx 2 * term1
        assert term2.item() == pytest.approx(2 * term1.item(), rel=1e-4)

    def test_auxiliary_lambda_sensitivity(self, pathway_tensors):
        """Verify that changing lambda actually scales the pathway component."""
        gene_preds, targets, pw_preds, pw_matrix, mask = pathway_tensors
        base = MaskedMSELoss()

        aux_low = AuxiliaryPathwayLoss(pw_matrix, base, lambda_pathway=0.1)
        aux_high = AuxiliaryPathwayLoss(pw_matrix, base, lambda_pathway=10.0)

        loss_low = aux_low(gene_preds, targets, mask=mask, pathway_preds=pw_preds)
        loss_high = aux_high(gene_preds, targets, mask=mask, pathway_preds=pw_preds)

        # high lambda should force a different loss value
        assert not torch.allclose(loss_low, loss_high)

        # Verify that lambda=0 exactly matches gene loss
        aux_zero = AuxiliaryPathwayLoss(pw_matrix, base, lambda_pathway=0.0)
        loss_zero = aux_zero(gene_preds, targets, mask=mask, pathway_preds=pw_preds)
        gene_only = base(gene_preds, targets, mask=mask)
        assert torch.allclose(loss_zero, gene_only)

    def test_hallmark_integration(self):
        """Test with a real (though small) MSigDB Hallmark matrix."""
        from spatial_transcript_former.data.pathways import get_pathway_init
        from spatial_transcript_former.training.losses import (
            AuxiliaryPathwayLoss,
            MaskedMSELoss,
        )

        # Subset of genes known to be in MSigDB Hallmarks
        gene_list = ["TP53", "MYC", "VEGFA", "VIM", "CDH1", "SNAI1", "AXIN2", "MLH1"]
        B, N, G = 2, 5, len(gene_list)

        # Get real Hallmark matrix (only for our 8 genes)
        matrix, names = get_pathway_init(gene_list, verbose=False)
        P = len(names)

        preds = torch.randn(B, N, G)
        targets = torch.randn(B, N, G).abs()
        pw_preds = torch.randn(B, N, P)

        loss_fn = AuxiliaryPathwayLoss(matrix, MaskedMSELoss())
        loss = loss_fn(preds, targets, pathway_preds=pw_preds)

        assert loss.isfinite()
        assert len(names) > 0

    def test_hallmark_signal_detection(self):
        """Verify that gene patterns aligned with a hallmark reduce the aux loss."""
        from spatial_transcript_former.training.losses import (
            AuxiliaryPathwayLoss,
            MaskedMSELoss,
        )

        # Define 4 genes and 2 pathways
        # P0: G0, G1
        # P1: G2, G3
        pw_matrix = torch.tensor(
            [[1.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 1.0]], dtype=torch.float32
        )
        gene_list = ["G0", "G1", "G2", "G3"]
        B, N, G = 1, 10, 4
        P = 2

        # Case 1: Random targets
        torch.manual_seed(42)
        targets = torch.randn(B, N, G).abs()
        # Predictions match targets for genes
        gene_preds = targets.clone()

        # Pathway preds are random
        pw_preds_random = torch.randn(B, N, P)

        loss_fn = AuxiliaryPathwayLoss(pw_matrix, MaskedMSELoss(), lambda_pathway=1.0)
        loss_random = loss_fn(gene_preds, targets, pathway_preds=pw_preds_random)

        # Case 2: Pathway preds perfectly match truth
        with torch.no_grad():
            means = targets.mean(dim=1, keepdim=True)
            stds = targets.std(dim=1, keepdim=True).clamp(min=1e-6)
            norm_genes = (targets - means) / stds
            pw_truth = torch.matmul(norm_genes, pw_matrix.T)
            member_counts = pw_matrix.sum(dim=1, keepdim=True).T.clamp(min=1.0)
            pw_truth = pw_truth / member_counts

        loss_perfect = loss_fn(gene_preds, targets, pathway_preds=pw_truth)

        # Case 3: Gene expression is specifically high for P0, and pw_preds are high for P0
        # If the spatial correlation is high, aux loss should be low.
        assert loss_perfect.item() < loss_random.item()


# --- From test_losses_robust.py ---


def test_zinb_gradient_flow():
    """Verify that gradients flow to all three parameters of ZINB."""
    zinb = ZINBLoss()

    B, G = 4, 10
    # Inputs require gradients
    pi = torch.full((B, G), 0.5, requires_grad=True)
    mu = torch.full((B, G), 10.0, requires_grad=True)
    theta = torch.full((B, G), 1.0, requires_grad=True)

    target = torch.randint(0, 100, (B, G)).float()

    loss = zinb((pi, mu, theta), target)
    loss.backward()

    assert pi.grad is not None
    assert mu.grad is not None
    assert theta.grad is not None

    assert not torch.allclose(pi.grad, torch.zeros_like(pi.grad))
    assert not torch.allclose(mu.grad, torch.zeros_like(mu.grad))
    assert not torch.allclose(theta.grad, torch.zeros_like(theta.grad))


def test_zinb_stability():
    """Verify ZINB handles zeros and very large counts without NaNs."""
    zinb = ZINBLoss()

    # Extreme inputs
    pi = torch.tensor([[1e-8, 0.5, 1.0 - 1e-8]])
    mu = torch.tensor([[1e-8, 100.0, 1e6]])
    theta = torch.tensor([[1e-8, 1.0, 1e6]])

    # Extreme targets
    target = torch.tensor([[0.0, 10.0, 1e6]])

    loss = zinb((pi, mu, theta), target)
    assert not torch.isnan(loss)
    assert not torch.isinf(loss)


def test_pcc_edge_cases():
    """Verify PCC fallback and masking edge cases."""
    pcc = PCCLoss()

    # N=1 Case (fallback to batch-wise)
    preds = torch.randn(4, 1, 10, requires_grad=True)
    target = torch.randn(4, 1, 10)

    loss = pcc(preds, target)
    assert loss.requires_grad
    assert not torch.isnan(loss)

    # mask all but 1 in spatial dim -> N=1 fallback
    preds_3d = torch.randn(2, 5, 3, requires_grad=True)
    target_3d = torch.randn(2, 5, 3)
    mask = torch.ones(2, 5, dtype=torch.bool)
    mask[:, 0] = False  # only first spot valid

    loss_masked = pcc(preds_3d, target_3d, mask=mask)
    assert not torch.isnan(loss_masked)
