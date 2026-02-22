"""
Tests for loss functions: MaskedMSELoss, PCCLoss, CompositeLoss.

Verifies correctness of masking, scale invariance, gradients,
and composite objective decomposition.
"""

import pytest
import torch
import torch.nn as nn
from spatial_transcript_former.training.losses import (
    MaskedMSELoss,
    PCCLoss,
    CompositeLoss,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


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
        """Negated inputs should give loss ≈ 2.0."""
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

    def test_gradient_flow(self, tensors_2d):
        """Gradients should flow through PCC loss."""
        preds, target = tensors_2d
        preds = preds.clone().requires_grad_(True)
        loss = PCCLoss()(preds, target)
        loss.backward()
        assert preds.grad is not None


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
