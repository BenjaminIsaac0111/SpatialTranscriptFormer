"""
Tests for loss functions: MaskedMSELoss, PCCLoss, CCCLoss, MaskedHuberLoss,
CLIPAlignmentLoss, and CompositeLoss.
"""

import pytest
import torch
import torch.nn as nn

from spatial_transcript_former.training.losses import (
    CCCLoss,
    CLIPAlignmentLoss,
    CompositeLoss,
    MaskedHuberLoss,
    MaskedMSELoss,
    PCCLoss,
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


# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# MaskedHuberLoss
# ---------------------------------------------------------------------------


class TestMaskedHuberLoss:
    def test_near_zero_matches_mse(self):
        """For small residuals (|x| << delta), Huber ≈ 0.5 * MSE."""
        torch.manual_seed(0)
        preds = torch.zeros(16, 50)
        target = preds + 0.01  # tiny residuals, well within quadratic zone
        huber = MaskedHuberLoss(delta=1.0)(preds, target)
        mse = MaskedMSELoss()(preds, target)
        # Huber = 0.5 * x^2, MSE = x^2 -> Huber ≈ 0.5 * MSE for small x
        assert huber.item() == pytest.approx(0.5 * mse.item(), rel=1e-3)

    def test_large_residuals_sub_quadratic(self):
        """For large residuals (|x| >> delta), Huber grows linearly, MSE quadratically."""
        preds = torch.zeros(8, 10)
        target = preds + 100.0  # far beyond delta=1.0
        huber = MaskedHuberLoss(delta=1.0)(preds, target)
        mse = MaskedMSELoss()(preds, target)
        # Huber should be much smaller than MSE for large errors
        assert huber.item() < mse.item()

    def test_mask_reduces_loss(self, tensors_3d):
        """Masking padded positions should change the loss value."""
        preds, target, mask = tensors_3d
        loss_no_mask = MaskedHuberLoss()(preds, target)
        loss_masked = MaskedHuberLoss()(preds, target, mask=mask)
        assert not torch.allclose(loss_no_mask, loss_masked)

    def test_gradient_flow(self, tensors_3d):
        """Gradients should flow through MaskedHuberLoss."""
        preds, target, mask = tensors_3d
        preds = preds.clone().requires_grad_(True)
        loss = MaskedHuberLoss()(preds, target, mask=mask)
        loss.backward()
        assert preds.grad is not None
        assert preds.grad.shape == preds.shape

    def test_perfect_predictions_zero(self):
        """Loss should be zero for identical preds and target."""
        x = torch.randn(8, 20)
        loss = MaskedHuberLoss()(x, x)
        assert loss.item() == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# CCCLoss
# ---------------------------------------------------------------------------


class TestCCCLoss:
    def test_perfect_predictions_zero(self):
        """Perfect predictions should give CCC=1, loss=0."""
        x = torch.randn(50, 100)
        loss = CCCLoss()(x, x)
        assert loss.item() == pytest.approx(0.0, abs=1e-5)

    def test_offset_penalised_more_than_pcc(self):
        """A constant offset prediction should have CCC loss > PCC loss."""
        torch.manual_seed(42)
        x = torch.randn(50, 100)
        # Shift predictions by a large constant — PCC is shift-invariant, CCC is not
        y = x + 5.0
        pcc_loss = PCCLoss()(x, y)
        ccc_loss = CCCLoss()(x, y)
        # PCC should be ~0 (perfect correlation), CCC should be larger
        assert pcc_loss.item() == pytest.approx(0.0, abs=1e-4)
        assert ccc_loss.item() > pcc_loss.item() + 0.1

    def test_3d_with_mask(self, tensors_3d):
        """CCCLoss should handle 3D inputs with mask."""
        preds, target, mask = tensors_3d
        loss = CCCLoss()(preds, target, mask=mask)
        assert loss.isfinite()
        assert 0.0 <= loss.item() <= 2.0

    def test_gradient_flow(self, tensors_2d):
        """Gradients should flow through CCCLoss."""
        preds, target = tensors_2d
        preds = preds.clone().requires_grad_(True)
        loss = CCCLoss()(preds, target)
        loss.backward()
        assert preds.grad is not None

    def test_anticorrelation(self):
        """Negated inputs should give loss > 1 (CCC is strongly negative)."""
        x = torch.randn(50, 100)
        loss = CCCLoss()(x, -x)
        # CCC of x and -x is negative, so 1 - CCC > 1
        assert loss.item() > 1.0


# ---------------------------------------------------------------------------
# CLIPAlignmentLoss
# ---------------------------------------------------------------------------


class TestCLIPAlignmentLoss:
    def test_batch_size_one_returns_zero(self):
        """B=1 should return 0.0 without crashing."""
        preds = torch.randn(1, 50)
        target = torch.randn(1, 50)
        loss = CLIPAlignmentLoss()(preds, target)
        assert loss.item() == pytest.approx(0.0, abs=1e-6)
        assert loss.requires_grad

    def test_identical_batch_has_high_loss(self):
        """If all predictions are identical, cross-entropy should be near log(B)."""
        B, G = 16, 50
        # All samples predict the same vector — worst case for CLIP
        preds = torch.ones(B, G)
        target = torch.randn(B, G)
        loss = CLIPAlignmentLoss(temperature=1.0)(preds, target)
        # Uniform distribution over B classes → cross-entropy ≈ log(B)
        import math

        assert loss.item() == pytest.approx(math.log(B), rel=0.05)

    def test_perfect_batch_has_low_loss(self):
        """Perfect predictions (preds == target) should give near-zero loss."""
        torch.manual_seed(7)
        x = torch.randn(8, 50)
        loss = CLIPAlignmentLoss(temperature=0.07)(x, x)
        # With identical embeddings the diagonal always wins → loss ≈ 0
        assert loss.item() < 0.05

    def test_3d_input_averaged(self, tensors_3d):
        """3D inputs should be averaged over spatial dim before CLIP loss."""
        preds, target, mask = tensors_3d
        loss = CLIPAlignmentLoss()(preds, target, mask=mask)
        assert loss.isfinite()

    def test_gradient_flow(self, tensors_2d):
        """Gradients should flow through CLIPAlignmentLoss."""
        preds, target = tensors_2d
        preds = preds.clone().requires_grad_(True)
        loss = CLIPAlignmentLoss()(preds, target)
        loss.backward()
        assert preds.grad is not None


# ---------------------------------------------------------------------------
# CompositeLoss — new variants
# ---------------------------------------------------------------------------


class TestCompositeLossVariants:
    def test_mse_ccc_gradients_flow(self, tensors_3d):
        """CompositeLoss(pcc_type='ccc') gradients flow through both terms."""
        preds, target, mask = tensors_3d
        preds = preds.clone().requires_grad_(True)
        loss = CompositeLoss(alpha=1.0, pcc_type="ccc")(preds, target, mask=mask)
        loss.backward()
        assert preds.grad is not None
        padded_grad = preds.grad[0, 80:, :]
        assert padded_grad.abs().sum() == 0.0

    def test_mse_ccc_clip_all_three_terms(self, tensors_2d):
        """CompositeLoss(pcc_type='ccc', clip_weight=0.5) should be > mse+ccc alone."""
        preds, target = tensors_2d
        torch.manual_seed(0)
        loss_no_clip = CompositeLoss(alpha=1.0, pcc_type="ccc", clip_weight=0.0)(
            preds, target
        )
        loss_with_clip = CompositeLoss(alpha=1.0, pcc_type="ccc", clip_weight=0.5)(
            preds, target
        )
        # CLIP term adds a positive value so combined loss should differ
        assert not torch.allclose(loss_no_clip, loss_with_clip)

    def test_mse_ccc_clip_gradients_flow(self, tensors_2d):
        """All three terms in mse+ccc+clip should contribute gradients."""
        preds, target = tensors_2d
        preds = preds.clone().requires_grad_(True)
        loss = CompositeLoss(alpha=1.0, pcc_type="ccc", clip_weight=0.5)(preds, target)
        loss.backward()
        assert preds.grad is not None

    def test_mse_huber_gradients_flow(self, tensors_2d):
        """CompositeLoss(mse_type='huber') should produce valid gradients."""
        preds, target = tensors_2d
        preds = preds.clone().requires_grad_(True)
        loss = CompositeLoss(alpha=1.0, mse_type="huber")(preds, target)
        loss.backward()
        assert preds.grad is not None

    def test_regression_mse_pcc_unchanged(self, tensors_2d):
        """Default CompositeLoss (pcc_type='pcc') should still equal MSE + PCC."""
        preds, target = tensors_2d
        mse_val = MaskedMSELoss()(preds, target)
        pcc_val = PCCLoss()(preds, target)
        expected = mse_val + 1.0 * pcc_val
        actual = CompositeLoss(alpha=1.0)(preds, target)
        assert torch.allclose(expected, actual, atol=1e-5)
