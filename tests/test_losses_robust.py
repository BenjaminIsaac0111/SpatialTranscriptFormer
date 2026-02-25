import torch
import torch.nn as nn
import pytest
from spatial_transcript_former.training.losses import ZINBLoss, PCCLoss


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


if __name__ == "__main__":
    pytest.main([__file__])
