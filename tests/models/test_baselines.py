"""
Tests for the spatial baselines (SpatialTransformerRegressor, KNNRetrievalBaseline).
"""

import torch
import pytest

from spatial_transcript_former.models import (
    SpatialTransformerRegressor,
    KNNRetrievalBaseline,
)


# ---------------------------------------------------------------------------
# SpatialTransformerRegressor
# ---------------------------------------------------------------------------


def test_spatial_transformer_dense_and_global_shapes():
    model = SpatialTransformerRegressor(
        input_dim=128, num_pathways=50, token_dim=64, n_heads=4, n_layers=2
    ).eval()
    B, S, D = 2, 7, 128
    feats = torch.randn(B, S, D)
    coords = torch.randn(B, S, 2)

    dense = model(feats, rel_coords=coords, return_dense=True)
    assert dense.shape == (B, S, 50)

    glob = model(feats, rel_coords=coords, return_dense=False)
    assert glob.shape == (B, 50)


def test_spatial_transformer_outputs_non_negative():
    """The Softplus head must keep predictions non-negative (target range)."""
    model = SpatialTransformerRegressor(
        input_dim=64, num_pathways=10, token_dim=32, use_spatial_pe=False
    ).eval()
    feats = torch.randn(2, 5, 64) * 10.0
    out = model(feats, return_dense=True)
    assert (out >= 0).all()


def test_spatial_transformer_mixes_patches():
    """h2h-style spatial mixing: patch 0 output should depend on patch 1 input."""
    model = SpatialTransformerRegressor(
        input_dim=32, num_pathways=10, token_dim=32, use_spatial_pe=False
    ).eval()
    feats = torch.randn(1, 3, 32, requires_grad=True)
    out = model(feats, return_dense=True)
    out[0, 0].sum().backward()
    assert feats.grad[0, 1].norm() > 0


def test_spatial_transformer_requires_coords_with_pe():
    model = SpatialTransformerRegressor(input_dim=32, use_spatial_pe=True)
    with pytest.raises(ValueError, match="rel_coords was not provided"):
        model(torch.randn(2, 4, 32), return_dense=True)


def test_spatial_transformer_runs_with_padding_mask():
    model = SpatialTransformerRegressor(
        input_dim=32, num_pathways=8, token_dim=32, use_spatial_pe=False
    ).eval()
    feats = torch.randn(2, 5, 32)
    mask = torch.zeros(2, 5, dtype=torch.bool)
    mask[0, 3:] = True  # pad last two spots of first slide
    out = model(feats, mask=mask, return_dense=False)
    assert out.shape == (2, 8)
    assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# KNNRetrievalBaseline
# ---------------------------------------------------------------------------


def test_knn_exact_retrieval_with_k1():
    """k=1 query equal to a bank row must return that row's target exactly."""
    bank_feats = torch.eye(4)  # 4 orthogonal feature vectors
    bank_tgts = torch.tensor([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [4.0, 0.0]])
    model = KNNRetrievalBaseline(bank_feats, bank_tgts, k=1, metric="cosine")

    query = bank_feats[2].unsqueeze(0)  # matches row 2 -> target [3, 0]
    pred = model(query)
    assert torch.allclose(pred[0], bank_tgts[2])


def test_knn_averages_k_neighbours():
    bank_feats = torch.tensor([[1.0, 0.0], [0.99, 0.01], [-1.0, 0.0]])
    bank_tgts = torch.tensor([[10.0], [20.0], [99.0]])
    model = KNNRetrievalBaseline(bank_feats, bank_tgts, k=2, metric="cosine")
    pred = model(torch.tensor([[1.0, 0.0]]))
    # Two nearest are rows 0 and 1 -> mean(10, 20) = 15, far row 2 excluded.
    assert torch.allclose(pred[0], torch.tensor([15.0]))


def test_knn_preserves_leading_shape_and_metrics():
    bank_feats = torch.randn(50, 16)
    bank_tgts = torch.randn(50, 5)
    for metric in ("cosine", "l2"):
        model = KNNRetrievalBaseline(bank_feats, bank_tgts, k=4, metric=metric)
        out = model(torch.randn(2, 6, 16))  # (B, N, D) -> (B, N, P)
        assert out.shape == (2, 6, 5)
        assert torch.isfinite(out).all()


def test_knn_k_clamped_to_bank_size():
    model = KNNRetrievalBaseline(torch.randn(3, 8), torch.randn(3, 2), k=16)
    assert model.k == 3


def test_knn_rejects_mismatched_banks():
    with pytest.raises(ValueError, match="same number of rows"):
        KNNRetrievalBaseline(torch.randn(5, 8), torch.randn(4, 2))
