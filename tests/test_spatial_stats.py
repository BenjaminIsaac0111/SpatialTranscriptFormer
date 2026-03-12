"""Unit tests for spatial_stats module (Moran's I computation)."""

import numpy as np
import pytest
from spatial_transcript_former.data.spatial_stats import (
    morans_i,
    morans_i_batch,
    spatial_coherence_score,
    _build_knn_weights,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_grid(rows=10, cols=10):
    """Create a regular (rows x cols) grid of 2D coordinates."""
    xs, ys = np.meshgrid(np.arange(cols), np.arange(rows))
    return np.column_stack([xs.ravel(), ys.ravel()]).astype(np.float64)


# ---------------------------------------------------------------------------
# Tests for _build_knn_weights
# ---------------------------------------------------------------------------


class TestBuildKnnWeights:
    def test_shape(self):
        coords = _make_grid(5, 5)
        W = _build_knn_weights(coords, k=4)
        assert W.shape == (25, 25)

    def test_row_normalised(self):
        """Each row should sum to approximately 1.0 (1/k * k neighbours)."""
        coords = _make_grid(5, 5)
        W = _build_knn_weights(coords, k=4)
        row_sums = np.array(W.sum(axis=1)).flatten()
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-10)

    def test_no_self_loops(self):
        """Diagonal of the weight matrix should be zero."""
        coords = _make_grid(5, 5)
        W = _build_knn_weights(coords, k=4)
        assert W.diagonal().sum() == 0.0


# ---------------------------------------------------------------------------
# Tests for morans_i
# ---------------------------------------------------------------------------


class TestMoransI:
    def test_uniform_expression(self):
        """Constant (uniform) expression → Moran's I should be 0."""
        coords = _make_grid(10, 10)
        x = np.ones(100)
        W = _build_knn_weights(coords, k=6)
        I = morans_i(x, W)
        assert I == pytest.approx(0.0, abs=1e-10)

    def test_spatially_clustered(self):
        """Left half high, right half low → strong positive autocorrelation."""
        coords = _make_grid(10, 10)
        x = np.zeros(100)
        # Left half (cols 0-4) = high, right half (cols 5-9) = low
        for i in range(100):
            col = i % 10
            x[i] = 10.0 if col < 5 else 0.0

        W = _build_knn_weights(coords, k=6)
        I = morans_i(x, W)
        # Strong spatial clustering → very high I
        assert I > 0.5, f"Expected Moran's I > 0.5 for clustered data, got {I}"

    def test_checkerboard(self):
        """Alternating checkerboard → negative autocorrelation."""
        coords = _make_grid(10, 10)
        x = np.zeros(100)
        for i in range(100):
            row, col = i // 10, i % 10
            x[i] = 1.0 if (row + col) % 2 == 0 else 0.0

        W = _build_knn_weights(coords, k=4)
        I = morans_i(x, W)
        assert I < -0.3, f"Expected Moran's I < -0.3 for checkerboard, got {I}"

    def test_gradient(self):
        """Smooth left-to-right gradient → positive autocorrelation."""
        coords = _make_grid(10, 10)
        x = np.array([i % 10 for i in range(100)], dtype=np.float64)

        W = _build_knn_weights(coords, k=6)
        I = morans_i(x, W)
        assert I > 0.3, f"Expected Moran's I > 0.3 for gradient, got {I}"


# ---------------------------------------------------------------------------
# Tests for morans_i_batch
# ---------------------------------------------------------------------------


class TestMoransIBatch:
    def test_output_shape(self):
        """Output should be (G,) for a (N, G) input."""
        coords = _make_grid(10, 10)
        expression = np.random.rand(100, 50)
        scores = morans_i_batch(expression, coords, k=6)
        assert scores.shape == (50,)

    def test_consistent_with_single(self):
        """Batch result should match individual morans_i calls."""
        coords = _make_grid(10, 10)
        expression = np.random.rand(100, 5)

        batch_scores = morans_i_batch(expression, coords, k=6)

        W = _build_knn_weights(coords, k=6)
        for g in range(5):
            single_score = morans_i(expression[:, g], W)
            assert batch_scores[g] == pytest.approx(single_score, abs=1e-10)

    def test_too_few_spots(self):
        """Fewer spots than k+1 should return all zeros gracefully."""
        coords = np.array([[0, 0], [1, 0], [0, 1]], dtype=np.float64)
        expression = np.random.rand(3, 10)
        scores = morans_i_batch(expression, coords, k=6)
        np.testing.assert_array_equal(scores, 0.0)

    def test_identifies_spatial_gene(self):
        """A clustered gene should score higher than a random gene."""
        coords = _make_grid(10, 10)
        n = 100

        # Gene 0: spatially clustered (left/right split)
        clustered = np.array([10.0 if i % 10 < 5 else 0.0 for i in range(n)])
        # Gene 1: random noise
        rng = np.random.RandomState(42)
        random_gene = rng.rand(n)

        expression = np.column_stack([clustered, random_gene])
        scores = morans_i_batch(expression, coords, k=6)

        assert scores[0] > scores[1], (
            f"Clustered gene I={scores[0]:.3f} should be > "
            f"random gene I={scores[1]:.3f}"
        )


# ---------------------------------------------------------------------------
# Tests for spatial_coherence_score
# ---------------------------------------------------------------------------


class TestSpatialCoherenceScore:
    def test_perfect_prediction(self):
        """Identical predictions and truth should score near 1.0."""
        coords = _make_grid(10, 10)
        n = 100
        rng = np.random.RandomState(42)

        # Create expression with a mix of spatial and random genes
        expression = np.column_stack([
            np.array([10.0 if i % 10 < 5 else 0.0 for i in range(n)]),  # clustered
            np.array([float(i % 10) for i in range(n)]),  # gradient
            rng.rand(n),  # random
            rng.rand(n),  # random
        ])

        score = spatial_coherence_score(
            predicted=expression,
            ground_truth=expression,
            coords=coords,
            k=6,
            top_k_genes=3,
        )
        assert score > 0.99, f"Perfect prediction should score ~1.0, got {score}"

    def test_random_prediction(self):
        """Random predictions should score near 0 (low coherence)."""
        coords = _make_grid(10, 10)
        n = 100
        rng = np.random.RandomState(42)

        # Ground truth: spatially structured
        gt = np.column_stack([
            np.array([10.0 if i % 10 < 5 else 0.0 for i in range(n)]),
            np.array([float(i // 10) for i in range(n)]),
            np.array([float(i % 10) for i in range(n)]),
        ])
        # Predictions: random noise
        pred = rng.rand(n, 3)

        score = spatial_coherence_score(
            predicted=pred,
            ground_truth=gt,
            coords=coords,
            k=6,
            top_k_genes=3,
        )
        # Random should be far from 1.0
        assert score < 0.5, f"Random prediction should score low, got {score}"

    def test_too_few_spots(self):
        """Gracefully return 0.0 with insufficient spots."""
        coords = np.array([[0, 0], [1, 0]], dtype=np.float64)
        expression = np.random.rand(2, 5)
        score = spatial_coherence_score(
            predicted=expression,
            ground_truth=expression,
            coords=coords,
            k=6,
        )
        assert score == 0.0

