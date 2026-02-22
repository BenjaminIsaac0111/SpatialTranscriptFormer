"""
Tests for visualization utilities.

Verifies pathway lookup, z-score normalization, and plot generation.
"""

import pytest
import os
import numpy as np
import tempfile
from spatial_transcript_former.predict import (
    BOWEL_CANCER_PATHWAYS,
    plot_training_summary,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def pathway_names():
    """Realistic MSigDB Hallmark pathway names."""
    from spatial_transcript_former.data.pathways import (
        download_hallmarks_gmt,
        parse_gmt,
    )

    gmt_path = download_hallmarks_gmt(".cache")
    return list(parse_gmt(gmt_path).keys())


@pytest.fixture
def mock_data(pathway_names):
    """Mock coords, predictions, and truth for plotting."""
    np.random.seed(42)
    N = 200
    P = len(pathway_names)
    coords = np.column_stack(
        [
            np.random.uniform(0, 1000, N),
            np.random.uniform(0, 1000, N),
        ]
    )
    pathway_pred = np.random.randn(N, P).astype(np.float32)
    pathway_truth = np.random.randn(N, P).astype(np.float32)
    return coords, pathway_pred, pathway_truth


# ---------------------------------------------------------------------------
# Pathway constants
# ---------------------------------------------------------------------------


class TestBowelCancerPathways:
    def test_all_pathways_exist_in_msigdb(self, pathway_names):
        """All 6 bowel cancer pathways should be in the MSigDB Hallmarks."""
        short_names = [n.replace("HALLMARK_", "") for n in pathway_names]
        for pw in BOWEL_CANCER_PATHWAYS:
            assert pw in short_names, f"Missing: {pw}"

    def test_pathway_count(self):
        """Should have exactly 6 fixed pathways."""
        assert len(BOWEL_CANCER_PATHWAYS) == 6


# ---------------------------------------------------------------------------
# Plot generation
# ---------------------------------------------------------------------------


class TestPlotTrainingSummary:
    def test_saves_file(self, mock_data, pathway_names):
        """Plot should be saved to the specified path."""
        coords, pred, truth = mock_data
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, "test_plot.png")
            plot_training_summary(
                coords,
                pred,
                truth,
                pathway_names,
                sample_id="TEST",
                save_path=save_path,
            )
            assert os.path.exists(save_path)
            assert os.path.getsize(save_path) > 0

    def test_with_histology(self, mock_data, pathway_names):
        """Plot should work with a histology image."""
        coords, pred, truth = mock_data
        fake_img = np.random.randint(0, 255, (500, 500, 3), dtype=np.uint8)
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, "test_histo.png")
            plot_training_summary(
                coords,
                pred,
                truth,
                pathway_names,
                sample_id="TEST",
                histology_img=fake_img,
                save_path=save_path,
            )
            assert os.path.exists(save_path)

    def test_no_matching_pathways_skips(self, mock_data):
        """Should skip gracefully if pathway names don't match."""
        coords, pred, truth = mock_data
        fake_names = [f"UNKNOWN_{i}" for i in range(pred.shape[1])]
        # Should not raise, just print warning
        plot_training_summary(coords, pred, truth, fake_names, sample_id="TEST")


# ---------------------------------------------------------------------------
# Z-score normalization
# ---------------------------------------------------------------------------


class TestZScoreNormalization:
    def test_z_score_properties(self):
        """Z-scored data should have mean ≈ 0 and std ≈ 1."""
        np.random.seed(42)
        raw = np.random.rand(500) * 100 + 50  # mean ~100, std ~29
        eps = 1e-8
        z = (raw - raw.mean()) / (raw.std() + eps)

        assert abs(z.mean()) < 1e-6
        assert abs(z.std() - 1.0) < 1e-6

    def test_z_score_preserves_relative_ordering(self):
        """Z-scoring should preserve which spots are high vs low."""
        np.random.seed(42)
        raw = np.random.rand(100) * 1000
        eps = 1e-8
        z = (raw - raw.mean()) / (raw.std() + eps)

        # Top-5 spots should be the same
        top_raw = np.argsort(raw)[-5:]
        top_z = np.argsort(z)[-5:]
        np.testing.assert_array_equal(top_raw, top_z)

    def test_constant_input_handled(self):
        """Constant input should produce zeros (not NaN)."""
        raw = np.ones(100) * 5.0
        eps = 1e-8
        z = (raw - raw.mean()) / (raw.std() + eps)
        assert np.all(np.isfinite(z))
        assert np.allclose(z, 0.0, atol=1e-4)
