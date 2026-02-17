"""
Tests for MSigDB pathway initialization and ground truth computation.

Verifies membership matrix structure, gene coverage, pathway truth consistency,
and the z-score normalization used for visualization.
"""
import pytest
import numpy as np
import torch
from spatial_transcript_former.data.pathways import get_pathway_init, download_hallmarks_gmt, parse_gmt


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def hallmarks():
    """Download and parse MSigDB Hallmarks (cached, shared across module)."""
    gmt_path = download_hallmarks_gmt(".cache")
    return parse_gmt(gmt_path)


@pytest.fixture(scope="module")
def gene_list():
    """A realistic gene list (subset of common genes)."""
    # These are known to appear in MSigDB Hallmarks
    known = ['TP53', 'MYC', 'VEGFA', 'CTNNB1', 'VIM', 'SNAI1',
             'MLH1', 'MSH2', 'CDH1', 'AXIN2', 'FLT1', 'TGFB1']
    # Add some filler genes
    filler = [f'GENE_{i}' for i in range(988)]
    return known + filler


@pytest.fixture(scope="module")
def pathway_result(gene_list):
    """Pathway initialization result: (matrix, names)."""
    return get_pathway_init(gene_list, verbose=False)


# ---------------------------------------------------------------------------
# GMT Parsing
# ---------------------------------------------------------------------------

class TestGMTParsing:
    def test_hallmarks_contains_50_pathways(self, hallmarks):
        """MSigDB Hallmarks should contain exactly 50 pathways."""
        assert len(hallmarks) == 50

    def test_pathway_names_have_prefix(self, hallmarks):
        """All pathway names should start with HALLMARK_."""
        for name in hallmarks:
            assert name.startswith('HALLMARK_'), f"Unexpected name: {name}"

    def test_each_pathway_has_genes(self, hallmarks):
        """Each pathway should contain at least one gene."""
        for name, genes in hallmarks.items():
            assert len(genes) > 0, f"{name} has no genes"


# ---------------------------------------------------------------------------
# Membership matrix
# ---------------------------------------------------------------------------

class TestMembershipMatrix:
    def test_shape(self, pathway_result, gene_list):
        """Matrix should be (50, G) where G = len(gene_list)."""
        matrix, names = pathway_result
        assert matrix.shape == (50, len(gene_list))

    def test_is_binary(self, pathway_result):
        """All values should be 0 or 1."""
        matrix, _ = pathway_result
        unique = torch.unique(matrix)
        assert set(unique.tolist()).issubset({0.0, 1.0})

    def test_known_gene_mapped(self, pathway_result, gene_list):
        """Known cancer genes should appear in at least one pathway."""
        matrix, _ = pathway_result
        known_genes = ['TP53', 'MYC', 'VEGFA', 'VIM']
        for gene in known_genes:
            if gene in gene_list:
                idx = gene_list.index(gene)
                column_sum = matrix[:, idx].sum().item()
                assert column_sum > 0, f"{gene} not mapped to any pathway"

    def test_pathway_count(self, pathway_result):
        """Should return exactly 50 pathway names."""
        _, names = pathway_result
        assert len(names) == 50

    def test_bowel_cancer_pathways_exist(self, pathway_result):
        """All 6 disease-relevant pathways should be in the names list."""
        _, names = pathway_result
        short_names = [n.replace('HALLMARK_', '') for n in names]
        required = [
            'EPITHELIAL_MESENCHYMAL_TRANSITION',
            'WNT_BETA_CATENIN_SIGNALING',
            'INFLAMMATORY_RESPONSE',
            'ANGIOGENESIS',
            'APOPTOSIS',
            'TNFA_SIGNALING_VIA_NFKB',
        ]
        for pw in required:
            assert pw in short_names, f"Missing pathway: {pw}"


# ---------------------------------------------------------------------------
# Pathway ground truth
# ---------------------------------------------------------------------------

class TestPathwayTruth:
    def test_consistent_across_calls(self, gene_list):
        """Ground truth from MSigDB membership should be identical across calls."""
        from spatial_transcript_former.visualization import _compute_pathway_truth

        np.random.seed(42)
        gene_truth = np.random.rand(200, len(gene_list)).astype(np.float32)

        result1, names1 = _compute_pathway_truth(gene_truth, gene_list)
        result2, names2 = _compute_pathway_truth(gene_truth, gene_list)

        np.testing.assert_array_equal(result1, result2)
        assert names1 == names2

    def test_output_shape(self, gene_list):
        """Pathway truth should be (N, P) where P=50."""
        from spatial_transcript_former.visualization import _compute_pathway_truth

        N = 150
        gene_truth = np.random.rand(N, len(gene_list)).astype(np.float32)
        result, names = _compute_pathway_truth(gene_truth, gene_list)

        assert result.shape == (N, 50)
        assert len(names) == 50

    def test_spatial_variation(self, gene_list):
        """Pathway truth should have spatial variation (non-zero std)."""
        from spatial_transcript_former.visualization import _compute_pathway_truth

        # Create gene expression with spatial patterns
        N = 200
        gene_truth = np.random.rand(N, len(gene_list)).astype(np.float32)
        # Add spatial structure to first few genes
        gene_truth[:100, 0] += 5.0
        gene_truth[100:, 1] += 5.0

        result, _ = _compute_pathway_truth(gene_truth, gene_list)

        # At least some pathways should have non-trivial spatial variation
        stds = np.std(result, axis=0)
        assert np.any(stds > 0.01), "Pathway truth has no spatial variation"
