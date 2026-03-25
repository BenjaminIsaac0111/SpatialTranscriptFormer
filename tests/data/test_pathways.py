"""
Merged tests: test_pathways.py, test_pathways_robust.py, test_pathway_stability.py
"""

import pytest
import numpy as np
import torch

from spatial_transcript_former.data.pathways import (
    get_pathway_init,
    download_msigdb_gmt,
    parse_gmt,
    MSIGDB_URLS,
)
from spatial_transcript_former.data.pathways import build_membership_matrix
from spatial_transcript_former.models.interaction import SpatialTranscriptFormer
from spatial_transcript_former.training.losses import (
    AuxiliaryPathwayLoss,
    MaskedMSELoss,
)

# --- From test_pathways.py ---


@pytest.fixture(scope="module")
def hallmarks():
    """Download and parse MSigDB Hallmarks (cached, shared across module)."""
    url = MSIGDB_URLS["hallmarks"]
    filename = url.split("/")[-1]
    gmt_path = download_msigdb_gmt(url, filename, ".cache")
    return parse_gmt(gmt_path)


@pytest.fixture(scope="module")
def gene_list():
    """A realistic gene list (subset of common genes)."""
    # These are known to appear in MSigDB Hallmarks
    known = [
        "TP53",
        "MYC",
        "VEGFA",
        "CTNNB1",
        "VIM",
        "SNAI1",
        "MLH1",
        "MSH2",
        "CDH1",
        "AXIN2",
        "FLT1",
        "TGFB1",
    ]
    # Add some filler genes
    filler = [f"GENE_{i}" for i in range(988)]
    return known + filler


@pytest.fixture(scope="module")
def pathway_result(gene_list):
    """Pathway initialization result: (matrix, names)."""
    return get_pathway_init(
        gene_list, gmt_urls=[MSIGDB_URLS["hallmarks"]], verbose=False
    )


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
            assert name.startswith("HALLMARK_"), f"Unexpected name: {name}"

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
        known_genes = ["TP53", "MYC", "VEGFA", "VIM"]
        for gene in known_genes:
            if gene in gene_list:
                idx = gene_list.index(gene)
                column_sum = matrix[:, idx].sum().item()
                assert column_sum > 0, f"{gene} not mapped to any pathway"

    def test_pathway_count(self, pathway_result):
        """Should return exactly 50 pathway names."""
        _, names = pathway_result
        assert len(names) == 50

    def test_core_pathways_exist(self, pathway_result):
        """All 6 representative pathways should be in the names list."""
        _, names = pathway_result
        short_names = [n.replace("HALLMARK_", "") for n in names]
        required = [
            "EPITHELIAL_MESENCHYMAL_TRANSITION",
            "WNT_BETA_CATENIN_SIGNALING",
            "INFLAMMATORY_RESPONSE",
            "ANGIOGENESIS",
            "APOPTOSIS",
            "TNFA_SIGNALING_VIA_NFKB",
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
        from unittest.mock import MagicMock

        args = MagicMock()
        args.sparsity_lambda = 0.0
        args.pathways = None

        np.random.seed(42)
        gene_truth = np.random.rand(200, len(gene_list)).astype(np.float32)

        result1, names1 = _compute_pathway_truth(gene_truth, gene_list, args)
        result2, names2 = _compute_pathway_truth(gene_truth, gene_list, args)

        np.testing.assert_array_equal(result1, result2)
        assert names1 == names2

    def test_output_shape(self, gene_list):
        """Pathway truth should be (N, P) where P=50 (Hallmarks default)."""
        from spatial_transcript_former.visualization import _compute_pathway_truth
        from unittest.mock import MagicMock

        args = MagicMock()
        args.sparsity_lambda = 0.0
        args.pathways = None

        N = 150
        gene_truth = np.random.rand(N, len(gene_list)).astype(np.float32)
        result, names = _compute_pathway_truth(gene_truth, gene_list, args)

        assert result.shape == (N, 50)
        assert len(names) == 50

    def test_spatial_variation(self, gene_list):
        """Pathway truth should have spatial variation (non-zero std)."""
        from spatial_transcript_former.visualization import _compute_pathway_truth
        from unittest.mock import MagicMock

        args = MagicMock()
        args.sparsity_lambda = 0.0
        args.pathways = None

        # Create gene expression with spatial patterns
        N = 200
        gene_truth = np.random.rand(N, len(gene_list)).astype(np.float32)
        # Add spatial structure to first few genes
        gene_truth[:100, 0] += 5.0
        gene_truth[100:, 1] += 5.0

        result, _ = _compute_pathway_truth(gene_truth, gene_list, args)

        # At least some pathways should have non-trivial spatial variation
        stds = np.std(result, axis=0)
        assert np.any(stds > 0.01), "Pathway truth has no spatial variation"


# --- From test_pathways_robust.py ---


def test_build_membership_matrix_integrity():
    """Verify that the membership matrix correctly maps genes to pathways."""
    pathway_dict = {
        "PATHWAY_A": ["GENE_1", "GENE_2"],
        "PATHWAY_B": ["GENE_2", "GENE_3"],
    }
    gene_list = ["GENE_1", "GENE_2", "GENE_3", "GENE_4"]

    matrix, names = build_membership_matrix(pathway_dict, gene_list)

    assert names == ["PATHWAY_A", "PATHWAY_B"]
    assert matrix.shape == (2, 4)

    # Pathway A: GENE_1, GENE_2
    assert matrix[0, 0] == 1.0
    assert matrix[0, 1] == 1.0
    assert matrix[0, 2] == 0.0
    assert matrix[0, 3] == 0.0

    # Pathway B: GENE_2, GENE_3
    assert matrix[1, 0] == 0.0
    assert matrix[1, 1] == 1.0
    assert matrix[1, 2] == 1.0
    assert matrix[1, 3] == 0.0


def test_build_membership_matrix_empty():
    """Check behavior with no matches."""
    pathway_dict = {"EMPTY": ["XYZ"]}
    gene_list = ["ABC", "DEF"]
    matrix, names = build_membership_matrix(pathway_dict, gene_list)
    assert matrix.sum() == 0
    assert names == ["EMPTY"]


# --- From test_pathway_stability.py ---


def test_pathway_initialization_stability_and_gradients():
    """
    Verifies that initializing the model with a binary pathway matrix:
    1. Does not cause predictions to exponentially explode (numerical stability).
    2. Allows gradients to flow properly when using AuxiliaryPathwayLoss.
    """
    torch.manual_seed(42)
    num_pathways = 50
    num_genes = 100

    # Create a synthetic MSigDB-style binary matrix
    pathway_matrix = (torch.rand(num_pathways, num_genes) > 0.8).float()
    # Ensure no empty pathways to avoid division by zero
    pathway_matrix[:, 0] = 1.0

    # Initialize model with pathway_init
    model = SpatialTranscriptFormer(
        num_genes=num_genes,
        num_pathways=num_pathways,
        pathway_init=pathway_matrix,
        use_spatial_pe=False,
        output_mode="counts",
        pretrained=False,
    )

    # Dummy inputs
    B, S, D = (
        2,
        10,
        2048,
    )  # Using D=2048 since backbone='resnet50' requires it natively, or provided features
    feats = torch.randn(B, S, D, requires_grad=True)
    coords = torch.randn(B, S, 2)
    target_genes = torch.randn(B, S, num_genes).abs()
    mask = torch.zeros(B, S, dtype=torch.bool)

    # Forward pass
    # return_pathways=True is needed to get the intermediate pathway preds for Auxiliary loss
    gene_preds, pathway_preds = model(
        feats, rel_coords=coords, return_dense=True, return_pathways=True
    )

    # 1. Numerical Stability Check
    # Without L1 normalization and removing temperature, predictions would explode.
    # With the fix, Softplus should keep outputs reasonably small.
    max_pred = gene_preds.max().item()
    print(f"Max prediction value at initialization: {max_pred:.2f}")
    assert (
        max_pred < 100.0
    ), f"Predictions exploded! Max value: {max_pred}. Check L1 normalization."
    assert not torch.isnan(gene_preds).any(), "Found NaNs in initial predictions."

    # 2. Gradient Flow Check (Compatibility with Training)
    loss_fn = AuxiliaryPathwayLoss(pathway_matrix, MaskedMSELoss(), lambda_pathway=1.0)
    loss = loss_fn(gene_preds, target_genes, mask=mask, pathway_preds=pathway_preds)

    assert loss.isfinite(), "Loss is not finite."

    loss.backward()

    # Verify gradients reached the core transformer layers
    target_layer_grad = model.fusion_engine.layers[0].linear1.weight.grad
    assert target_layer_grad is not None, "Gradients did not reach the fusion engine."
    assert target_layer_grad.norm() > 0, "Vanishing gradients in the fusion engine."
    assert torch.isfinite(
        target_layer_grad
    ).all(), "Exploding/NaN gradients in fusion engine."

    # Verify gradients reached the final reconstructor layer
    recon_grad = model.gene_reconstructor.weight.grad
    assert recon_grad is not None, "Gradients did not reach the gene reconstructor."
    assert recon_grad.norm() > 0, "Vanishing gradients in the gene reconstructor."
    assert torch.isfinite(
        recon_grad
    ).all(), "Exploding/NaN gradients in gene reconstructor."

    print("Pathway initialization is fully stable and compatible with NN training.")
