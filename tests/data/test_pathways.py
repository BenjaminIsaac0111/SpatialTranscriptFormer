"""
Tests for MSigDB pathway parsing and membership matrix construction.
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


# ---------------------------------------------------------------------------
# Pathway Moran's I
# ---------------------------------------------------------------------------


class TestPathwayMoransI:
    """Tests for per-pathway Moran's I computation and H5 serialisation."""

    def test_spatially_coherent_pathway_has_high_morans(self):
        """A pathway with strong spatial structure should have high Moran's I."""
        from spatial_transcript_former.recipes.hest.compute_pathway_activities import (
            _compute_pathway_morans_i,
        )

        np.random.seed(42)
        # Grid of 100 spots
        xs, ys = np.meshgrid(np.arange(10), np.arange(10))
        coords = np.column_stack([xs.ravel(), ys.ravel()]).astype(np.float64)

        n_spots = 100
        n_pathways = 3

        activities = np.zeros((n_spots, n_pathways), dtype=np.float32)
        # Pathway 0: strong spatial gradient (high Moran's I)
        activities[:, 0] = coords[:, 0].astype(np.float32)
        # Pathway 1: random noise (low Moran's I)
        activities[:, 1] = np.random.randn(n_spots).astype(np.float32)
        # Pathway 2: constant (zero Moran's I)
        activities[:, 2] = 1.0

        morans = _compute_pathway_morans_i(activities, coords, k=6)
        assert morans.shape == (n_pathways,)
        assert morans.dtype == np.float32
        # Spatially coherent pathway should have high I
        assert morans[0] > 0.5
        # Random pathway should have low I (clipped to >= 0)
        assert morans[1] >= 0.0
        assert morans[1] < 0.3
        # Constant pathway: zero
        assert morans[2] == pytest.approx(0.0, abs=1e-6)

    def test_too_few_spots_returns_zeros(self):
        """With fewer spots than k+1, should return zeros."""
        from spatial_transcript_former.recipes.hest.compute_pathway_activities import (
            _compute_pathway_morans_i,
        )

        activities = np.random.randn(3, 5).astype(np.float32)
        coords = np.array([[0, 0], [1, 0], [0, 1]], dtype=np.float64)

        morans = _compute_pathway_morans_i(activities, coords, k=6)
        assert morans.shape == (5,)
        np.testing.assert_array_equal(morans, 0.0)

    def test_negative_morans_clipped_to_zero(self):
        """Negative Moran's I should be clipped to 0."""
        from spatial_transcript_former.recipes.hest.compute_pathway_activities import (
            _compute_pathway_morans_i,
        )

        np.random.seed(123)
        # Checkerboard pattern produces negative Moran's I
        xs, ys = np.meshgrid(np.arange(10), np.arange(10))
        coords = np.column_stack([xs.ravel(), ys.ravel()]).astype(np.float64)
        n_spots = 100

        activities = np.zeros((n_spots, 1), dtype=np.float32)
        activities[:, 0] = ((xs.ravel() + ys.ravel()) % 2).astype(np.float32)

        morans = _compute_pathway_morans_i(activities, coords, k=4)
        # Should be clipped to 0, not negative
        assert morans[0] >= 0.0

    def test_h5_roundtrip_morans(self, tmp_path):
        """pathway_morans_i should survive write/read roundtrip."""
        import h5py
        from spatial_transcript_former.recipes.hest.compute_pathway_activities import (
            load_pathway_activities,
            PATHWAY_FILE_VERSION,
        )

        n_spots, n_pathways = 50, 5
        acts = np.random.randn(n_spots, n_pathways).astype(np.float32)
        barcodes_raw = [f"SPOT_{i}" for i in range(n_spots)]
        barcodes_bytes = np.array(barcodes_raw, dtype="S")
        pw_names = np.array([f"PW_{i}" for i in range(n_pathways)], dtype="S")
        morans_orig = np.array([0.1, 0.5, 0.0, 0.8, 0.3], dtype=np.float32)

        h5_path = str(tmp_path / "test_sample.h5")
        with h5py.File(h5_path, "w") as f:
            f.create_dataset("activities", data=acts)
            f.create_dataset("barcodes", data=barcodes_bytes)
            f.create_dataset("pathway_names", data=pw_names)
            f.create_dataset("pathway_morans_i", data=morans_orig)
            f.attrs["format_version"] = PATHWAY_FILE_VERSION

        _, _, _, morans_loaded = load_pathway_activities(h5_path, barcodes_raw)
        assert morans_loaded is not None
        np.testing.assert_array_almost_equal(morans_loaded, morans_orig)

    def test_h5_missing_morans_returns_none(self, tmp_path):
        """A current-version H5 without pathway_morans_i should return None."""
        import h5py
        from spatial_transcript_former.recipes.hest.compute_pathway_activities import (
            load_pathway_activities,
            PATHWAY_FILE_VERSION,
        )

        n_spots, n_pathways = 20, 3
        acts = np.random.randn(n_spots, n_pathways).astype(np.float32)
        barcodes_raw = [f"SPOT_{i}" for i in range(n_spots)]
        barcodes_bytes = np.array(barcodes_raw, dtype="S")
        pw_names = np.array([f"PW_{i}" for i in range(n_pathways)], dtype="S")

        h5_path = str(tmp_path / "test_old_sample.h5")
        with h5py.File(h5_path, "w") as f:
            f.create_dataset("activities", data=acts)
            f.create_dataset("barcodes", data=barcodes_bytes)
            f.create_dataset("pathway_names", data=pw_names)
            f.attrs["format_version"] = PATHWAY_FILE_VERSION
            # No pathway_morans_i dataset

        _, _, _, morans_loaded = load_pathway_activities(h5_path, barcodes_raw)
        assert morans_loaded is None


# ---------------------------------------------------------------------------
# Score construction & file format
# ---------------------------------------------------------------------------


class TestPathwayScoring:
    """Tests for `_score_pathways` semantics and on-disk file versioning."""

    def test_score_is_simple_member_mean(self):
        """activities[s, p] should equal mean of log1p CP10k expression of pathway p
        members, computed on the input matrix as-is — no per-gene rescaling."""
        from spatial_transcript_former.recipes.hest.compute_pathway_activities import (
            _score_pathways,
        )

        # 3 spots, 4 genes
        expr = np.array(
            [
                [1.0, 2.0, 3.0, 0.0],
                [4.0, 0.0, 1.0, 2.0],
                [0.0, 5.0, 2.0, 1.0],
            ],
            dtype=np.float32,
        )
        gene_names = ["A", "B", "C", "D"]
        # Two pathways. PW1 has 3 members so is scored; PW2 has 1 so is skipped.
        pathway_dict = {
            "PW1": ["A", "B", "C"],
            "PW2": ["D"],
        }

        activities, names, n_scored = _score_pathways(
            expr, gene_names, pathway_dict, min_genes=3
        )

        assert activities.shape == (3, 2)
        assert names == ["PW1", "PW2"]
        assert n_scored == 1

        expected_pw1 = expr[:, [0, 1, 2]].mean(axis=1)
        np.testing.assert_array_almost_equal(activities[:, 0], expected_pw1)
        # PW2 below min_genes — left at zero
        np.testing.assert_array_almost_equal(activities[:, 1], np.zeros(3))

    def test_score_is_slide_stationary(self):
        """Adding a per-slide constant to expression shifts the score by the same
        constant — so a scoring-only z-score (which would centre this away) is
        not happening."""
        from spatial_transcript_former.recipes.hest.compute_pathway_activities import (
            _score_pathways,
        )

        np.random.seed(0)
        expr = np.random.rand(8, 5).astype(np.float32)
        gene_names = [f"G{i}" for i in range(5)]
        pathway_dict = {"PW": ["G0", "G1", "G2"]}

        acts_a, _, _ = _score_pathways(expr, gene_names, pathway_dict, min_genes=3)
        acts_b, _, _ = _score_pathways(
            expr + 5.0, gene_names, pathway_dict, min_genes=3
        )

        # Plain mean → constant-shift propagates exactly. (A z-scored variant
        # would centre to ~0 for both, breaking this property.)
        np.testing.assert_array_almost_equal(acts_b[:, 0] - acts_a[:, 0], 5.0)

    def test_load_rejects_missing_version(self, tmp_path):
        """A file with no format_version attribute must be rejected."""
        import h5py
        from spatial_transcript_former.recipes.hest.compute_pathway_activities import (
            load_pathway_activities,
        )

        n_spots, n_pathways = 5, 2
        h5_path = str(tmp_path / "no_version.h5")
        with h5py.File(h5_path, "w") as f:
            f.create_dataset(
                "activities",
                data=np.zeros((n_spots, n_pathways), dtype=np.float32),
            )
            f.create_dataset(
                "barcodes",
                data=np.array([f"S{i}" for i in range(n_spots)], dtype="S"),
            )
            f.create_dataset(
                "pathway_names",
                data=np.array([f"P{i}" for i in range(n_pathways)], dtype="S"),
            )
            # Intentionally omit f.attrs["format_version"]

        with pytest.raises(ValueError, match="format_version"):
            load_pathway_activities(h5_path, [f"S{i}" for i in range(n_spots)])

    def test_load_rejects_old_version(self, tmp_path):
        """A file with format_version=1 (old z-scored layout) must be rejected."""
        import h5py
        from spatial_transcript_former.recipes.hest.compute_pathway_activities import (
            load_pathway_activities,
        )

        n_spots, n_pathways = 5, 2
        h5_path = str(tmp_path / "v1.h5")
        with h5py.File(h5_path, "w") as f:
            f.create_dataset(
                "activities",
                data=np.zeros((n_spots, n_pathways), dtype=np.float32),
            )
            f.create_dataset(
                "barcodes",
                data=np.array([f"S{i}" for i in range(n_spots)], dtype="S"),
            )
            f.create_dataset(
                "pathway_names",
                data=np.array([f"P{i}" for i in range(n_pathways)], dtype="S"),
            )
            f.attrs["format_version"] = 1

        with pytest.raises(ValueError, match="format_version"):
            load_pathway_activities(h5_path, [f"S{i}" for i in range(n_spots)])


# ---------------------------------------------------------------------------
# Gene-identifier resolution (Ensembl-indexed slides)
# ---------------------------------------------------------------------------


def test_resolve_gene_symbols_maps_ensembl_via_symbol_column():
    """Ensembl-indexed slides must resolve to symbols, not score 0 pathways.

    HEST mixes conventions: some slides index var by Ensembl gene ID and keep
    the HGNC symbol in a var column. Hallmark sets are symbol-based, so
    without resolution such a slide matches zero members, trips the
    min_pathways guard and is dropped -- while the batch summary still reports
    "0 failed". That is silent corpus loss, so it is guarded here.
    """
    import anndata
    import numpy as np
    import pandas as pd

    from spatial_transcript_former.recipes.hest.compute_pathway_activities import (
        _resolve_gene_symbols,
    )

    ens = [f"ENSG{i:011d}" for i in range(6)]
    syms = ["TP53", "MYC", "EGFR", "VIM", "ACTB", "CD8A"]
    adata = anndata.AnnData(
        X=np.ones((3, 6), dtype=np.float32),
        var=pd.DataFrame({"SYMBOL": syms}, index=ens),
    )
    assert _resolve_gene_symbols(adata, "T") == syms


def test_resolve_gene_symbols_passes_through_symbol_index():
    """A slide already indexed by symbol must be returned unchanged."""
    import anndata
    import numpy as np
    import pandas as pd

    from spatial_transcript_former.recipes.hest.compute_pathway_activities import (
        _resolve_gene_symbols,
    )

    syms = ["TP53", "MYC", "EGFR"]
    adata = anndata.AnnData(
        X=np.ones((2, 3), dtype=np.float32),
        var=pd.DataFrame(index=syms),
    )
    assert _resolve_gene_symbols(adata, "T") == syms


def test_resolve_gene_symbols_without_symbol_column_is_graceful():
    """Ensembl index with no symbol column must warn, not raise."""
    import anndata
    import numpy as np
    import pandas as pd

    from spatial_transcript_former.recipes.hest.compute_pathway_activities import (
        _resolve_gene_symbols,
    )

    ens = [f"ENSG{i:011d}" for i in range(4)]
    adata = anndata.AnnData(
        X=np.ones((2, 4), dtype=np.float32),
        var=pd.DataFrame({"unrelated": list(range(4))}, index=ens),
    )
    assert _resolve_gene_symbols(adata, "T") == ens


def test_clean_gene_name_strips_multigenome_prefixes():
    """Multi-genome CellRanger prefixes use inconsistent separators in HEST.

    Real files carry ``GRCh38_``, ``GRCh38__`` and ``GRCh38______``. A cleaner
    that assumed a single underscore left ``_OR4F5``, which matches no Hallmark
    member -- so the sample scored 0/50 and was dropped while the batch summary
    still said "0 failed".
    """
    from spatial_transcript_former.recipes.hest.compute_pathway_activities import (
        clean_gene_name,
    )

    for raw in ("GRCh38_OR4F5", "GRCh38__OR4F5", "GRCh38______OR4F5", "OR4F5"):
        assert clean_gene_name(raw) == "OR4F5", raw
    assert clean_gene_name("mm10___Actb") == "ACTB"
    assert clean_gene_name("wuhCor1__ORF1ab") == "ORF1AB"


def test_clean_gene_name_preserves_hyphenated_symbols():
    """Real HGNC symbols contain hyphens and digits; they must survive intact."""
    from spatial_transcript_former.recipes.hest.compute_pathway_activities import (
        clean_gene_name,
    )

    for sym in ("HLA-DRB1", "NKX2-1", "TP53", "MT-CO1"):
        assert clean_gene_name(sym) == sym.upper()


def test_score_pathways_matches_genome_prefixed_genes():
    """End-to-end: a genome-prefixed matrix must score, not silently yield 0."""
    import numpy as np

    from spatial_transcript_former.recipes.hest.compute_pathway_activities import (
        _score_pathways,
    )

    genes = [
        "GRCh38______TP53",
        "GRCh38______MYC",
        "GRCh38______EGFR",
        "GRCh38______VIM",
        "GRCh38______ACTB",
        "GRCh38______CD8A",
    ]
    expr = np.random.default_rng(0).random((10, len(genes))).astype(np.float32)
    pathways = {"HALLMARK_TEST": ["TP53", "MYC", "EGFR", "VIM", "ACTB"]}

    activities, names, n_scored = _score_pathways(expr, genes, pathways, min_genes=5)
    assert n_scored == 1, "genome-prefixed genes failed to match the pathway"
    assert activities.shape == (10, 1)
    assert np.isfinite(activities).all()
