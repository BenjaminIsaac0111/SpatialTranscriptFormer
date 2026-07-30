"""
Tests for the spatial-attribution module (docs/EXPERIMENT_SPATIAL_ATTRIBUTION.md).
"""

import numpy as np
import torch

from spatial_transcript_former.attribution import (
    gradient_saliency,
    pathway_attention_map,
    remove_shared_component,
    shared_attention_map,
    spatial_pattern_fidelity,
)
from spatial_transcript_former.models import (
    AttentionMIL,
    SpatialTranscriptFormer,
    TransMIL,
)


def _make_stf(num_pathways=6, feat_dim=16, token_dim=32, n_heads=2, n_layers=2):
    model = SpatialTranscriptFormer(
        num_pathways=num_pathways,
        backbone_name="resnet50",
        pretrained=False,
        token_dim=token_dim,
        n_heads=n_heads,
        n_layers=n_layers,
        use_spatial_pe=True,
    )
    # Precomputed-feature mode never runs the (2048-dim resnet50) backbone,
    # so swap image_proj to match the small synthetic feature dim used here.
    model.image_proj = torch.nn.Linear(feat_dim, token_dim)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# pathway_attention_map — shape and normalisation
# ---------------------------------------------------------------------------


def test_pathway_attention_map_mean_shape_and_mask():
    n, d, p = 50, 16, 6
    model = _make_stf(num_pathways=p, feat_dim=d, token_dim=d)
    feats = torch.randn(1, n, d)
    coords = torch.rand(1, n, 2)
    mask = torch.zeros(1, n, dtype=torch.bool)
    mask[0, -10:] = True  # last 10 patches padded

    attn = pathway_attention_map(model, feats, coords, mask, reduce="mean")
    assert attn.shape == (n - 10, p)
    assert np.isfinite(attn).all()


def test_pathway_attention_map_rollout_shape():
    n, d, p = 40, 16, 4
    model = _make_stf(num_pathways=p, feat_dim=d, token_dim=d)
    feats = torch.randn(1, n, d)
    coords = torch.rand(1, n, 2)

    attn = pathway_attention_map(model, feats, coords, mask=None, reduce="rollout")
    assert attn.shape == (n, p)
    assert np.isfinite(attn).all()


def test_pathway_attention_map_unbatched_input():
    """Accepts (S, D)/(S, 2) tensors without an explicit batch dim."""
    n, d, p = 30, 16, 3
    model = _make_stf(num_pathways=p, feat_dim=d, token_dim=d)
    feats = torch.randn(n, d)
    coords = torch.rand(n, 2)

    attn = pathway_attention_map(model, feats, coords, mask=None)
    assert attn.shape == (n, p)


# ---------------------------------------------------------------------------
# shared_attention_map — MIL models
# ---------------------------------------------------------------------------


def test_shared_attention_map_attention_mil_shape():
    n, d = 25, 16
    model = AttentionMIL(input_dim=d, hidden_dim=16, output_dim=5)
    model.eval()
    feats = torch.randn(1, n, d)

    attn = shared_attention_map(model, feats)
    assert attn.shape == (n,)
    # AttentionMIL softmaxes over all N patches -> weights sum to ~1.
    assert np.isclose(attn.sum(), 1.0, atol=1e-4)


def test_shared_attention_map_respects_mask():
    n, d = 25, 16
    model = AttentionMIL(input_dim=d, hidden_dim=16, output_dim=5)
    model.eval()
    feats = torch.randn(1, n, d)
    mask = torch.zeros(1, n, dtype=torch.bool)
    mask[0, -5:] = True

    attn = shared_attention_map(model, feats, mask)
    assert attn.shape == (n - 5,)


def test_shared_attention_map_transmil_shape():
    n, d = 300, 16  # NystromAttention needs a reasonably large N
    model = TransMIL(input_dim=d, output_dim=5)
    model.eval()
    feats = torch.randn(1, n, d)

    attn = shared_attention_map(model, feats)
    assert attn.shape == (n,)


# ---------------------------------------------------------------------------
# gradient_saliency — pathway-specific, runs for STF and MIL
# ---------------------------------------------------------------------------


def test_gradient_saliency_stf_shape():
    n, d, p = 20, 16, 5
    model = _make_stf(num_pathways=p, feat_dim=d, token_dim=d)
    feats = torch.randn(1, n, d)
    coords = torch.rand(1, n, 2)

    sal = gradient_saliency(model, feats, coords, pathway_idx=2)
    assert sal.shape == (n,)
    assert np.isfinite(sal).all()


def test_gradient_saliency_is_pathway_specific():
    """Saliency for two different pathways should generally differ (the
    backward pass is seeded from a different scalar output each time) — this
    is the property that makes gradient saliency a *fair* per-pathway signal
    even for MIL models whose raw attention is a single shared map (doc §7)."""
    n, d, p = 20, 16, 5
    model = _make_stf(num_pathways=p, feat_dim=d, token_dim=d)
    feats = torch.randn(1, n, d)
    coords = torch.rand(1, n, 2)

    sal_0 = gradient_saliency(model, feats, coords, pathway_idx=0)
    sal_1 = gradient_saliency(model, feats, coords, pathway_idx=1)
    assert not np.allclose(sal_0, sal_1)


def test_gradient_saliency_mil_runs():
    n, d = 20, 16
    model = AttentionMIL(input_dim=d, hidden_dim=16, output_dim=4)
    model.eval()
    feats = torch.randn(1, n, d)

    sal = gradient_saliency(model, feats, pathway_idx=1)
    assert sal.shape == (n,)
    assert np.isfinite(sal).all()


def test_gradient_saliency_grad_input_reduce():
    n, d, p = 15, 16, 3
    model = _make_stf(num_pathways=p, feat_dim=d, token_dim=d)
    feats = torch.randn(1, n, d)
    coords = torch.rand(1, n, 2)

    sal = gradient_saliency(model, feats, coords, pathway_idx=0, reduce="grad_input")
    assert sal.shape == (n,)


# ---------------------------------------------------------------------------
# remove_shared_component — PC1 removal correctness
# ---------------------------------------------------------------------------


def test_remove_shared_component_recovers_rank1_signal():
    """A rank-1-plus-small-noise matrix should have its shared factor almost
    fully removed, and the fitted loading should match the true (equal-
    magnitude) generating direction."""
    rng = np.random.default_rng(0)
    n, p = 200, 10
    loading = rng.choice([-1.0, 1.0], size=p) / np.sqrt(p)
    scores = rng.normal(size=n)
    rank1 = np.outer(scores, loading) * 5.0
    noise = rng.normal(size=(n, p)) * 0.1
    data = rank1 + noise

    residual, pc1_scores, pc1_loadings = remove_shared_component(data)

    cos_sim = abs(np.dot(pc1_loadings, loading))
    assert cos_sim > 0.95

    standardized = (data - data.mean(0)) / data.std(0)
    assert residual.var() < standardized.var() * 0.1
    assert pc1_scores.shape == (n,)


def test_remove_shared_component_applies_given_loadings():
    """Passing a pre-fit pc1_loadings must project onto that fixed
    direction rather than fitting a new one (doc §8: same axis removed from
    both ground truth and signal)."""
    rng = np.random.default_rng(1)
    n, p = 50, 8
    fixed_loading = rng.normal(size=p)
    fixed_loading /= np.linalg.norm(fixed_loading)

    data = rng.normal(size=(n, p))
    _, _, returned_loadings = remove_shared_component(data, pc1_loadings=fixed_loading)
    assert np.allclose(returned_loadings, fixed_loading)


def test_remove_shared_component_zero_variance_column_safe():
    """A constant column must not produce NaN/inf (division-by-zero guard)."""
    data = np.zeros((30, 4))
    data[:, 0] = 5.0  # constant column
    data[:, 1:] = np.random.default_rng(2).normal(size=(30, 3))

    residual, pc1_scores, pc1_loadings = remove_shared_component(data)
    assert np.isfinite(residual).all()
    assert np.isfinite(pc1_loadings).all()


# ---------------------------------------------------------------------------
# spatial_pattern_fidelity — perfect signal -> ~1, shuffled -> ~0
# ---------------------------------------------------------------------------


def test_spatial_pattern_fidelity_perfect_signal():
    rng = np.random.default_rng(3)
    n, p = 200, 10
    coords = rng.random(size=(n, 2))
    target = rng.normal(size=(n, p)) + np.outer(rng.normal(size=n), rng.normal(size=p))

    result = spatial_pattern_fidelity(target.copy(), target, coords, residual=True)
    assert result["pearson"] > 0.99
    assert result["spearman"] > 0.99
    assert result["per_pathway"]["pearson"].shape == (p,)


def test_spatial_pattern_fidelity_shuffled_signal_near_zero():
    rng = np.random.default_rng(4)
    n, p = 300, 10
    coords = rng.random(size=(n, 2))
    target = rng.normal(size=(n, p)) + np.outer(rng.normal(size=n), rng.normal(size=p))
    shuffled = target[rng.permutation(n)]

    result = spatial_pattern_fidelity(shuffled, target, coords, residual=True)
    assert abs(result["pearson"]) < 0.3
    assert abs(result["spearman"]) < 0.3


def test_spatial_pattern_fidelity_raw_vs_residual_differ():
    """Raw and residual fidelity must be computed independently — a
    regression test for a bug where the tertiary Moran's-I metric silently
    reused the raw maps regardless of the residual flag."""
    rng = np.random.default_rng(5)
    n, p = 150, 6
    coords = rng.random(size=(n, 2))
    shared = np.outer(rng.normal(size=n), rng.normal(size=p)) * 5.0
    target = shared + rng.normal(size=(n, p)) * 0.5
    signal = (
        shared + rng.normal(size=(n, p)) * 2.0
    )  # tracks the shared factor, not detail

    raw = spatial_pattern_fidelity(signal, target, coords, residual=False)
    residual = spatial_pattern_fidelity(signal, target, coords, residual=True)

    assert raw["pearson"] != residual["pearson"]
    assert raw["morans_i_agreement"] != residual["morans_i_agreement"]


# ---------------------------------------------------------------------------
# Signal/target pathway-ordering alignment (regression, doc §6/§8)
# ---------------------------------------------------------------------------


def test_eval_pathway_indices_align_signal_and_target(tmp_path):
    """Regression: `scripts/evaluate_spatial_attribution.py` derives
    `pathway_indices` from the MODEL's output ordering, so the ground-truth
    target tensor must be loaded in that same ordering.

    Loading targets in raw .h5 file order instead silently grades each
    signal column against a *different* pathway's ground truth whenever a
    run was trained on a pathway subset (model order != file order). This
    asserts the dataset honours a requested pathway ordering, which is the
    invariant the fix relies on.
    """
    import h5py
    import numpy as np

    from spatial_transcript_former.recipes.hest.compute_pathway_activities import (
        PATHWAY_FILE_VERSION,
        load_pathway_activities,
    )

    file_order = ["P_A", "P_B", "P_C", "P_D"]
    barcodes = [b"bc0", b"bc1", b"bc2"]
    # Column j is filled with the constant j, so a column's identity is
    # recoverable from its values alone.
    acts = np.tile(np.arange(len(file_order), dtype=np.float32), (len(barcodes), 1))

    h5_path = tmp_path / "sample.h5"
    with h5py.File(h5_path, "w") as f:
        f.attrs["format_version"] = PATHWAY_FILE_VERSION
        f.create_dataset("activities", data=acts)
        f.create_dataset("barcodes", data=np.array(barcodes))
        f.create_dataset(
            "pathway_names", data=np.array([n.encode() for n in file_order])
        )

    loaded, names, _, _ = load_pathway_activities(str(h5_path), list(barcodes))
    assert names == file_order

    # A model trained on this subset outputs columns in THIS order, which is
    # deliberately not the file order.
    model_order = ["P_C", "P_A"]
    indices = [names.index(n) for n in model_order]
    reordered = loaded[:, indices]

    # Each selected column must carry its own pathway's values (C -> 2, A -> 0).
    assert np.allclose(reordered[:, 0], 2.0), "P_C column lost its identity"
    assert np.allclose(reordered[:, 1], 0.0), "P_A column lost its identity"

    # And naive positional indexing (the bug) must be demonstrably different,
    # so this test fails loudly if the two paths are ever conflated again.
    naive = loaded[:, [0, 1]]
    assert not np.allclose(naive, reordered)


# ---------------------------------------------------------------------------
# regress_out_covariate — the depth-confound remover (doc §2b, §4)
# ---------------------------------------------------------------------------


def test_regress_out_covariate_zeroes_the_confound():
    """Projecting out a measured covariate must leave exactly zero residual
    correlation with it — the property PC1 removal only approximates."""
    from spatial_transcript_former.attribution import regress_out_covariate

    rng = np.random.default_rng(7)
    n, p = 300, 12
    depth = rng.normal(size=n)
    # every pathway carries a depth component plus its own signal
    maps = np.outer(depth, rng.normal(size=p) + 2.0) + rng.normal(size=(n, p)) * 0.5

    residual, beta = regress_out_covariate(maps, depth)

    for j in range(p):
        if residual[:, j].std() > 1e-12:
            assert abs(np.corrcoef(residual[:, j], depth)[0, 1]) < 1e-9
    assert beta.shape == (p,)


def test_regress_out_covariate_preserves_orthogonal_signal():
    """Signal orthogonal to the covariate must survive untouched."""
    from spatial_transcript_former.attribution import regress_out_covariate

    rng = np.random.default_rng(8)
    n = 400
    depth = rng.normal(size=n)
    orthogonal = rng.normal(size=n)
    orthogonal -= orthogonal.dot(depth) / depth.dot(depth) * depth  # exactly orthogonal

    maps = np.stack([orthogonal, depth, orthogonal + depth], axis=1)
    residual, _ = regress_out_covariate(maps, depth)

    # column 0 is pure orthogonal signal -> unchanged up to standardisation
    assert abs(np.corrcoef(residual[:, 0], orthogonal)[0, 1]) > 0.999
    # column 1 is pure depth -> annihilated
    assert residual[:, 1].std() < 1e-9


def test_regress_out_covariate_constant_covariate_is_safe():
    """A constant covariate carries no information and must not divide by zero."""
    from spatial_transcript_former.attribution import regress_out_covariate

    maps = np.random.default_rng(9).normal(size=(50, 4))
    residual, beta = regress_out_covariate(maps, np.full(50, 3.0))
    assert np.isfinite(residual).all()
    assert np.allclose(beta, 0.0)


def test_regress_out_covariate_beats_pc1_on_depth_removal():
    """Regression removes a measured confound more completely than PC1 removal.

    This is the empirical claim behind doc §2b, reduced to a deterministic
    synthetic case: PC1 is only an approximation of the depth direction, so it
    leaves residual correlation that explicit regression does not.
    """
    from spatial_transcript_former.attribution import regress_out_covariate

    rng = np.random.default_rng(10)
    n, p = 500, 20
    depth = rng.normal(size=n)
    loading = rng.uniform(0.5, 2.0, size=p)  # uneven -> PC1 != depth axis
    maps = np.outer(depth, loading) + rng.normal(size=(n, p)) * 0.8

    res_reg, _ = regress_out_covariate(maps, depth)
    res_pc1, _, _ = remove_shared_component(maps)

    def leak(m):
        return np.nanmean(
            [
                abs(np.corrcoef(m[:, j], depth)[0, 1])
                for j in range(p)
                if m[:, j].std() > 1e-12
            ]
        )

    assert leak(res_reg) < 1e-9
    assert leak(res_reg) < leak(res_pc1)


def test_normalize_features_is_applied_after_qc_filtering(tmp_path):
    """Regression: per-slide feature normalisation must survive the QC rebuild.

    The QC-validity filter rebuilds ``self.features`` partway through
    ``_load_data``. Normalising before that point is silently discarded --
    the flag appears to work, the stats are unchanged, and the intended batch
    correction never happens. This asserts the ordering.
    """
    import inspect

    from spatial_transcript_former.recipes.hest import dataset as ds_mod

    src = inspect.getsource(ds_mod.HEST_FeatureDataset._load_data)
    norm_at = src.find("if self.normalize_features")
    rebuild_at = src.rfind("self.features = features[mask_bool]")
    assert norm_at != -1, "normalisation block missing"
    assert rebuild_at != -1, "QC rebuild missing"
    assert norm_at > rebuild_at, (
        "feature normalisation runs before the QC rebuild of self.features, "
        "so it will be silently overwritten"
    )
