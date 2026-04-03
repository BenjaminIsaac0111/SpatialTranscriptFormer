"""
Tests for the public Python API surface.

Covers:
    - Package-level imports (__init__.py)
    - Config serialization (save_pretrained / load_pretrained round-trip)
    - from_pretrained class method
    - Predictor (patch and WSI mode)
    - FeatureExtractor
    - inject_predictions (AnnData integration)
"""

import json
import os
import tempfile

import numpy as np
import pytest
import torch

from spatial_transcript_former import (
    SpatialTranscriptFormer,
    Predictor,
    FeatureExtractor,
    save_pretrained,
    load_pretrained,
    inject_predictions,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def small_model():
    """A minimal SpatialTranscriptFormer for fast tests."""
    return SpatialTranscriptFormer(
        num_pathways=10,
        backbone_name="resnet50",
        pretrained=False,
        token_dim=64,
        n_heads=4,
        n_layers=2,
        use_spatial_pe=True,
    )


@pytest.fixture
def checkpoint_dir(small_model):
    """Save a small model to a temp directory and return the path."""
    with tempfile.TemporaryDirectory() as tmpdir:
        pathway_names = [f"PATHWAY_{i}" for i in range(10)]
        save_pretrained(small_model, tmpdir, pathway_names=pathway_names)
        yield tmpdir


# ---------------------------------------------------------------------------
# Package imports
# ---------------------------------------------------------------------------


class TestPackageImports:
    def test_model_importable(self):
        from spatial_transcript_former import SpatialTranscriptFormer

        assert SpatialTranscriptFormer is not None

    def test_predictor_importable(self):
        from spatial_transcript_former import Predictor

        assert Predictor is not None

    def test_feature_extractor_importable(self):
        from spatial_transcript_former import FeatureExtractor

        assert FeatureExtractor is not None

    def test_checkpoint_functions_importable(self):
        from spatial_transcript_former import save_pretrained, load_pretrained

        assert callable(save_pretrained)
        assert callable(load_pretrained)

    def test_inject_predictions_importable(self):
        from spatial_transcript_former import inject_predictions

        assert callable(inject_predictions)


# ---------------------------------------------------------------------------
# Config serialization round-trip
# ---------------------------------------------------------------------------


class TestCheckpointSerialization:
    def test_save_creates_files(self, small_model):
        """save_pretrained should create config.json and model.pth."""
        with tempfile.TemporaryDirectory() as tmpdir:
            save_pretrained(small_model, tmpdir)
            assert os.path.isfile(os.path.join(tmpdir, "config.json"))
            assert os.path.isfile(os.path.join(tmpdir, "model.pth"))

    def test_save_with_pathway_names(self, small_model):
        """save_pretrained should create pathway_names.json when provided."""
        with tempfile.TemporaryDirectory() as tmpdir:
            names = [f"G{i}" for i in range(10)]
            save_pretrained(small_model, tmpdir, pathway_names=names)
            path = os.path.join(tmpdir, "pathway_names.json")
            assert os.path.isfile(path)
            with open(path) as f:
                loaded = json.load(f)
            assert loaded == names

    def test_save_pathway_names_length_mismatch(self, small_model):
        """Mismatched pathway_names length should raise ValueError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(ValueError, match="pathway_names length"):
                save_pretrained(small_model, tmpdir, pathway_names=["A", "B"])

    def test_config_json_contents(self, small_model):
        """config.json should contain all expected architecture keys."""
        with tempfile.TemporaryDirectory() as tmpdir:
            save_pretrained(small_model, tmpdir)
            with open(os.path.join(tmpdir, "config.json")) as f:
                config = json.load(f)
            assert config["num_pathways"] == 10
            assert config["token_dim"] == 64
            assert config["n_heads"] == 4
            assert config["n_layers"] == 2
            assert config["use_spatial_pe"] is True

    def test_round_trip_weights(self, small_model, checkpoint_dir):
        """Weights should be identical after save → load."""
        loaded = load_pretrained(checkpoint_dir, device="cpu")
        for (n1, p1), (n2, p2) in zip(
            small_model.named_parameters(), loaded.named_parameters()
        ):
            assert n1 == n2, f"Parameter name mismatch: {n1} vs {n2}"
            assert torch.allclose(p1, p2), f"Weight mismatch in {n1}"

    def test_round_trip_pathway_names(self, checkpoint_dir):
        """gene_names should survive the round trip."""
        model = load_pretrained(checkpoint_dir)
        assert model.pathway_names is not None
        assert len(model.pathway_names) == 10
        assert model.pathway_names[0] == "PATHWAY_0"

    def test_load_missing_config_raises(self):
        """Loading from empty directory should raise FileNotFoundError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(FileNotFoundError, match="config.json"):
                load_pretrained(tmpdir)


# ---------------------------------------------------------------------------
# from_pretrained
# ---------------------------------------------------------------------------


class TestFromPretrained:
    def test_from_pretrained_returns_model(self, checkpoint_dir):
        """from_pretrained should return a SpatialTranscriptFormer in eval mode."""
        model = SpatialTranscriptFormer.from_pretrained(checkpoint_dir)
        assert isinstance(model, SpatialTranscriptFormer)
        assert not model.training  # should be in eval mode

    def test_from_pretrained_with_overrides(self, checkpoint_dir):
        """Overriding dropout should be reflected in loaded model."""
        model = SpatialTranscriptFormer.from_pretrained(checkpoint_dir, dropout=0.0)
        # The first transformer layer should use the override
        layer = model.fusion_engine.layers[0]
        # dropout is an attribute of the layer
        assert layer.dropout.p == 0.0


# ---------------------------------------------------------------------------
# Predictor
# ---------------------------------------------------------------------------


class TestPredictor:
    def test_predict_patch(self, small_model):
        """predict_patch should return (1, G) tensor."""
        predictor = Predictor(small_model, device="cpu")
        image = torch.randn(1, 3, 224, 224)
        result = predictor.predict_patch(image)
        assert result.shape == (1, 10)

    def test_predict_patch_no_batch_dim(self, small_model):
        """predict_patch should accept (3, H, W) without batch dim."""
        predictor = Predictor(small_model, device="cpu")
        image = torch.randn(3, 224, 224)
        result = predictor.predict_patch(image)
        assert result.shape == (1, 10)

    def test_predict_wsi(self, small_model):
        """predict_wsi should return (1, G) tensor for global mode."""
        predictor = Predictor(small_model, device="cpu")
        features = torch.randn(20, small_model.image_proj.in_features)
        coords = torch.randn(20, 2)
        result = predictor.predict_wsi(features, coords)
        assert result.shape == (1, 10)

    def test_predict_wsi_dense(self, small_model):
        """predict_wsi with return_dense should return (1, N, G)."""
        predictor = Predictor(small_model, device="cpu")
        n_patches = 15
        features = torch.randn(n_patches, small_model.image_proj.in_features)
        coords = torch.randn(n_patches, 2)
        result = predictor.predict_wsi(features, coords, return_dense=True)
        assert result.shape == (1, n_patches, 10)

    def test_predict_wsi_feature_dim_mismatch(self, small_model):
        """Wrong feature dim should raise ValueError with helpful message."""
        predictor = Predictor(small_model, device="cpu")
        features = torch.randn(10, 999)  # wrong dim
        coords = torch.randn(10, 2)
        with pytest.raises(ValueError, match="Feature dimension mismatch"):
            predictor.predict_wsi(features, coords)

    def test_predict_unified_dispatch_image(self, small_model):
        """predict() should dispatch to patch mode for 4D image input."""
        predictor = Predictor(small_model, device="cpu")
        image = torch.randn(1, 3, 224, 224)
        result = predictor.predict(image)
        assert result.shape == (1, 10)

    def test_predict_unified_dispatch_features(self, small_model):
        """predict() should dispatch to WSI mode for 2D features."""
        predictor = Predictor(small_model, device="cpu")
        features = torch.randn(10, small_model.image_proj.in_features)
        coords = torch.randn(10, 2)
        result = predictor.predict(features, coords)
        assert result.shape == (1, 10)

    def test_predict_features_without_coords_raises(self, small_model):
        """predict() on features without coords should raise."""
        predictor = Predictor(small_model, device="cpu")
        features = torch.randn(10, small_model.image_proj.in_features)
        with pytest.raises(ValueError, match="coords are required"):
            predictor.predict(features)

    def test_pathway_names_exposed(self, checkpoint_dir):
        """Predictor should expose gene_names from the model."""
        model = SpatialTranscriptFormer.from_pretrained(checkpoint_dir)
        predictor = Predictor(model)
        assert predictor.pathway_names is not None
        assert len(predictor.pathway_names) == 10


# ---------------------------------------------------------------------------
# inject_predictions (AnnData)
# ---------------------------------------------------------------------------


class TestInjectPredictions:
    def test_basic_injection(self):
        """Should set adata.X and adata.obsm['spatial']."""
        anndata = pytest.importorskip("anndata")
        import pandas as pd

        n, g = 100, 50
        adata = anndata.AnnData(obs=pd.DataFrame(index=[f"spot_{i}" for i in range(n)]))
        coords = np.random.rand(n, 2)
        predictions = np.random.rand(n, g).astype(np.float32)

        inject_predictions(adata, coords, predictions)

        assert adata.X is not None
        assert adata.X.shape == (n, g)
        np.testing.assert_array_equal(adata.obsm["spatial"], coords)

    def test_with_gene_names(self):
        """Gene names should populate adata.var_names."""
        anndata = pytest.importorskip("anndata")
        import pandas as pd

        n, g = 50, 20
        adata = anndata.AnnData(obs=pd.DataFrame(index=[f"s{i}" for i in range(n)]))
        gene_names = [f"GENE_{i}" for i in range(g)]
        inject_predictions(
            adata,
            np.zeros((n, 2)),
            np.zeros((n, g)),
            pathway_names=gene_names,
        )
        assert list(adata.var_names) == gene_names

    def test_with_pathway_scores(self):
        """Pathway scores should go into adata.obsm['spatial_pathways']."""
        anndata = pytest.importorskip("anndata")
        import pandas as pd

        n, g, p = 30, 10, 5
        adata = anndata.AnnData(obs=pd.DataFrame(index=[f"s{i}" for i in range(n)]))
        pathway_scores = np.random.rand(n, p).astype(np.float32)
        pathway_names = [f"PW_{i}" for i in range(p)]

        inject_predictions(adata, np.zeros((n, 2)), pathway_scores, pathway_names=pathway_names)
        assert adata.X.shape == (n, p)
        assert list(adata.var_names) == pathway_names

    def test_shape_mismatch_raises(self):
        """Mismatched row counts should raise ValueError."""
        anndata = pytest.importorskip("anndata")
        import pandas as pd

        adata = anndata.AnnData(obs=pd.DataFrame(index=[f"s{i}" for i in range(10)]))
        with pytest.raises(ValueError, match="coords has"):
            inject_predictions(adata, np.zeros((5, 2)), np.zeros((10, 20)))

    def test_torch_tensor_input(self):
        """Should accept torch tensors and convert them."""
        anndata = pytest.importorskip("anndata")
        import pandas as pd

        n, g = 20, 10
        adata = anndata.AnnData(obs=pd.DataFrame(index=[f"s{i}" for i in range(n)]))
        coords = torch.rand(n, 2)
        preds = torch.rand(n, g)
        inject_predictions(adata, coords, preds)
        assert adata.X.shape == (n, g)
