import os
import h5py
import numpy as np
import pytest
from unittest.mock import patch, MagicMock
from spatial_transcript_former.recipes.hest.io import (
    get_hest_data_dir,
    decode_h5_string,
    load_h5ad_metadata,
    get_image_from_h5ad,
)


def test_decode_h5_string():
    assert decode_h5_string(b"hello") == "hello"
    assert decode_h5_string("world") == "world"
    assert decode_h5_string(123) == "123"


@patch("spatial_transcript_former.recipes.hest.io.get_config")
@patch("os.path.exists")
def test_get_hest_data_dir_from_config(mock_exists, mock_get_config):
    # Mock config to return a specific path
    mock_get_config.return_value = ["/mock/data/dir"]
    # Mock exists to say the dir and a representative file exist
    mock_exists.side_effect = lambda p: p in [
        "/mock/data/dir",
        os.path.join("/mock/data/dir", "HEST_v1_3_0.csv"),
    ]

    assert get_hest_data_dir() == "/mock/data/dir"
    mock_get_config.assert_called_with("data_dirs", [])


@patch("spatial_transcript_former.recipes.hest.io.get_config")
@patch("os.path.exists")
def test_get_hest_data_dir_fallbacks(mock_exists, mock_get_config):
    mock_get_config.return_value = []
    # Mock exists to only return True for a fallback path
    fallback_path = "hest_data"
    mock_exists.side_effect = lambda p: p in [
        fallback_path,
        os.path.join(fallback_path, "st"),
    ]

    assert get_hest_data_dir() == fallback_path


@patch("spatial_transcript_former.recipes.hest.io.get_config")
@patch("os.path.exists")
def test_get_hest_data_dir_not_found(mock_exists, mock_get_config):
    mock_get_config.return_value = []
    mock_exists.return_value = False

    with pytest.raises(FileNotFoundError, match="Could not find HEST data directory"):
        get_hest_data_dir()


def test_load_h5ad_metadata(tmp_path):
    h5_path = str(tmp_path / "test.h5ad")
    with h5py.File(h5_path, "w") as f:
        # Create obs with index
        obs = f.create_group("obs")
        obs.create_dataset("_index", data=np.array([b"spot1", b"spot2"]))

        # Create var with index
        var = f.create_group("var")
        var.create_dataset("index", data=np.array([b"geneA", b"geneB"]))

        # Create uns/spatial
        uns = f.create_group("uns")
        spatial = uns.create_group("spatial")
        sample = spatial.create_group("sample1")

        imgs = sample.create_group("images")
        imgs.create_dataset("hires", data=np.zeros((10, 10, 3)))

        sfs = sample.create_group("scalefactors")
        sfs.create_dataset("tissue_hires_scalef", data=0.5)

    metadata = load_h5ad_metadata(h5_path)
    assert metadata["barcodes"] == ["spot1", "spot2"]
    assert metadata["gene_names"] == ["geneA", "geneB"]
    assert "spatial" in metadata
    assert metadata["spatial"]["sample_id"] == "sample1"
    assert "hires" in metadata["spatial"]["images"]
    assert metadata["spatial"]["images"]["hires"]["shape"] == (10, 10, 3)
    assert metadata["spatial"]["scalefactors"]["tissue_hires_scalef"] == 0.5


def test_get_image_from_h5ad(tmp_path):
    h5_path = str(tmp_path / "test_img.h5ad")
    img_data = np.random.rand(10, 10, 3).astype(np.float32)
    with h5py.File(h5_path, "w") as f:
        spatial = f.create_group("uns/spatial/sample1")
        imgs = spatial.create_group("images")
        imgs.create_dataset("lowres", data=img_data)

        sfs = spatial.create_group("scalefactors")
        sfs.create_dataset("tissue_lowres_scalef", data=0.1)

    img, scalef = get_image_from_h5ad(h5_path, img_type="lowres")
    assert np.allclose(img, img_data)
    assert scalef == 0.1

    # Test auto-selection if img_type is None
    img, scalef = get_image_from_h5ad(h5_path, img_type=None)
    assert np.allclose(img, img_data)
    assert scalef == 0.1
