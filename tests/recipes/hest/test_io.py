"""
Merged tests: test_io.py, test_download.py, test_download_script.py
"""

import os
from unittest.mock import patch, MagicMock
import unittest
from unittest.mock import patch, MagicMock, call
import sys
import tempfile

import h5py
import numpy as np
import pytest
import pandas as pd
import shutil
import io
import sys
import os
sys.path.append(os.path.abspath("."))
sys.path.append(os.path.abspath("scripts"))
from download_hest import main

from spatial_transcript_former.recipes.hest.io import (
    get_hest_data_dir,
    decode_h5_string,
    load_h5ad_metadata,
    get_image_from_h5ad,
)
from spatial_transcript_former.recipes.hest.download import (
    download_metadata,
    filter_samples,
    download_hest_subset,
)


# --- From test_io.py ---


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

# --- From test_download.py ---


class TestDownload(unittest.TestCase):

    def setUp(self):
        # Create a temporary directory for testing file operations
        self.test_dir = tempfile.mkdtemp()
        self.metadata_path = os.path.join(self.test_dir, "HEST_v1_3_0.csv")

    def tearDown(self):
        # Clean up the temporary directory
        shutil.rmtree(self.test_dir)

    @patch("spatial_transcript_former.recipes.hest.download.hf_hub_download")
    @patch("os.path.exists")
    def test_download_metadata_exists(self, mock_exists, mock_download):
        # Test case where metadata already exists
        mock_exists.return_value = True

        result = download_metadata(self.test_dir)

        self.assertEqual(result, self.metadata_path)
        mock_download.assert_not_called()

    @patch("spatial_transcript_former.recipes.hest.download.hf_hub_download")
    @patch("os.path.exists")
    def test_download_metadata_missing(self, mock_exists, mock_download):
        # Test case where metadata is missing and needs download
        mock_exists.return_value = False
        mock_download.return_value = self.metadata_path

        result = download_metadata(self.test_dir)

        self.assertEqual(result, self.metadata_path)
        mock_download.assert_called_once()

    def test_filter_samples(self):
        # Create a mock CSV file
        data = {
            "id": ["S1", "S2", "S3", "S4"],
            "organ": ["Bowel", "Kindey", "Bowel", "Lung"],
            "disease_state": ["Cancer", "Cancer", "Healthy", "Cancer"],
            "st_technology": ["Visium", "Visium", "Visium", "Xenium"],
        }
        df = pd.DataFrame(data)
        df.to_csv(self.metadata_path, index=False)

        # Test filtering by organ
        samples = filter_samples(self.metadata_path, organ="Bowel")
        self.assertEqual(sorted(samples), ["S1", "S3"])

        # Test filtering by disease
        samples = filter_samples(self.metadata_path, disease_state="Cancer")
        self.assertEqual(sorted(samples), ["S1", "S2", "S4"])

        # Test filtering by multiple criteria
        samples = filter_samples(
            self.metadata_path, organ="Bowel", disease_state="Cancer"
        )
        self.assertEqual(samples, ["S1"])

        # Test no match
        samples = filter_samples(self.metadata_path, organ="Brain")
        self.assertEqual(samples, [])

    @patch("spatial_transcript_former.recipes.hest.download.snapshot_download")
    def test_download_hest_subset_calls(self, mock_snapshot):
        # Test that snapshot_download is called with correct patterns
        sample_ids = ["S1", "S2"]
        additional_patterns = ["extra_file.txt"]

        download_hest_subset(sample_ids, self.test_dir, additional_patterns)

        mock_snapshot.assert_called_once()
        call_args = mock_snapshot.call_args
        _, kwargs = call_args

        self.assertEqual(kwargs["repo_id"], "MahmoodLab/hest")
        self.assertEqual(kwargs["local_dir"], self.test_dir)

        patterns = kwargs["allow_patterns"]
        # Check standard recursive patterns
        self.assertIn("**/S1.*", patterns)
        self.assertIn("**/S1_*", patterns)
        self.assertIn("**/S2.*", patterns)
        self.assertIn("README.md", patterns)
        # Check additional patterns
        self.assertIn("extra_file.txt", patterns)

    @patch("spatial_transcript_former.recipes.hest.download.snapshot_download")
    @patch("zipfile.ZipFile")
    @patch("os.listdir")
    @patch("os.path.exists")
    def test_unzip_logic(self, mock_exists, mock_listdir, mock_zipfile, mock_snapshot):
        # Simulate zip files existing in cellvit_seg
        mock_exists.side_effect = (
            lambda p: p.endswith("cellvit_seg")
            or p.endswith("xenium_seg")
            or p.endswith("tissue_seg")
        )
        mock_listdir.return_value = ["file.zip", "other.txt"]

        # Mock the zip file context manager
        mock_zip_instance = MagicMock()
        mock_zipfile.return_value.__enter__.return_value = mock_zip_instance

        download_hest_subset(["S1"], self.test_dir)

        # Check that ZipFile was called for file.zip in checked directories
        # download_hest_subset iterates over ['cellvit_seg', 'xenium_seg', 'tissue_seg']
        # mock_exists returns True for all of them.
        # mock_listdir returns ['file.zip', 'other.txt'] for all of them.
        # So we expect 3 calls (one per directory).

        self.assertEqual(mock_zipfile.call_count, 3)
        mock_zip_instance.extractall.assert_called()

# --- From test_download_script.py ---


@pytest.fixture
def mock_metadata():
    csv_content = """id,organ,disease_state,st_technology,species,preservation_method
TENX1,Breast,Cancer,Visium,Homo sapiens,FFPE
TENX2,Breast,Healthy,Visium,Homo sapiens,FFPE
TENX3,Bowel,Cancer,Visium,Homo sapiens,Fresh Frozen
TENX4,Bowel,Cancer,Xenium,Homo sapiens,FFPE
TENX5,Lung,Cancer,Visium,Mus musculus,FFPE
"""
    return pd.read_csv(io.StringIO(csv_content))


@patch("download_hest.download_metadata")
@patch("download_hest.download_hest_subset")
@patch("pandas.read_csv")
def test_download_filter_organ(
    mock_read_csv, mock_subset, mock_metadata_download, mock_metadata
):
    mock_read_csv.return_value = mock_metadata
    mock_metadata_download.return_value = "mock.csv"

    # Simulate: python scripts/download_hest.py --organ Bowel --dry-run
    test_args = ["scripts/download_hest.py", "--organ", "Bowel", "--dry-run"]
    with patch.object(sys, "argv", test_args):
        main()

    # Filtered should be TENX3 and TENX4
    # We can't easily check prints here without more setup, so we check if subset would be called if dry_run was off
    # Let's test non-dry-run but mock the download part

    test_args = ["scripts/download_hest.py", "--organ", "Bowel", "--yes"]
    with patch.object(sys, "argv", test_args):
        main()

    # Capture the ids passed to download_hest_subset
    args, kwargs = mock_subset.call_args
    downloaded_ids = args[0]

    assert set(downloaded_ids) == {"TENX3", "TENX4"}


@patch("download_hest.download_metadata")
@patch("download_hest.download_hest_subset")
@patch("pandas.read_csv")
def test_download_filter_multi(
    mock_read_csv, mock_subset, mock_metadata_download, mock_metadata
):
    mock_read_csv.return_value = mock_metadata
    mock_metadata_download.return_value = "mock.csv"

    # Filter by organ AND technology
    test_args = [
        "scripts/download_hest.py",
        "--organ",
        "Bowel",
        "--tech",
        "Xenium",
        "--yes",
    ]
    with patch.object(sys, "argv", test_args):
        main()

    args, _ = mock_subset.call_args
    assert args[0] == ["TENX4"]


@patch("download_hest.download_metadata")
@patch("download_hest.download_hest_subset")
@patch("pandas.read_csv")
def test_download_limit(
    mock_read_csv, mock_subset, mock_metadata_download, mock_metadata
):
    mock_read_csv.return_value = mock_metadata
    mock_metadata_download.return_value = "mock.csv"

    # Filter by breast and limit to 1
    test_args = [
        "scripts/download_hest.py",
        "--organ",
        "Breast",
        "--limit",
        "1",
        "--yes",
    ]
    with patch.object(sys, "argv", test_args):
        main()

    args, _ = mock_subset.call_args
    assert len(args[0]) == 1
    assert args[0][0] == "TENX1"


@patch("download_hest.download_metadata")
@patch("pandas.read_csv")
def test_list_options(mock_read_csv, mock_metadata_download, mock_metadata, capsys):
    mock_read_csv.return_value = mock_metadata
    mock_metadata_download.return_value = "mock.csv"

    test_args = ["scripts/download_hest.py", "--list-options"]
    with patch.object(sys, "argv", test_args):
        main()

    captured = capsys.readouterr()
    assert "ORGAN (3 options)" in captured.out
    assert "Bowel" in captured.out
    assert "Breast" in captured.out
    assert "Lung" in captured.out
    assert "Homo sapiens" in captured.out


@patch("download_hest.download_metadata")
@patch("download_hest.download_hest_subset")
@patch("pandas.read_csv")
def test_refresh_metadata(
    mock_read_csv, mock_subset, mock_metadata_download, mock_metadata
):
    mock_read_csv.return_value = mock_metadata
    mock_metadata_download.return_value = "mock.csv"

    test_args = ["scripts/download_hest.py", "--refresh-metadata", "--dry-run"]
    with patch.object(sys, "argv", test_args):
        main()

    # Verify that download_metadata was called with force=True
    _, kwargs = mock_metadata_download.call_args
    assert kwargs["force"] is True

    # Verify it didn't call subset download (since no filters provided)
    assert mock_subset.call_count == 0


@patch("download_hest.download_metadata")
@patch("download_hest.download_hest_subset")
@patch("pandas.read_csv")
def test_download_all(
    mock_read_csv, mock_subset, mock_metadata_download, mock_metadata
):
    mock_read_csv.return_value = mock_metadata
    mock_metadata_download.return_value = "mock.csv"

    test_args = ["scripts/download_hest.py", "--all", "--yes"]
    with patch.object(sys, "argv", test_args):
        main()

    args, _ = mock_subset.call_args
    assert len(args[0]) == len(mock_metadata)
