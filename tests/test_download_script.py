import pytest
import pandas as pd
import io
from unittest.mock import patch, MagicMock
import sys
import os

# Add project root and scripts to path
sys.path.append(os.path.abspath("."))
sys.path.append(os.path.abspath("scripts"))

from download_hest import main


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
