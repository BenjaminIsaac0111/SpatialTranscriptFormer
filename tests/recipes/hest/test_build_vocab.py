"""
Merged tests: test_build_vocab.py
"""

import os
from unittest.mock import patch

import pytest

from spatial_transcript_former.recipes.hest.build_vocab import scan_h5ad_files


# --- From test_build_vocab.py ---


def test_scan_h5ad_files_success(tmp_path):
    # Set up mock st directory
    data_dir = tmp_path / "mock_data"
    st_dir = data_dir / "st"
    st_dir.mkdir(parents=True)

    # Create mock h5ad files
    (st_dir / "sample1.h5ad").touch()
    (st_dir / "sample2.h5ad").touch()
    (st_dir / "not_an_h5ad.txt").touch()

    # Test scanning
    sample_ids = scan_h5ad_files(str(data_dir))

    assert len(sample_ids) == 2
    assert "sample1" in sample_ids
    assert "sample2" in sample_ids
    assert "not_an_h5ad.txt" not in sample_ids


def test_scan_h5ad_files_missing_dir(tmp_path, capsys):
    # Directory does not exist
    data_dir = tmp_path / "empty_dir"

    sample_ids = scan_h5ad_files(str(data_dir))

    # Should return empty and print warning
    assert len(sample_ids) == 0
    captured = capsys.readouterr()
    assert "Directory not found:" in captured.out
    assert "DATA_FORMAT.md" in captured.out
