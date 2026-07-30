"""
Tests for patient-wise dataset splitting and data leakage prevention.
"""

import os
import tempfile
import sys
from unittest.mock import patch

import pandas as pd
import pytest

from spatial_transcript_former.recipes.hest.splitting import split_hest_patients
from spatial_transcript_former.recipes.hest.splitting import split_hest_patients, main
from spatial_transcript_former.recipes.hest.splitting import (
    split_hest_by_dataset,
    iter_loso_folds,
    split_colonmap_vs_rest,
    DEFAULT_COLONMAP_TITLE,
)

# ---------------------------------------------------------------------------
# Core Logic
# ---------------------------------------------------------------------------


def test_split_hest_patients():
    # Handle path to data
    # Assuming running from root
    metadata_path = r"A:\hest_data\HEST_v1_3_0.csv"
    if not os.path.exists(metadata_path):
        metadata_path = os.path.join("hest_data", "HEST_v1_3_0.csv")

    if not os.path.exists(metadata_path):
        pytest.skip("Metadata file not found, skipping test.")

    train_ids, val_ids, test_ids = split_hest_patients(metadata_path)

    assert len(train_ids) > 0
    assert len(val_ids) > 0
    # Patient leakage check is already done inside split_hest_patients


# ---------------------------------------------------------------------------
# Robustness
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_metadata():
    """Create a temporary metadata CSV with known patient structure."""
    data = {
        "id": ["S1", "S2", "S3", "S4", "S5", "S6"],
        "patient": ["P1", "P1", "P2", "P2", "P3", None],  # S6 has no patient
    }
    df = pd.DataFrame(data)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        df.to_csv(f.name, index=False)
        return f.name


def test_split_hest_patients_isolation(mock_metadata):
    """Verify that patients are strictly isolated."""
    # With 3 patients (+ 1 unique fallback), one patient in val is 25%
    train, val, test = split_hest_patients(mock_metadata, val_ratio=0.25, seed=42)

    # Check that no sample is in both
    assert set(train).isdisjoint(set(val))

    # Map back to patients
    df = pd.read_csv(mock_metadata)
    df["patient_filled"] = df["patient"].fillna(df["id"])

    train_patients = set(df[df["id"].isin(train)]["patient_filled"])
    val_patients = set(df[df["id"].isin(val)]["patient_filled"])

    # Critical check: No patient overlap
    assert train_patients.isdisjoint(val_patients)

    # Cleanup
    os.remove(mock_metadata)


def test_split_hest_patients_missing_id_fallback():
    """Verify that samples with missing patient IDs are treated as unique."""
    data = {"id": ["S1", "S2", "S3"], "patient": [None, None, None]}
    df = pd.DataFrame(data)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        df.to_csv(f.name, index=False)
        path = f.name

    # With 3 unique "patients", split. Since test_size=0.34, 1 should be in val.
    train, val, test = split_hest_patients(path, val_ratio=0.34, seed=42)
    # Ensure total is 3 and val/train are not empty (since 0.34 * 3 = 1.02)
    assert len(train) + len(val) == 3
    assert len(val) >= 1
    assert len(train) >= 1

    os.remove(path)


def test_splitting_main_cli(mock_metadata):
    """Verify that the CLI main function runs without error and respects args."""
    test_args = ["prog", mock_metadata, "--val_ratio", "0.5", "--seed", "123"]
    with patch.object(sys, "argv", test_args):
        # Should not raise exception
        main()
    os.remove(mock_metadata)


# ---------------------------------------------------------------------------
# Dataset-grouped splitting (LOSO / COLON-MAP-vs-rest, doc §3)
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_dataset_metadata():
    """Mock metadata with 3 studies of uneven size, plus rows that should be
    filtered out by organ/technology/species (mirroring the real corpus's
    need to exclude Xenium/Visium HD and non-human samples, doc §2/§3)."""
    data = {
        "id": ["S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8", "X1", "M1"],
        "dataset_title": [
            "COLON MAP: Colon Molecular Atlas Project",
            "COLON MAP: Colon Molecular Atlas Project",
            "COLON MAP: Colon Molecular Atlas Project",
            "Study B",
            "Study B",
            "Study C",
            "Study C",
            "Study C",
            "Study B",  # Xenium row within Study B — should be excluded by technology filter
            "Study C",  # mouse row within Study C — should be excluded by species filter
        ],
        "organ": ["Bowel"] * 8 + ["Bowel", "Bowel"],
        "st_technology": ["Visium"] * 8 + ["Xenium", "Visium"],
        "species": ["Homo sapiens"] * 8 + ["Homo sapiens", "Mus musculus"],
    }
    df = pd.DataFrame(data)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        df.to_csv(f.name, index=False)
        return f.name


def test_split_hest_by_dataset_isolation(mock_dataset_metadata):
    """Holding out one study must put every one of its samples (and only
    its samples) in val, with no overlap with train."""
    train, val = split_hest_by_dataset(
        mock_dataset_metadata,
        held_out="COLON MAP: Colon Molecular Atlas Project",
        organ="Bowel",
        technology="Visium",
    )
    assert set(val) == {"S1", "S2", "S3"}
    assert set(train) == {"S4", "S5", "S6", "S7", "S8"}
    assert set(train).isdisjoint(set(val))
    os.remove(mock_dataset_metadata)


def test_split_hest_by_dataset_filters_technology_and_species(mock_dataset_metadata):
    """Xenium and non-human rows must be excluded even when their
    dataset_title matches a study included in the split."""
    train, val = split_hest_by_dataset(
        mock_dataset_metadata, held_out="Study B", organ="Bowel", technology="Visium"
    )
    all_ids = set(train) | set(val)
    assert "X1" not in all_ids  # Xenium
    assert "M1" not in all_ids  # mouse
    os.remove(mock_dataset_metadata)


def test_split_hest_by_dataset_unknown_study_raises(mock_dataset_metadata):
    with pytest.raises(ValueError, match="Unknown dataset_title"):
        split_hest_by_dataset(
            mock_dataset_metadata,
            held_out="Not A Real Study",
            organ="Bowel",
            technology="Visium",
        )
    os.remove(mock_dataset_metadata)


def test_split_hest_by_dataset_multiple_held_out(mock_dataset_metadata):
    """held_out accepts a list, for the COLON-MAP-vs-rest 'reverse' direction
    style usage (hold out several studies at once)."""
    train, val = split_hest_by_dataset(
        mock_dataset_metadata,
        held_out=["Study B", "Study C"],
        organ="Bowel",
        technology="Visium",
    )
    assert set(val) == {"S4", "S5", "S6", "S7", "S8"}
    assert set(train) == {"S1", "S2", "S3"}
    os.remove(mock_dataset_metadata)


def test_iter_loso_folds_no_leakage_and_full_coverage(mock_dataset_metadata):
    """Every LOSO fold must have disjoint train/val, and every fold's
    train+val together must cover the full (filtered) corpus."""
    folds = list(
        iter_loso_folds(mock_dataset_metadata, organ="Bowel", technology="Visium")
    )
    studies = {f[0] for f in folds}
    assert studies == {
        "COLON MAP: Colon Molecular Atlas Project",
        "Study B",
        "Study C",
    }
    all_filtered_ids = {"S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8"}
    for study, train, val in folds:
        assert set(train).isdisjoint(set(val)), f"leakage in fold {study!r}"
        assert set(train) | set(val) == all_filtered_ids
    os.remove(mock_dataset_metadata)


def test_split_colonmap_vs_rest_forward_and_reverse(mock_dataset_metadata):
    train, val = split_colonmap_vs_rest(
        mock_dataset_metadata, organ="Bowel", technology="Visium"
    )
    assert set(val) == {"S1", "S2", "S3"}

    train_r, val_r = split_colonmap_vs_rest(
        mock_dataset_metadata, organ="Bowel", technology="Visium", reverse=True
    )
    assert train_r == val and val_r == train
    os.remove(mock_dataset_metadata)


def test_split_hest_by_dataset_real_metadata():
    """Smoke test against the real HEST metadata, matching the exact counts
    audited in docs/EXPERIMENT_SPATIAL_ATTRIBUTION.md §3 (73 Bowel/Visium/
    human slides, 10 studies, COLON MAP = 41)."""
    metadata_path = r"A:\hest_data\HEST_v1_3_0.csv"
    if not os.path.exists(metadata_path):
        pytest.skip("Metadata file not found, skipping test.")

    train, val = split_colonmap_vs_rest(
        metadata_path, organ="Bowel", technology="Visium"
    )
    assert len(train) == 32
    assert len(val) == 41

    folds = list(iter_loso_folds(metadata_path, organ="Bowel", technology="Visium"))
    assert len(folds) == 10
    assert sum(len(v) for _, _, v in folds) == 73
    for study, tr, va in folds:
        assert set(tr).isdisjoint(set(va))
