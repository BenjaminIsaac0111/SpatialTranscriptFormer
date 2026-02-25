import pandas as pd
import pytest
import os
import tempfile
from spatial_transcript_former.data.splitting import split_hest_patients, main
import sys
from unittest.mock import patch


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


if __name__ == "__main__":
    pytest.main([__file__])
