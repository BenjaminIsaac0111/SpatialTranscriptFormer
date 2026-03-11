import pandas as pd
import os
import pytest
from spatial_transcript_former.recipes.hest.splitting import split_hest_patients


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


if __name__ == "__main__":
    test_split_hest_patients()
