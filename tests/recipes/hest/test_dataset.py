"""
Merged tests: test_dataset_logic.py, test_dataset_mocks.py, test_dataloader_h5ad.py, test_qc_filtering.py
"""

from unittest.mock import MagicMock, patch
import os

import torch
import numpy as np
import pytest

from spatial_transcript_former.recipes.hest.dataset import (
    apply_dihedral_augmentation,
    apply_dihedral_to_tensor,
    normalize_coordinates,
)
from spatial_transcript_former.recipes.hest.dataset import (
    HEST_Dataset,
    HEST_FeatureDataset,
)
from spatial_transcript_former.recipes.hest.dataset import get_hest_dataloader


# --- From test_dataset_logic.py ---


def test_apply_dihedral_augmentation_all_ops():
    """Verify all 8 dihedral operations against expected transformations."""
    # Unit square coordinates
    coords = torch.tensor([[1.0, 1.0]])

    # Expected results for (1, 1) under each op
    expected = {
        0: [1.0, 1.0],  # Identity
        1: [1.0, -1.0],  # 90 CCW: (x,y) -> (y,-x)
        2: [-1.0, -1.0],  # 180: (x,y) -> (-x,-y)
        3: [-1.0, 1.0],  # 270 CCW: (x,y) -> (-y,x)
        4: [-1.0, 1.0],  # Flip H: (-x,y)
        5: [1.0, -1.0],  # Flip V: (x,-y)
        6: [1.0, 1.0],  # Transpose: (y,x)
        7: [-1.0, -1.0],  # Anti-transpose: (-y,-x)
    }

    for op, exp in expected.items():
        out, _ = apply_dihedral_augmentation(coords, op=op)
        assert torch.allclose(out, torch.tensor([exp])), f"Failed op {op}"


def test_dihedral_composition_properties():
    """Verify mathematical properties of the D4 group."""
    coords = torch.randn(10, 2)

    # Flip H (4) twice is identity
    out, _ = apply_dihedral_augmentation(coords, op=4)
    out2, _ = apply_dihedral_augmentation(out, op=4)
    assert torch.allclose(out2, coords)

    # Rotate 90 (1) four times is identity
    curr = coords
    for _ in range(4):
        curr, _ = apply_dihedral_augmentation(curr, op=1)
    assert torch.allclose(curr, coords)

    # Transpose (6) is its own inverse
    out, _ = apply_dihedral_augmentation(coords, op=6)
    out2, _ = apply_dihedral_augmentation(out, op=6)
    assert torch.allclose(out2, coords)


def test_normalize_coordinates_boundaries():
    """Verify step_size thresholds (0.5 and 2.0)."""
    # Test step_size < 2.0 (Identity)
    # x_vals: [0, 1.9] -> step 1.9
    c1 = np.array([[0.0, 0.0], [1.9, 0.0]])
    assert np.allclose(normalize_coordinates(c1), c1)

    # Test step_size == 2.0 (Normalize)
    c2 = np.array([[0.0, 0.0], [2.0, 0.0]])
    assert np.allclose(normalize_coordinates(c2), [[0, 0], [1, 0]])

    # Test valid_steps filtering (steps <= 0.5 are ignored)
    # x_vals: [0, 0.5, 3.0] -> steps [0.5, 2.5].
    # valid_steps should only see 2.5
    c3 = np.array([[0.0, 0.0], [0.5, 0.0], [3.0, 0.0]])
    # step_size = 2.5. 0.5/2.5 = 0.2 -> rounds to 0. 3.0/2.5 = 1.2 -> rounds to 1
    assert np.allclose(normalize_coordinates(c3), [[0, 0], [0, 0], [1, 0]])

    # x_vals: [0, 0.51, 3.0] -> steps [0.51, 2.49].
    # step_size = 0.51. 0.51/0.51 = 1. 3.0/0.51 = 5.88 -> 6
    # But wait, step_size 0.51 < 2.0, so it remains identity
    c4 = np.array([[0.0, 0.0], [0.51, 0.0], [3.0, 0.0]])
    assert np.allclose(normalize_coordinates(c4), c4)


def test_apply_dihedral_to_tensor_consistency():
    """Verify all tensor ops match coordinate ops for a single point."""
    # Use a 3x3 tensor with a single hot spot at (2,0) -> row 0, col 2
    # Coordinates in centered frame for 3x3:
    # (-1, -1) (0, -1) (1, -1)
    # (-1,  0) (0,  0) (1,  0)
    # (-1,  1) (0,  1) (1,  1)
    # Point (1, 0) is index [1, 2]

    img = torch.zeros((1, 3, 3))
    img[0, 1, 2] = 1.0
    coords = torch.tensor([[1.0, 0.0]])

    for op in range(8):
        # Transform coord
        aug_coords, _ = apply_dihedral_augmentation(coords, op=op)
        ax, ay = aug_coords[0]

        # Transform image
        aug_img = apply_dihedral_to_tensor(img, op)

        # Map back to indices: row = ay + 1, col = ax + 1
        row, col = int(ay + 1), int(ax + 1)
        assert aug_img[0, row, col] == 1.0, f"Inconsistent mapping for op {op}"

# --- From test_dataset_mocks.py ---


@pytest.fixture
def mock_h5_file():
    with patch("h5py.File") as mock_file:
        mock_instance = mock_file.return_value
        # Mock image data
        mock_instance.__getitem__.side_effect = lambda key: {
            "img": np.zeros((10, 224, 224, 3), dtype=np.uint8)
        }[key]
        yield mock_instance


def test_hest_dataset_augmentation_consistency(mock_h5_file):
    """Verify that HEST_Dataset applies the same augmentation to pixels and coords."""
    # We need neighborhood_indices to trigger apply_dihedral_augmentation
    coords = np.array([[10.0, 20.0], [30.0, 40.0]])
    genes = np.zeros((2, 100))
    indices = np.array([0, 1])
    neighborhood_indices = np.array([[1]])  # center 0 has neighbor 1

    ds = HEST_Dataset(
        h5_path="mock.h5",
        spatial_coords=coords,
        gene_matrix=genes,
        indices=indices,
        neighborhood_indices=neighborhood_indices,
        coords_all=coords,
        augment=True,
    )

    # We want to check if apply_dihedral_to_tensor and apply_dihedral_augmentation
    # are called with the same 'op'.
    with (
        patch(
            "spatial_transcript_former.recipes.hest.dataset.apply_dihedral_to_tensor"
        ) as mock_tensor_aug,
        patch(
            "spatial_transcript_former.recipes.hest.dataset.apply_dihedral_augmentation"
        ) as mock_coord_aug,
    ):

        mock_tensor_aug.side_effect = lambda img, op: img
        mock_coord_aug.side_effect = lambda coords, op: (coords, op)

        _ = ds[0]

        # Check that both mocks were called
        assert mock_tensor_aug.called, "apply_dihedral_to_tensor was not called"
        assert mock_coord_aug.called, "apply_dihedral_augmentation was not called"

        # Check that the 'op' argument matches
        tensor_op = mock_tensor_aug.call_args[0][1]
        coord_op = mock_coord_aug.call_args[1]["op"]
        assert tensor_op == coord_op


def test_hest_feature_dataset_neighborhood_dropout():
    """Verify that HEST_FeatureDataset correctly zeros out neighbors during augmentation."""
    n_neighbors = 2
    # Ensure features, coords, and barcodes all match in length (3)
    feats = torch.ones((3, 128))
    coords = torch.zeros((3, 2))
    barcodes = [b"p0", b"p1", b"p2"]

    mock_gene_matrix = np.zeros((3, 10))
    mock_mask = [True, True, True]  # Must match length of barcodes
    mock_names = ["gene1"]

    with (
        patch("torch.load") as mock_load,
        patch(
            "spatial_transcript_former.recipes.hest.dataset.load_gene_expression_matrix"
        ) as mock_gene_load,
    ):

        mock_load.return_value = {
            "features": feats,
            "coords": coords,
            "barcodes": barcodes,
        }
        mock_gene_load.return_value = (mock_gene_matrix, mock_mask, mock_names)

        ds = HEST_FeatureDataset(
            feature_path="mock.pt",
            h5ad_path="mock.h5ad",
            n_neighbors=n_neighbors,
            augment=True,
        )

        # Run multiple times to trigger the stochastic dropout
        dropout_occurred = False
        for _ in range(100):
            f, _, _ = ds[0]
            # Center (index 0) should NEVER be zero
            assert not torch.all(f[0] == 0)

            # Check if any neighbor is zero
            if torch.any(torch.all(f[1:] == 0, dim=1)):
                dropout_occurred = True

        assert dropout_occurred, "Neighborhood dropout augmentation was never triggered"


def test_hest_dataset_log1p_logic(mock_h5_file):
    """Verify that log1p is applied to genes when enabled."""
    coords = np.array([[10.0, 20.0]])
    genes = np.array([[10.0]])

    ds_no_log = HEST_Dataset("mock.h5", coords, genes, log1p=False)
    _, g_no_log, _ = ds_no_log[0]
    assert g_no_log[0] == 10.0

    ds_log = HEST_Dataset("mock.h5", coords, genes, log1p=True)
    _, g_log, _ = ds_log[0]
    assert torch.allclose(g_log[0], torch.log1p(torch.tensor(10.0)))

# --- From test_dataloader_h5ad.py ---


data_dir = r"A:\hest_data"
# Use a sample ID we know exists
sample_ids = ["MEND29"]  # Start with just one

print(f"Testing DataLoader with ID: {sample_ids}")

try:
    loader = get_hest_dataloader(data_dir, sample_ids, batch_size=4, num_genes=100)
    print(f"DataLoader created with {len(loader)} batches.")

    for i, (images, targets) in enumerate(loader):
        print(f"Batch {i}:")
        print(f"  Images shape: {images.shape}")
        print(f"  Targets shape: {targets.shape}")
        print(f"  Target range: {targets.min()} - {targets.max()}")
        if i >= 0:
            break  # Just one batch

except Exception as e:
    print(f"Error: {e}")
    import traceback

    traceback.print_exc()
