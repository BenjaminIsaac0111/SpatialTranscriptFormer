import torch
import numpy as np
import pytest
from unittest.mock import MagicMock, patch
from spatial_transcript_former.recipes.hest.dataset import (
    HEST_Dataset,
    HEST_FeatureDataset,
)


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
