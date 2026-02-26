import torch
import numpy as np
import pytest
from spatial_transcript_former.data.dataset import (
    apply_dihedral_augmentation,
    apply_dihedral_to_tensor,
    normalize_coordinates,
)


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


if __name__ == "__main__":
    pytest.main([__file__])
