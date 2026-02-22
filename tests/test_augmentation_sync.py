import torch
import numpy as np
import pytest
from spatial_transcript_former.data.dataset import (
    apply_dihedral_augmentation,
    apply_dihedral_to_tensor,
)


def test_sync_logic():
    # 1. Create a dummy image with a point at (scale, 0)
    # C, H, W = 1, 10, 10. Center is (5, 5)
    img = torch.zeros((1, 11, 11))
    img[0, 5, 8] = 1.0  # Point at +3 on X from center

    # Coordinates relative to center
    # Center (5, 5) -> (0, 0)
    # Point (5, 8) -> (+3, 0) (Note: HEST uses [x, y], but arrays use [y, x])
    # [3.0, 0.0]
    rel_coords = torch.tensor([[0.0, 0.0], [3.0, 0.0]])

    for op in range(8):
        # Transform Image
        aug_img = apply_dihedral_to_tensor(img, op)

        # Transform Coords
        aug_coords, _ = apply_dihedral_augmentation(rel_coords, op=op)

        # Find where the pixel moved
        # Original (5, 8) -> relative (0, +3) in y-down (indexing [5, 8])
        # After rotation, find the new index of '1.0'
        new_pos = torch.nonzero(aug_img[0])
        if len(new_pos) > 0:
            ny, nx = new_pos[0]  # Pixel indexing
            # Convert to relative to center (5, 5)
            rx, ry = nx.item() - 5, ny.item() - 5

            # Check against augmented coords
            # Note: aug_coords[1] is the moved point
            ax, ay = aug_coords[1].numpy()

            print(
                f"Op {op} ({aug_img.shape}): Found pixel at ({nx}, {ny}) -> Rel=({rx}, {ry}) | Expected Coords=({ax}, {ay})"
            )

            assert rx == pytest.approx(ax), f"X mismatch in op {op}"
            assert ry == pytest.approx(ay), f"Y mismatch in op {op}"
        else:
            # Should not happen as identity/rotations/flips preserve count
            assert False, f"Pixel lost in op {op}"


if __name__ == "__main__":
    test_sync_logic()
