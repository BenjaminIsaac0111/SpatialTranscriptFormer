import torch
import torch.nn as nn
import numpy as np
import pytest
from spatial_transcript_former.models.interaction import LocalPatchMixer


def test_local_patch_mixer_basic():
    """Test that output shape is correct and forward pass works."""
    B, N, D = 2, 10, 32
    mixer = LocalPatchMixer(dim=D, kernel_size=3)

    # Grid of coordinates (2x5)
    coords = []
    for b in range(B):
        bc = []
        for y in range(2):
            for x in range(5):
                bc.append([x, y])
        coords.append(bc)
    coords = torch.tensor(coords, dtype=torch.float32)

    x = torch.randn(B, N, D)
    out = mixer(x, coords)

    assert out.shape == x.shape
    # Check that residual connection worked (output is not zero)
    assert not torch.allclose(out, torch.zeros_like(out))


def test_local_patch_mixer_sparse():
    """Test that gaps in coordinates are handled."""
    dim = 8
    mixer = LocalPatchMixer(dim=dim, kernel_size=3)

    # 3 patches in a line with a gap: (0,0), (2,0), (4,0)
    x = torch.zeros(1, 3, dim)
    x[0, 0, 0] = 1.0
    x[0, 1, 1] = 1.0
    x[0, 2, 2] = 1.0

    coords = torch.tensor([[[0, 0], [2, 0], [4, 0]]], dtype=torch.float32)

    with torch.no_grad():
        mixer.conv.weight.fill_(0.0)
        mixer.conv.bias.fill_(0.0)
        # Set weight to look at neighbor at x+1 (weight[1, 2] in 3x3)
        mixer.conv.weight[:, 0, 1, 2] = 1.0

    out = mixer(x, coords)

    # The patch at 2.0 looks at 3.0 (empty). Conv result should be 0.
    # Out = x + 0 = x.
    # Check that ALL values are exactly as original x
    torch.testing.assert_close(out, x, atol=1e-4, rtol=1e-4)


def test_local_patch_mixer_neighbor_influence():
    """Verify that neighbors actually influence the center patch."""
    dim = 4
    mixer = LocalPatchMixer(dim=dim, kernel_size=3)

    # 3 patches adjacent: (0,0), (1,0), (2,0)
    x = torch.zeros(1, 3, dim)
    # Patch 1 (center) has signal in channel 0
    x[0, 1, 0] = 1.0

    coords = torch.tensor([[[0, 0], [1, 0], [2, 0]]], dtype=torch.float32)

    with torch.no_grad():
        mixer.conv.weight.fill_(0.0)
        mixer.conv.bias.fill_(0.0)
        # Weight[1, 0] looks at neighbor x-1
        # So output at Grid(1,0) looks at Grid(0,0)
        # Output at Grid(2,0) looks at Grid(1,0)
        mixer.conv.weight[:, 0, 1, 0] = 10.0

    out = mixer(x, coords)

    # Patch 2 (at 2,0) should receive signal from Patch 1 (at 1,0)
    # Patch 1 had 1.0. Conv result at 2,0 should be 1.0 * 10.0 = 10.0.
    # After GELU: GELU(10) is approx 10.0.
    # Residual: 0.0 + 10.0 = 10.0.
    assert out[0, 2, 0] > 9.0

    # Patch 1 (at 1,0) should receive signal from Patch 0 (at 0,0) which is 0.
    # Residual: 1.0 + 0.0 = 1.0.
    assert out[0, 1, 0] == 1.0


if __name__ == "__main__":
    pytest.main([__file__])
