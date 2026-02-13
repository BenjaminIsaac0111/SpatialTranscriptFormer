import pytest
import torch
import sys
import os

# Add src to sys.path so we can import spatial_transcript_former
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

@pytest.fixture
def mock_image_batch():
    """Returns a mock batch of center patches: (B, C, H, W)"""
    return torch.randn(4, 3, 224, 224)

@pytest.fixture
def mock_neighborhood_batch():
    """Returns a mock batch of neighbors: (B, S, C, H, W) where S=9 (center + 8 neighbors)"""
    return torch.randn(2, 9, 3, 224, 224)

@pytest.fixture
def mock_rel_coords():
    """Returns mock relative coordinates: (B, S, 2)"""
    B, S = 2, 9
    return torch.randn(B, S, 2)
