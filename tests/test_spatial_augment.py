import torch
import numpy as np
import pytest
from spatial_transcript_former.data.dataset import apply_dihedral_augmentation

def test_apply_dihedral_augmentation_torch():
    coords = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    
    # Run multiple times to see different ops
    results = []
    for _ in range(100):
        out, _ = apply_dihedral_augmentation(coords)
        results.append(out.numpy().tolist())
    
    # Check that we get more than 1 unique result (proving it's stochastic)
    unique_results = []
    for r in results:
        if r not in unique_results:
            unique_results.append(r)
    
    print(f"Unique results found: {len(unique_results)}")
    assert len(unique_results) > 1
    assert len(unique_results) <= 8
    
    # Check a specific transformation (e.g. Identity should be in there)
    assert coords.tolist() in unique_results

def test_apply_dihedral_augmentation_numpy():
    coords = np.array([[1.0, 0.0], [0.0, 1.0]])
    out, _ = apply_dihedral_augmentation(coords)
    assert isinstance(out, np.ndarray)
    assert out.shape == coords.shape

if __name__ == "__main__":
    pytest.main([__file__])
