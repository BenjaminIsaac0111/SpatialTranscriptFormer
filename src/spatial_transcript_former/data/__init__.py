"""
Data abstractions for SpatialTranscriptFormer.

Core exports:
    - :class:`SpatialDataset` — abstract base class implementing the data contract
    - :func:`apply_dihedral_augmentation` — D4 coordinate augmentation
    - :func:`apply_dihedral_to_tensor` — D4 image augmentation
    - :func:`normalize_coordinates` — auto-normalise spatial coordinates

HEST-specific exports (backward compatibility — prefer ``recipes.hest``):
    - :class:`HEST_Dataset`, :func:`get_hest_dataloader`
    - :func:`split_hest_patients`
    - :func:`download_hest_subset`, :func:`download_metadata`, :func:`filter_samples`
"""

# Core abstractions (framework)
from .base import (
    SpatialDataset,
    apply_dihedral_augmentation,
    apply_dihedral_to_tensor,
    normalize_coordinates,
)
