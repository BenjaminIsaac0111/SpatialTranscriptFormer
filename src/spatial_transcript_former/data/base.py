"""
Abstract data contracts for SpatialTranscriptFormer.

Defines the :class:`SpatialDataset` ABC that any spatial transcriptomics
dataset must implement.  The training engine, ``Trainer``, and ``Predictor``
all depend only on this contract — never on a specific data source.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset


class SpatialDataset(Dataset, ABC):
    """Abstract base class for spatial transcriptomics datasets.

    Any dataset used with SpatialTranscriptFormer must subclass this and
    implement :meth:`__getitem__` and :meth:`__len__`.

    ``__getitem__`` must return a 3-tuple::

        (features, gene_counts, rel_coords)

    where:

    * **features** — ``(S, D)`` float tensor of patch embeddings
      (``S`` = 1 + K neighbours, ``D`` = backbone feature dim), or
      ``(3, H, W)`` / ``(S, 3, H, W)`` image tensor in raw-patch mode.
    * **gene_counts** — ``(G,)`` float tensor of gene expression targets.
    * **rel_coords** — ``(S, 2)`` float tensor of spatial coordinates
      relative to the centre patch (centre is always ``[0, 0]``).

    Subclasses SHOULD also expose :attr:`num_genes` and (optionally)
    :attr:`gene_names` as properties.

    Example::

        class MyVisiumDataset(SpatialDataset):
            def __init__(self, slide_path, genes, coords, features):
                self._features = features   # (N, D)
                self._genes = genes         # (N, G)
                self._coords = coords       # (N, 2)

            def __len__(self):
                return len(self._features)

            def __getitem__(self, idx):
                center = self._coords[idx]
                rel = self._coords - center   # simplest: no neighbour selection
                return self._features[idx:idx+1], self._genes[idx], rel[idx:idx+1]

            @property
            def num_genes(self):
                return self._genes.shape[1]

            @property
            def gene_names(self):
                return None  # or a list of gene symbols
    """

    @abstractmethod
    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``(features, gene_counts, rel_coords)`` for index ``idx``."""
        ...

    @abstractmethod
    def __len__(self) -> int:
        """Return the number of items in the dataset."""
        ...

    # ------------------------------------------------------------------
    # Optional attributes — subclasses can set these as instance
    # attributes (self.num_genes = ...) or override as properties.
    # ------------------------------------------------------------------
    #: Number of gene expression targets.  Set in __init__ or override.
    num_genes: int = 0
    #: Ordered list of gene symbols, or None if unavailable.
    gene_names: Optional[List[str]] = None


# ---------------------------------------------------------------------------
# Generic augmentation helpers (shared across recipes)
# ---------------------------------------------------------------------------


def apply_dihedral_augmentation(coords, op=None):
    """Apply one of the 8 dihedral (D4) symmetries to 2-D coordinates.

    Args:
        coords: ``(N, 2)`` array or tensor of (x, y) coordinates.
        op: Integer in ``[0, 7]`` or ``None`` (random).

    Returns:
        Tuple of (augmented_coords, op).
    """
    is_torch = isinstance(coords, torch.Tensor)
    if is_torch:
        x, y = coords[..., 0].clone(), coords[..., 1].clone()
    else:
        x, y = coords[..., 0].copy(), coords[..., 1].copy()

    if op is None:
        op = np.random.randint(0, 8)

    if op == 0:
        pass
    elif op == 1:
        x, y = y, -x
    elif op == 2:
        x, y = -x, -y
    elif op == 3:
        x, y = -y, x
    elif op == 4:
        x = -x
    elif op == 5:
        y = -y
    elif op == 6:
        x, y = y, x
    elif op == 7:
        x, y = -y, -x

    if is_torch:
        return torch.stack([x, y], dim=-1), op
    else:
        return np.stack([x, y], axis=-1), op


def apply_dihedral_to_tensor(img, op):
    """Apply a dihedral operation to a ``(C, H, W)`` image tensor.

    Each operation matches :func:`apply_dihedral_augmentation` so that pixel
    content and spatial coordinates stay aligned after augmentation.
    """
    if op == 0:
        return img
    if op == 1:
        return torch.rot90(img, k=1, dims=[1, 2])
    if op == 2:
        return torch.rot90(img, k=2, dims=[1, 2])
    if op == 3:
        return torch.rot90(img, k=3, dims=[1, 2])
    if op == 4:
        return torch.flip(img, dims=[2])
    if op == 5:
        return torch.flip(img, dims=[1])
    if op == 6:
        return img.transpose(1, 2)
    if op == 7:
        return img.transpose(1, 2).flip(dims=[1, 2])
    return img


def normalize_coordinates(coords: np.ndarray) -> np.ndarray:
    """Auto-normalize physical coordinates to integer grid indices."""
    if len(coords) == 0:
        return coords

    x_vals = np.unique(coords[:, 0])
    y_vals = np.unique(coords[:, 1])

    dx = x_vals[1:] - x_vals[:-1]
    dy = y_vals[1:] - y_vals[:-1]

    steps = np.concatenate([dx, dy])
    valid_steps = steps[steps > 0.5]

    if len(valid_steps) == 0:
        return coords

    step_size = valid_steps.min()
    if step_size >= 2.0:
        return np.round(coords / step_size).astype(coords.dtype)
    return coords
