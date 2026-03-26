"""
Centralised path resolution utilities for SpatialTranscriptFormer.

Eliminates duplicated path-discovery logic that was previously scattered
across ``train.py``, ``builder.py``, ``utils.py``, and ``dataset.py``.
"""

import os
from typing import List, Optional


def find_project_root() -> str:
    """Return the absolute path to the project root directory.

    Walks upward from the installed package location
    (``src/spatial_transcript_former/data/``) to the repository root.
    """
    return os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    )


def resolve_gene_vocab_path(*search_dirs: str) -> str:
    """Locate ``global_genes.json`` by searching multiple directories.

    Searches (in order):
    1. Each directory in ``search_dirs``.
    2. The project root (detected automatically).
    3. The current working directory.

    Args:
        *search_dirs: Additional directories to search first.

    Returns:
        Absolute path to the first ``global_genes.json`` found.

    Raises:
        FileNotFoundError: If the file cannot be found anywhere.
    """
    candidates = list(search_dirs) + [find_project_root(), os.getcwd()]

    for d in candidates:
        path = os.path.join(d, "global_genes.json")
        if os.path.exists(path):
            return os.path.abspath(path)

    searched = ", ".join(repr(d) for d in candidates)
    raise FileNotFoundError(
        f"global_genes.json not found. Searched: {searched}. "
        "Run `stf-build-vocab` or place the file in your data directory."
    )


def resolve_feature_dir(
    data_dir: str,
    backbone: str = "resnet50",
    feature_dir: Optional[str] = None,
) -> str:
    """Resolve the path to pre-computed backbone features.

    Search order:
    1. ``feature_dir`` if explicitly provided and exists.
    2. ``<data_dir>/he_features`` (ResNet-50) or
       ``<data_dir>/he_features_<backbone>``.
    3. ``<data_dir>/patches/he_features[_<backbone>]``.

    Args:
        data_dir: Root data directory (e.g. ``A:\\hest_data``).
        backbone: Backbone identifier.
        feature_dir: Explicit override; returned directly if it exists.

    Returns:
        Absolute path to the feature directory.

    Raises:
        FileNotFoundError: If no valid feature directory can be found.
    """
    # 1. Explicit override
    if feature_dir and os.path.exists(feature_dir):
        return os.path.abspath(feature_dir)

    # 2. Standard location
    feat_dir_name = (
        "he_features" if backbone == "resnet50" else f"he_features_{backbone}"
    )
    candidate = os.path.join(data_dir, feat_dir_name)
    if os.path.exists(candidate):
        return os.path.abspath(candidate)

    # 3. Inside patches/ subdirectory
    candidate = os.path.join(data_dir, "patches", feat_dir_name)
    if os.path.exists(candidate):
        return os.path.abspath(candidate)

    raise FileNotFoundError(
        f"Feature directory '{feat_dir_name}' not found in "
        f"'{data_dir}' or '{os.path.join(data_dir, 'patches')}'. "
        f"Run feature extraction first or pass --feature-dir explicitly."
    )
