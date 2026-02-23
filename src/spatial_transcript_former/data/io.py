"""
Low-level I/O utilities for the HEST dataset.

Centralizes data directory discovery, HDF5 metadata extraction,
and string decoding to eliminate redundancy across scripts and tests.
"""

import os
import h5py
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from ..config import get_config


def get_hest_data_dir() -> str:
    """
    Find the HEST data directory by searching common local and remote paths.
    Prioritizes paths defined in config.yaml under 'data_dirs'.

    Returns:
        str: Absolute path to the data directory.

    Raises:
        FileNotFoundError: If no valid data directory is found.
    """
    # Prefer paths from config.yaml
    potential_dirs = get_config("data_dirs", [])

    # Built-in fallbacks
    potential_dirs += [
        r"A:\hest_data",
        "hest_data",
        "../hest_data",
        os.path.abspath("data"),
    ]

    for d in potential_dirs:
        if os.path.exists(d):
            # Verify it contains at least one of the expected subdirs
            if any(
                os.path.exists(os.path.join(d, s))
                for s in ["st", "patches", "HEST_v1_3_0.csv"]
            ):
                return d

    raise FileNotFoundError(
        f"Could not find HEST data directory in any of {potential_dirs}. "
        "Please ensure your data is mounted or exists in a 'hest_data' folder."
    )


def decode_h5_string(s: Any) -> str:
    """Safely decode HDF5 string outcomes (bytes or str) to UTF-8."""
    if isinstance(s, bytes):
        return s.decode("utf-8")
    return str(s)


def load_h5ad_metadata(h5ad_path: str) -> Dict[str, Any]:
    """
    Extract core metadata from a HEST .h5ad file without loading the full matrix.

    Args:
        h5ad_path: Path to the .h5ad file.

    Returns:
        Dict containing:
            - 'barcodes': List[str]
            - 'gene_names': List[str]
            - 'spatial': Dict of images/scalefactors if available
    """
    metadata = {}
    with h5py.File(h5ad_path, "r") as f:
        # 1. Barcodes (obs index)
        if "obs" in f:
            idx_key = "_index" if "_index" in f["obs"] else "index"
            if idx_key in f["obs"]:
                metadata["barcodes"] = [
                    decode_h5_string(b) for b in f["obs"][idx_key][:]
                ]

        # 2. Gene Names (var index)
        if "var" in f:
            idx_key = "_index" if "_index" in f["var"] else "index"
            if idx_key in f["var"]:
                metadata["gene_names"] = [
                    decode_h5_string(g) for g in f["var"][idx_key][:]
                ]

        # 3. Spatial Metadata
        if "uns" in f and "spatial" in f["uns"]:
            spatial_group = f["uns/spatial"]
            sample_key = list(spatial_group.keys())[0]
            sample_data = spatial_group[sample_key]

            spatial_meta = {"sample_id": sample_key, "images": {}, "scalefactors": {}}

            if "images" in sample_data:
                for img_key in sample_data["images"].keys():
                    # We store the shape and dtype instead of the full image
                    img_ds = sample_data["images"][img_key]
                    spatial_meta["images"][img_key] = {
                        "shape": img_ds.shape,
                        "dtype": str(img_ds.dtype),
                    }

            if "scalefactors" in sample_data:
                sf_group = sample_data["scalefactors"]
                for sf_key in sf_group.keys():
                    spatial_meta["scalefactors"][sf_key] = sf_group[sf_key][()]

            metadata["spatial"] = spatial_meta

    return metadata


def get_image_from_h5ad(
    h5ad_path: str, img_type: Optional[str] = None
) -> Tuple[np.ndarray, float]:
    """
    Extract the histology image and corresponding scalefactor from H5AD.
    """
    with h5py.File(h5ad_path, "r") as f:
        spatial = f["uns/spatial"]
        sample_key = list(spatial.keys())[0]
        sample_data = spatial[sample_key]

        # Determine image
        img_group = sample_data["images"]
        if img_type is None:
            img_type = (
                "downscaled_fullres"
                if "downscaled_fullres" in img_group
                else list(img_group.keys())[0]
            )

        img = img_group[img_type][:]

        # Determine scalefactor
        sf_group = sample_data["scalefactors"]
        # Match "tissue_IMGTYPE_scalef"
        sf_key = f"tissue_{img_type}_scalef"
        if sf_key not in sf_group:
            # Fallback: find any matching "scalef" or take first
            matches = [k for k in sf_group.keys() if img_type in k and "scalef" in k]
            sf_key = matches[0] if matches else list(sf_group.keys())[0]

        scalef = sf_group[sf_key][()]

    return img, scalef
