"""
Tests for biological coverage and spatial data integrity.
"""

import os
import pytest
import torch
import numpy as np

from spatial_transcript_former.recipes.hest.io import (
    get_hest_data_dir,
    load_h5ad_metadata,
)
from spatial_transcript_former.recipes.hest.dataset import load_global_genes
from spatial_transcript_former.data.pathways import (
    download_msigdb_gmt,
    parse_gmt,
    MSIGDB_URLS,
)


@pytest.fixture
def data_dir():
    try:
        return get_hest_data_dir()
    except FileNotFoundError:
        pytest.skip("HEST data directory not found. Skipping data integrity tests.")


def test_gene_pathway_coverage(data_dir):
    """
    Verify that the top 1000 global genes provide sufficient coverage
    of biological Hallmark pathways.
    """
    num_genes = 1000
    try:
        genes = load_global_genes(data_dir, num_genes)
    except FileNotFoundError:
        pytest.skip("global_genes.json not found. Skipping data integrity test.")

    # Load Hallmarks
    url = MSIGDB_URLS["hallmarks"]
    gmt_path = download_msigdb_gmt(
        url, "h.all.v2024.1.Hs.symbols.gmt", os.path.join(data_dir, ".cache")
    )
    pathway_dict = parse_gmt(gmt_path)

    unique_hallmark_genes = set()
    for p_genes in pathway_dict.values():
        unique_hallmark_genes.update(p_genes)

    overlap = set(genes).intersection(unique_hallmark_genes)
    # Percentage of our global genes that are in the Hallmark sets
    relevance = len(overlap) / len(genes)

    print(
        f"Hallmark Gene Relevance: {relevance*100:.1f}% of global genes are Hallmarks"
    )
    # We expect at least 45% of our top 1000 genes to be Hallmarks to ensure biological focus
    assert (
        relevance > 0.45
    ), f"Global gene relevance to Hallmarks is too low ({relevance*100:.1f}%)"


def test_coordinate_alignment_bounds(data_dir):
    """
    Verify that patch coordinates align with histology image dimensions.
    Checks if projected coordinates stay within the image canvas (with patch-size tolerance).
    """
    st_dir = os.path.join(data_dir, "st")
    if not os.path.exists(st_dir):
        pytest.skip("ST directory not found.")

    # Find first available sample with features
    sample_ids = [
        f.replace(".h5ad", "") for f in os.listdir(st_dir) if f.endswith(".h5ad")
    ]

    found_valid_sample = False
    for sample_id in sample_ids:
        h5ad_path = os.path.join(st_dir, f"{sample_id}.h5ad")
        feat_path = os.path.join(data_dir, "he_features_ctranspath", f"{sample_id}.pt")

        if os.path.exists(feat_path):
            sample_id_to_test = sample_id
            found_valid_sample = True
            break

    if not found_valid_sample:
        pytest.skip("No samples with matching .pt features found for alignment test.")

    # Load metadata and features
    metadata = load_h5ad_metadata(h5ad_path)
    if "spatial" not in metadata:
        pytest.skip(f"No spatial metadata found in {sample_id_to_test}.h5ad")

    spatial = metadata["spatial"]
    # Usually 'hires' or 'lowres'
    img_key = (
        "hires" if "hires" in spatial["images"] else list(spatial["images"].keys())[0]
    )
    img_shape = spatial["images"][img_key]["shape"]  # (H, W, C)
    H, W = img_shape[0], img_shape[1]

    # Find matching scalefactor (tissue_hires_scalef or similar)
    sf = None
    sf_key = f"tissue_{img_key}_scalef"
    if sf_key in spatial["scalefactors"]:
        sf = spatial["scalefactors"][sf_key]
    else:
        # Fallback to any matching key or first available
        matches = [k for k in spatial["scalefactors"] if img_key in k]
        sf = (
            spatial["scalefactors"][matches[0]]
            if matches
            else list(spatial["scalefactors"].values())[0]
        )

    # Load pre-computed features and coordinates
    data = torch.load(feat_path, map_location="cpu", weights_only=True)
    coords = data["coords"].numpy()  # (N, 2), typically [X, Y]

    # Project coordinates to the image space
    vis_coords = coords * sf

    # Verify bounds
    max_x = vis_coords[:, 0].max()
    max_y = vis_coords[:, 1].max()

    # We allow a tolerance of 224px (standard patch size) because coordinates
    # may represent patch centers or top-left corners, and can slightly exceed
    # the image if the sliding window went to the edge.
    PATCH_SIZE = 224
    assert max_x < W + PATCH_SIZE, f"X coords exceed image width: {max_x} > {W}"
    assert max_y < H + PATCH_SIZE, f"Y coords exceed image height: {max_y} > {H}"
    assert vis_coords.min() >= -PATCH_SIZE, "Coordinates are excessively negative"
