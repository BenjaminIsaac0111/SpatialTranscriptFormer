"""
HEST-1k recipe for SpatialTranscriptFormer.

Provides HEST-specific dataset classes, download utilities, and dataloader
factories.  All components implement the generic :class:`SpatialDataset`
contract from ``spatial_transcript_former.data.base``.

Quick start::

    from spatial_transcript_former.recipes.hest import (
        HEST_Dataset,
        HEST_FeatureDataset,
        get_hest_dataloader,
        get_hest_feature_dataloader,
        get_sample_ids,
        setup_dataloaders,
        download_hest_subset,
    )
"""

# Dataset classes and DataLoader factories
from spatial_transcript_former.recipes.hest.dataset import (
    HEST_Dataset,
    HEST_FeatureDataset,
    get_hest_dataloader,
    get_hest_feature_dataloader,
    load_gene_expression_matrix,
    load_global_genes,
)

# I/O utilities
from spatial_transcript_former.recipes.hest.io import (
    get_hest_data_dir,
    load_h5ad_metadata,
    get_image_from_h5ad,
    decode_h5_string,
)

# Download
from spatial_transcript_former.recipes.hest.download import (
    download_hest_subset,
    download_metadata,
    filter_samples,
)

# Sample discovery and dataloader setup
from spatial_transcript_former.recipes.hest.utils import (
    get_sample_ids,
    setup_dataloaders,
)

# Splitting
from spatial_transcript_former.recipes.hest.splitting import split_hest_patients

# Vocab building
from spatial_transcript_former.recipes.hest.build_vocab import scan_h5ad_files

__all__ = [
    # Datasets
    "HEST_Dataset",
    "HEST_FeatureDataset",
    "get_hest_dataloader",
    "get_hest_feature_dataloader",
    "load_gene_expression_matrix",
    "load_global_genes",
    # I/O
    "get_hest_data_dir",
    "load_h5ad_metadata",
    "get_image_from_h5ad",
    "decode_h5_string",
    # Download
    "download_hest_subset",
    "download_metadata",
    "filter_samples",
    # Utils
    "get_sample_ids",
    "setup_dataloaders",
    "split_hest_patients",
    "scan_h5ad_files",
]
