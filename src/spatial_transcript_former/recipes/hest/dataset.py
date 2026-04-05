"""
Dataset classes and DataLoader factories for the HEST spatial transcriptomics benchmark.

Two loading paths are supported:

* **Raw-patch path** – ``HEST_Dataset`` / ``get_hest_dataloader``
  Reads histology tiles directly from ``.h5`` files and applies pixel-space
  augmentations.  Suitable for training models that take image crops as input.

* **Pre-computed feature path** – ``HEST_FeatureDataset`` / ``get_hest_feature_dataloader``
  Reads backbone feature vectors from ``.pt`` files, skipping repeated CNN
  forward passes.  The default path used by the SpatialTranscriptFormer
  training pipeline (``--precomputed``).

Both paths return ``(features_or_patches, gene_counts, pathway_activities,
relative_coords)`` tuples, keeping the rest of the codebase agnostic to the
loading strategy.  ``pathway_activities`` is ``None`` when no
``pathway_targets_dir`` is supplied.
"""

import os
import sys
import h5py
import json
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from typing import List, Optional
from .io import decode_h5_string, load_h5ad_metadata
from spatial_transcript_former.data.base import (
    SpatialDataset,
    apply_dihedral_augmentation,
    apply_dihedral_to_tensor,
    normalize_coordinates,
)
from torch.utils.data import DataLoader, ConcatDataset
from scipy.sparse import csr_matrix, issparse
from typing import List, Optional, Tuple, Union
from scipy.spatial import KDTree
import torch.nn.functional as F

# Augmentation helpers and normalize_coordinates are now in data.base
# and imported above. Kept here for backward compatibility:
# from spatial_transcript_former.recipes.hest.dataset import apply_dihedral_augmentation
# still works via the import at the top of this file.


# ---------------------------------------------------------------------------
# Raw-patch dataset
# ---------------------------------------------------------------------------


class HEST_Dataset(SpatialDataset):
    """PyTorch Dataset that loads raw histology patches from a HEST ``.h5`` file.

    Each item is a tuple ``(patches, gene_counts, rel_coords)`` where:

    * **patches** – ``(3, H, W)`` tensor for the centre patch, or
      ``(1+K, 3, H, W)`` when neighbours are included.
    * **gene_counts** – ``(G,)`` float tensor of gene expression counts.
    * **rel_coords** – ``(1+K, 2)`` float tensor of coordinates relative to
      the centre patch (centre is always ``[0, 0]``).

    Uses *lazy* file opening so that ``h5py`` handles are created inside each
    DataLoader worker process, avoiding pickling issues.

    Args:
        h5_path (str): Path to the HEST ``.h5`` file.
        spatial_coords (np.ndarray): ``(N, 2)`` (x, y) coordinates of the
            subset of patches in this dataset.
        gene_matrix (np.ndarray): ``(N, G)`` gene expression matrix aligned
            to ``spatial_coords``.
        indices (np.ndarray, optional): ``(N,)`` mapping from dataset index to
            the corresponding row index inside the ``.h5`` file.  If ``None``,
            the dataset index is used directly.
        transform (callable, optional): Torchvision-style transform applied to
            each patch tensor after normalisation to ``[0, 1]``.
        neighborhood_indices (np.ndarray, optional): ``(N, K)`` pre-computed
            file-level indices of the K nearest neighbours for each patch.
        coords_all (np.ndarray, optional): ``(N_all, 2)`` coordinates of *all*
            patches in the file, required for computing relative coordinates
            when neighbours are used.
        augment (bool): If ``True``, apply a random dihedral transformation
            (same op applied to both pixels and coordinates).
    """

    def __init__(
        self,
        h5_path: str,
        spatial_coords: np.ndarray,
        indices: Optional[np.ndarray] = None,
        transform=None,
        neighborhood_indices: Optional[np.ndarray] = None,
        coords_all: Optional[np.ndarray] = None,
        augment: bool = False,
    ):
        self.h5_path = h5_path
        self.transform = transform
        self.coords = spatial_coords
        self.indices = indices
        self.neighborhood_indices = neighborhood_indices
        self.coords_all = coords_all
        self.augment = augment

        # Opened lazily inside each DataLoader worker (see __getitem__).
        self.h5_file = None

    def __len__(self):
        return len(self.coords)

    def __getitem__(self, idx):
        # Open the HDF5 file on first access within this worker process.
        if self.h5_file is None:
            self.h5_file = h5py.File(self.h5_path, "r")

        # Map dataset index to the row index inside the HDF5 file.
        center_file_idx = self.indices[idx] if self.indices is not None else idx

        # Sample a single augmentation op and apply it to both pixels and
        # coordinates to keep them in sync.
        aug_op = np.random.randint(0, 8) if self.augment else 0

        if self.neighborhood_indices is not None:
            # --- Neighbourhood mode: centre patch + K neighbours ---
            neighbor_file_indices = self.neighborhood_indices[idx]
            file_indices = [center_file_idx] + list(neighbor_file_indices)

            patch_sequence = []
            for f_idx in file_indices:
                p = self.h5_file["img"][f_idx]
                p = torch.from_numpy(p).permute(2, 0, 1).float() / 255.0
                if self.transform:
                    p = self.transform(p)
                if self.augment:
                    p = apply_dihedral_to_tensor(p, aug_op)
                patch_sequence.append(p)

            data = torch.stack(patch_sequence)  # (1+K, 3, H, W)

            if self.coords_all is not None:
                center_coord = self.coords[idx]
                neighbor_coords = self.coords_all[neighbor_file_indices]
                rel_coords = np.concatenate(
                    [[[0, 0]], neighbor_coords - center_coord], axis=0
                )
                rel_coords = torch.from_numpy(rel_coords).float()
                rel_coords, _ = apply_dihedral_augmentation(rel_coords, op=aug_op)
            else:
                rel_coords = torch.zeros((len(file_indices), 2))

        else:
            # --- Single-patch mode ---
            patch = self.h5_file["img"][center_file_idx]
            patch = torch.from_numpy(patch).permute(2, 0, 1).float() / 255.0
            if self.transform:
                patch = self.transform(patch)
            if self.augment:
                patch = apply_dihedral_to_tensor(patch, aug_op)
            data = patch  # (3, H, W)
            rel_coords = torch.zeros((1, 2))

        # gene_counts removed (pathway-only)
        return data, None, rel_coords

    def __del__(self):
        if self.h5_file:
            self.h5_file.close()


# ---------------------------------------------------------------------------
# Barcode masking utilities
# ---------------------------------------------------------------------------


def get_h5ad_valid_mask(
    h5ad_path: str,
    patch_barcodes: List[bytes],
):
    """Determine which patch barcodes exist in the .h5ad (after QC)."""
    from .io import load_h5ad_metadata, decode_h5_string
    import numpy as np

    metadata = load_h5ad_metadata(h5ad_path)
    st_barcodes = metadata["barcodes"]
    st_barcode_to_idx = {b: i for i, b in enumerate(st_barcodes)}

    # Map patch barcodes to row indices
    patch_barcodes_decoded = [decode_h5_string(b) for b in patch_barcodes]
    valid_patch_mask = [pb in st_barcode_to_idx for pb in patch_barcodes_decoded]
    return np.array(valid_patch_mask)


# normalize_coordinates is now in data.base and imported above.


def get_h5ad_valid_mask(
    h5ad_path: str,
    patch_barcodes: List[bytes],
):
    """Determine which patch barcodes exist in the .h5ad (after QC)."""
    from .io import load_h5ad_metadata, decode_h5_string
    import numpy as np

    metadata = load_h5ad_metadata(h5ad_path)
    st_barcodes = metadata["barcodes"]
    st_barcode_to_idx = {b: i for i, b in enumerate(st_barcodes)}

    # Map patch barcodes to row indices
    patch_barcodes_decoded = [decode_h5_string(b) for b in patch_barcodes]
    valid_patch_mask = [pb in st_barcode_to_idx for pb in patch_barcodes_decoded]
    return np.array(valid_patch_mask)


# ---------------------------------------------------------------------------
# Raw-patch DataLoader factory
# ---------------------------------------------------------------------------


def get_hest_dataloader(
    root_dir: str,
    ids: List[str],
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
    transform=None,
    n_neighbors: int = 0,
    augment: bool = False,
):
    """Build a DataLoader over raw histology patches for a list of HEST sample IDs.

    Patches and gene expression are loaded from ``<root_dir>/patches/<id>.h5``
    and ``<root_dir>/st/<id>.h5ad`` respectively.  All samples are aligned to
    the global list of pathway-relevant features.

    Args:
        root_dir (str): Root directory containing ``patches/`` and ``st/``
            subdirectories (or flat layout).
        ids (List[str]): Sample IDs to include.
        batch_size (int): Number of items per batch.
        shuffle (bool): Whether to shuffle at each epoch.
        num_workers (int): DataLoader worker processes.
        transform (callable, optional): Transform applied to each patch tensor.
        num_genes (int): Number of genes per sample.
        n_neighbors (int): Number of spatial neighbours to include per patch.
            ``0`` disables neighbourhood mode.
        augment (bool): Whether to apply dihedral augmentations.

    Returns:
        DataLoader: Yields ``(patches, None, rel_coords)`` tuples.
    """
    datasets = []

    patches_dir = os.path.join(root_dir, "patches")
    st_dir = os.path.join(root_dir, "st")
    if not os.path.exists(patches_dir):
        patches_dir = root_dir
        st_dir = os.path.join(os.path.dirname(root_dir), "st")

    pbar = tqdm(ids, desc="Loading patches", file=sys.stdout, dynamic_ncols=True)
    for sample_id in pbar:
        h5_path = os.path.join(patches_dir, f"{sample_id}.h5")
        h5ad_path = os.path.join(st_dir, f"{sample_id}.h5ad")

        if not os.path.exists(h5_path):
            continue
        if not os.path.exists(h5ad_path):
            print(f"Warning: ST data for {sample_id} not found at {h5ad_path}")
            continue

        try:
            with h5py.File(h5_path, "r") as f:
                if "coords" not in f or "img" not in f:
                    print(f"Warning: {sample_id}.h5 missing 'coords' or 'img'")
                    continue

                patch_barcodes = f["barcode"][:].flatten()
                coords_all = f["coords"][:]
                coords_all = normalize_coordinates(coords_all)

            mask = get_h5ad_valid_mask(
                h5ad_path,
                patch_barcodes,
            )

            coords_subset = coords_all[mask]
            indices_subset = np.where(mask)[0]

            # Pre-compute KD-tree neighbours if requested
            neighborhood_indices = None
            if n_neighbors > 0:
                print(f"Pre-computing {n_neighbors} neighbours for {sample_id}...")
                tree = KDTree(coords_all)
                # Query k+1 so we can exclude the centre point itself
                _, idxs = tree.query(coords_subset, k=n_neighbors + 1)

                final_neighbors = []
                for i, center_idx in enumerate(indices_subset):
                    n_idxs = idxs[i]
                    n_idxs = n_idxs[n_idxs != center_idx][:n_neighbors]
                    if len(n_idxs) < n_neighbors:
                        padding = [center_idx] * (n_neighbors - len(n_idxs))
                        n_idxs = np.concatenate([n_idxs, padding]).astype(np.int64)
                    final_neighbors.append(n_idxs)

                neighborhood_indices = np.array(final_neighbors)

            ds = HEST_Dataset(
                h5_path,
                coords_subset,
                None,  # gene_matrix removed
                indices=indices_subset,
                transform=transform,
                neighborhood_indices=neighborhood_indices,
                coords_all=coords_all,
                augment=augment,
            )
            datasets.append(ds)

        except Exception as e:
            print(f"Error loading {sample_id}: {e}")
            continue

    if not datasets:
        raise ValueError("No valid datasets found.")

    concat_ds = ConcatDataset(datasets)
    return DataLoader(
        concat_ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
    )


# ---------------------------------------------------------------------------
# Pre-computed feature dataset
# ---------------------------------------------------------------------------


class HEST_FeatureDataset(SpatialDataset):
    """Dataset for pre-computed backbone feature vectors.

    Loads CTransPath (or any other backbone) feature vectors from a ``.pt``
    file produced by the HEST feature extraction pipeline and pairs them with
    the corresponding gene expression targets from the ``.h5ad`` file.

    Each item is a tuple ``(feats, gene_counts, rel_coords)`` where:

    * **feats** – ``(1+K, D)`` float tensor; row 0 is the centre patch, rows
      1..K are its nearest neighbours (``K = n_neighbors``).
    * **gene_counts** – ``(G,)`` float tensor for the centre patch.
    * **rel_coords** – ``(1+K, 2)`` float tensor; row 0 is always ``[0, 0]``.

    **Whole-slide mode** (``whole_slide_mode=True``): ``__len__`` returns 1
    and ``__getitem__(0)`` returns all patches on the slide as
    ``(N, D), (N, G), (N, 2)`` tensors.  Data is loaded fresh on every call to
    avoid caching large tensors in RAM.

    Args:
        feature_path (str): Path to the ``.pt`` feature file.
        h5ad_path (str): Path to the corresponding ``.h5ad`` expression file.
        num_genes (int): Number of genes expected in targets.
        selected_gene_names (List[str], optional): Gene names to align targets
            to.  ``None`` uses discovery mode on the first loaded sample.
        n_neighbors (int): Spatial neighbours to include per patch.
        use_global_context (bool): Whether to append randomly sampled
            slide-level context patches to each neighbourhood sequence.
        global_context_size (int): Number of patches in the global context.
        whole_slide_mode (bool): Return all patches as a single item.
        augment (bool): Apply dihedral and neighbourhood dropout augmentations.
    """

    def __init__(
        self,
        feature_path: str,
        h5ad_path: str,
        n_neighbors: int = 6,
        use_global_context: bool = False,
        global_context_size: int = 256,
        whole_slide_mode: bool = False,
        augment: bool = False,
        pathway_targets_dir: Optional[str] = None,
        pathway_names: Optional[List[str]] = None,
    ):
        self.feature_path = feature_path
        self.h5ad_path = h5ad_path
        self.n_neighbors = n_neighbors
        self.use_global_context = use_global_context
        self.global_context_size = global_context_size
        self.whole_slide_mode = whole_slide_mode
        self.augment = augment
        self.pathway_targets_dir = pathway_targets_dir
        self.target_pathway_names = pathway_names

        self._load_data()

    def _load_data(self):
        """Load features and gene expression into memory and build the KD-tree."""
        saved_data = torch.load(
            self.feature_path, map_location="cpu", weights_only=True
        )
        features = saved_data["features"]  # (N, D)
        coords = saved_data["coords"]  # (N, 2)
        barcodes = saved_data["barcodes"]

        mask = get_h5ad_valid_mask(
            self.h5ad_path,
            barcodes,
        )

        mask_bool = np.array(mask, dtype=bool)
        self.features = features[mask_bool]  # (N_valid, D)
        coords_valid = coords[mask_bool].numpy()
        self.coords = torch.from_numpy(
            normalize_coordinates(coords_valid)
        )  # (N_valid, 2)
        self.genes = None  # gene_matrix removed
        self.kdtree = KDTree(self.coords.numpy())

        # Load pathway activity targets if a directory is provided
        self.pathway_activities = None
        self.pathway_morans_i = None
        if self.pathway_targets_dir is not None:
            sample_id = os.path.splitext(os.path.basename(self.feature_path))[0]
            h5_path = os.path.join(self.pathway_targets_dir, f"{sample_id}.h5")
            if os.path.exists(h5_path):
                from .compute_pathway_activities import load_pathway_activities

                acts, pw_names, _, pw_morans = load_pathway_activities(
                    h5_path, list(barcodes)
                )

                if self.target_pathway_names is not None:
                    # Filter pathways to match the requested subset
                    indices = []
                    for name in self.target_pathway_names:
                        if name in pw_names:
                            indices.append(pw_names.index(name))
                        else:
                            # If a requested pathway is missing, we'll use a zero column
                            indices.append(-1)

                    # Subset and handle missing pathways (as zeros)
                    p = len(self.target_pathway_names)
                    subset_acts = np.zeros((acts.shape[0], p), dtype=np.float32)
                    subset_morans = np.zeros(p, dtype=np.float32)

                    for i, idx in enumerate(indices):
                        if idx != -1:
                            subset_acts[:, i] = acts[:, idx]
                            if pw_morans is not None:
                                subset_morans[i] = pw_morans[idx]

                    self.pathway_activities = torch.tensor(
                        subset_acts[mask_bool], dtype=torch.float32
                    )
                    self.pathway_morans_i = torch.tensor(
                        subset_morans, dtype=torch.float32
                    )
                else:
                    # No subsetting: load all pathways
                    self.pathway_activities = torch.tensor(
                        acts[mask_bool], dtype=torch.float32
                    )
                    if pw_morans is not None:
                        self.pathway_morans_i = torch.tensor(
                            pw_morans, dtype=torch.float32
                        )

    def __len__(self):
        return 1 if self.whole_slide_mode else len(self.coords)

    def __getitem__(self, idx):
        if self.whole_slide_mode:
            return self._getitem_whole_slide()
        return self._getitem_patch(idx)

    def _getitem_whole_slide(self):
        """Return all patches on the slide as a single (N, ...) item."""
        co = self.coords.clone()
        if self.augment:
            co, _ = apply_dihedral_augmentation(co)

        return (
            self.features.clone(),
            None,  # genes removed
            (
                self.pathway_activities.clone()
                if self.pathway_activities is not None
                else None
            ),
            co,
            (
                self.pathway_morans_i.clone()
                if self.pathway_morans_i is not None
                else None
            ),
        )

    def _getitem_patch(self, idx):
        """Return a single patch together with its neighbourhood context."""
        # --- Build neighbourhood index sequence ---
        # Query k+1 neighbours; the first result is usually the point itself.
        dists, neighbor_idxs = self.kdtree.query(
            self.coords[idx], k=self.n_neighbors + 1
        )

        if self.n_neighbors == 0:
            dists = np.array([dists])
            neighbor_idxs = np.array([neighbor_idxs])

        # Pad when the slide has fewer patches than requested neighbours.
        # Center (idx) must remain first so rel_coords[0] == [0, 0].
        if len(self.coords) < self.n_neighbors + 1:
            pad_len = (self.n_neighbors + 1) - len(self.coords)
            others = [i for i in range(len(self.coords)) if i != idx]
            neighbor_idxs = np.array(
                [idx] + others[: self.n_neighbors] + [idx] * pad_len
            )

        # --- Optional global context ---
        if self.use_global_context:
            total_patches = len(self.coords)
            global_idxs = (
                np.random.choice(
                    total_patches, size=self.global_context_size, replace=True
                )
                if total_patches > 0
                else np.full(self.global_context_size, idx)
            )
            combined_idxs = np.concatenate([neighbor_idxs, global_idxs])
        else:
            combined_idxs = neighbor_idxs

        feats = self.features[combined_idxs]

        # --- Neighbourhood dropout augmentation ---
        # Randomly zero out 0–2 neighbour feature vectors (never the centre).
        if self.augment and self.n_neighbors > 1:
            n_to_drop = np.random.randint(0, min(3, self.n_neighbors))
            if n_to_drop > 0:
                drop_idxs = np.random.choice(
                    range(1, self.n_neighbors + 1), size=n_to_drop, replace=False
                )
                feats[drop_idxs] = 0.0

        # --- Relative coordinates ---
        center_coord = self.coords[idx]
        rel_coords = self.coords[combined_idxs] - center_coord  # (S, 2)

        if self.augment:
            # Dihedral rotation / flip
            rel_coords, _ = apply_dihedral_augmentation(rel_coords)

        target_genes = None
        pathway_acts = (
            self.pathway_activities[idx]
            if self.pathway_activities is not None
            else None
        )
        pathway_morans = self.pathway_morans_i  # (P,) or None — same for all spots
        return feats, target_genes, pathway_acts, rel_coords, pathway_morans


# ---------------------------------------------------------------------------
# Collate helpers
# ---------------------------------------------------------------------------


def collate_fn_patch(batch):
    """Collate ``(feats, genes, pathway_acts, coords, pathway_morans)`` tuples.

    Handles ``pathway_acts=None`` and ``pathway_morans=None`` (when no
    pathway targets dir is configured) by passing ``None`` through.

    Args:
        batch: List of ``(feats, genes, pathway_acts, coords, pathway_morans)``
            tuples.

    Returns:
        tuple: ``(feats, genes, pathway_acts, coords, pathway_morans)`` where
        ``pathway_acts`` and ``pathway_morans`` are stacked tensors or ``None``.
    """
    feats = torch.stack([item[0] for item in batch])
    genes = (
        torch.stack([item[1] for item in batch]) if batch[0][1] is not None else None
    )
    has_pathways = batch[0][2] is not None
    pathways = torch.stack([item[2] for item in batch]) if has_pathways else None
    coords = torch.stack([item[3] for item in batch])
    has_morans = batch[0][4] is not None
    morans = torch.stack([item[4] for item in batch]) if has_morans else None
    return feats, genes, pathways, coords, morans


# ---------------------------------------------------------------------------
# Pre-computed feature DataLoader factory
# ---------------------------------------------------------------------------


def get_hest_feature_dataloader(
    root_dir: str,
    ids: List[str],
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
    n_neighbors: int = 6,
    use_global_context: bool = False,
    global_context_size: int = 256,
    whole_slide_mode: bool = False,
    augment: bool = False,
    feature_dir: Optional[str] = None,
    pathway_targets_dir: Optional[str] = None,
    pathway_names: Optional[List[str]] = None,
):
    """Build a DataLoader over pre-computed feature vectors for a list of HEST sample IDs.

    Feature files are looked up in ``<root_dir>/he_features/<id>.pt`` by
    default, or in ``feature_dir`` when explicitly specified.  barcode valid
    masks are determined from ``<root_dir>/st/<id>.h5ad``.

    In **whole-slide mode** (``whole_slide_mode=True``) slides of varying
    length are padded to the longest slide in each mini-batch and a boolean
    padding mask is appended.  Batches therefore yield four tensors:
    ``(feats, None, coords, padding_mask)``.

    In **patch mode** each item is a neighbourhood sequence of fixed length
    ``(1 + n_neighbors [+ global_context_size], D)`` and batching is standard.

    Args:
        root_dir (str): Root data directory.
        ids (List[str]): Sample IDs to include.
        batch_size (int): Batch size.
        shuffle (bool): Shuffle at each epoch.
        num_workers (int): DataLoader worker processes.
        n_neighbors (int): Spatial neighbours per patch (patch mode only).
        use_global_context (bool): Append global context patches (patch mode).
        global_context_size (int): Number of global context patches.
        whole_slide_mode (bool): Return full slides instead of individual patches.
        augment (bool): Apply data augmentations.
        feature_dir (str, optional): Explicit feature directory; overrides the
            default ``<root_dir>/he_features`` location.

    Returns:
        DataLoader: Batches of ``(feats, None, coords)`` in patch mode, or
        ``(feats, None, coords, mask)`` in whole-slide mode.
    """
    datasets = []

    if feature_dir is None:
        features_dir = os.path.join(root_dir, "he_features")
        if not os.path.exists(features_dir):
            features_dir = root_dir
    else:
        features_dir = feature_dir

    st_dir = os.path.join(root_dir, "st")

    for sample_id in tqdm(
        ids, desc="Loading slide data", file=sys.stdout, dynamic_ncols=True
    ):
        pt_path = os.path.join(features_dir, f"{sample_id}.pt")
        h5ad_path = os.path.join(st_dir, f"{sample_id}.h5ad")

        if os.path.exists(pt_path) and os.path.exists(h5ad_path):
            ds = HEST_FeatureDataset(
                pt_path,
                h5ad_path,
                n_neighbors=n_neighbors,
                use_global_context=use_global_context,
                global_context_size=global_context_size,
                whole_slide_mode=whole_slide_mode,
                augment=augment,
                pathway_targets_dir=pathway_targets_dir,
                pathway_names=pathway_names,
            )
            datasets.append(ds)
            if len(datasets) % 50 == 0:
                print(f"Loaded {len(datasets)}/{len(ids)} slides...")

    if not datasets:
        raise ValueError("No valid feature datasets found.")

    concat_ds = ConcatDataset(datasets)

    if whole_slide_mode:

        def collate_fn_ws(batch):
            """Pad variable-length slides to the longest in the batch.

            Args:
                batch: List of ``(feats, genes, pathway_acts, coords, pathway_morans)``
                    tuples where each tensor has a variable first dimension
                    (number of patches).  ``pathway_acts`` and
                    ``pathway_morans`` may be ``None``.

            Returns:
                tuple: ``(padded_feats, padded_genes, padded_pathway_acts,
                padded_coords, mask, pathway_morans)`` where ``mask`` is
                ``True`` for padding positions.  ``padded_pathway_acts`` and
                ``pathway_morans`` are ``None`` when not loaded.
            """
            # lengths and common dims
            lengths = [item[0].shape[0] for item in batch]
            max_len = max(lengths)
            d_dim = batch[0][0].shape[1]
            has_pathways = batch[0][2] is not None
            has_morans = batch[0][4] is not None
            bs = len(batch)

            padded_feats = torch.zeros(bs, max_len, d_dim)
            padded_coords = torch.zeros(bs, max_len, 2)
            # True = padding, False = valid data  (matches nn.MultiheadAttention convention)
            mask = torch.ones(bs, max_len, dtype=torch.bool)

            if has_pathways:
                p_dim = batch[0][2].shape[1]
                padded_pathways = torch.zeros(bs, max_len, p_dim)
            else:
                padded_pathways = None

            # pathway_morans is per-sample (P,), no spatial padding needed
            if has_morans:
                stacked_morans = torch.stack([item[4] for item in batch])  # (B, P)
            else:
                stacked_morans = None

            for i, (f, g, pw, c, _pm) in enumerate(batch):
                l = lengths[i]
                padded_feats[i, :l] = f
                padded_coords[i, :l] = c
                mask[i, :l] = False
                if has_pathways:
                    padded_pathways[i, :l] = pw

            return (
                padded_feats,
                None,  # genes removed
                padded_pathways,
                padded_coords,
                mask,
                stacked_morans,
            )

        return DataLoader(
            concat_ds,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=collate_fn_ws,
            pin_memory=True,
            persistent_workers=(num_workers > 0),
        )

    return DataLoader(
        concat_ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn_patch,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
    )
