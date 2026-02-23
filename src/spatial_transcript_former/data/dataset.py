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

Both paths return ``(features_or_patches, gene_counts, relative_coords)``
tuples, keeping the rest of the codebase agnostic to the loading strategy.
"""

import os
import h5py
import json
import torch
import pandas as pd
import numpy as np
from .io import decode_h5_string, load_h5ad_metadata
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from scipy.sparse import csr_matrix
from typing import List, Optional, Tuple, Union
from scipy.spatial import KDTree
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Spatial augmentation helpers
# ---------------------------------------------------------------------------


def apply_dihedral_augmentation(coords, op=None):
    """Apply one of the 8 dihedral symmetries to a set of 2-D coordinates.

    The dihedral group D4 contains 4 rotations and 4 reflections, which leave
    a square grid invariant.  Applying the same operation to both pixel tensors
    and coordinate tensors keeps spatial relationships consistent after
    augmentation.

    Args:
        coords (torch.Tensor or np.ndarray): Shape ``(N, 2)`` array of (x, y)
            coordinates defined in a *centred* frame (i.e. the origin is the
            centre of the slide region, not the top-left corner).
        op (int, optional): Integer in ``[0, 7]`` selecting the operation.
            If ``None``, one is chosen uniformly at random.

    Returns:
        tuple:
            - **augmented_coords** – same type and shape as the input.
            - **op** (*int*) – the operation that was applied (useful for
              applying the same transformation to the corresponding image).

    Operations
    ----------
    =====  ==============
    Index  Description
    =====  ==============
    0      Identity
    1      Rotate 90° CCW
    2      Rotate 180°
    3      Rotate 270° CCW
    4      Flip horizontal  (negate x)
    5      Flip vertical    (negate y)
    6      Transpose        (swap x and y)
    7      Anti-transpose   (swap and negate both)
    =====  ==============
    """
    is_torch = isinstance(coords, torch.Tensor)
    if is_torch:
        x, y = coords[..., 0].clone(), coords[..., 1].clone()
    else:
        x, y = coords[..., 0].copy(), coords[..., 1].copy()

    if op is None:
        op = np.random.randint(0, 8)

    if op == 0:  # Identity
        pass
    elif op == 1:  # Rotate 90° CCW
        x, y = y, -x
    elif op == 2:  # Rotate 180°
        x, y = -x, -y
    elif op == 3:  # Rotate 270° CCW
        x, y = -y, x
    elif op == 4:  # Flip horizontal
        x = -x
    elif op == 5:  # Flip vertical
        y = -y
    elif op == 6:  # Transpose
        x, y = y, x
    elif op == 7:  # Anti-transpose
        x, y = -y, -x

    if is_torch:
        return torch.stack([x, y], dim=-1), op
    else:
        return np.stack([x, y], axis=-1), op


def apply_dihedral_to_tensor(img, op):
    """Apply a dihedral operation to a ``(C, H, W)`` image tensor.

    Each operation matches the coordinate transform in
    :func:`apply_dihedral_augmentation` so that pixel content and spatial
    coordinates stay aligned after augmentation.

    Args:
        img (torch.Tensor): Image tensor of shape ``(C, H, W)``.
        op (int): Operation index in ``[0, 7]``.

    Returns:
        torch.Tensor: Transformed image tensor, same shape as ``img``.
    """
    if op == 0:
        return img
    if op == 1:
        return torch.rot90(img, k=1, dims=[1, 2])  # Rotate 90° CCW
    if op == 2:
        return torch.rot90(img, k=2, dims=[1, 2])  # Rotate 180°
    if op == 3:
        return torch.rot90(img, k=3, dims=[1, 2])  # Rotate 270° CCW
    if op == 4:
        return torch.flip(img, dims=[2])  # Flip horizontal (width axis)
    if op == 5:
        return torch.flip(img, dims=[1])  # Flip vertical   (height axis)
    if op == 6:
        return img.transpose(1, 2)  # Transpose
    if op == 7:
        return img.transpose(1, 2).flip(dims=[1, 2])  # Anti-transpose
    return img


# ---------------------------------------------------------------------------
# Raw-patch dataset
# ---------------------------------------------------------------------------


class HEST_Dataset(Dataset):
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
        gene_matrix: np.ndarray,
        indices: Optional[np.ndarray] = None,
        transform=None,
        neighborhood_indices: Optional[np.ndarray] = None,
        coords_all: Optional[np.ndarray] = None,
        augment: bool = False,
        log1p: bool = False,
    ):
        self.h5_path = h5_path
        self.transform = transform
        self.coords = spatial_coords
        self.genes = gene_matrix
        self.indices = indices
        self.neighborhood_indices = neighborhood_indices
        self.coords_all = coords_all
        self.augment = augment
        self.log1p = log1p

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

        gene_counts = torch.tensor(self.genes[idx], dtype=torch.float32)
        if self.log1p:
            gene_counts = torch.log1p(gene_counts)

        return data, gene_counts, rel_coords

    def __del__(self):
        if self.h5_file:
            self.h5_file.close()


# ---------------------------------------------------------------------------
# Gene-expression loading utilities
# ---------------------------------------------------------------------------


def load_gene_expression_matrix(
    h5ad_path: str,
    patch_barcodes: List[bytes],
    selected_gene_names: Optional[List[str]] = None,
    num_genes: int = 1000,
):
    """Load and align a gene expression matrix from a HEST ``.h5ad`` file.

    Depending on whether ``selected_gene_names`` is provided, the function
    operates in one of two modes:

    **Discovery mode** (``selected_gene_names=None``)
        Selects the top ``num_genes`` genes by total count across the valid
        patches in this file and returns their names.

    **Alignment mode** (``selected_gene_names`` provided)
        Projects the file's expression matrix onto the supplied gene list,
        filling any missing genes with zeros.  This ensures all samples in a
        training run share an identical feature space.

    Args:
        h5ad_path (str): Path to the ``.h5ad`` file.
        patch_barcodes (List[bytes]): Barcodes identifying the patches whose
            expression should be retrieved (order matches the ``.h5`` file).
        selected_gene_names (List[str], optional): Target gene list for
            alignment mode.  ``None`` activates discovery mode.
        num_genes (int): Number of genes to select in discovery mode, or the
            expected output width in alignment mode.

    Returns:
        tuple:
            - **dense_matrix** (*np.ndarray*) – ``(N_valid, num_genes)`` float32
              gene expression matrix.
            - **valid_mask** (*np.ndarray*) – ``(N_patches,)`` boolean array;
              ``True`` where a patch barcode was found in the ``.h5ad`` file.
            - **selected_names** (*List[str]*) – Names of the genes in the
              output columns (discovery mode) or the unchanged input list
              (alignment mode).
    """
    metadata = load_h5ad_metadata(h5ad_path)

    st_barcodes = metadata["barcodes"]
    st_barcode_to_idx = {b: i for i, b in enumerate(st_barcodes)}

    current_gene_names = metadata["gene_names"]
    gene_name_to_idx = {name: i for i, name in enumerate(current_gene_names)}

    with h5py.File(h5ad_path, "r") as f:
        # --- Map patch barcodes to row indices ---
        patch_barcodes_decoded = [decode_h5_string(b) for b in patch_barcodes]
        patch_indices = []
        valid_patch_mask = []
        for pb in patch_barcodes_decoded:
            if pb in st_barcode_to_idx:
                patch_indices.append(st_barcode_to_idx[pb])
                valid_patch_mask.append(True)
            else:
                patch_indices.append(0)  # placeholder; excluded by mask
                valid_patch_mask.append(False)

        valid_patch_mask = np.array(valid_patch_mask)
        patch_indices_array = np.array(patch_indices)

        # --- Load expression matrix (sparse or dense) ---
        X = f["X"]
        if isinstance(X, h5py.Group):
            # Stored as a CSR group (data / indices / indptr)
            mat = csr_matrix(
                (X["data"][:], X["indices"][:], X["indptr"][:]),
                shape=(len(st_barcodes), len(current_gene_names)),
            )
        elif isinstance(X, h5py.Dataset):
            mat = X[:]  # Dense; assumed to fit in memory
        else:
            raise ValueError("Unknown X format in h5ad file")

        # Slice rows to the valid patches only
        valid_indices = patch_indices_array[valid_patch_mask]
        mat_subset = mat[valid_indices]

        # --- Select / align genes ---
        if selected_gene_names is None:
            # Discovery mode: pick top-N genes by total count
            if isinstance(mat_subset, csr_matrix):
                gene_sums = np.array(mat_subset.sum(axis=0)).flatten()
            else:
                gene_sums = np.sum(mat_subset, axis=0)

            top_indices = np.argsort(gene_sums)[-num_genes:][::-1]
            selected_names = [current_gene_names[i] for i in top_indices]

            final_subset_raw = mat_subset[:, top_indices]
            if isinstance(final_subset_raw, csr_matrix):
                final_subset_raw = final_subset_raw.toarray()

            # Pad with zeros if fewer than num_genes are available
            n_found = final_subset_raw.shape[1]
            if n_found < num_genes:
                final_subset = np.zeros(
                    (final_subset_raw.shape[0], num_genes), dtype=np.float32
                )
                final_subset[:, :n_found] = final_subset_raw
                selected_names += [f"pad_gene_{j}" for j in range(num_genes - n_found)]
            else:
                final_subset = final_subset_raw

        else:
            # Alignment mode: project onto the provided gene list
            selected_names = selected_gene_names
            valid_src_indices = []
            valid_dst_indices = []
            for i, name in enumerate(selected_names):
                if name in gene_name_to_idx:
                    valid_src_indices.append(gene_name_to_idx[name])
                    valid_dst_indices.append(i)

            n_valid_patches = len(valid_indices)
            final_subset = np.zeros((n_valid_patches, num_genes), dtype=np.float32)

            if valid_src_indices:
                if isinstance(mat_subset, csr_matrix):
                    cols_data = mat_subset[:, valid_src_indices].toarray()
                else:
                    cols_data = mat_subset[:, valid_src_indices]
                final_subset[:, valid_dst_indices] = cols_data

        return final_subset, valid_patch_mask, selected_names


def normalize_coordinates(coords: np.ndarray) -> np.ndarray:
    """Auto-normalizes physical coordinates to integer grid indices."""
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


def load_global_genes(root_dir: str, num_genes: int = 1000) -> List[str]:
    """Load a globally consistent gene list from ``global_genes.json``.

    Searches ``root_dir`` first, then the current working directory.

    Args:
        root_dir (str): Primary directory to look for ``global_genes.json``.
        num_genes (int): Maximum number of genes to return (truncates the list
            if it is longer than this value).

    Returns:
        List[str]: Ordered list of gene names.

    Raises:
        FileNotFoundError: If ``global_genes.json`` cannot be found.
        RuntimeError: If the file cannot be parsed.
    """
    global_genes_path = os.path.join(root_dir, "global_genes.json")
    if not os.path.exists(global_genes_path):
        global_genes_path = "global_genes.json"

    if not os.path.exists(global_genes_path):
        raise FileNotFoundError(
            f"global_genes.json not found in '{root_dir}' or the current directory. "
            "Ensure global_genes.json exists to maintain consistent gene representation."
        )

    try:
        with open(global_genes_path, "r") as f:
            genes = json.load(f)
        genes = genes[:num_genes]
        print(f"Loaded {len(genes)} global genes from {global_genes_path}")
        return genes
    except Exception as e:
        raise RuntimeError(f"Error loading global genes from {global_genes_path}: {e}")


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
    num_genes: int = 1000,
    n_neighbors: int = 0,
    augment: bool = False,
    log1p: bool = False,
):
    """Build a DataLoader over raw histology patches for a list of HEST sample IDs.

    Patches and gene expression are loaded from ``<root_dir>/patches/<id>.h5``
    and ``<root_dir>/st/<id>.h5ad`` respectively.  All samples are aligned to
    the global gene list loaded via :func:`load_global_genes`.

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
        DataLoader: Yields ``(patches, gene_counts, rel_coords)`` tuples.
    """
    datasets = []

    patches_dir = os.path.join(root_dir, "patches")
    st_dir = os.path.join(root_dir, "st")
    if not os.path.exists(patches_dir):
        patches_dir = root_dir
        st_dir = os.path.join(os.path.dirname(root_dir), "st")

    common_gene_names = load_global_genes(root_dir, num_genes)

    for sample_id in ids:
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

            gene_matrix, mask, _ = load_gene_expression_matrix(
                h5ad_path,
                patch_barcodes,
                selected_gene_names=common_gene_names,
                num_genes=num_genes,
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
                gene_matrix,
                indices=indices_subset,
                transform=transform,
                neighborhood_indices=neighborhood_indices,
                coords_all=coords_all,
                augment=augment,
                log1p=log1p,
            )
            datasets.append(ds)

        except Exception as e:
            print(f"Error loading {sample_id}: {e}")
            continue

    if not datasets:
        raise ValueError("No valid datasets found.")

    concat_ds = ConcatDataset(datasets)
    return DataLoader(
        concat_ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
    )


# ---------------------------------------------------------------------------
# Pre-computed feature dataset
# ---------------------------------------------------------------------------


class HEST_FeatureDataset(Dataset):
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
        log1p (bool): Apply ``log1p`` transform to gene expression counts.
    """

    def __init__(
        self,
        feature_path: str,
        h5ad_path: str,
        num_genes: int = 1000,
        selected_gene_names: Optional[List[str]] = None,
        n_neighbors: int = 6,
        use_global_context: bool = False,
        global_context_size: int = 256,
        whole_slide_mode: bool = False,
        augment: bool = False,
        log1p: bool = False,
    ):
        self.feature_path = feature_path
        self.h5ad_path = h5ad_path
        self.num_genes = num_genes
        self.selected_gene_names = selected_gene_names
        self.n_neighbors = n_neighbors
        self.use_global_context = use_global_context
        self.global_context_size = global_context_size
        self.whole_slide_mode = whole_slide_mode
        self.augment = augment
        self.log1p = log1p

        if not self.whole_slide_mode:
            self._load_data()
        else:
            # Defer loading to __getitem__ to avoid caching whole slides in RAM.
            self.features = None
            self.coords = None
            self.genes = None
            self.kdtree = None

    def _load_data(self):
        """Load features and gene expression into memory and build the KD-tree."""
        saved_data = torch.load(
            self.feature_path, map_location="cpu", weights_only=True
        )
        features = saved_data["features"]  # (N, D)
        coords = saved_data["coords"]  # (N, 2)
        barcodes = saved_data["barcodes"]

        gene_matrix, mask, selected_names = load_gene_expression_matrix(
            self.h5ad_path,
            barcodes,
            selected_gene_names=self.selected_gene_names,
            num_genes=self.num_genes,
        )

        if self.log1p:
            gene_matrix = np.log1p(gene_matrix)

        if self.selected_gene_names is None:
            self.selected_gene_names = selected_names

        mask_bool = np.array(mask, dtype=bool)
        self.features = features[mask_bool]  # (N_valid, D)
        coords_valid = coords[mask_bool].numpy()
        self.coords = torch.from_numpy(
            normalize_coordinates(coords_valid)
        )  # (N_valid, 2)
        self.genes = torch.tensor(gene_matrix, dtype=torch.float32)  # (N_valid, G)
        self.kdtree = KDTree(self.coords.numpy())

    def __len__(self):
        return 1 if self.whole_slide_mode else len(self.coords)

    def __getitem__(self, idx):
        if self.whole_slide_mode:
            return self._getitem_whole_slide()
        return self._getitem_patch(idx)

    def _getitem_whole_slide(self):
        """Return all patches on the slide as a single (N, ...) item."""
        saved_data = torch.load(
            self.feature_path, map_location="cpu", weights_only=True
        )
        features = saved_data["features"]
        coords = saved_data["coords"]
        barcodes = saved_data["barcodes"]
        del saved_data

        gene_matrix, mask, _ = load_gene_expression_matrix(
            self.h5ad_path,
            barcodes,
            selected_gene_names=self.selected_gene_names,
            num_genes=self.num_genes,
        )

        if self.log1p:
            gene_matrix = np.log1p(gene_matrix)

        mask_bool = np.array(mask, dtype=bool)
        feats = features[mask_bool]
        co = coords[mask_bool].numpy()
        co = torch.from_numpy(normalize_coordinates(co))
        g = torch.tensor(gene_matrix, dtype=torch.float32)
        del gene_matrix

        if self.augment:
            co, _ = apply_dihedral_augmentation(co)

        return feats, g, co

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

        # Pad when the slide has fewer patches than requested neighbours
        if len(self.coords) < self.n_neighbors + 1:
            pad_len = (self.n_neighbors + 1) - len(self.coords)
            neighbor_idxs = np.array(list(range(len(self.coords))) + [idx] * pad_len)

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
            # Coordinate jitter (applied to neighbours only; centre stays at origin)
            jitter = torch.randn_like(rel_coords) * 5.0
            jitter[0] = 0.0
            rel_coords = rel_coords + jitter

            # Dihedral rotation / flip
            rel_coords, _ = apply_dihedral_augmentation(rel_coords)

        target_genes = self.genes[idx]
        return feats, target_genes, rel_coords


# ---------------------------------------------------------------------------
# Pre-computed feature DataLoader factory
# ---------------------------------------------------------------------------


def get_hest_feature_dataloader(
    root_dir: str,
    ids: List[str],
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
    num_genes: int = 1000,
    n_neighbors: int = 6,
    use_global_context: bool = False,
    global_context_size: int = 256,
    whole_slide_mode: bool = False,
    augment: bool = False,
    log1p: bool = False,
    feature_dir: Optional[str] = None,
):
    """Build a DataLoader over pre-computed feature vectors for a list of HEST sample IDs.

    Feature files are looked up in ``<root_dir>/he_features/<id>.pt`` by
    default, or in ``feature_dir`` when explicitly specified.  Gene expression
    targets are loaded from ``<root_dir>/st/<id>.h5ad``.

    In **whole-slide mode** (``whole_slide_mode=True``) slides of varying
    length are padded to the longest slide in each mini-batch and a boolean
    padding mask is appended.  Batches therefore yield four tensors:
    ``(feats, genes, coords, padding_mask)``.

    In **patch mode** each item is a neighbourhood sequence of fixed length
    ``(1 + n_neighbors [+ global_context_size], D)`` and batching is standard.

    Args:
        root_dir (str): Root data directory.
        ids (List[str]): Sample IDs to include.
        batch_size (int): Batch size.
        shuffle (bool): Shuffle at each epoch.
        num_workers (int): DataLoader worker processes.
        num_genes (int): Number of genes per sample.
        n_neighbors (int): Spatial neighbours per patch (patch mode only).
        use_global_context (bool): Append global context patches (patch mode).
        global_context_size (int): Number of global context patches.
        whole_slide_mode (bool): Return full slides instead of individual patches.
        augment (bool): Apply data augmentations.
        log1p (bool): Apply ``log1p`` transform to gene expression counts.
        feature_dir (str, optional): Explicit feature directory; overrides the
            default ``<root_dir>/he_features`` location.

    Returns:
        DataLoader: Batches of ``(feats, genes, coords)`` in patch mode, or
        ``(feats, genes, coords, mask)`` in whole-slide mode.
    """
    datasets = []

    if feature_dir is None:
        features_dir = os.path.join(root_dir, "he_features")
        if not os.path.exists(features_dir):
            features_dir = root_dir
    else:
        features_dir = feature_dir

    st_dir = os.path.join(root_dir, "st")
    common_gene_names = load_global_genes(root_dir, num_genes)

    for sample_id in ids:
        pt_path = os.path.join(features_dir, f"{sample_id}.pt")
        h5ad_path = os.path.join(st_dir, f"{sample_id}.h5ad")

        if os.path.exists(pt_path) and os.path.exists(h5ad_path):
            ds = HEST_FeatureDataset(
                pt_path,
                h5ad_path,
                num_genes=num_genes,
                selected_gene_names=common_gene_names,
                n_neighbors=n_neighbors,
                use_global_context=use_global_context,
                global_context_size=global_context_size,
                whole_slide_mode=whole_slide_mode,
                augment=augment,
                log1p=log1p,
            )
            datasets.append(ds)

    if not datasets:
        raise ValueError("No valid feature datasets found.")

    concat_ds = ConcatDataset(datasets)

    if whole_slide_mode:

        def collate_fn_ws(batch):
            """Pad variable-length slides to the longest in the batch.

            Args:
                batch: List of ``(feats, genes, coords)`` tuples where each
                    tensor has a variable first dimension (number of patches).

            Returns:
                tuple: ``(padded_feats, padded_genes, padded_coords, mask)``
                    where ``mask`` is ``True`` for padding positions and
                    ``False`` for real data.
            """
            lengths = [item[0].shape[0] for item in batch]
            max_len = max(lengths)
            d_dim = batch[0][0].shape[1]
            g_dim = batch[0][1].shape[1]
            bs = len(batch)

            padded_feats = torch.zeros(bs, max_len, d_dim)
            padded_genes = torch.zeros(bs, max_len, g_dim)
            padded_coords = torch.zeros(bs, max_len, 2)
            # True = padding, False = valid data  (matches nn.MultiheadAttention convention)
            mask = torch.ones(bs, max_len, dtype=torch.bool)

            for i, (f, g, c) in enumerate(batch):
                l = lengths[i]
                padded_feats[i, :l] = f
                padded_genes[i, :l] = g
                padded_coords[i, :l] = c
                mask[i, :l] = False

            return padded_feats, padded_genes, padded_coords, mask

        return DataLoader(
            concat_ds,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=collate_fn_ws,
        )

    return DataLoader(
        concat_ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
    )
