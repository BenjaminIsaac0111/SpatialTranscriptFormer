import os
import h5py
import json
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from scipy.sparse import csr_matrix
from typing import List, Optional, Tuple, Union
from scipy.spatial import KDTree

class HEST_Dataset(Dataset):
    def __init__(self, h5_path: str, spatial_coords: np.ndarray, 
                 gene_matrix: np.ndarray, indices: Optional[np.ndarray] = None, 
                 transform=None, neighborhood_indices: Optional[np.ndarray] = None,
                 coords_all: Optional[np.ndarray] = None):
        """
        Args:
            h5_path (str): Path to the HEST .h5 file containing patches.
            spatial_coords (array): (N_subset, 2) array of (x,y) coordinates of center patches.
            gene_matrix (array): (N_subset, n_genes) array of gene counts.
            indices (array, optional): (N_subset,) array mapping dataset index to file index.
            transform (callable, optional): Optional transform.
            neighborhood_indices (array, optional): (N_subset, K) indices of neighbors in the file.
            coords_all (array, optional): (N_all, 2) all coordinates in the file for relative distance calculation.
        """
        self.h5_path = h5_path
        self.transform = transform
        self.coords = spatial_coords
        self.genes = gene_matrix
        self.indices = indices
        self.neighborhood_indices = neighborhood_indices
        self.coords_all = coords_all
        
        # Lazy loading handles for file objects
        self.h5_file = None

    def __len__(self):
        return len(self.coords)

    def __getitem__(self, idx):
        if self.h5_file is None:
            # Lazy loading inside the worker
            self.h5_file = h5py.File(self.h5_path, 'r')
            
        # Map idx to file_idx for center patch
        center_file_idx = idx
        if self.indices is not None:
            center_file_idx = self.indices[idx]
            
        # Load patches
        if self.neighborhood_indices is not None:
            # We have neighbors! neighbor_indices[idx] is a list of k file_indices
            neighbor_file_indices = self.neighborhood_indices[idx]
            
            # Sequence of patches: [center, n1, n2, ..., nk]
            patch_sequence = []
            file_indices = [center_file_idx] + list(neighbor_file_indices)
            
            for f_idx in file_indices:
                p = self.h5_file['img'][f_idx]
                p = torch.from_numpy(p).permute(2, 0, 1).float() / 255.0
                if self.transform:
                    p = self.transform(p)
                patch_sequence.append(p)
            
            data = torch.stack(patch_sequence) # (1+K, 3, H, W)
            
            # Relative coordinates for masking/positional encoding
            if self.coords_all is not None:
                center_coord = self.coords[idx]
                neighbor_coords = self.coords_all[neighbor_file_indices]
                rel_coords = np.concatenate([[[0, 0]], neighbor_coords - center_coord], axis=0)
                rel_coords = torch.from_numpy(rel_coords).float()
            else:
                rel_coords = torch.zeros((len(file_indices), 2))
                
        else:
            # Standard single-patch mode
            patch = self.h5_file['img'][center_file_idx] 
            patch = torch.from_numpy(patch).permute(2, 0, 1).float() / 255.0
            if self.transform:
                patch = self.transform(patch)
            data = patch # (3, H, W)
            rel_coords = torch.zeros((1, 2))
        
        # Get Genes
        gene_counts = self.genes[idx]
        gene_counts = torch.tensor(gene_counts, dtype=torch.float32)
        
        return data, gene_counts, rel_coords

    def __del__(self):
        if self.h5_file:
            self.h5_file.close()

def load_gene_expression_matrix(h5ad_path: str, patch_barcodes: List[bytes], selected_gene_names: Optional[List[str]] = None, num_genes: int = 1000):
    """
    Loads gene expression from .h5ad file.
    If selected_gene_names is None, selects top `num_genes` matches and returns them + their names.
    If selected_gene_names is provided, aligns current file to those genes (filling missing with 0).
    
    Returns:
        dense_matrix (np.ndarray): (N_patches, num_genes)
        valid_mask (np.ndarray): (N_patches,) boolean mask of found barcodes
        selected_names (List[str]): Names of genes used
    """
    with h5py.File(h5ad_path, 'r') as f:
        # Load barcodes from obs/_index
        if 'obs' in f and '_index' in f['obs']:
            st_barcodes = f['obs']['_index'][:]
        elif 'obs' in f and 'index' in f['obs']:
             st_barcodes = f['obs']['index'][:]
        else:
            raise ValueError(f"Could not find barcodes in {h5ad_path}")
            
        # Decode barcodes
        st_barcodes = [b.decode('utf-8') if isinstance(b, bytes) else str(b) for b in st_barcodes]
        st_barcode_to_idx = {b: i for i, b in enumerate(st_barcodes)}
        
        # Load Gene Names (var index)
        if 'var' in f and '_index' in f['var']:
             gene_names_raw = f['var']['_index'][:]
        elif 'var' in f and 'index' in f['var']:
             gene_names_raw = f['var']['index'][:]
        else:
             # Fallback: create dummy names? No, critical failure.
             raise ValueError(f"Could not find gene names (var index) in {h5ad_path}")
             
        # Decode gene names
        current_gene_names = [g.decode('utf-8') if isinstance(g, bytes) else str(g) for g in gene_names_raw]
        # Map name -> index for fast lookup
        gene_name_to_idx = {name: i for i, name in enumerate(current_gene_names)}

        # Find indices for patch barcodes
        patch_indices = []
        valid_patch_mask = [] 
        
        patch_barcodes_decoded = [b.decode('utf-8') if isinstance(b, bytes) else str(b) for b in patch_barcodes]
        
        for pb in patch_barcodes_decoded:
            if pb in st_barcode_to_idx:
                patch_indices.append(st_barcode_to_idx[pb])
                valid_patch_mask.append(True)
            else:
                patch_indices.append(0) 
                valid_patch_mask.append(False)
        
        valid_patch_mask = np.array(valid_patch_mask)
        patch_indices_array = np.array(patch_indices)
        
        # Load Data
        X = f['X']
        # Convert to CSR for flexible slicing if it's a Group
        if isinstance(X, h5py.Group):
            data = X['data'][:]
            indices = X['indices'][:]
            indptr = X['indptr'][:]
            n_obs = len(st_barcodes)
            n_vars = len(current_gene_names)
            mat = csr_matrix((data, indices, indptr), shape=(n_obs, n_vars))
        elif isinstance(X, h5py.Dataset):
            mat = X[:] # Dense load, hoping it fits in memory
            # If dense, convert to CSR? Or keep dense.
            # Convert to CSR for uniform handling of gene alignment later?
            # Actually if we just use fancy indexing, dense is fine.
        else:
             raise ValueError("Unknown X format")

        # Slice Rows (valid patches only)
        valid_indices = patch_indices_array[valid_patch_mask]
        
        if isinstance(mat, csr_matrix):
            mat_subset = mat[valid_indices]
        else:
            mat_subset = mat[valid_indices]

        # Determine Target Genes
        if selected_gene_names is None:
            # Select Top N genes from THIS sample
            if isinstance(mat_subset, csr_matrix):
                gene_sums = np.array(mat_subset.sum(axis=0)).flatten()
            else:
                gene_sums = np.sum(mat_subset, axis=0)
                
            top_indices = np.argsort(gene_sums)[-num_genes:][::-1]
            selected_names = [current_gene_names[i] for i in top_indices]
            
            # Slice columns
            final_subset_raw = mat_subset[:, top_indices]
            if isinstance(final_subset_raw, csr_matrix):
                final_subset_raw = final_subset_raw.toarray()
            
            # Ensure exactly num_genes columns (pad with zeros if sample is too small)
            n_found = final_subset_raw.shape[1]
            if n_found < num_genes:
                final_subset = np.zeros((final_subset_raw.shape[0], num_genes), dtype=np.float32)
                final_subset[:, :n_found] = final_subset_raw
                # Pad gene names too
                selected_names += [f"pad_gene_{j}" for j in range(num_genes - n_found)]
            else:
                final_subset = final_subset_raw
                
        else:
            # ALIGNMENT LOGIC
            selected_names = selected_gene_names
            target_indices = []
            zeros_mask = [] # Indices in the OUTPUT vector that need to be zero (missing gene)
            
            # For each target gene, find its index in current file
            valid_src_indices = []
            valid_dst_indices = []
            
            for i, name in enumerate(selected_names):
                if name in gene_name_to_idx:
                    valid_src_indices.append(gene_name_to_idx[name])
                    valid_dst_indices.append(i)
            
            # Create the final container
            n_valid_patches = len(valid_indices)
            final_subset = np.zeros((n_valid_patches, num_genes), dtype=np.float32)
            
            if valid_src_indices:
                # Extract valid columns
                if isinstance(mat_subset, csr_matrix):
                     # slicing CSR with list of columns
                     cols_data = mat_subset[:, valid_src_indices].toarray()
                else:
                     cols_data = mat_subset[:, valid_src_indices]
                
                # Place into final container
                final_subset[:, valid_dst_indices] = cols_data
                
        return final_subset, valid_patch_mask, selected_names


def load_global_genes(root_dir: str, num_genes: int = 1000) -> List[str]:
    """
    Load global gene list from global_genes.json.
    Checks in root_dir and CWD.

    Args:
        root_dir (str): Root directory to search for global_genes.json.
        num_genes (int): Number of genes to return.

    Returns:
        List[str]: List of gene names.
    
    Raises:
        FileNotFoundError: If global_genes.json is not found.
        RuntimeError: If there's an error reading the file.
    """
    global_genes_path = os.path.join(root_dir, 'global_genes.json')
    if not os.path.exists(global_genes_path):
        global_genes_path = 'global_genes.json' # Check CWD
    
    if os.path.exists(global_genes_path):
        try:
            with open(global_genes_path, 'r') as f:
                genes = json.load(f)
            if len(genes) > num_genes:
                genes = genes[:num_genes]
            print(f"Loaded {len(genes)} global genes from {global_genes_path}")
            return genes
        except Exception as e:
            raise RuntimeError(f"Error loading global genes from {global_genes_path}: {e}")
    else:
        raise FileNotFoundError(
            f"global_genes.json not found in {root_dir} or current directory. "
            "Please ensure global_genes.json exists to maintain consistent gene representation for the entire data."
        )

def get_hest_dataloader(
    root_dir: str,
    ids: List[str],
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
    transform = None,
    num_genes: int = 1000,
    n_neighbors: int = 0
):
    datasets = []
    
    patches_dir = os.path.join(root_dir, 'patches')
    st_dir = os.path.join(root_dir, 'st')
    
    if not os.path.exists(patches_dir):
        patches_dir = root_dir
        st_dir = os.path.join(os.path.dirname(root_dir), 'st')
        
    common_gene_names = load_global_genes(root_dir, num_genes)
    
    for i, sample_id in enumerate(ids):
        h5_path = os.path.join(patches_dir, f"{sample_id}.h5") 
        h5ad_path = os.path.join(st_dir, f"{sample_id}.h5ad")
        
        if not os.path.exists(h5_path):
            continue
        if not os.path.exists(h5ad_path):
             print(f"Warning: ST data for {sample_id} not found at {h5ad_path}")
             continue
            
        try:
             with h5py.File(h5_path, 'r') as f:
                 if 'coords' not in f or 'img' not in f:
                     print(f"Warning: {sample_id}.h5 missing coords or img")
                     continue
                 
                 patch_barcodes = f['barcode'][:].flatten()
                 coords_all = f['coords'][:]
                 
                 # Load and align genes
                 gene_matrix, mask, selected_names = load_gene_expression_matrix(
                     h5ad_path, patch_barcodes, selected_gene_names=common_gene_names, num_genes=num_genes
                 )
                 
                 
                 coords_subset = coords_all[mask]
                 indices_subset = np.where(mask)[0]
                 
                 neighborhood_indices = None
                 if n_neighbors > 0:
                     print(f"Pre-computing {n_neighbors} neighbors for {sample_id}...")
                     tree = KDTree(coords_all)
                     # Find k+1 because query point is usually in the tree
                     # dists, idxs = tree.query(coords_subset, k=n_neighbors + 1)
                     # Often the first index is the point itself. Let's be explicit.
                     dists, idxs = tree.query(coords_subset, k=n_neighbors + 1)
                     
                     # Extract true neighbors (excluding the point itself by looking at indices)
                     final_neighbors = []
                     for i, center_idx in enumerate(indices_subset):
                         n_idxs = idxs[i]
                         # Filter out the center_idx if it exists in n_idxs
                         n_idxs = n_idxs[n_idxs != center_idx]
                         # Take top n_neighbors
                         current_neighbors = n_idxs[:n_neighbors]
                         # Pad if too few neighbors
                         if len(current_neighbors) < n_neighbors:
                             padding = [center_idx] * (n_neighbors - len(current_neighbors))
                             current_neighbors = np.concatenate([current_neighbors, padding]).astype(np.int64)
                         final_neighbors.append(current_neighbors)
                     
                     neighborhood_indices = np.array(final_neighbors)
                 
                 ds = HEST_Dataset(h5_path, coords_subset, gene_matrix, 
                                  indices=indices_subset, transform=transform,
                                  neighborhood_indices=neighborhood_indices,
                                  coords_all=coords_all)
                 datasets.append(ds)
                 
        except Exception as e:
            print(f"Error loading {sample_id}: {e}")
            continue
            
    if not datasets:
        raise ValueError("No valid datasets found.")
        
    concat_ds = ConcatDataset(datasets)
    loader = DataLoader(concat_ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    
    return loader

    loader = DataLoader(concat_ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return loader

class HEST_FeatureDataset(Dataset):
    """
    Dataset for Pre-Computed Features.
    Serves (Center + Neighbors + Global Context) features for a single patch.
    
    Args:
        feature_path (str): Path to the .pt file containing features, coords, barcodes.
        h5ad_path (str): Path to the .h5ad file containing gene expression data.
        num_genes (int): Number of genes to select/align to.
        selected_gene_names (Optional[List[str]]): Pre-defined list of gene names for alignment.
        n_neighbors (int): Number of neighbors to retrieve for each patch.
        use_global_context (bool): Whether to include global context features.
        global_context_size (int): Number of patches to sample for global context.
        whole_slide_mode (bool): If True, __getitem__ returns the whole slide data.
        augment (bool): Whether to apply data augmentation.
        log1p (bool): Whether to log1p transform the gene expression counts.
    """
    def __init__(self, feature_path: str, h5ad_path: str, num_genes: int = 1000, 
                 selected_gene_names: Optional[List[str]] = None, n_neighbors: int = 6,
                 use_global_context: bool = False,
                 global_context_size: int = 256,
                 whole_slide_mode: bool = False,
                 augment: bool = False,
                 log1p: bool = False):
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
        
        # Load Data
        if not self.whole_slide_mode:
            self._load_data()
        else:
            # In whole_slide_mode, we delay loading until __getitem__
            self.features = None
            self.coords = None
            self.genes = None
            self.kdtree = None

    def _load_data(self):
        saved_data = torch.load(self.feature_path, map_location='cpu', weights_only=True)
        features = saved_data['features'] # (N, D)
        coords = saved_data['coords'] # (N, 2)
        barcodes = saved_data['barcodes']
        
        # Load Genes
        gene_matrix, mask, selected_names = load_gene_expression_matrix(
            self.h5ad_path, barcodes, selected_gene_names=self.selected_gene_names, num_genes=self.num_genes
        )
        
        if self.log1p:
            gene_matrix = np.log1p(gene_matrix)
        
        if self.selected_gene_names is None:
            self.selected_gene_names = selected_names
            
        # Filter valid patches
        mask_bool = np.array(mask, dtype=bool)
        
        self.features = features[mask_bool] # (N_valid, D)
        self.coords = coords[mask_bool] # (N_valid, 2)
        self.genes = torch.tensor(gene_matrix, dtype=torch.float32) # (N_valid, G)
        
        # Build KDTree for neighbors
        # Use simple KDTree on CPU
        if not self.whole_slide_mode:
            self.kdtree = KDTree(self.coords.numpy())

        
    def __len__(self):
        if self.whole_slide_mode:
            return 1
        return len(self.coords)

    def __getitem__(self, idx):
        if self.whole_slide_mode:
            # Strictly load-on-demand to save RAM.
            # We do NOT cache in self.features.
            
            saved_data = torch.load(self.feature_path, map_location='cpu', weights_only=True)
            features = saved_data['features']
            coords = saved_data['coords']
            barcodes = saved_data['barcodes']
            
            gene_matrix, mask, _ = load_gene_expression_matrix(
                self.h5ad_path, barcodes, selected_gene_names=self.selected_gene_names, num_genes=self.num_genes
            )
            
            if self.log1p:
                gene_matrix = np.log1p(gene_matrix)

            mask_bool = np.array(mask, dtype=bool)
            
            feats = features[mask_bool]
            co = coords[mask_bool]
            g = torch.tensor(gene_matrix, dtype=torch.float32)
            
            # Cleanup raw load
            del saved_data
            del gene_matrix
            
            return feats, g, co
            
        # 1. Get Neighbors
            
        # 1. Get Neighbors
        dists, neighbor_idxs = self.kdtree.query(self.coords[idx], k=self.n_neighbors + 1)
        
        if self.n_neighbors == 0:
            dists = np.array([dists])
            neighbor_idxs = np.array([neighbor_idxs])
            
        # Robust check for self-inclusion
        if neighbor_idxs[0] != idx:
            pass
            
        # Pad if needed
        if len(self.coords) < self.n_neighbors + 1:
             pad_len = (self.n_neighbors + 1) - len(self.coords)
             neighbor_idxs = list(range(len(self.coords))) + [idx] * pad_len
             neighbor_idxs = np.array(neighbor_idxs)

        # 2. Get Global Context (if enabled)
        global_idxs = []
        if self.use_global_context:
            # Sample random indices from the whole slide
            total_patches = len(self.coords)
            if total_patches > 0:
                # Random replacement sampling is faster and fine for context
                global_idxs = np.random.choice(total_patches, size=self.global_context_size, replace=True)
            else:
                global_idxs = np.array([idx] * self.global_context_size)

        # 3. Combine Indices
        # Format: [Self, Neighbor1, ..., NeighborK, Global1, ..., GlobalM]
        # Ensure we flatten everything to 1D array of indices
        if self.use_global_context:
            combined_idxs = np.concatenate([neighbor_idxs, global_idxs])
        else:
            combined_idxs = neighbor_idxs
            
        # Get Features
        feats = self.features[combined_idxs]
        
        # 4. Interaction Augmentations
        if self.augment and not self.whole_slide_mode:
            # A. Neighborhood Dropout: Randomly zero out 1-2 neighbors
            # Index 0 is 'Self', neighbors are 1:n_neighbors+1
            if self.n_neighbors > 1:
                # We don't drop 'Self' (index 0)
                n_to_drop = np.random.randint(0, min(3, self.n_neighbors))
                if n_to_drop > 0:
                    drop_idxs = np.random.choice(range(1, self.n_neighbors + 1), size=n_to_drop, replace=False)
                    # We can "zero out" by replacing with the center patch feature 
                    # or literally zeros. Replacing with center patch (Self) is often better 
                    # than zeros as it doesn't break distribution as much.
                    # Or we just zero them. Let's use zeros for clear signal.
                    feats[drop_idxs] = 0.0

        # Get Relative Coords
        center_coord = self.coords[idx]
        nbr_coords = self.coords[combined_idxs]
        rel_coords = nbr_coords - center_coord # (S, 2)

        if self.augment and not self.whole_slide_mode:
            # B. Coordinate Jitter: Add small Gaussian noise
            # Typical HEST coordinates are in pixels (e.g. 0 to 10000)
            # A jitter of 5-10 pixels is subtle but effective.
            jitter = torch.randn_like(rel_coords) * 5.0
            # Don't jitter the center (index 0)
            jitter[0] = 0.0
            rel_coords = rel_coords + jitter
            
        # Get Target (Center Patch Gene Expression)
        target_genes = self.genes[idx]
        
        return feats, target_genes, rel_coords


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
    feature_dir: Optional[str] = None
):
    """
    Returns a DataLoader.
    If whole_slide_mode=True, each item is (N_patches, D), (N_patches, G), (N_patches, 2).
    Else, each item is (Sequence, Target, RelCoords).
    
    Args:
        root_dir (str): Root directory.
        ids (List[str]): List of sample IDs.
        batch_size (int): Batch size.
        shuffle (bool): Shuffle.
        num_workers (int): Workers.
        num_genes (int): Number of genes.
        n_neighbors (int): Number of neighbors.
        use_global_context (bool): Global context.
        global_context_size (int): Size.
        whole_slide_mode (bool): Whole slide.
        augment (bool): Data augmentation.
        log1p (bool): Log transform targets.
        feature_dir (Optional[str]): Feature directory.
    """
    datasets = []
    
    # Check paths
    if feature_dir is None:
        features_dir = os.path.join(root_dir, 'he_features')
        if not os.path.exists(features_dir):
            features_dir = root_dir
    else:
        features_dir = feature_dir
        
    st_dir = os.path.join(root_dir, 'st')
        
    # 1. Load common genes
    common_gene_names = load_global_genes(root_dir, num_genes)
        
    # 2. Create Datasets
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
                log1p=log1p
            )
            datasets.append(ds)
            
    if not datasets:
        raise ValueError("No valid feature datasets found.")
        
    concat_ds = ConcatDataset(datasets)
    
    # If whole_slide_mode is True, we must use batch_size=1 and custom collate
    if whole_slide_mode:
        # We now support batch_size > 1 with padding
            
        def collate_fn_ws(batch):
            """
            Pads sequences to the longest in the batch.
            Batch items: (feats, genes, coords)
            feats: (N, D)
            genes: (N, G)
            coords: (N, 2)
            """
            # Find max length
            lengths = [item[0].shape[0] for item in batch]
            max_len = max(lengths)
            
            # Feature dim
            d_dim = batch[0][0].shape[1]
            g_dim = batch[0][1].shape[1]
            
            # Prepare padded tensors
            batch_size = len(batch)
            padded_feats = torch.zeros(batch_size, max_len, d_dim)
            padded_genes = torch.zeros(batch_size, max_len, g_dim)
            padded_coords = torch.zeros(batch_size, max_len, 2)
            
            # Create mask (True = Padding)
            mask = torch.ones(batch_size, max_len, dtype=torch.bool)
            
            for i, (f, g, c) in enumerate(batch):
                l = lengths[i]
                padded_feats[i, :l, :] = f
                padded_genes[i, :l, :] = g
                padded_coords[i, :l, :] = c
                mask[i, :l] = False # Valid data is False, Padding is True
                
            return padded_feats, padded_genes, padded_coords, mask
            
        loader = DataLoader(concat_ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=collate_fn_ws)
    else:
        # Standard DataLoader (batching logic applies)
        loader = DataLoader(concat_ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    
    return loader
