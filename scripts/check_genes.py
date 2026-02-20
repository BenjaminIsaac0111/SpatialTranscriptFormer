import json
import os
import sys
import numpy as np
import h5py
from scipy.sparse import csr_matrix

# Add src to path just in case
sys.path.append(os.path.abspath('src'))
from spatial_transcript_former.data.pathways import download_hallmarks_gmt, parse_gmt, build_membership_matrix
from spatial_transcript_former.data.dataset import load_global_genes

data_dir = r"A:\hest_data"
num_genes = 1000

# 1. Load global genes
genes = load_global_genes(data_dir, num_genes)
print(f"Loaded {len(genes)} global genes.")

# 2. Load pathways
gmt_path = download_hallmarks_gmt(os.path.join(data_dir, '.cache'))
pathway_dict = parse_gmt(gmt_path)

# 3. Check coverage
unique_hallmark_genes = set()
for p_genes in pathway_dict.values():
    unique_hallmark_genes.update(p_genes)

print(f"Unique genes in all 50 Hallmarks: {len(unique_hallmark_genes)}")

overlap = set(genes).intersection(unique_hallmark_genes)
print(f"Overlap with sum-top {num_genes} global genes: {len(overlap)} / {len(unique_hallmark_genes)} ({len(overlap)/len(unique_hallmark_genes)*100:.1f}%)")

# Pathway specific coverage
matrix, names = build_membership_matrix(pathway_dict, genes)
covered_per_pathway = matrix.sum(dim=1).numpy()
print("\nPathway coverage (genes present in top 1000):")
for name, count in zip(names, covered_per_pathway):
    print(f"{name}: {int(count)} / {len(pathway_dict[name])}")

# 4. Check expression sparsity
st_dir = os.path.join(data_dir, 'st')
test_file = os.path.join(st_dir, 'TENX156.h5ad')
if os.path.exists(test_file):
    print(f"\nAnalyzing expression in {test_file}")
    
    with h5py.File(test_file, 'r') as f:
        # Get matrix
        # Get gene names
        if 'var' in f and '_index' in f['var']:
            gene_names_raw = f['var']['_index'][:]
        elif 'var' in f and 'index' in f['var']:
            gene_names_raw = f['var']['index'][:]
        
        current_gene_names = [g.decode('utf-8') if isinstance(g, bytes) else str(g) for g in gene_names_raw]
        gene_name_to_idx = {name: i for i, name in enumerate(current_gene_names)}
        
        valid_indices = [gene_name_to_idx[g] for g in genes if g in gene_name_to_idx]
        print(f"\nFound {len(valid_indices)} of {num_genes} global genes in this sample.")
        
        X = f['X']
        if isinstance(X, h5py.Group):
            data = X['data'][:]
            indices = X['indices'][:]
            indptr = X['indptr'][:]
            
            num_rows = f['obs']['_index'].shape[0] if '_index' in f['obs'] else f['obs']['index'].shape[0]
            num_cols = len(current_gene_names)
            mat = csr_matrix((data, indices, indptr), shape=(num_rows, num_cols))
            
            mat_sub = mat[:, valid_indices]
            total_elements = mat_sub.shape[0] * mat_sub.shape[1]
            non_zeros = mat_sub.nnz
            sparsity = 1.0 - (non_zeros / total_elements)
            print(f"Sparsity in top {num_genes} global genes: {sparsity*100:.2f}%")
            
            # Print mean and max
            dense_sub = mat_sub.toarray()
            print(f"Mean expression: {dense_sub.mean():.4f}, Max: {dense_sub.max():.4f}")
            print(f"Variance: {dense_sub.var():.4f}")

