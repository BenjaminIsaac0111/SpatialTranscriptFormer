import os
import argparse
import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm
import json
from collections import defaultdict

def get_sample_ids(data_dir, filter_organ=None):
    """
    Find HEST sample IDs based on metadata. Similar to train.py but simplified.
    """
    patches_dir = os.path.join(data_dir, 'patches')
    search_dir = patches_dir if os.path.isdir(patches_dir) else data_dir

    all_files = [f for f in os.listdir(search_dir) if f.endswith('.h5')]
    all_ids = [f.replace('.h5', '') for f in all_files]
    
    if not all_ids:
        raise ValueError(f"No .h5 files found in {search_dir}")
        
    metadata_path = os.path.join(data_dir, "HEST_v1_3_0.csv")
    if os.path.exists(metadata_path):
        df = pd.read_csv(metadata_path)
        available_ids = set(all_ids)
        
        # Filter for existing files and Homo sapiens
        df_filtered = df[df['id'].isin(available_ids)]
        df_human = df_filtered[df_filtered['species'] == 'Homo sapiens']
        
        if filter_organ:
            print(f"Filtering for organ: {filter_organ}")
            df_final = df_human[df_human['organ'].str.contains(filter_organ, case=False, na=False)]
        else:
            df_final = df_human
            
        final_ids = df_final['id'].tolist()
        print(f"Found {len(final_ids)} samples matching criteria.")
    else:
        print("Metadata not found, using all files.")
        final_ids = all_ids

    return final_ids

def calculate_global_genes(data_dir, ids, num_genes=1000):
    st_dir = os.path.join(data_dir, 'st')
    if not os.path.exists(st_dir):
        st_dir = os.path.join(data_dir, '..', 'st') # Try parent/st if in patches

    print(f"Scanning {len(ids)} samples in {st_dir}...")
    
    gene_totals = defaultdict(float)
    gene_occurences = defaultdict(int)
    
    for sample_id in tqdm(ids):
        h5ad_path = os.path.join(st_dir, f"{sample_id}.h5ad")
        if not os.path.exists(h5ad_path):
            continue
            
        try:
            with h5py.File(h5ad_path, 'r') as f:
                # Load Gene Names
                if 'var' in f and '_index' in f['var']:
                     gene_names_raw = f['var']['_index'][:]
                elif 'var' in f and 'index' in f['var']:
                     gene_names_raw = f['var']['index'][:]
                else:
                    continue
                
                gene_names = [g.decode('utf-8') if isinstance(g, bytes) else str(g) for g in gene_names_raw]
                
                # Load Data (Sum)
                X = f['X']
                if isinstance(X, h5py.Group):
                    data = X['data'][:]
                    indices = X['indices'][:]
                    indptr = X['indptr'][:]
                    # We want column sums. 
                    # CSR matrix sum(axis=0) is efficient.
                    # But implementing CSR sum manually without scipy potentially?
                    # Let's import scipy, it's installed.
                    from scipy.sparse import csr_matrix
                    n_obs = f['obs']['_index'].shape[0] if '_index' in f['obs'] else f['obs']['index'].shape[0]
                    n_vars = len(gene_names)
                    mat = csr_matrix((data, indices, indptr), shape=(n_obs, n_vars))
                    sums = np.array(mat.sum(axis=0)).flatten()
                elif isinstance(X, h5py.Dataset):
                    mat = X[:]
                    sums = np.sum(mat, axis=0)
                
                for i, gene in enumerate(gene_names):
                    gene_totals[gene] += float(sums[i])
                    gene_occurences[gene] += 1
                    
        except Exception as e:
            print(f"Error processing {sample_id}: {e}")
            
    print(f"Aggregated counts for {len(gene_totals)} unique genes.")
    
    # Sort by total expression
    sorted_genes = sorted(gene_totals.items(), key=lambda x: x[1], reverse=True)
    
    top_genes = [g[0] for g in sorted_genes[:num_genes]]
    
    return top_genes, sorted_genes

def main():
    parser = argparse.ArgumentParser(description="Calculate Global Top Genes")
    parser.add_argument('--data-dir', type=str, required=True)
    parser.add_argument('--organ', type=str, default=None, help="Filter by organ (e.g., Bowel)")
    parser.add_argument('--num-genes', type=int, default=1000)
    parser.add_argument('--output', type=str, default='global_genes.json')
    
    args = parser.parse_args()
    
    ids = get_sample_ids(args.data_dir, args.organ)
    top_genes, all_stats = calculate_global_genes(args.data_dir, ids, args.num_genes)
    
    print(f"Saving top {len(top_genes)} genes to {args.output}")
    with open(args.output, 'w') as f:
        json.dump(top_genes, f, indent=4)
        
    # Optional: Save stats
    stats_df = pd.DataFrame(all_stats, columns=['gene', 'total_counts'])
    stats_df.to_csv(args.output.replace('.json', '_stats.csv'), index=False)
    print("Saved stats to CSV.")

if __name__ == "__main__":
    main()
