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

def calculate_global_genes(data_dir, ids, num_genes=1000, target_pathways=None):
    st_dir = os.path.join(data_dir, 'st')
    if not os.path.exists(st_dir):
        st_dir = os.path.join(data_dir, '..', 'st') # Try parent/st if in patches

    print(f"Scanning {len(ids)} samples in {st_dir}...")
    
    gene_totals = defaultdict(float)
    
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
                    
        except Exception as e:
            print(f"Error processing {sample_id}: {e}")
            
    print(f"Aggregated counts for {len(gene_totals)} unique genes.")
    
    prioritized_genes = set()
    if target_pathways:
        from spatial_transcript_former.data.pathways import download_msigdb_gmt, parse_gmt, MSIGDB_URLS
        print(f"Prioritizing genes from pathways: {target_pathways}")
        
        # Load C2 KEGG first
        kegg_path = download_msigdb_gmt(MSIGDB_URLS['c2_kegg'], MSIGDB_URLS['c2_kegg'].split('/')[-1], os.path.join(data_dir, '.cache'))
        dict_kegg = parse_gmt(kegg_path)
        
        kegg_med_path = download_msigdb_gmt(MSIGDB_URLS['c2_medicus'], MSIGDB_URLS['c2_medicus'].split('/')[-1], os.path.join(data_dir, '.cache'))
        dict_med = parse_gmt(kegg_med_path)
        
        # Load C2 CGP
        cgp_path = download_msigdb_gmt(MSIGDB_URLS['c2_cgp'], MSIGDB_URLS['c2_cgp'].split('/')[-1], os.path.join(data_dir, '.cache'))
        dict_cgp = parse_gmt(cgp_path)
        
        # Also load hallmarks just in case
        h_path = download_msigdb_gmt(MSIGDB_URLS['hallmarks'], 'h.all.v2024.1.Hs.symbols.gmt', os.path.join(data_dir, '.cache'))
        dict_h = parse_gmt(h_path)
        
        combined_dict = {**dict_kegg, **dict_med, **dict_cgp, **dict_h}
        
        for p in target_pathways:
            if p in combined_dict:
                for pw_gene in combined_dict[p]:
                    if pw_gene in gene_totals:
                        prioritized_genes.add(pw_gene)
            else:
                print(f"Warning: Pathway {p} not found in MSIGDB dictionaries.")
                
        print(f"Found {len(prioritized_genes)} valid target pathway genes.")

    # Sort all by total expression
    sorted_all = sorted(gene_totals.items(), key=lambda x: x[1], reverse=True)
    
    top_genes = list(prioritized_genes)
    for g, _ in sorted_all:
        if len(top_genes) >= num_genes:
            break
        if g not in prioritized_genes:
            top_genes.append(g)
            
    print(f"Final set: {len(prioritized_genes)} pathway genes + {len(top_genes) - len(prioritized_genes)} global genes")
    
    return top_genes, sorted_all

def main():
    parser = argparse.ArgumentParser(description="Calculate Global Top Genes")
    parser.add_argument('--data-dir', type=str, required=True)
    parser.add_argument('--organ', type=str, default=None, help="Filter by organ (e.g., Bowel)")
    parser.add_argument('--num-genes', type=int, default=1000)
    parser.add_argument('--output', type=str, default='global_genes.json')
    parser.add_argument('--pathways', nargs='+', default=None, help='List of MSigDB pathway names to explicitly include')
    
    args = parser.parse_args()
    
    # Must add src to path to import MSIGDB logic
    import sys
    sys.path.append(os.path.abspath('src'))
    
    ids = get_sample_ids(args.data_dir, args.organ)
    top_genes, all_stats = calculate_global_genes(args.data_dir, ids, args.num_genes, target_pathways=args.pathways)
    
    print(f"Saving top {len(top_genes)} genes to {args.output}")
    with open(args.output, 'w') as f:
        json.dump(top_genes, f, indent=4)
        
    stats_df = pd.DataFrame(all_stats, columns=['gene', 'total_counts'])
    stats_df.to_csv(args.output.replace('.json', '_stats.csv'), index=False)
    print("Saved stats to CSV.")

if __name__ == "__main__":
    main()
