import pandas as pd
import h5py
import os
import numpy as np

def get_top_genes(h5ad_path, num_genes=1000):
    try:
        with h5py.File(h5ad_path, 'r') as f:
            # Load Gene Names
            if 'var' in f and '_index' in f['var']:
                gene_names = f['var']['_index'][:]
            elif 'var' in f and 'index' in f['var']:
                gene_names = f['var']['index'][:]
            else:
                return set()
            
            gene_names = [g.decode('utf-8') if isinstance(g, bytes) else str(g) for g in gene_names]
            
            # Load X to find top genes
            X = f['X']
            if isinstance(X, h5py.Group):
                # Sparse
                data = X['data'][:]
                indices = X['indices'][:]
                indptr = X['indptr'][:]
                
                # Simple sum per column
                sums = np.zeros(len(gene_names))
                for i in range(len(indptr)-1):
                    row_data = data[indptr[i]:indptr[i+1]]
                    row_indices = indices[indptr[i]:indptr[i+1]]
                    sums[row_indices] += row_data
            else:
                # Dense
                sums = np.sum(X[:], axis=0)
            
            top_indices = np.argsort(sums)[-num_genes:]
            return set([gene_names[i] for i in top_indices])
    except Exception as e:
        print(f"Error reading {h5ad_path}: {e}")
        return set()

def main():
    metadata_path = "HEST_v1_3_0.csv"
    if not os.path.exists(metadata_path):
        metadata_path = r"A:\hest_data\HEST_v1_3_0.csv"
    
    df = pd.read_csv(metadata_path)
    human_bowel = df[(df['organ'].str.lower() == 'bowel') & (df['species'] == 'Homo sapiens')]
    sample_ids = human_bowel['id'].tolist()
    
    st_dir = r"A:\hest_data\st"
    
    all_gene_sets = []
    processed_ids = []
    
    # Analyze a subset of samples to get an idea
    for sid in sample_ids[:10]:
        h5ad_path = os.path.join(st_dir, f"{sid}.h5ad")
        if os.path.exists(h5ad_path):
            print(f"Processing {sid}...")
            genes = get_top_genes(h5ad_path)
            if genes:
                all_gene_sets.append(genes)
                processed_ids.append(sid)
    
    if not all_gene_sets:
        print("No gene sets found.")
        return
    
    intersection = all_gene_sets[0]
    for gs in all_gene_sets[1:]:
        intersection = intersection.intersection(gs)
    
    union = set().union(*all_gene_sets)
    
    print(f"\nAnalysis of {len(processed_ids)} human bowel samples:")
    print(f"Total unique genes in top 1000 of all samples: {len(union)}")
    print(f"Genes common to top 1000 of ALL {len(processed_ids)} samples: {len(intersection)}")
    print(f"Common genes (first 50): {sorted(list(intersection))[:50]}")
    
    # Check total pool overlap
    total_intersection = all_gene_sets[0]
    # Re-fetch all genes (not just top 1000)
    all_genes_ref = []
    for sid in processed_ids:
        h5ad_path = os.path.join(st_dir, f"{sid}.h5ad")
        with h5py.File(h5ad_path, 'r') as f:
            if 'var' in f and '_index' in f['var']:
                g = f['var']['_index'][:]
            else:
                g = f['var']['index'][:]
            all_genes_ref.append(set([x.decode('utf-8') if isinstance(x, bytes) else str(x) for x in g]))
    
    global_intersection = all_genes_ref[0]
    for gs in all_genes_ref[1:]:
        global_intersection = global_intersection.intersection(gs)
    
    print(f"\nTotal genes available in ALL {len(processed_ids)} samples (entire pool): {len(global_intersection)}")
    print(f"First 20 available genes: {sorted(list(global_intersection))[:20]}")

    # Find genes common to top 2000 in at least 50% of samples
    # ...

if __name__ == "__main__":
    main()
