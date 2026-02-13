import pandas as pd
import h5py
import os

def main():
    df = pd.read_csv('HEST_v1_3_0.csv')
    hb_v = df[(df['organ'].str.lower() == 'bowel') & 
              (df['species'] == 'Homo sapiens') & 
              (df['st_technology'].str.contains('Visium'))]
    ids = hb_v['id'].tolist()
    
    st_dir = r"A:\hest_data\st"
    gene_sets = []
    processed_ids = []
    
    for sid in ids:
        p = os.path.join(st_dir, sid + '.h5ad')
        if os.path.exists(p):
            with h5py.File(p, 'r') as f:
                if 'var' in f:
                    g = f['var']['_index'][:] if '_index' in f['var'] else f['var']['index'][:]
                    gene_sets.append(set([x.decode('utf-8') if isinstance(x, bytes) else str(x) for x in g]))
                    processed_ids.append(sid)
    
    if not gene_sets:
        print("No Visium samples found.")
        return
        
    common = gene_sets[0]
    for gs in gene_sets[1:]:
        common = common.intersection(gs)
        
    print(f"Analysis of {len(processed_ids)} Human Bowel Visium samples:")
    print(f"Total unique genes across all: {len(set().union(*gene_sets))}")
    print(f"Common genes (entire pool): {len(common)}")
    
    # Check top 1000 overlap
    # (Just reusing the idea from before but simplified)
    
if __name__ == "__main__":
    main()
