import h5py
import os

file_path = r"A:\hest_data\st\MISC54.h5ad"

if not os.path.exists(file_path):
    print(f"File not found: {file_path}")
else:
    try:
        with h5py.File(file_path, 'r') as f:
            print(f"Keys in {file_path}: {list(f.keys())}")
            if 'var' in f:
                if '_index' in f['var']:
                    print(f"Var index shape: {f['var']['_index'].shape}")
                
            # check ref sample
            ref_path = r"A:\hest_data\st\MEND29.h5ad" # MEND29 was used in test? Or MEND92
            with h5py.File(ref_path, 'r') as f2:
                 print(f"MEND29 Var index shape: {f2['var']['_index'].shape}")

    except Exception as e:
        print(f"Error opening file: {e}")
