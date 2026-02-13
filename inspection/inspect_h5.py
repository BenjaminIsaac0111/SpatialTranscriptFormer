import h5py
import os

file_path = r"A:\hest_data\patches\MEND29.h5"

if not os.path.exists(file_path):
    print(f"File not found: {file_path}")
else:
    try:
        with h5py.File(file_path, 'r') as f:
            print(f"Keys in {file_path}: {list(f.keys())}")
            for key in f.keys():
                print(f"  Shape of {key}: {f[key].shape}")
    except Exception as e:
        print(f"Error opening file: {e}")
