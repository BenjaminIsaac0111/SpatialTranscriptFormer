import h5py
import os

file_path = r"A:\hest_data\st\MEND29.h5ad"

if not os.path.exists(file_path):
    print(f"File not found: {file_path}")
else:
    try:
        with h5py.File(file_path, "r") as f:
            print(f"Keys in {file_path}: {list(f.keys())}")

            def print_structure(name, obj):
                print(name)
                if isinstance(obj, h5py.Dataset):
                    print(f"  Shape: {obj.shape}")

            f.visititems(print_structure)

    except Exception as e:
        print(f"Error opening file: {e}")
