import os
import h5py
import numpy as np

data_dir = r"A:\hest_data\st"
files = [f for f in os.listdir(data_dir) if f.endswith(".h5ad")]
if not files:
    print("No .h5ad files found")
    exit()

sample_path = os.path.join(data_dir, files[0])
print(f"Inspecting: {sample_path}")

with h5py.File(sample_path, "r") as f:
    if "X" in f:
        X = f["X"]
        if isinstance(X, h5py.Group):
            data = X["data"][:1000]  # Sample
        else:
            data = X[:1000].flatten()  # Sample

        print(f"Data Type: {data.dtype}")
        print(f"Min: {data.min()}")
        print(f"Max: {data.max()}")
        print(f"Mean: {data.mean()}")
        print(f"Integers only? {np.all(np.mod(data, 1) == 0)}")
    else:
        print("No X found")
