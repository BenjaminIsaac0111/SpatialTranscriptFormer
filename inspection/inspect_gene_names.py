import h5py
import numpy as np

file_path = r"A:\hest_data\st\MEND29.h5ad"

try:
    with h5py.File(file_path, "r") as f:
        if "var" in f and "_index" in f["var"]:
            gene_names = f["var"]["_index"][:]
            print(f"First 5 gene names in MEND29: {gene_names[:5]}")
            print(f"Type: {gene_names.dtype}")

    file_path_2 = r"A:\hest_data\st\MISC54.h5ad"
    with h5py.File(file_path_2, "r") as f:
        if "var" in f and "_index" in f["var"]:
            gene_names = f["var"]["_index"][:]
            print(f"First 5 gene names in MISC54: {gene_names[:5]}")

except Exception as e:
    print(f"Error: {e}")
