import torch
import os
import h5py

pt_path = r'A:\hest_data\he_features\MEND29.pt'
h5ad_path = r'A:\hest_data\st\MEND29.h5ad'

print(f"Loading {pt_path}...")
data = torch.load(pt_path, map_location='cpu')
barcodes = data['barcodes']
print(f"Loaded {len(barcodes)} barcodes from PT.")
print("First 5 PT barcodes:", barcodes[:5])
print("Type:", type(barcodes[0]))

print(f"Loading {h5ad_path}...")
with h5py.File(h5ad_path, 'r') as f:
    if 'obs' in f and '_index' in f['obs']:
        st_barcodes = f['obs']['_index'][:]
    elif 'obs' in f and 'index' in f['obs']:
        st_barcodes = f['obs']['index'][:]
    else:
        print("Could not find barcodes in h5ad")
        exit()

st_barcodes = [b.decode('utf-8') if isinstance(b, bytes) else str(b) for b in st_barcodes]
print(f"Loaded {len(st_barcodes)} barcodes from H5AD.")
print("First 5 H5AD barcodes:", st_barcodes[:5])

# Check intersection
st_set = set(st_barcodes)
matches = sum(1 for b in barcodes if b in st_set)
print(f"Matches: {matches}")
