import os
from spatial_transcript_former.data.dataset import get_hest_dataloader
import torch

data_dir = r"A:\hest_data"
# Use a sample ID we know exists
sample_ids = ["MEND29"] # Start with just one

print(f"Testing DataLoader with ID: {sample_ids}")

try:
    loader = get_hest_dataloader(data_dir, sample_ids, batch_size=4, num_genes=100)
    print(f"DataLoader created with {len(loader)} batches.")
    
    for i, (images, targets) in enumerate(loader):
        print(f"Batch {i}:")
        print(f"  Images shape: {images.shape}")
        print(f"  Targets shape: {targets.shape}")
        print(f"  Target range: {targets.min()} - {targets.max()}")
        if i >= 0: break # Just one batch

except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
