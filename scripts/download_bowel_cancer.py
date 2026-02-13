import pandas as pd
import os
import sys

# Ensure we can import from src
# Assuming this script is run from project root, or we add src to path
# If run from scripts/ directory:
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
src_path = os.path.join(project_root, 'src')
if src_path not in sys.path:
    sys.path.append(src_path)

from spatial_transcript_former.data import download_hest_subset, download_metadata

# Define the local directory for downloads
local_dir = r'A:\hest_data'

def main():
    # Ensure metadata is present
    try:
        metadata_path = download_metadata(local_dir)
    except Exception as e:
        print(f"Error downloading metadata: {e}")
        return

    meta_df = pd.read_csv(metadata_path)
    
    # Filter for bowel cancer samples
    # Based on research, 'Bowel' is the primary organ name in the dataset
    bowel_df = meta_df[meta_df['organ'].str.lower() == 'bowel']
    
    ids_to_query = bowel_df['id'].tolist()
    print(f"Found {len(ids_to_query)} samples for Bowel.")
    
    if not ids_to_query:
        print("No samples found for 'Bowel' organ.")
        return

    print(f"Starting download to '{local_dir}'...")
    download_hest_subset(ids_to_query, local_dir)

if __name__ == "__main__":
    main()
