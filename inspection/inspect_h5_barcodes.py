import h5py
import numpy as np
import os
import argparse

def main():
    parser = argparse.ArgumentParser(description="Inspect barcodes in a HEST .h5 patch file.")
    parser.add_argument("--id", type=str, default="MEND29", help="Sample ID to inspect (default: MEND29)")
    parser.add_argument("--data_dir", type=str, help="Base directory for HEST data (optional)")
    args = parser.parse_args()

    sample_id = args.id
    
    # Try to find the file in multiple locations
    potential_dirs = []
    if args.data_dir:
        potential_dirs.append(args.data_dir)
    
    potential_dirs.extend([
        "hest_data",             # Local directory if run from project root
        "../hest_data",          # Local directory if run from inspection folder
        r"A:\hest_data"          # Hardcoded path on A: drive
    ])
    
    file_path = None
    for d in potential_dirs:
        # The newer download script places patches in a 'patches' subdirectory
        test_path = os.path.join(d, "patches", f"{sample_id}.h5")
        if os.path.exists(test_path):
            file_path = test_path
            break
        # Fallback to direct path in case it's in the root of data_dir
        test_path = os.path.join(d, f"{sample_id}.h5")
        if os.path.exists(test_path):
            file_path = test_path
            break

    if not file_path:
        print(f"Error: Could not find {sample_id}.h5 in any of the potential directories: {potential_dirs}")
        return

    print(f"Inspecting file: {file_path}")
    try:
        with h5py.File(file_path, 'r') as f:
            print(f"Keys: {list(f.keys())}")
            if 'barcode' not in f:
                print("Error: 'barcode' key not found.")
                return
                
            barcodes = f['barcode'][:]
            print(f"Barcodes shape: {barcodes.shape}")
            
            # Show first 5 barcodes, decoded if they are bytes
            print("First 5 barcodes:")
            for b in barcodes[:5]:
                # Barcodes can be (N, 1) or (N,)
                val = b[0] if hasattr(b, '__len__') and not isinstance(b, (bytes, str)) else b
                if isinstance(val, bytes):
                    print(f"  {val.decode('utf-8')}")
                else:
                    print(f"  {val}")
            
            # Check dtype
            print(f"Barcode dtype: {barcodes.dtype}")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
