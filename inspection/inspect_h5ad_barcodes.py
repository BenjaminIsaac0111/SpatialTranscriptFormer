import h5py
import numpy as np
import os
import argparse

def main():
    parser = argparse.ArgumentParser(description="Inspect barcodes in a HEST .h5ad ST file.")
    parser.add_argument("--id", type=str, default="MEND29", help="Sample ID to inspect (default: MEND29)")
    parser.add_argument("--data_dir", type=str, help="Base directory for HEST data (optional)")
    args = parser.parse_args()

    sample_id = args.id
    potential_dirs = []
    if args.data_dir:
        potential_dirs.append(args.data_dir)
    potential_dirs.extend(["hest_data", "../hest_data", r"A:\hest_data"])

    file_path = None
    for d in potential_dirs:
        # The newer download script places ST data in an 'st' subdirectory
        test_path = os.path.join(d, "st", f"{sample_id}.h5ad")
        if os.path.exists(test_path):
            file_path = test_path
            break
        # Fallback to direct path
        test_path = os.path.join(d, f"{sample_id}.h5ad")
        if os.path.exists(test_path):
            file_path = test_path
            break

    if not file_path:
        print(f"Error: Could not find {sample_id}.h5ad in any of {potential_dirs}")
        return

    print(f"Inspecting file: {file_path}")
    try:
        with h5py.File(file_path, 'r') as f:
            print(f"Keys: {list(f.keys())}")
            if 'obs' in f:
                target_key = None
                if '_index' in f['obs']:
                    target_key = 'obs/_index'
                elif 'index' in f['obs']:
                    target_key = 'obs/index'
                
                if target_key:
                    barcodes = f[target_key][:]
                    print(f"Obs index shape: {barcodes.shape}")
                    print("First 5 barcodes:")
                    for b in barcodes[:5]:
                        if isinstance(b, bytes):
                            print(f"  {b.decode('utf-8')}")
                        else:
                            print(f"  {b}")
                else:
                    print("No known index key found in obs.")
                    print(f"Keys in obs: {list(f['obs'].keys())}")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
