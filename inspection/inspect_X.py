import h5py
import os
import argparse

def main():
    parser = argparse.ArgumentParser(description="Inspect 'X' (gene expression) in a HEST .h5ad file.")
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
            if 'X' not in f:
                print("Error: 'X' not found in file.")
                return
            
            x_node = f['X']
            print(f"Type of X: {type(x_node)}")
            if isinstance(x_node, h5py.Dataset):
                print(f"X is a Dataset with shape: {x_node.shape}, dtype: {x_node.dtype}")
            elif isinstance(x_node, h5py.Group):
                print(f"X is a Group (Sparse Matrix format). Keys: {list(x_node.keys())}")
                for k in x_node.keys():
                    if hasattr(x_node[k], 'shape'):
                        print(f"  {k}: {x_node[k].shape}")
                    else:
                        print(f"  {k}: {type(x_node[k])}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
