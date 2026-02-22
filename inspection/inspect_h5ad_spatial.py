import h5py
import numpy as np


def inspect_spatial(h5ad_path):
    print(f"Inspecting {h5ad_path}...")
    with h5py.File(h5ad_path, "r") as f:
        if "uns" not in f or "spatial" not in f["uns"]:
            print("No spatial data found in uns.")
            return

        spatial = f["uns/spatial"]
        for sample_key in spatial.keys():
            print(f"\nSample Key: {sample_key}")
            group = spatial[sample_key]
            print(f"  Keys: {list(group.keys())}")

            if "images" in group:
                print(f"  Images found: {list(group['images'].keys())}")
                for img_type in group["images"].keys():
                    img = group["images"][img_type]
                    print(f"    - {img_type}: shape {img.shape}, dtype {img.dtype}")

            if "scalefactors" in group:
                print(f"  Scalefactors: {list(group['scalefactors'].keys())}")
                for sf_key in group["scalefactors"].keys():
                    print(f"    - {sf_key}: {group['scalefactors'][sf_key][()]}")


if __name__ == "__main__":
    inspect_spatial("A:/hest_data/st/MEND29.h5ad")
