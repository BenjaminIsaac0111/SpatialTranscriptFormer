from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
import torch


def inspect_keys():
    repo_id = "1aurent/swin_tiny_patch4_window7_224.CTransPath"
    filename = "model.safetensors"
    path = hf_hub_download(repo_id=repo_id, filename=filename)
    sd = load_file(path)

    keys = sorted(sd.keys())
    for k in keys:
        if (
            "downsample" in k
            or "patch_embed" in k
            or "layers.0" in k
            or "layers.1" in k
        ):
            print(f"{k}: {sd[k].shape}")


if __name__ == "__main__":
    inspect_keys()
