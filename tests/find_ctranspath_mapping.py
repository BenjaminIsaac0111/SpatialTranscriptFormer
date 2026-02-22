import timm
import torch
import torch.nn as nn
from spatial_transcript_former.models.backbones import ConvStem
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file


def find_mapping():
    # 1. Get Timm model
    model = timm.create_model("swin_tiny_patch4_window7_224", pretrained=False)
    model.patch_embed = ConvStem(embed_dim=96, norm_layer=None, flatten=True)
    timm_sd = model.state_dict()

    # 2. Get Checkpoint
    repo_id = "1aurent/swin_tiny_patch4_window7_224.CTransPath"
    path = hf_hub_download(repo_id=repo_id, filename="model.safetensors")
    ckpt_sd = load_file(path)

    print("--- TIMM STAGES ---")
    for k in sorted(timm_sd.keys()):
        if "blocks.0.norm1.weight" in k or "downsample.reduction.weight" in k:
            print(f"Timm: {k} -> {timm_sd[k].shape}")

    print("\n--- CKPT STAGES ---")
    for k in sorted(ckpt_sd.keys()):
        if "blocks.0.norm1.weight" in k or "downsample.reduction.weight" in k:
            print(f"Ckpt: {k} -> {ckpt_sd[k].shape}")


if __name__ == "__main__":
    find_mapping()
