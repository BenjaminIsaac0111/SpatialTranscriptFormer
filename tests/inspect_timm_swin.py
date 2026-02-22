import timm
import torch
import torch.nn as nn
from spatial_transcript_former.models.backbones import ConvStem


def inspect_timm_swin():
    model = timm.create_model("swin_tiny_patch4_window7_224", pretrained=False)
    model.patch_embed = ConvStem(embed_dim=96, norm_layer=None, flatten=True)

    print("Timm Swin-T Keys:")
    for k, v in model.state_dict().items():
        if "downsample" in k or "layers.0" in k or "layers.1" in k:
            print(f"{k}: {v.shape}")


if __name__ == "__main__":
    inspect_timm_swin()
