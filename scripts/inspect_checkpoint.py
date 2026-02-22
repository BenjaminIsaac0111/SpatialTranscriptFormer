import torch
import os

ckpt_path = (
    r"z:\Projects\SpatialTranscriptFormer\results_long_run\latest_model_interaction.pth"
)

if os.path.exists(ckpt_path):
    print(f"Inspecting {ckpt_path}...")
    try:
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        if isinstance(checkpoint, dict):
            print(f"Keys: {list(checkpoint.keys())}")
            if "epoch" in checkpoint:
                print(f"Epoch: {checkpoint['epoch']}")
            else:
                print("No 'epoch' key found.")
        else:
            print("Checkpoint is not a dict (legacy save format?)")
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
else:
    print("Checkpoint not found.")
