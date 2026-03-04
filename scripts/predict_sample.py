#!/usr/bin/env python
import argparse
import os
import torch
import json
from spatial_transcript_former.visualization import run_inference_plot


# Dummy class to hold loaded arguments
class RunArgs:
    def __init__(self, **entries):
        self.__dict__.update(entries)


def parse_args():
    parser = argparse.ArgumentParser("Predict sample pathways")
    parser.add_argument(
        "--sample-id",
        required=True,
        type=str,
        help="Sample ID to run inference on (e.g. TENX156)",
    )
    parser.add_argument(
        "--run-dir",
        required=True,
        type=str,
        help="Directory containing model weights and args.json",
    )
    parser.add_argument(
        "--output-dir", type=str, default=".", help="Where to save the output plot"
    )
    parser.add_argument(
        "--epoch", type=int, default=0, help="Epoch number to label the plot with"
    )
    return parser.parse_args()


def main():
    cli_args = parse_args()

    # Load args from run_dir
    args_path = os.path.join(cli_args.run_dir, "results_summary.json")
    if not os.path.exists(args_path):
        raise FileNotFoundError(f"Missing {args_path}")

    with open(args_path, "r") as f:
        summary_dict = json.load(f)
        run_args_dict = summary_dict.get("config", {})

    run_args = RunArgs(**run_args_dict)
    run_args.output_dir = cli_args.output_dir
    run_args.run_dir = cli_args.run_dir

    # Optional arguments that might be missing from older args.json
    if not hasattr(run_args, "log_transform"):
        run_args.log_transform = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Re-initialize the model based on run_args
    if run_args.model == "baseline":
        from spatial_transcript_former.models import SpatialTranscriptFormer

        model = SpatialTranscriptFormer(
            backbone=run_args.backbone,
            num_genes=run_args.num_genes,
            dropout=run_args.dropout,
            n_neighbors=run_args.n_neighbors,
        )
    elif run_args.model == "interaction":
        from spatial_transcript_former.models import SpatialTranscriptFormer

        model = SpatialTranscriptFormer(
            num_genes=run_args.num_genes,
            backbone_name=run_args.backbone,
            pretrained=run_args.pretrained,
            token_dim=getattr(run_args, "token_dim", 384),
            n_heads=getattr(run_args, "n_heads", 6),
            n_layers=getattr(run_args, "n_layers", 4),
            num_pathways=getattr(run_args, "num_pathways", 0),
            use_spatial_pe=getattr(run_args, "use_spatial_pe", True),
            output_mode="zinb" if getattr(run_args, "loss", "") == "zinb" else "counts",
            interactions=getattr(run_args, "interactions", None),
        )
    else:
        raise ValueError(f"Unknown model type: {run_args.model}")

    model.to(device)

    # Note: we explicitly load the *best* model if it exists, otherwise the latest
    ckpt_path = os.path.join(cli_args.run_dir, f"best_model_{run_args.model}.pth")
    if not os.path.exists(ckpt_path):
        ckpt_path = os.path.join(cli_args.run_dir, f"latest_model_{run_args.model}.pth")

    if os.path.exists(ckpt_path):
        print(f"Loading checkpoint from {ckpt_path}...")
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=True)
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)
    else:
        print(
            f"Warning: No checkpoint found in {cli_args.run_dir}. Using untrained model."
        )

    print(f"Running inference for sample {cli_args.sample_id}...")
    run_inference_plot(model, run_args, cli_args.sample_id, cli_args.epoch, device)


if __name__ == "__main__":
    main()
