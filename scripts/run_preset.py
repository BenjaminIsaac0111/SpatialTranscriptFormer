import subprocess
import argparse
import sys
import os

from spatial_transcript_former.config import get_config


def make_stf_params(n_layers: int, token_dim: int, n_heads: int, batch_size: int):
    """Helper to create standard SpatialTranscriptFormer parameters."""
    return {
        "model": "interaction",
        "backbone": "ctranspath",
        "precomputed": True,
        "whole-slide": True,
        "pathway-init": True,
        "use-amp": True,
        "log-transform": True,
        "loss": "mse_pcc",
        "resume": True,
        "n-layers": n_layers,
        "token-dim": token_dim,
        "n-heads": n_heads,
        "batch-size": batch_size,
        "vis_sample": 'TENX29',
    }


PRESETS = {
    # --- Baselines ---
    "he2rna_baseline": {
        "model": "he2rna",
        "backbone": "resnet50",
        "batch-size": 64,
    },
    "vit_baseline": {
        "model": "vit_st",
        "backbone": "vit_b_16",
        "batch-size": 32,
    },
    "attention_mil": {
        "model": "attention_mil",
        "whole-slide": True,
        "precomputed": True,
        "batch-size": 1,
    },
    "transmil": {
        "model": "transmil",
        "whole-slide": True,
        "precomputed": True,
        "batch-size": 1,
    },
    # --- SpatialTranscriptFormer Variants ---
    "stf_tiny": make_stf_params(n_layers=2, token_dim=256, n_heads=4, batch_size=8),
    "stf_small": make_stf_params(n_layers=4, token_dim=384, n_heads=8, batch_size=4),
    "stf_medium": make_stf_params(n_layers=6, token_dim=512, n_heads=8, batch_size=2),
    "stf_large": make_stf_params(n_layers=12, token_dim=768, n_heads=12, batch_size=1),
}


def params_to_args(params_dict):
    """Convert a parameter dictionary to a list of CLI arguments."""
    args = []
    for key, value in params_dict.items():
        arg_name = f"--{key.replace('_', '-')}"
        if value is True:
            args.append(arg_name)
        elif value is False or value is None:
            continue
        else:
            args.extend([arg_name, str(value)])
    return args


def main():
    parser = argparse.ArgumentParser(
        description="Run Spatial TranscriptFormer training presets"
    )
    parser.add_argument(
        "--preset",
        type=str,
        choices=list(PRESETS.keys()),
        required=True,
        help="Preset configuration",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=get_config("data_dirs", ["A:\\hest_data"])[0],
        help="Data directory",
    )
    parser.add_argument("--epochs", type=int, default=2000, help="Number of epochs")
    parser.add_argument("--max-samples", type=int, default=None, help="Limit samples")
    parser.add_argument(
        "--output-dir", type=str, default=None, help="Override output dir"
    )

    # Custom args allow appending more arguments to train.py
    args, unknown = parser.parse_known_args()

    cmd = [
        sys.executable,
        "-m",
        "spatial_transcript_former.train",
        "--data-dir",
        args.data_dir,
        "--epochs",
        str(args.epochs),
    ]

    if args.max_samples:
        cmd += ["--max-samples", str(args.max_samples)]

    if args.output_dir:
        cmd += ["--output-dir", args.output_dir]
    else:
        # Default output dir based on preset
        cmd += ["--output-dir", f"./runs/{args.preset}"]

    # Add preset arguments
    cmd += params_to_args(PRESETS[args.preset])

    # Add any unknown arguments passed to this script
    cmd += unknown

    print(f"Executing: {' '.join(cmd)}")

    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Training failed with error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
