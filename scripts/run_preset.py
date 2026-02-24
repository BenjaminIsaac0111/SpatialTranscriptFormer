import subprocess
import argparse
import sys
import os

from spatial_transcript_former.config import get_config

# Common flags for all STF interaction models
STF_COMMON = [
    "--model",
    "interaction",
    "--backbone",
    "ctranspath",
    "--precomputed",
    "--whole-slide",
    "--pathway-init",
    "--use-amp",
    "--log-transform",
    "--loss",
    "mse_pcc",
    "--resume",
]

PRESETS = {
    # --- Baselines ---
    "he2rna_baseline": [
        "--model",
        "he2rna",
        "--backbone",
        "resnet50",
        "--batch-size",
        "64",
    ],
    "vit_baseline": [
        "--model",
        "vit_st",
        "--backbone",
        "vit_b_16",
        "--batch-size",
        "32",
    ],
    "attention_mil": [
        "--model",
        "attention_mil",
        "--whole-slide",
        "--precomputed",
        "--batch-size",
        "1",
    ],
    "transmil": [
        "--model",
        "transmil",
        "--whole-slide",
        "--precomputed",
        "--batch-size",
        "1",
    ],
    # --- Interaction Models (Layer Scaling) ---
    "stf_interaction_l2": STF_COMMON
    + [
        "--n-layers",
        "2",
        "--token-dim",
        "256",
        "--n-heads",
        "4",
        "--batch-size",
        "4",
    ],
    "stf_interaction_l4": STF_COMMON
    + [
        "--n-layers",
        "4",
        "--token-dim",
        "384",
        "--n-heads",
        "8",
        "--batch-size",
        "4",
    ],
    "stf_interaction_l6": STF_COMMON
    + [
        "--n-layers",
        "6",
        "--token-dim",
        "512",
        "--n-heads",
        "8",
        "--batch-size",
        "2",  # Reduced batch size for large model memory
    ],
    # --- Specific Configurations ---
    "stf_interaction_zinb": [
        "--model",
        "interaction",
        "--backbone",
        "ctranspath",
        "--precomputed",
        "--whole-slide",
        "--pathway-init",
        "--sparsity-lambda",
        "0",
        "--lr",
        "1e-4",
        "--batch-size",
        "4",
        "--epochs",
        "2000",
        "--use-amp",
        "--loss",
        "zinb",
        "--log-transform",
        "--pathway-loss-weight",
        "0.5",
        "--interactions",
        "p2p",
        "p2h",
        "h2p",
        "h2h",
        "--plot-pathways",
        "--resume",
    ],
    # Legacy / Specialized
    "stf_pathway_nystrom": STF_COMMON
    + [
        "--sparsity-lambda",
        "0.05",
        "--lr",
        "1e-4",
        "--batch-size",
        "8",
        "--epochs",
        "2000",
    ],
}

# Add alias for the one currently running to ensure backward compatibility for монитор
PRESETS["stf_interaction_mse_pcc"] = PRESETS["stf_interaction_l2"] + [
    "--pathway-loss-weight",
    "0.5",
    "--plot-pathways",
]


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
    cmd += PRESETS[args.preset]

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
