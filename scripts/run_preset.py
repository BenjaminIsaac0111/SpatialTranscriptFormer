import argparse
import sys

from spatial_transcript_former.config import get_config

# Curated list of MSigDB Hallmarks with strong evidence of involvement in Colorectal/Bowel Cancer
CRC_PATHWAYS = [
    "HALLMARK_WNT_BETA_CATENIN_SIGNALING",
    "HALLMARK_TGF_BETA_SIGNALING",
    "HALLMARK_KRAS_SIGNALING_UP",
    "HALLMARK_KRAS_SIGNALING_DN",
    "HALLMARK_PI3K_AKT_MTOR_SIGNALING",
    "HALLMARK_EPITHELIAL_MESENCHYMAL_TRANSITION",
    "HALLMARK_ANGIOGENESIS",
    "HALLMARK_APICAL_JUNCTION",
    "HALLMARK_INFLAMMATORY_RESPONSE",
    "HALLMARK_IL6_JAK_STAT3_SIGNALING",
    "HALLMARK_APOPTOSIS",
    "HALLMARK_P53_PATHWAY",
    "HALLMARK_DNA_REPAIR",
    "HALLMARK_HYPOXIA",
]


def make_stf_params(n_layers: int, token_dim: int, n_heads: int, batch_size: int):
    """Helper to create standard SpatialTranscriptFormer parameters."""
    return {
        "model": "interaction",
        "backbone": "ctranspath",
        "precomputed": True,
        "whole-slide": True,
        "use-amp": True,
        "compile": True,
        "loss": "mse_pcc",
        "resume": True,
        "n-layers": n_layers,
        "token-dim": token_dim,
        "n-heads": n_heads,
        "batch-size": batch_size,
        "vis_sample": "TENX29",
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
    "stf_small": make_stf_params(n_layers=4, token_dim=384, n_heads=8, batch_size=8),
    "stf_medium": make_stf_params(n_layers=6, token_dim=512, n_heads=8, batch_size=8),
    "stf_large": make_stf_params(n_layers=12, token_dim=768, n_heads=12, batch_size=8),
    # --- Biologically-Prioritized Variants (e.g. Colorectal Cancer) ---
    "stf_crc_tiny": {**make_stf_params(2, 256, 4, 8), "pathways": CRC_PATHWAYS},
    "stf_crc_small": {**make_stf_params(4, 384, 8, 8), "pathways": CRC_PATHWAYS},
    "stf_crc_medium": {**make_stf_params(6, 512, 8, 8), "pathways": CRC_PATHWAYS},
    "stf_crc_large": {**make_stf_params(12, 768, 12, 8), "pathways": CRC_PATHWAYS},
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
        elif isinstance(value, list) or isinstance(value, tuple):
            args.append(arg_name)
            args.extend([str(v) for v in value])
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

    # Build the argument list as if it were CLI args to train.py
    train_args = [
        "--data-dir",
        args.data_dir,
        "--epochs",
        str(args.epochs),
    ]

    if args.max_samples:
        train_args += ["--max-samples", str(args.max_samples)]

    if args.output_dir:
        train_args += ["--output-dir", args.output_dir]
    else:
        # Default output dir based on preset
        train_args += ["--output-dir", f"./runs/{args.preset}"]

    # Add preset arguments
    train_args += params_to_args(PRESETS[args.preset])

    # Add any unknown arguments passed to this script
    train_args += unknown

    print(f"Preset: {args.preset}")
    print(f"Arguments: {' '.join(train_args)}")

    # Inject args and call train.main() directly in-process.
    # This preserves the terminal's TTY handle so tqdm progress bars
    # render correctly (subprocess breaks isatty() on Windows).
    from spatial_transcript_former.train import main as train_main

    sys.argv = ["stf-train"] + train_args

    print(f"\n[run_preset] Starting training for preset '{args.preset}'...", flush=True)
    train_main()


if __name__ == "__main__":
    main()
