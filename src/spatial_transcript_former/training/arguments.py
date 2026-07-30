import os
import argparse
from spatial_transcript_former.config import get_config


def parse_args():
    parser = argparse.ArgumentParser(description="Train Spatial TranscriptFormer")

    # Data
    g = parser.add_argument_group("Data")
    g.add_argument(
        "--data-dir",
        type=str,
        default=get_config("data_dirs", ["hest_data"])[0],
        help="Root directory of HEST data",
    )
    g.add_argument(
        "--feature-dir",
        type=str,
        default=None,
        help="Explicit feature directory (overrides auto-detection)",
    )
    g.add_argument(
        "--max-samples", type=int, default=None, help="Limit samples for debugging"
    )
    g.add_argument(
        "--precomputed", action="store_true", help="Use pre-computed features"
    )
    g.add_argument(
        "--whole-slide", action="store_true", help="Dense whole-slide prediction"
    )
    g.add_argument("--seed", type=int, default=42)
    g.add_argument("--organ", type=str, default=None, help="Filter samples by organ")
    g.add_argument(
        "--technology",
        type=str,
        default=None,
        help="Filter samples by st_technology (e.g. Visium)",
    )
    g.add_argument(
        "--held-out-study",
        nargs="+",
        default=None,
        help="Dataset-grouped Leave-One-Study-Out split: hold out these "
        "dataset_title(s) as the validation set instead of the patient-aware "
        "split (docs/EXPERIMENT_SPATIAL_ATTRIBUTION.md §3). Mutually "
        "exclusive with --colonmap-vs-rest.",
    )
    g.add_argument(
        "--colonmap-vs-rest",
        type=str,
        default=None,
        choices=["forward", "reverse"],
        help="Dataset-grouped COLON-MAP-vs-rest headline split (doc §3). "
        "'forward' trains on the 9 non-COLON-MAP studies and evaluates on "
        "COLON MAP; 'reverse' swaps the direction. Mutually exclusive with "
        "--held-out-study.",
    )

    # Loss
    parser.add_argument(
        "--loss",
        type=str,
        default="mse_pcc",
        choices=[
            "mse",
            "pcc",
            "ccc",
            "mse_pcc",
            "mse_ccc",
            "mse_ccc_clip",
            "mse_huber",
        ],
    )
    parser.add_argument(
        "--pcc-weight",
        type=float,
        default=1.0,
        help="Weight for PCC/CCC term in composite losses",
    )
    parser.add_argument(
        "--clip-weight",
        type=float,
        default=0.5,
        help="Weight for CLIP alignment term (mse_ccc_clip only)",
    )
    parser.add_argument(
        "--clip-temp",
        type=float,
        default=0.07,
        help="Temperature τ for CLIP alignment loss",
    )
    parser.add_argument(
        "--residualize-depth",
        action="store_true",
        help="Regress measured sequencing depth out of the pathway targets. "
        "The raw score correlates with library size at |r|~0.93, so without "
        "this the model largely learns to predict depth from H&E. Applies to "
        "training, checkpoint selection and evaluation alike, keeping all "
        "three on the same quantity. Requires pathway files of "
        "format_version>=3 (which store total_counts).",
    )
    parser.add_argument(
        "--normalize-features",
        action="store_true",
        help="Standardise patch features per slide. A cheap batch correction: "
        "study identity is ~98.5%% decodable from these features, and per-slide "
        "z-scoring lifts cross-study PCA+Ridge from -0.026 to +0.046.",
    )
    parser.add_argument(
        "--pathway-targets-dir",
        type=str,
        default=None,
        help="Directory of pre-computed pathway activity .h5 files",
    )

    # Model
    g = parser.add_argument_group("Model")
    g.add_argument(
        "--model",
        type=str,
        default="he2rna",
        choices=[
            "he2rna",
            "vit_st",
            "interaction",
            "spatial_transformer",
            "attention_mil",
            "transmil",
        ],
    )
    g.add_argument("--backbone", type=str, default="resnet50")
    g.add_argument("--no-pretrained", action="store_false", dest="pretrained")
    g.set_defaults(pretrained=True)
    g.add_argument("--num-pathways", type=int, default=50)
    g.add_argument(
        "--pathway-prior",
        type=str,
        default="hallmarks",
        choices=["hallmarks", "progeny"],
        help="Pathway prior for token initialisation. "
        "'progeny' sets num-pathways=14 automatically.",
    )
    g.add_argument("--token-dim", type=int, default=256)
    g.add_argument("--n-heads", type=int, default=4)
    g.add_argument("--n-layers", type=int, default=2)
    g.add_argument(
        "--use-spatial-pe",
        action="store_true",
        help="Enable spatial positional encoding",
    )
    g.add_argument(
        "--output-activation",
        type=str,
        default="auto",
        choices=["auto", "softplus", "linear"],
        help="Output head. 'softplus' suits the non-negative raw targets; "
        "'linear' is required for depth-residualised targets, which are signed "
        "(a positive-only head cannot represent them and collapses to a "
        "constant). 'auto' (default) follows --residualize-depth.",
    )
    g.add_argument(
        "--interactions",
        nargs="+",
        default=None,
        help="Attention interactions to enable: p2p, p2h, h2p, h2h (default: all)",
    )

    # Training
    g = parser.add_argument_group("Training")
    g.add_argument("--epochs", type=int, default=get_config("training.epochs", 10))
    g.add_argument(
        "--batch-size", type=int, default=get_config("training.batch_size", 32)
    )
    g.add_argument(
        "--num-workers", type=int, default=4, help="DataLoader worker processes"
    )
    g.add_argument("--grad-accum-steps", type=int, default=1)
    g.add_argument(
        "--lr", type=float, default=get_config("training.learning_rate", 1e-4)
    )
    g.add_argument("--weight-decay", type=float, default=0.0)
    g.add_argument("--warmup-epochs", type=int, default=10)
    g.add_argument("--augment", action="store_true")
    g.add_argument("--use-amp", action="store_true")
    g.add_argument(
        "--output-dir",
        type=str,
        default=get_config("training.output_dir", "./checkpoints"),
    )
    g.add_argument("--compile", action="store_true")
    g.add_argument("--resume", action="store_true")
    g.add_argument(
        "--early-stopping-patience",
        type=int,
        default=None,
        help="Number of epochs to wait for val_ccc improvement before early stopping (default: None/disabled)",
    )
    g.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Name used for checkpoint files and logs (defaults to --model if unset). "
        "Set automatically by run_preset.py to the preset name.",
    )

    # Advanced
    g = parser.add_argument_group("Advanced")
    g.add_argument("--n-neighbors", type=int, default=0)
    g.add_argument("--use-global-context", action="store_true")
    g.add_argument("--global-context-size", type=int, default=128)
    g.add_argument("--compile-backend", type=str, default="inductor")
    g.add_argument("--plot-pathways", action="store_true")
    g.add_argument(
        "--plot-pathways-list",
        nargs="+",
        default=None,
        help="List of pathway names to exclusively visualize (e.g. HALLMARK_HYPOXIA). Defaults to the first 6 if None.",
    )
    g.add_argument("--plot-attention", action="store_true")
    g.add_argument(
        "--return-attention",
        action="store_true",
        help="Extract and return attention maps during forward pass",
    )
    g.add_argument(
        "--weak-supervision", action="store_true", help="Bag-level training for MIL"
    )
    g.add_argument(
        "--interaction-type",
        type=str,
        default=None,
        help="Interaction architecture type (placeholder for future experiments)",
    )
    g.add_argument(
        "--pathway-sparsity",
        type=str,
        default=None,
        help="Pathway sparsity topology (placeholder for future experiments)",
    )
    g.add_argument(
        "--pathways",
        nargs="+",
        default=None,
        help="List of selected pathway names to define expected input/output dimension.",
    )
    g.add_argument(
        "--vis-interval",
        type=int,
        default=1,
        help="Epoch interval for generating validation plots",
    )
    g.add_argument(
        "--vis-sample",
        type=str,
        default=None,
        help="Sample ID to use for periodic visualization",
    )

    args = parser.parse_args()
    if args.pathway_targets_dir is None:
        args.pathway_targets_dir = os.path.join(args.data_dir, "pathway_activities")
    return args
