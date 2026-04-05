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

    # Loss
    parser.add_argument(
        "--loss",
        type=str,
        default="mse_pcc",
        choices=[
            "mse",
            "pcc",
            "mse_pcc",
            "poisson",
            "logcosh",
        ],
    )
    parser.add_argument(
        "--pcc-weight",
        type=float,
        default=1.0,
        help="Weight for PCC term in mse_pcc loss",
    )
    parser.add_argument(
        "--pathway-targets-dir",
        type=str,
        default=None,
        help="Directory of pre-computed pathway activity .h5 files",
    )
    parser.add_argument(
        "--morans-pathway-weight",
        action="store_true",
        help="Weight MSE loss per-pathway by Moran's I spatial autocorrelation. "
        "Requires pathway .h5 files to contain pathway_morans_i "
        "(re-run stf-compute-pathways --overwrite to add them).",
    )

    # Model
    g = parser.add_argument_group("Model")
    g.add_argument(
        "--model",
        type=str,
        default="he2rna",
        choices=["he2rna", "vit_st", "interaction", "attention_mil", "transmil"],
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
