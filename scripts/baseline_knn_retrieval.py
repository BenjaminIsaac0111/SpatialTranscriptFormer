"""
Non-parametric k-NN retrieval baseline for pathway-activity prediction.

Builds a bank of training-spot features + pathway targets, then predicts each
validation spot as the mean target of its k nearest training spots in
backbone-feature space (a BLEEP-style retrieval baseline, Xie et al. 2023).

There is no training. Metrics are computed with the *same* engine ``validate``
used for the learned models, so the numbers are directly comparable to a
``stf-train`` run (val MAE vs. baseline, per-pathway PCC/CCC, spatial coherence).

Usage::

    python scripts/baseline_knn_retrieval.py --data-dir A:\\hest_data --k 16
    python scripts/baseline_knn_retrieval.py --data-dir A:\\hest_data --metric l2 \
        --pathways HALLMARK_HYPOXIA HALLMARK_ANGIOGENESIS
"""

import argparse
import json
import os
from datetime import datetime

import torch

from spatial_transcript_former.utils import set_seed
from spatial_transcript_former.data.paths import resolve_feature_dir
from spatial_transcript_former.recipes.hest.utils import get_train_val_ids
from spatial_transcript_former.recipes.hest.dataset import get_hest_feature_dataloader
from spatial_transcript_former.models import KNNRetrievalBaseline
from spatial_transcript_former.training.engine import validate
from spatial_transcript_former.training.losses import CompositeLoss


def build_bank(loader, device):
    """Concatenate all valid (non-padded) train spots into feature/target banks.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: ``(features (M, D), targets (M, P))``.
    """
    feats_all, tgts_all = [], []
    for feats, _, pathways, _, mask in loader:
        if pathways is None:
            raise ValueError(
                "Pathway targets are required. Did you pass --pathway-targets-dir "
                "or run stf-compute-pathways?"
            )
        for b in range(feats.shape[0]):
            valid = (
                ~mask[b]
                if mask is not None
                else torch.ones(feats.shape[1], dtype=torch.bool)
            )
            feats_all.append(feats[b][valid])
            tgts_all.append(pathways[b][valid])
    feature_bank = torch.cat(feats_all, dim=0).to(device)
    target_bank = torch.cat(tgts_all, dim=0).to(device)
    return feature_bank, target_bank


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--backbone", type=str, default="ctranspath")
    parser.add_argument("--feature-dir", type=str, default=None)
    parser.add_argument("--pathway-targets-dir", type=str, default=None)
    parser.add_argument(
        "--pathways",
        nargs="+",
        default=None,
        help="Restrict to a subset of pathway names (must match the target files).",
    )
    parser.add_argument("--k", type=int, default=16, help="Neighbours to average")
    parser.add_argument(
        "--metric", type=str, default="cosine", choices=["cosine", "l2"]
    )
    parser.add_argument("--chunk", type=int, default=1024, help="Query rows per batch")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--organ", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="./runs/knn_retrieval")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    set_seed(args.seed)

    pathway_targets_dir = args.pathway_targets_dir or os.path.join(
        args.data_dir, "pathway_activities"
    )
    feat_dir = resolve_feature_dir(args.data_dir, args.backbone, args.feature_dir)

    # Same discovery + patient-aware split as the learned runs.
    train_ids, val_ids = get_train_val_ids(
        args.data_dir,
        precomputed=True,
        backbone=args.backbone,
        feature_dir=feat_dir,
        max_samples=args.max_samples,
        organ=args.organ,
        seed=args.seed,
    )
    print(f"Split: {len(train_ids)} train, {len(val_ids)} val")
    if not train_ids or not val_ids:
        raise RuntimeError("Need non-empty train and val splits.")

    loader_kwargs = dict(
        batch_size=1,
        num_workers=0,
        whole_slide_mode=True,
        feature_dir=feat_dir,
        pathway_targets_dir=pathway_targets_dir,
        pathway_names=args.pathways,
    )
    train_loader = get_hest_feature_dataloader(
        args.data_dir, train_ids, shuffle=False, augment=False, **loader_kwargs
    )
    val_loader = get_hest_feature_dataloader(
        args.data_dir, val_ids, shuffle=False, augment=False, **loader_kwargs
    )

    print("Building retrieval bank from training spots...")
    feature_bank, target_bank = build_bank(train_loader, device)
    print(
        f"Bank: {feature_bank.shape[0]} spots x {feature_bank.shape[1]} dims "
        f"-> {target_bank.shape[1]} pathways"
    )

    model = KNNRetrievalBaseline(
        feature_bank, target_bank, k=args.k, metric=args.metric, chunk=args.chunk
    ).to(device)

    # val_loss is informational for a non-parametric model; reuse the standard
    # composite so the rest of the metric dict matches the learned runs.
    criterion = CompositeLoss(alpha=1.0, pcc_type="ccc").to(device)

    print(f"Evaluating k-NN (k={args.k}, metric={args.metric})...")
    metrics = validate(model, val_loader, criterion, device, whole_slide=True)

    os.makedirs(args.output_dir, exist_ok=True)
    summary = {
        "timestamp": datetime.now().isoformat(),
        "baseline": "knn_retrieval",
        "config": {
            "data_dir": args.data_dir,
            "backbone": args.backbone,
            "k": args.k,
            "metric": args.metric,
            "max_samples": args.max_samples,
            "organ": args.organ,
            "seed": args.seed,
            "num_train_spots": int(feature_bank.shape[0]),
            "num_pathways": int(target_bank.shape[1]),
            "n_train_slides": len(train_ids),
            "n_val_slides": len(val_ids),
        },
        "metrics": {
            k: v
            for k, v in metrics.items()
            if k not in ("val_pcc_per_pathway", "val_ccc_per_pathway")
        },
    }
    out_path = os.path.join(args.output_dir, "results_summary.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved to {out_path}")
    print(
        f"val_mae={metrics['val_mae']:.4f} "
        f"(baseline {metrics.get('val_baseline_mae')}) | "
        f"PCC={metrics.get('val_pcc')} | CCC={metrics.get('val_ccc')}"
    )


if __name__ == "__main__":
    main()
