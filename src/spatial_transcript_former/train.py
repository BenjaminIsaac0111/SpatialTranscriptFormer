"""
Training script for SpatialTranscriptFormer and baselines.

Usage:
    stf-train --model interaction --data-dir /path/to/hest --precomputed --whole-slide --pathway-prior hallmarks
"""

import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from spatial_transcript_former.config import get_config
from spatial_transcript_former.models import HE2RNA, ViT_ST, SpatialTranscriptFormer
from spatial_transcript_former.utils import set_seed
from spatial_transcript_former.training.engine import train_one_epoch, validate
from spatial_transcript_former.training.experiment_logger import ExperimentLogger
from spatial_transcript_former.visualization import run_inference_plot
from spatial_transcript_former.recipes.hest.utils import (
    get_train_val_ids,
    setup_dataloaders,
)

from spatial_transcript_former.training.arguments import parse_args
from spatial_transcript_former.training.builder import setup_model, setup_criterion
from spatial_transcript_former.training.checkpoint import (
    save_checkpoint,
    load_checkpoint,
)
from spatial_transcript_former.checkpoint import save_pretrained
from spatial_transcript_former.recipes.hest.compute_pathway_activities import (
    PATHWAY_FILE_VERSION,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve_pathway_names(args):
    """Best-effort recovery of pathway names for ``save_pretrained``.

    Order:
      1. ``args.pathways`` if it's a non-empty list.
      2. The ``pathway_names`` dataset of the first .h5 in
         ``args.pathway_targets_dir``.
      3. ``None`` (skip writing pathway_names.json).
    """
    explicit = getattr(args, "pathways", None)
    if explicit and isinstance(explicit, (list, tuple)) and len(explicit) > 0:
        return list(explicit)

    targets_dir = getattr(args, "pathway_targets_dir", None)
    if targets_dir and os.path.isdir(targets_dir):
        for fname in sorted(os.listdir(targets_dir)):
            if not fname.endswith(".h5"):
                continue
            try:
                import h5py

                with h5py.File(os.path.join(targets_dir, fname), "r") as f:
                    if "pathway_names" in f:
                        return [
                            n.decode() if isinstance(n, bytes) else n
                            for n in f["pathway_names"][:]
                        ]
            except Exception:
                pass
            break  # only inspect the first .h5
    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    set_seed(args.seed)

    set_seed(args.seed)

    # 1. Data — discover sample IDs and split (recipe handles splitting strategy)
    train_ids, val_ids = get_train_val_ids(
        args.data_dir,
        precomputed=args.precomputed,
        backbone=args.backbone,
        feature_dir=args.feature_dir,
        max_samples=args.max_samples,
        organ=args.organ,
        seed=args.seed,
    )
    print(f"Split: {len(train_ids)} train, {len(val_ids)} val")

    train_loader, val_loader, val_whole_slide = setup_dataloaders(
        args, train_ids, val_ids
    )

    # 2. Model, Loss, Optimizer
    model = setup_model(args, device)
    criterion = setup_criterion(args).to(device)
    optimizer = optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    # LR scheduler: cosine annealing with optional linear warmup
    warmup_epochs = args.warmup_epochs
    warmup_scheduler = optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.01, total_iters=max(1, warmup_epochs)
    )
    cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, args.epochs - warmup_epochs), eta_min=1e-6
    )

    if warmup_epochs > 0:
        main_scheduler = optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_epochs],
        )
    else:
        main_scheduler = cosine_scheduler

    scaler = torch.amp.GradScaler("cuda") if args.use_amp else None
    print(f"Loss: {criterion.__class__.__name__}")
    print(f"LR schedule: {warmup_epochs}-epoch warmup -> cosine annealing to 1e-6")
    print(
        f"Targets: pathway_format_version={PATHWAY_FILE_VERSION} "
        "(mean log1p CP10k of pathway members). "
        "Validation MAE/loss are in those units; best-model selection uses CCC (concordance)."
    )

    # 3. Output & Logger
    os.makedirs(args.output_dir, exist_ok=True)
    config_dict = vars(args)
    logger = ExperimentLogger(args.output_dir, config_dict)

    # 4. Resume
    # ``best_val_metric`` tracks the highest val_ccc seen so far (CCC is
    # higher-is-better and offset-sensitive (concordance-measuring); preferable to
    # MSE-based selection now that targets live in raw log1p CP10k units).
    start_epoch, best_val_metric = 0, -float("inf")
    schedulers = {"main": main_scheduler}
    if args.resume:
        start_epoch, best_val_metric, loaded_schedulers = load_checkpoint(
            model, optimizer, scaler, schedulers, args.output_dir, args.model, device
        )

        # Fallback for old checkpoints: manually step the scheduler to catch up
        if start_epoch > 0 and main_scheduler.last_epoch < start_epoch:
            print(
                f"Old checkpoint detected. Manually stepping scheduler {start_epoch} times to catch up..."
            )
            for _ in range(start_epoch):
                main_scheduler.step()

    epochs_no_improve = 0
    # 5. Training Loop
    for epoch in range(start_epoch, args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")

        train_loss = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            whole_slide=args.whole_slide,
            scaler=scaler,
            grad_accum_steps=args.grad_accum_steps,
        )

        val_metrics = validate(
            model,
            val_loader,
            criterion,
            device,
            whole_slide=val_whole_slide,
            use_amp=args.use_amp,
        )
        val_loss = val_metrics["val_loss"]

        print(
            f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, LR: {optimizer.param_groups[0]['lr']:.2e}"
        )

        # Step LR scheduler
        main_scheduler.step()

        # Log epoch
        epoch_row = {
            "train_loss": train_loss,
            "val_loss": val_loss,
            "lr": optimizer.param_groups[0]["lr"],
        }
        if val_metrics.get("val_mae") is not None:
            epoch_row["val_mae"] = round(val_metrics["val_mae"], 4)
        if val_metrics.get("val_pcc") is not None:
            epoch_row["val_pcc"] = round(val_metrics["val_pcc"], 4)
        if val_metrics.get("val_ccc") is not None:
            epoch_row["val_ccc"] = round(val_metrics["val_ccc"], 4)
        if val_metrics.get("pred_variance") is not None:
            epoch_row["pred_variance"] = round(val_metrics["pred_variance"], 6)
        if val_metrics.get("spatial_coherence") is not None:
            epoch_row["spatial_coherence"] = round(val_metrics["spatial_coherence"], 4)
        if val_metrics.get("attn_correlation") is not None:
            epoch_row["attn_correlation"] = round(val_metrics["attn_correlation"], 4)

        # Hardware Resource Monitoring
        try:
            import psutil

            epoch_row["sys_cpu_percent"] = psutil.cpu_percent()
            epoch_row["sys_ram_percent"] = psutil.virtual_memory().percent
        except ImportError:
            pass

        if torch.cuda.is_available():
            epoch_row["sys_gpu_mem_mb"] = round(
                torch.cuda.memory_allocated() / (1024**2), 2
            )

        logger.log_epoch(epoch + 1, epoch_row)

        # Save best — selection driven by CCC (higher is better)
        val_ccc = val_metrics.get("val_ccc")
        if val_ccc is not None and val_ccc > best_val_metric:
            best_val_metric = val_ccc
            epochs_no_improve = 0

            # Legacy state_dict path (kept for tools that still load .pth directly)
            best_path = os.path.join(args.output_dir, f"best_model_{args.model}.pth")
            torch.save(model.state_dict(), best_path)

            # Self-contained checkpoint directory (config.json + model.pth +
            # optional pathway_names.json) so inference can rebuild the model
            # without re-specifying architecture flags.
            best_dir = os.path.join(args.output_dir, f"best_{args.model}")
            try:
                save_pretrained(
                    model,
                    best_dir,
                    pathway_names=_resolve_pathway_names(args),
                )
            except Exception as e:
                print(f"  (skipped save_pretrained bundle: {e})")
            print(f"Saved best model (val_ccc={val_ccc:.4f}) -> {best_path}")
        else:
            if val_ccc is not None:
                epochs_no_improve += 1

        # Save latest
        save_checkpoint(
            model,
            optimizer,
            scaler,
            schedulers,
            epoch,
            best_val_metric,
            args.output_dir,
            args.model,
        )

        if (
            args.early_stopping_patience is not None
            and epochs_no_improve >= args.early_stopping_patience
        ):
            print(
                f"Early stopping triggered: val_ccc has not improved for {args.early_stopping_patience} epochs."
            )
            break

        # Periodic visualization
        if val_ids and (epoch + 1) % args.vis_interval == 0:
            if not getattr(model, "weak_supervision", False):
                vis_id = args.vis_sample if args.vis_sample else val_ids[0]
                print(f"Generating visualization for sample {vis_id}...")
                run_inference_plot(model, args, vis_id, epoch + 1, device)

    # 6. Finalize
    logger.finalize(best_val_metric)
    if best_val_metric == -float("inf"):
        print("\nTraining complete. No valid CCC was recorded.")
    else:
        print(f"\nTraining complete. Best val CCC: {best_val_metric:.4f}")


if __name__ == "__main__":
    main()
