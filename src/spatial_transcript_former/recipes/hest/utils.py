import os
import pandas as pd
from torchvision import transforms
from spatial_transcript_former.data.paths import resolve_feature_dir
from .dataset import get_hest_dataloader, get_hest_feature_dataloader


def get_sample_ids(
    data_dir,
    precomputed=False,
    backbone="resnet50",
    feature_dir=None,
    max_samples=None,
    organ=None,
):
    """
    Find and filter HEST sample IDs based on metadata and data availability.

    Args:
        data_dir (str): Root directory of HEST data.
        precomputed (bool): Whether to filter for samples with precomputed features.
        backbone (str): Backbone name to use for finding precomputed features.
        feature_dir (str, optional): Explicit custom feature directory.
        max_samples (int): Maximum number of samples to return.

    Returns:
        list: List of filtered sample IDs.
    """
    # Check for patches subdirectory
    patches_dir = os.path.join(data_dir, "patches")
    search_dir = patches_dir if os.path.isdir(patches_dir) else data_dir

    all_files = [f for f in os.listdir(search_dir) if f.endswith(".h5")]
    all_ids = [f.replace(".h5", "") for f in all_files]

    if not all_ids:
        raise ValueError(f"No .h5 files found in {search_dir}")

    # Load metadata to filter samples
    metadata_path = os.path.join(data_dir, "HEST_v1_3_0.csv")
    if os.path.exists(metadata_path):
        df = pd.read_csv(metadata_path)
        available_ids = set(all_ids)

        # Filter for existing files and Homo sapiens
        df_filtered = df[df["id"].isin(available_ids)]
        df_human = df_filtered[df_filtered["species"] == "Homo sapiens"]

        if organ:
            if organ == "Colorectal":
                organ = "Bowel"
            print(f"Filtering for organ: {organ}")
            df_human = df_human[df_human["organ"] == organ]
            if df_human.empty:
                print(f"Warning: No samples found for organ '{organ}'.")

        human_ids = df_human["id"].tolist()
        final_ids = human_ids
        if not final_ids:
            print("Warning: No Human samples found. Using all files.")
            final_ids = all_ids
    else:
        print("Metadata not found, using all files.")
        final_ids = all_ids

    # Filter for precomputed features
    if precomputed:
        try:
            feat_dir = resolve_feature_dir(data_dir, backbone, feature_dir)
        except FileNotFoundError as e:
            print(f"Warning: {e}")
            return final_ids

        available_pts = set(
            f.replace(".pt", "") for f in os.listdir(feat_dir) if f.endswith(".pt")
        )
        original_count = len(final_ids)
        final_ids = [fid for fid in final_ids if fid in available_pts]
        print(
            f"Refined to {len(final_ids)}/{original_count} based on features in "
            f"{feat_dir}"
        )

    if max_samples is not None:
        final_ids = final_ids[:max_samples]

    return final_ids


def get_train_val_ids(
    data_dir,
    precomputed=False,
    backbone="resnet50",
    feature_dir=None,
    max_samples=None,
    organ=None,
    val_ratio=0.2,
    seed=42,
):
    """Discover sample IDs and split into train/val sets.

    Uses patient-aware splitting when HEST metadata is available,
    otherwise falls back to a naive random split.

    Args:
        data_dir: Root data directory.
        precomputed: If True, filter for samples with pre-computed features.
        backbone: Backbone identifier for feature directory discovery.
        feature_dir: Explicit feature directory override.
        max_samples: Optional cap on total sample count.
        organ: Optional organ filter.
        val_ratio: Fraction of samples for validation.
        seed: Random seed for reproducible splits.

    Returns:
        Tuple of (train_ids, val_ids).
    """
    import numpy as np

    final_ids = get_sample_ids(
        data_dir,
        precomputed=precomputed,
        backbone=backbone,
        feature_dir=feature_dir,
        max_samples=max_samples,
        organ=organ,
    )

    if len(final_ids) <= 1:
        return final_ids, final_ids

    # Patient-aware splitting (prevents data leakage)
    metadata_path = os.path.join(data_dir, "HEST_v1_3_0.csv")
    if os.path.exists(metadata_path):
        from .splitting import split_hest_patients

        train_ids, val_ids, _ = split_hest_patients(
            metadata_path, val_ratio=val_ratio, seed=seed
        )
        # Intersect with available IDs (after feature/organ filtering)
        available = set(final_ids)
        train_ids = [i for i in train_ids if i in available]
        val_ids = [i for i in val_ids if i in available]
        print(f"Patient-aware split: {len(train_ids)} train, {len(val_ids)} val")
        return train_ids, val_ids

    # Fallback: naive random split
    np.random.seed(seed)
    np.random.shuffle(final_ids)
    split_idx = int(len(final_ids) * (1.0 - val_ratio))
    return final_ids[:split_idx], final_ids[split_idx:]


def get_dataset_grouped_ids(
    data_dir,
    held_out=None,
    colonmap_vs_rest=None,
    precomputed=False,
    backbone="resnet50",
    feature_dir=None,
    max_samples=None,
    organ=None,
    technology=None,
):
    """Discover sample IDs and split them by source study (``dataset_title``)
    instead of by patient.

    This is the LOSO / COLON-MAP-vs-rest protocol required by
    docs/EXPERIMENT_SPATIAL_ATTRIBUTION.md §3, since ``patient`` is not a
    usable grouping key for the Bowel/Visium corpus (most patient labels are
    blank, and where present they collide across different real people in
    different studies). Mirrors ``get_train_val_ids``'s
    discover-then-intersect-with-available-features shape, but swaps in the
    dataset-grouped splitters from ``splitting.py``.

    Args:
        data_dir: Root data directory.
        held_out: One ``dataset_title`` (or list of them) to hold out as the
            validation set — a single Leave-One-Study-Out fold. Mutually
            exclusive with ``colonmap_vs_rest``.
        colonmap_vs_rest: ``"forward"`` (train on the 9 non-COLON-MAP
            studies, evaluate on COLON MAP) or ``"reverse"`` (swap
            direction). Mutually exclusive with ``held_out``.
        precomputed: If True, filter for samples with pre-computed features.
        backbone: Backbone identifier for feature directory discovery.
        feature_dir: Explicit feature directory override.
        max_samples: Optional cap on total sample count (applied before the
            split, like ``get_train_val_ids``).
        organ: Optional organ filter (e.g. ``"Bowel"``).
        technology: Optional ``st_technology`` filter (e.g. ``"Visium"``).

    Returns:
        Tuple of (train_ids, val_ids).
    """
    if (held_out is None) == (colonmap_vs_rest is None):
        raise ValueError("Exactly one of held_out or colonmap_vs_rest must be given.")

    from .splitting import split_colonmap_vs_rest, split_hest_by_dataset

    final_ids = get_sample_ids(
        data_dir,
        precomputed=precomputed,
        backbone=backbone,
        feature_dir=feature_dir,
        max_samples=max_samples,
        organ=organ,
    )

    metadata_path = os.path.join(data_dir, "HEST_v1_3_0.csv")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(
            f"Dataset-grouped splitting requires HEST metadata at {metadata_path}"
        )

    if held_out is not None:
        train_ids, val_ids = split_hest_by_dataset(
            metadata_path, held_out=held_out, organ=organ, technology=technology
        )
    else:
        train_ids, val_ids = split_colonmap_vs_rest(
            metadata_path,
            reverse=(colonmap_vs_rest == "reverse"),
            organ=organ,
            technology=technology,
        )

    # Intersect with available IDs (after feature/organ/precomputed filtering)
    available = set(final_ids)
    train_ids = [i for i in train_ids if i in available]
    val_ids = [i for i in val_ids if i in available]
    print(
        f"Dataset-grouped split (post feature-availability filter): "
        f"{len(train_ids)} train, {len(val_ids)} val"
    )
    return train_ids, val_ids


def resolve_split_ids(args):
    """Resolve train/val sample IDs for a run, dispatching to the
    dataset-grouped (LOSO / COLON-MAP-vs-rest) or patient-aware splitter
    based on which fields are set on ``args``.

    This is the exact branch ``stf-train`` uses (see
    docs/EXPERIMENT_SPATIAL_ATTRIBUTION.md §3), factored out so a finished
    run's own train/val split can be reconstructed later purely from its
    saved ``results_summary.json`` config (used by
    ``scripts/evaluate_spatial_attribution.py``).

    Args:
        args: Namespace-like object with ``data_dir``, ``precomputed``,
            ``backbone``, ``feature_dir``, ``max_samples``, ``organ``, and
            optionally ``held_out_study``, ``colonmap_vs_rest``,
            ``technology``, ``seed``.

    Returns:
        Tuple of (train_ids, val_ids).
    """
    held_out_study = getattr(args, "held_out_study", None)
    colonmap_vs_rest = getattr(args, "colonmap_vs_rest", None)
    if held_out_study or colonmap_vs_rest:
        return get_dataset_grouped_ids(
            args.data_dir,
            held_out=held_out_study,
            colonmap_vs_rest=colonmap_vs_rest,
            precomputed=args.precomputed,
            backbone=args.backbone,
            feature_dir=args.feature_dir,
            max_samples=args.max_samples,
            organ=args.organ,
            technology=getattr(args, "technology", None),
        )
    return get_train_val_ids(
        args.data_dir,
        precomputed=args.precomputed,
        backbone=args.backbone,
        feature_dir=args.feature_dir,
        max_samples=args.max_samples,
        organ=args.organ,
        seed=args.seed,
    )


def setup_dataloaders(args, train_ids, val_ids):
    """
    Create training and validation dataloaders.

    When using precomputed features, the validation loader is always built
    in whole-slide mode so that spatial PCC is computed per-slide (across
    the N spatial positions) rather than across a mixed batch of patches
    from different slides.

    Args:
        args (argparse.Namespace): CLI arguments.
        train_ids (list): IDs for training.
        val_ids (list): IDs for validation.

    Returns:
        tuple: ``(train_loader, val_loader, val_whole_slide)`` where
        ``val_whole_slide`` indicates the mode the val_loader was built in.
    """
    if args.precomputed:
        # Use centralised path resolution
        try:
            feat_dir = resolve_feature_dir(
                args.data_dir,
                args.backbone,
                getattr(args, "feature_dir", None),
            )
        except FileNotFoundError as e:
            raise RuntimeError(f"Cannot setup dataloaders: {e}")

        pathway_targets_dir = getattr(args, "pathway_targets_dir", None)
        pathway_names = getattr(args, "pathways", None)

        if args.whole_slide:
            train_loader = (
                get_hest_feature_dataloader(
                    args.data_dir,
                    train_ids,
                    batch_size=args.batch_size,
                    shuffle=True,
                    num_workers=0,  # Cached in-RAM, no workers needed (faster/safer on Windows)
                    n_neighbors=args.n_neighbors,
                    whole_slide_mode=True,
                    augment=args.augment,
                    feature_dir=feat_dir,
                    pathway_targets_dir=pathway_targets_dir,
                    pathway_names=pathway_names,
                    residualize_depth=getattr(args, "residualize_depth", False),
                    normalize_features=getattr(args, "normalize_features", False),
                )
                if train_ids
                else None
            )
            val_loader = (
                get_hest_feature_dataloader(
                    args.data_dir,
                    val_ids,
                    batch_size=args.batch_size,
                    shuffle=False,
                    num_workers=0,  # Cached in-RAM, no workers needed
                    n_neighbors=args.n_neighbors,
                    whole_slide_mode=True,
                    augment=False,
                    feature_dir=feat_dir,
                    pathway_targets_dir=pathway_targets_dir,
                    pathway_names=pathway_names,
                    residualize_depth=getattr(args, "residualize_depth", False),
                    normalize_features=getattr(args, "normalize_features", False),
                )
                if val_ids
                else None
            )
        else:
            train_loader = (
                get_hest_feature_dataloader(
                    args.data_dir,
                    train_ids,
                    batch_size=args.batch_size,
                    shuffle=True,
                    num_workers=args.num_workers,
                    n_neighbors=args.n_neighbors,
                    use_global_context=args.use_global_context,
                    global_context_size=args.global_context_size,
                    augment=args.augment,
                    feature_dir=feat_dir,
                    pathway_targets_dir=pathway_targets_dir,
                    pathway_names=pathway_names,
                    residualize_depth=getattr(args, "residualize_depth", False),
                    normalize_features=getattr(args, "normalize_features", False),
                )
                if train_ids
                else None
            )
            # Always validate in whole-slide mode so spatial PCC is
            # computed per-slide rather than across a mixed batch.
            val_loader = (
                get_hest_feature_dataloader(
                    args.data_dir,
                    val_ids,
                    batch_size=args.batch_size,
                    shuffle=False,
                    num_workers=0,  # Cached in-RAM, no workers needed
                    whole_slide_mode=True,
                    augment=False,
                    feature_dir=feat_dir,
                    pathway_targets_dir=pathway_targets_dir,
                    pathway_names=pathway_names,
                    residualize_depth=getattr(args, "residualize_depth", False),
                    normalize_features=getattr(args, "normalize_features", False),
                )
                if val_ids
                else None
            )
    else:
        # Base normalization
        norm = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

        train_transform = norm
        if args.augment:
            train_transform = transforms.Compose(
                [
                    # Rotations/Flips handled DIHEDRALLY inside HEST_Dataset
                    transforms.ColorJitter(brightness=0.1, contrast=0.1),
                    norm,
                ]
            )
            print("Enabled image augmentations for training.")

        val_transform = norm

        if args.use_global_context:
            print("Warning: Global context only supported with pre-computed features.")
        train_loader = (
            get_hest_dataloader(
                args.data_dir,
                train_ids,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=args.num_workers,
                n_neighbors=args.n_neighbors,
                transform=train_transform,
                augment=args.augment,
                pathway_targets_dir=getattr(args, "pathway_targets_dir", None),
                pathway_names=getattr(args, "pathways", None),
            )
            if train_ids
            else None
        )
        val_loader = (
            get_hest_dataloader(
                args.data_dir,
                val_ids,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                n_neighbors=args.n_neighbors,
                transform=val_transform,
                pathway_targets_dir=getattr(args, "pathway_targets_dir", None),
                pathway_names=getattr(args, "pathways", None),
            )
            if val_ids
            else None
        )

    # Precomputed features always validate in whole-slide mode for
    # proper spatial PCC.  Raw-patch mode has no whole-slide support.
    val_whole_slide = True if args.precomputed else args.whole_slide
    return train_loader, val_loader, val_whole_slide
