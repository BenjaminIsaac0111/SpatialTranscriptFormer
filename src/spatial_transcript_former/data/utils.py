import os
import argparse
import pandas as pd
from torchvision import transforms
from .dataset import get_hest_dataloader, get_hest_feature_dataloader

def get_sample_ids(data_dir, precomputed=False, backbone='resnet50', max_samples=None):
    """
    Find and filter HEST sample IDs based on metadata and data availability.

    Args:
        data_dir (str): Root directory of HEST data.
        precomputed (bool): Whether to filter for samples with precomputed features.
        backbone (str): Backbone name to use for finding precomputed features.
        max_samples (int): Maximum number of samples to return.

    Returns:
        list: List of filtered sample IDs.
    """
    # Check for patches subdirectory
    patches_dir = os.path.join(data_dir, 'patches')
    search_dir = patches_dir if os.path.isdir(patches_dir) else data_dir

    all_files = [f for f in os.listdir(search_dir) if f.endswith('.h5')]
    all_ids = [f.replace('.h5', '') for f in all_files]
    
    if not all_ids:
        raise ValueError(f"No .h5 files found in {search_dir}")
        
    # Load metadata to filter samples
    metadata_path = os.path.join(data_dir, "HEST_v1_3_0.csv")
    if os.path.exists(metadata_path):
        df = pd.read_csv(metadata_path)
        available_ids = set(all_ids)
        
        # Filter for existing files and Homo sapiens
        df_filtered = df[df['id'].isin(available_ids)]
        df_human = df_filtered[df_filtered['species'] == 'Homo sapiens']
        human_ids = df_human['id'].tolist()
        
        # Filter for Human Bowel
        df_bowel = df_human[df_human['organ'].str.contains('Bowel', case=False, na=False)]
        
        if not df_bowel.empty:
            print(f"Filtering for Human Bowel samples ({len(df_bowel)} found)...")
            final_ids = df_bowel['id'].tolist()
        else:
            print("No Human Bowel samples found, falling back to all Human samples...")
            final_ids = human_ids
            
        if not final_ids:
             print("Warning: No Human samples found. Using all files.")
             final_ids = all_ids
    else:
        print("Metadata not found, using all files.")
        final_ids = all_ids

    # Filter for precomputed features
    if precomputed:
        feat_dir_name = 'he_features' if backbone == 'resnet50' else f"he_features_{backbone}"
        feat_dir = os.path.join(data_dir, feat_dir_name)
        if not os.path.exists(feat_dir):
             feat_dir = os.path.join(data_dir, 'patches', feat_dir_name)
        
        if os.path.exists(feat_dir):
            available_pts = set(f.replace('.pt', '') for f in os.listdir(feat_dir) if f.endswith('.pt'))
            original_count = len(final_ids)
            final_ids = [fid for fid in final_ids if fid in available_pts]
            print(f"Refined to {len(final_ids)}/{original_count} based on features in {feat_dir}")
        else:
            print(f"Warning: Feature directory '{feat_dir_name}' not found.")

    if max_samples is not None:
        final_ids = final_ids[:max_samples]

    return final_ids

def setup_dataloaders(args, train_ids, val_ids):
    """
    Create training and validation dataloaders.

    Args:
        args (argparse.Namespace): CLI arguments.
        train_ids (list): IDs for training.
        val_ids (list): IDs for validation.
        augment (bool): Whether to enable augmentations.

    Returns:
        tuple: (train_loader, val_loader)
    """
    if args.precomputed:
        # Use explicit feature_dir if provided, otherwise auto-detect
        if getattr(args, 'feature_dir', None) and os.path.exists(args.feature_dir):
            feat_dir = args.feature_dir
        else:
            feat_dir_name = 'he_features' if args.backbone == 'resnet50' else f"he_features_{args.backbone}"
            feat_dir = os.path.join(args.data_dir, feat_dir_name)
            if not os.path.exists(feat_dir):
                 feat_dir = os.path.join(args.data_dir, 'patches', feat_dir_name)
             
        if args.whole_slide:
            train_loader = get_hest_feature_dataloader(
                args.data_dir, train_ids, batch_size=args.batch_size, shuffle=True,
                num_genes=args.num_genes, n_neighbors=args.n_neighbors,
                whole_slide_mode=True, augment=args.augment, feature_dir=feat_dir,
                log1p=args.log_transform
            )
            val_loader = get_hest_feature_dataloader(
                args.data_dir, val_ids, batch_size=args.batch_size, shuffle=False,
                num_genes=args.num_genes, n_neighbors=args.n_neighbors,
                whole_slide_mode=True, augment=False, feature_dir=feat_dir,
                log1p=args.log_transform
            )
        else:
            train_loader = get_hest_feature_dataloader(
                args.data_dir, train_ids, batch_size=args.batch_size, shuffle=True,
                num_genes=args.num_genes, n_neighbors=args.n_neighbors,
                use_global_context=args.use_global_context, global_context_size=args.global_context_size,
                augment=args.augment, feature_dir=feat_dir,
                log1p=args.log_transform
            )
            val_loader = get_hest_feature_dataloader(
                args.data_dir, val_ids, batch_size=args.batch_size, shuffle=False,
                num_genes=args.num_genes, n_neighbors=args.n_neighbors,
                use_global_context=args.use_global_context, global_context_size=args.global_context_size,
                augment=False, feature_dir=feat_dir,
                log1p=args.log_transform
            )
    else:
        # Base normalization
        norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
        train_transform = norm
        if args.augment:
            train_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(90),
                transforms.ColorJitter(brightness=0.1, contrast=0.1),
                norm
            ])
            print("Enabled image augmentations for training.")
            
        val_transform = norm
        
        if args.use_global_context:
            print("Warning: Global context only supported with pre-computed features.")
        train_loader = get_hest_dataloader(args.data_dir, train_ids, batch_size=args.batch_size, shuffle=True, 
                                          num_genes=args.num_genes, n_neighbors=args.n_neighbors,
                                          transform=train_transform)
        val_loader = get_hest_dataloader(args.data_dir, val_ids, batch_size=args.batch_size, shuffle=False, 
                                        num_genes=args.num_genes, n_neighbors=args.n_neighbors,
                                        transform=val_transform)
        
    return train_loader, val_loader
