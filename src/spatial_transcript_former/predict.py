import argparse
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader

from spatial_transcript_former.models.regression import HE2RNA, ViT_ST
from spatial_transcript_former.models.interaction import SpatialTranscriptFormer
from spatial_transcript_former.data.dataset import get_hest_dataloader, HEST_Dataset, load_gene_expression_matrix
import h5py

def plot_spatial_genes(coords, truth, pred, gene_names, sample_id, save_path=None):
    """
    Plots spatial maps of gene expression (dot plot format).
    """
    num_plots = min(len(gene_names), 5)
    fig, axes = plt.subplots(num_plots, 2, figsize=(12, 4 * num_plots))
    if num_plots == 1:
        axes = np.expand_dims(axes, axis=0)

    for i in range(num_plots):
        gene_name = gene_names[i]
        
        # Truth
        sns.scatterplot(x=coords[:, 1], y=coords[:, 0], hue=truth[:, i], 
                        ax=axes[i, 0], palette='viridis', legend=False, s=10)
        axes[i, 0].set_title(f"{sample_id} - {gene_name} (TRUTH)")
        axes[i, 0].invert_yaxis()
        axes[i, 0].set_aspect('equal')
        
        # Prediction
        sns.scatterplot(x=coords[:, 1], y=coords[:, 0], hue=pred[:, i], 
                        ax=axes[i, 1], palette='viridis', legend=False, s=10)
        axes[i, 1].set_title(f"{sample_id} - {gene_name} (PRED)")
        axes[i, 1].invert_yaxis()
        axes[i, 1].set_aspect('equal')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    else:
        plt.show()
    plt.close(fig)
    plt.close('all')

def plot_histology_overlay(image, coords, values, gene_names, sample_id, scalef=1.0, save_path=None):
    """
    Plots predictions as an overlay on the histology image.
    """
    num_plots = min(len(gene_names), 3)
    fig, axes = plt.subplots(1, num_plots, figsize=(6 * num_plots, 6))
    if num_plots == 1:
        axes = [axes]

    # Project coords to image pixels
    pixel_coords = coords * scalef

    for i in range(num_plots):
        gene_name = gene_names[i]
        ax = axes[i]
        
        # Display Histology
        ax.imshow(image)
        
        # Overlay Predictions
        # pixel_coords[:, 0] = X, pixel_coords[:, 1] = Y
        sc = ax.scatter(pixel_coords[:, 0], pixel_coords[:, 1], c=values[:, i], 
                        cmap='viridis', alpha=0.6, s=15, edgecolors='none')
        
        ax.set_title(f"{sample_id} - {gene_name} Overlay")
        ax.axis('off')
        plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"Overlay saved to {save_path}")
    else:
        plt.show()
    plt.close(fig)
    plt.close('all')

def main():
    parser = argparse.ArgumentParser(description="Predict and Visualize Spatial Transcriptomics")
    parser.add_argument('--data-dir', type=str, required=True)
    parser.add_argument('--sample-id', type=str, required=True)
    parser.add_argument('--model-path', type=str, required=True)
    parser.add_argument('--model-type', type=str, default='he2rna', choices=['he2rna', 'vit_st', 'interaction'])
    parser.add_argument('--num-genes', type=int, default=1000)
    parser.add_argument('--output-dir', type=str, default='./results')
    parser.add_argument('--n-neighbors', type=int, default=0, help='Number of spatial neighbors to use')
    parser.add_argument('--use-nystrom', action='store_true', help='Use Nystrom attention for linear complexity')
    parser.add_argument('--fusion-mode', type=str, default='decoder', choices=['decoder', 'jaume'], help='Fusion mode for SpatialTranscriptFormer')
    parser.add_argument('--num-pathways', type=int, default=50, help='Number of pathways in the bottleneck')
    
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Paths
    patches_dir = os.path.join(args.data_dir, 'patches')
    if not os.path.exists(patches_dir):
        patches_dir = args.data_dir
    st_dir = os.path.join(args.data_dir, 'st')
    
    h5_path = os.path.join(patches_dir, f"{args.sample_id}.h5")
    h5ad_path = os.path.join(st_dir, f"{args.sample_id}.h5ad")
    
    # Load Model
    if args.model_type == 'he2rna':
        model = HE2RNA(num_genes=args.num_genes)
    elif args.model_type == 'vit_st':
        model = ViT_ST(num_genes=args.num_genes)
    elif args.model_type == 'interaction':
        model = SpatialTranscriptFormer(
            num_genes=args.num_genes,
            use_nystrom=args.use_nystrom,
            fusion_mode=args.fusion_mode,
            num_pathways=args.num_pathways
        )
        
    state_dict = torch.load(args.model_path, map_location=device)
    # Handle possible torch.compile prefix
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('_orig_mod.'):
            new_state_dict[k[len('_orig_mod.'):]] = v
        else:
            new_state_dict[k] = v
            
    model.load_state_dict(new_state_dict)
    model.to(device)
    model.eval()
    
    # Load Sample
    with h5py.File(h5_path, 'r') as f:
        patch_barcodes = f['barcode'][:].flatten()
        coords = f['coords'][:]
        
    gene_matrix, mask, gene_names = load_gene_expression_matrix(
        h5ad_path, patch_barcodes, num_genes=args.num_genes
    )
    
    coord_subset = coords[mask]
    indices = np.where(mask)[0]
    
    # Neighborhood computation if requested
    neighborhood_indices = None
    if args.n_neighbors > 0:
        from scipy.spatial import KDTree
        with h5py.File(h5_path, 'r') as f:
            coords_all = f['coords'][:]
        tree = KDTree(coords_all)
        dists, idxs = tree.query(coord_subset, k=args.n_neighbors + 1)
        final_neighbors = []
        for i, center_idx in enumerate(indices):
            n_idxs = idxs[i]
            n_idxs = n_idxs[n_idxs != center_idx]
            final_neighbors.append(n_idxs[:args.n_neighbors])
        neighborhood_indices = np.array(final_neighbors)
    else:
        with h5py.File(h5_path, 'r') as f:
            coords_all = f['coords'][:]

    dataset = HEST_Dataset(h5_path, coord_subset, gene_matrix, indices=indices, 
                           neighborhood_indices=neighborhood_indices, coords_all=coords_all)
    loader = DataLoader(dataset, batch_size=32, shuffle=False)
    
    all_preds = []
    all_truth = []
    
    print(f"Running inference on {len(dataset)} patches...")
    with torch.no_grad():
        for images, targets, rel_coords in loader:
            images = images.to(device)
            rel_coords = rel_coords.to(device)
            
            if isinstance(model, SpatialTranscriptFormer):
                outputs = model(images, rel_coords=rel_coords)
            else:
                outputs = model(images)
                
            all_preds.append(outputs.cpu().numpy())
            all_truth.append(targets.numpy())
            
    all_preds = np.concatenate(all_preds, axis=0)
    all_truth = np.concatenate(all_truth, axis=0)
    
    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Standard Plot
    save_dot_path = os.path.join(args.output_dir, f"{args.sample_id}_spatial_inference.png")
    plot_spatial_genes(coord_subset, all_truth, all_preds, gene_names[:5], args.sample_id, save_path=save_dot_path)
    
    # Histology Overlay Plot
    try:
        # Construct h5ad path robustly
        h5ad_overlay_path = h5_path.replace('.h5', '.h5ad')
        if 'patches' in h5ad_overlay_path:
             h5ad_overlay_path = h5ad_overlay_path.replace('patches', 'st')
        
        if os.path.exists(h5ad_overlay_path):
            with h5py.File(h5ad_overlay_path, 'r') as f:
                # Robust group access
                if 'uns' in f and 'spatial' in f['uns']:
                    spatial = f['uns/spatial']
                    sample_key = list(spatial.keys())[0] if len(spatial.keys()) > 0 else None
                    if sample_key:
                        img_group = spatial[sample_key]['images']
                        img_key = 'downscaled_fullres' if 'downscaled_fullres' in img_group else list(img_group.keys())[0]
                        img = img_group[img_key][:]
                        
                        scale_group = spatial[sample_key]['scalefactors']
                        scale_key = 'tissue_downscaled_fullres_scalef' if 'tissue_downscaled_fullres_scalef' in scale_group else list(scale_group.keys())[0]
                        scalef = scale_group[scale_key][()]
                        
                        save_overlay_path = os.path.join(args.output_dir, f"{args.sample_id}_histology_overlay.png")
                        print(f"Generating histology overlay for {args.sample_id}...")
                        plot_histology_overlay(img, coord_subset, all_preds, gene_names, args.sample_id, scalef=scalef, save_path=save_overlay_path)
                else:
                    print(f"No 'uns/spatial' found in {h5ad_overlay_path}. Keys: {list(f.keys())}")
        else:
            print(f"H5AD file not found for overlay: {h5ad_overlay_path}")
    except Exception as e:
        print(f"Warning: Could not generate histology overlay: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
