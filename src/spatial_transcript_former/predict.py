import argparse
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader

from spatial_transcript_former.models.regression import HE2RNA, ViT_ST
from spatial_transcript_former.models.interaction import SpatialTranscriptFormer
from spatial_transcript_former.data.dataset import get_hest_dataloader, HEST_Dataset, load_gene_expression_matrix, load_global_genes
import h5py

def plot_spatial_genes(coords, truth, pred, gene_names, sample_id, save_path=None, cmap='viridis'):
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
        sc = axes[i, 0].scatter(coords[:, 1], coords[:, 0], c=truth[:, i],
                                cmap=cmap, s=10, edgecolors='none')
        axes[i, 0].set_title(f"{sample_id} - {gene_name} (TRUTH)")
        axes[i, 0].invert_yaxis()
        axes[i, 0].set_aspect('equal')
        plt.colorbar(sc, ax=axes[i, 0], shrink=0.6)
        
        # Prediction
        sc = axes[i, 1].scatter(coords[:, 1], coords[:, 0], c=pred[:, i],
                                cmap=cmap, s=10, edgecolors='none')
        axes[i, 1].set_title(f"{sample_id} - {gene_name} (PRED)")
        axes[i, 1].invert_yaxis()
        axes[i, 1].set_aspect('equal')
        plt.colorbar(sc, ax=axes[i, 1], shrink=0.6)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    else:
        plt.show()
    plt.close(fig)
    plt.close('all')

def plot_histology_overlay(image, coords, values, gene_names, sample_id, scalef=1.0, save_path=None, cmap='viridis'):
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
                        cmap=plt.get_cmap(cmap), alpha=0.6, s=15, edgecolors='none')
        
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

def plot_spatial_pathways(coords, activations, sample_id, save_path=None, cmap='jet',
                          pathway_names=None, pathway_truth=None):
    """
    Plots spatial maps of pathway activations (top 5 by variance).
    If pathway_truth is provided, shows truth (top row) vs prediction (bottom row).
    """
    num_pathways = activations.shape[1]
    # Select top 5 pathways by prediction variance
    vars_pred = np.var(activations, axis=0)
    top_indices = np.argsort(vars_pred)[::-1][:5]
    
    num_plots = min(len(top_indices), 5)
    
    if num_plots == 0:
        print("No pathways to plot.")
        return

    has_truth = pathway_truth is not None
    nrows = 2 if has_truth else 1
    fig, axes = plt.subplots(nrows, num_plots, figsize=(5 * num_plots, 5 * nrows))
    if num_plots == 1 and nrows == 1:
        axes = np.array([[axes]])
    elif num_plots == 1:
        axes = axes.reshape(nrows, 1)
    elif nrows == 1:
        axes = axes.reshape(1, num_plots)
        
    for i, idx in enumerate(top_indices):
        if pathway_names is not None and idx < len(pathway_names):
            label = pathway_names[idx].replace("HALLMARK_", "")
        else:
            label = f"Pathway {idx}"

        # Truth row
        if has_truth:
            ax_t = axes[0, i]
            sc_t = ax_t.scatter(coords[:, 1], coords[:, 0], c=pathway_truth[:, idx],
                                cmap=cmap, s=15, edgecolors='none')
            ax_t.set_title(f"{label}\n(TRUTH)", fontsize=9)
            ax_t.invert_yaxis()
            ax_t.set_aspect('equal')
            ax_t.axis('off')
            plt.colorbar(sc_t, ax=ax_t, shrink=0.6)

        # Prediction row
        row = 1 if has_truth else 0
        ax_p = axes[row, i]
        sc_p = ax_p.scatter(coords[:, 1], coords[:, 0], c=activations[:, idx],
                            cmap=cmap, s=15, edgecolors='none')
        ax_p.set_title(f"{label}\n(PRED, Var: {vars_pred[idx]:.2f})", fontsize=9)
        ax_p.invert_yaxis()
        ax_p.set_aspect('equal')
        ax_p.axis('off')
        plt.colorbar(sc_p, ax=ax_p, shrink=0.6)

    plt.suptitle(f"{sample_id} — Top Pathway Activations", fontweight='bold')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Pathway plot saved to {save_path}")
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
    parser.add_argument('--backbone', type=str, default='resnet50', help='Backbone for feature extraction')
    parser.add_argument('--plot-pathways', action='store_true', help='Visualize pathway activations')
    
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
        model = HE2RNA(num_genes=args.num_genes, backbone=args.backbone)
    elif args.model_type == 'vit_st':
        model = ViT_ST(num_genes=args.num_genes, model_name=args.backbone)
    elif args.model_type == 'interaction':
        model = SpatialTranscriptFormer(
            num_genes=args.num_genes,
            use_nystrom=args.use_nystrom,
            fusion_mode=args.fusion_mode,
            num_pathways=args.num_pathways,
            backbone_name=args.backbone
        )
        
    state_dict = torch.load(args.model_path, map_location=device, weights_only=True)
    # Handle possible torch.compile prefix
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('_orig_mod.'):
            new_state_dict[k[len('_orig_mod.'):]] = v
        else:
            new_state_dict[k] = v
    # Handle legacy checkpoints missing keys? No, usually extra keys are issue or missing
    # But if we added components (pathways?) no they are same architecture just different forward.
            
    model.load_state_dict(new_state_dict)
    model.to(device)
    model.eval()
    
    # Load Sample
    with h5py.File(h5_path, 'r') as f:
        patch_barcodes = f['barcode'][:].flatten()
        coords = f['coords'][:]
        
    # Load common genes for consistency
    try:
        common_gene_names = load_global_genes(args.data_dir, args.num_genes)
    except Exception as e:
        print(f"Warning: Could not load global genes, falling back to sample top genes: {e}")
        common_gene_names = None

    gene_matrix, mask, gene_names = load_gene_expression_matrix(
        h5ad_path, patch_barcodes, selected_gene_names=common_gene_names, num_genes=args.num_genes
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
    all_pathways = []
    
    print(f"Running inference on {len(dataset)} patches...")
    with torch.no_grad():
        for images, targets, rel_coords in loader:
            images = images.to(device)
            rel_coords = rel_coords.to(device)
            
            if isinstance(model, SpatialTranscriptFormer):
                # Request pathways if argument set
                output = model(images, rel_coords=rel_coords, return_pathways=args.plot_pathways)
                if isinstance(output, tuple):
                    preds = output[0]
                    pathways = output[1]
                    all_pathways.append(pathways.cpu().numpy())
                else:
                    preds = output
            else:
                preds = model(images)
                
            all_preds.append(preds.cpu().numpy())
            all_truth.append(targets.numpy())
            
    all_preds = np.concatenate(all_preds, axis=0)
    all_truth = np.concatenate(all_truth, axis=0)
    if all_pathways:
        all_pathways = np.concatenate(all_pathways, axis=0)
    
    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Standard Plot
    save_dot_path = os.path.join(args.output_dir, f"{args.sample_id}_spatial_inference.png")
    plot_spatial_genes(coord_subset, all_truth, all_preds, gene_names[:5], args.sample_id, save_path=save_dot_path, cmap='jet')
    
    if args.plot_pathways and len(all_pathways) > 0:
        save_pathway_path = os.path.join(args.output_dir, f"{args.sample_id}_pathway_activations.png")
        plot_spatial_pathways(coord_subset, all_pathways, args.sample_id, save_path=save_pathway_path, cmap='jet')
    
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
                        plot_histology_overlay(img, coord_subset, all_preds, gene_names, args.sample_id, scalef=scalef, save_path=save_overlay_path, cmap='jet')
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
