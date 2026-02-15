import os
import torch
import numpy as np
import h5py
from torch.utils.data import DataLoader
from spatial_transcript_former.data.dataset import load_gene_expression_matrix, HEST_FeatureDataset, HEST_Dataset, load_global_genes
from spatial_transcript_former.models import SpatialTranscriptFormer
from spatial_transcript_former.predict import plot_spatial_genes, plot_histology_overlay, plot_spatial_pathways

def run_inference_plot(model, args, sample_id, epoch, device):
    """
    Run inference on a single sample and save a visualization plot.

    Args:
        model (nn.Module): The model to use.
        args (argparse.Namespace): CLI arguments.
        sample_id (str): ID of the sample to plot.
        epoch (int): Current epoch (for filename).
        device (torch.device): Device to run inference on.
    """
    try:
        plot_pathways = getattr(args, 'plot_pathways', False)
        
        with torch.no_grad():
            model.eval()
            
            # Setup paths for this sample
            patches_dir = os.path.join(args.data_dir, 'patches') if os.path.isdir(os.path.join(args.data_dir, 'patches')) else args.data_dir
            st_dir = os.path.join(args.data_dir, 'st')
            h5_path = os.path.join(patches_dir, f"{sample_id}.h5")
            h5ad_path = os.path.join(st_dir, f"{sample_id}.h5ad")
            
            with h5py.File(h5_path, 'r') as f:
                patch_barcodes = f['barcode'][:].flatten()
                coords = f['coords'][:]
            
            # Load global genes for consistent alignment
            try:
                common_gene_names = load_global_genes(args.data_dir, args.num_genes)
            except Exception as e:
                print(f"Warning: Could not load global genes in visualization: {e}")
                common_gene_names = None

            gene_matrix, mask, gene_names = load_gene_expression_matrix(
                h5ad_path, patch_barcodes, selected_gene_names=common_gene_names, num_genes=args.num_genes
            )
            
            preds = []
            pathways_list = []
            
            if args.precomputed:
                feat_dir_name = 'he_features' if args.backbone == 'resnet50' else f"he_features_{args.backbone}"
                feature_path = os.path.join(args.data_dir, feat_dir_name, f"{sample_id}.pt")
                if not os.path.exists(feature_path):
                     feature_path = os.path.join(args.data_dir, 'patches', feat_dir_name, f"{sample_id}.pt")
                     
                ds = HEST_FeatureDataset(
                    feature_path, h5ad_path, num_genes=args.num_genes, 
                    selected_gene_names=common_gene_names,
                    n_neighbors=args.n_neighbors, whole_slide_mode=args.whole_slide
                )
                
                if args.whole_slide:
                    feats, _, _ = ds[0]
                    feats = feats.unsqueeze(0).to(device)
                    if hasattr(model, 'forward_dense'):
                        # Handle potential tuple return (preds, pathways)
                        output = model.forward_dense(feats, return_pathways=plot_pathways)
                        if isinstance(output, tuple):
                            preds.append(output[0].detach().cpu().squeeze(0))
                            pathways_list.append(output[1].detach().cpu().squeeze(0))
                        else:
                            preds.append(output.detach().cpu().squeeze(0))
                else:
                    dl = DataLoader(ds, batch_size=32, shuffle=False)
                    for feats, _, rel_coords_batch in dl:
                        if isinstance(model, SpatialTranscriptFormer):
                            # Handle potential tuple return (preds, pathways)
                            output = model(feats.to(device), rel_coords=rel_coords_batch.to(device), return_pathways=plot_pathways)
                            if isinstance(output, tuple):
                                p = output[0].cpu()
                                path = output[1].cpu()
                                pathways_list.append(path)
                            else:
                                p = output.cpu()
                        else:
                            p = model(feats.to(device)).cpu()
                        preds.append(p)
            else:
                coord_subset = coords[mask]
                ds = HEST_Dataset(h5_path, coord_subset, gene_matrix, indices=np.where(mask)[0])
                dl = DataLoader(ds, batch_size=32, shuffle=False)
                for imgs, _, rel_coords_batch in dl:
                    if isinstance(model, SpatialTranscriptFormer):
                        output = model(imgs.to(device), rel_coords=rel_coords_batch.to(device), return_pathways=plot_pathways)
                        if isinstance(output, tuple):
                            p = output[0].cpu()
                            path = output[1].cpu()
                            pathways_list.append(path)
                        else:
                            p = output.cpu()
                    else:
                        p = model(imgs.to(device)).cpu()
                    preds.append(p)

            if preds:
                preds = torch.cat(preds, dim=0).numpy()
                coord_mask = np.array(mask, dtype=bool)
                coord_subset = coords[coord_mask]
                
                plot_spatial_genes(coord_subset, gene_matrix, preds, gene_names[:5], sample_id, 
                                   save_path=os.path.join(args.output_dir, f"{sample_id}_epoch_{epoch+1}.png"),
                                   cmap='jet')
                
                if plot_pathways and pathways_list:
                    pathways = torch.cat(pathways_list, dim=0).numpy()
                    save_pathway_path = os.path.join(args.output_dir, f"{sample_id}_pathway_epoch_{epoch+1}.png")
                    plot_spatial_pathways(coord_subset, pathways, sample_id, save_path=save_pathway_path, cmap='jet')

                # Histology Overlay
                try:
                    if os.path.exists(h5ad_path):
                        with h5py.File(h5ad_path, 'r') as f:
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
                                    
                                    save_overlay_path = os.path.join(args.output_dir, f"{sample_id}_overlay_epoch_{epoch+1}.png")
                                    plot_histology_overlay(img, coord_subset, preds, gene_names, sample_id, scalef=scalef, save_path=save_overlay_path, cmap='jet')
                except Exception as e_overlay:
                    print(f"Warning: Could not create overlay: {e_overlay}")
    except Exception as e:
        print(f"Warning: Failed to generate validation plot: {e}")
        import traceback
        traceback.print_exc()
