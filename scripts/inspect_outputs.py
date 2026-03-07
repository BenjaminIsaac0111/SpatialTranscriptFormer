import torch
import json
import os
import argparse
import numpy as np
from spatial_transcript_former.models import SpatialTranscriptFormer
from spatial_transcript_former.recipes.hest.utils import get_sample_ids, setup_dataloaders


class Args:
    pass


args = Args()
args.data_dir = "A:\\hest_data"
args.epochs = 2000
args.output_dir = "runs/stf_tiny"
args.model = "interaction"
args.backbone = "ctranspath"
args.precomputed = True
args.whole_slide = True
args.pathway_init = True
args.use_amp = True
args.log_transform = True
args.loss = "mse_pcc"
args.resume = True
args.n_layers = 2
args.token_dim = 256
args.n_heads = 4
args.batch_size = 1
args.vis_sample = "TENX29"
args.max_samples = 1
args.organ = None
args.num_genes = 1000
args.n_neighbors = 6
args.use_global_context = False
args.global_context_size = 0
args.augment = False
args.feature_dir = None
args.seed = 42
args.warmup_epochs = 10
args.sparsity_lambda = 0.0

device = "cuda" if torch.cuda.is_available() else "cpu"

genes_path = "global_genes.json"
with open(genes_path, "r") as f:
    gene_list = json.load(f)[:1000]
args.num_genes = len(gene_list)

final_ids = get_sample_ids(
    args.data_dir, precomputed=args.precomputed, backbone=args.backbone, max_samples=1
)
train_loader, _ = setup_dataloaders(args, final_ids, [])

model = SpatialTranscriptFormer(
    num_genes=args.num_genes,
    backbone_name=args.backbone,
    pretrained=False,
    token_dim=args.token_dim,
    n_heads=args.n_heads,
    n_layers=args.n_layers,
    num_pathways=50,
    use_spatial_pe=True,
    output_mode="counts",
)

ckpt_path = os.path.join(args.output_dir, "latest_model_interaction.pth")
if os.path.exists(ckpt_path):
    print("Loading", ckpt_path)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
else:
    print("No ckpt found!")

model.to(device)
model.eval()

with torch.no_grad():
    for batch in train_loader:
        feats, genes, coords, mask = [x.to(device) for x in batch]
        out = model(feats, rel_coords=coords, mask=mask, return_dense=True)
        preds = out

        preds = torch.expm1(preds) if args.log_transform else preds
        targets = torch.expm1(genes) if args.log_transform else genes

        patch_idx = None
        for i in range(mask.shape[1]):
            if not mask[0, i]:
                patch_idx = i
                break

        with open(
            "C:/Users/wispy/.gemini/antigravity/brain/6a31ec6d-2f34-4f97-96b8-e437c2640219/model_output_sample.md",
            "w",
        ) as f:
            f.write("# Model Output Sample (stf_tiny with simplifications)\n\n")
            if patch_idx is not None:
                f.write("### Target vs Prediction for a Single Valid Patch\n")
                f.write("Showing the first 20 genes (absolute expression counts).\n\n")

                f.write("| Gene Index | Target Count (True) | Predicted Count |\n")
                f.write("|------------|----------------------|-----------------|\n")

                t_vals = targets[0, patch_idx, :20].cpu().numpy()
                p_vals = preds[0, patch_idx, :20].cpu().numpy()

                for i in range(20):
                    f.write(f"| {i} | {t_vals[i]:.2f} | {p_vals[i]:.2f} |\n")

                f.write("\n### Summary Statistics Across All Patches in Batch\n")
                f.write(f"- Target Mean: {targets[~mask].mean().item():.4f}\n")
                f.write(f"- Target Max:  {targets[~mask].max().item():.4f}\n")
                f.write(f"- Pred Mean:   {preds[~mask].mean().item():.4f}\n")
                f.write(f"- Pred Max:    {preds[~mask].max().item():.4f}\n")
                f.write(f"- Pred Min:    {preds[~mask].min().item():.4f}\n")
            else:
                f.write("No valid patches found in sample.\n")

        print("Sample logic written to artifact.")
        break
