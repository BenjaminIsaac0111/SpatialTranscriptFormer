import os
import torch
import torch.nn as nn
from spatial_transcript_former.models import HE2RNA, ViT_ST, SpatialTranscriptFormer
from spatial_transcript_former.training.losses import (
    PCCLoss,
    CompositeLoss,
    MaskedMSELoss,
)


def _resolve_num_pathways(args):
    """Determine the number of pathway targets expected."""
    if getattr(args, "pathway_prior", "hallmarks") == "progeny":
        return 14
    if getattr(args, "pathways", None):
        return len(args.pathways)
    return 50  # Default Hallmarks


def setup_model(args, device):
    """Initialize and optionally compile the model."""
    args.num_pathways = _resolve_num_pathways(args)

    if args.model == "he2rna":
        model = HE2RNA(
            num_genes=args.num_pathways,
            backbone=args.backbone,
            pretrained=args.pretrained,
        )
    elif args.model == "vit_st":
        model = ViT_ST(
            num_genes=args.num_pathways,
            model_name=args.backbone if "vit_" in args.backbone else "vit_b_16",
            pretrained=args.pretrained,
        )
    elif args.model == "interaction":
        print(
            f"Initializing SpatialTranscriptFormer ({args.backbone}, pretrained={args.pretrained}, num_pathways={args.num_pathways})"
        )

        model = SpatialTranscriptFormer(
            num_pathways=args.num_pathways,
            backbone_name=args.backbone,
            pretrained=args.pretrained,
            token_dim=args.token_dim,
            n_heads=args.n_heads,
            n_layers=args.n_layers,
            use_spatial_pe=args.use_spatial_pe,
            interactions=getattr(args, "interactions", None),
        )
    elif args.model == "attention_mil":
        from spatial_transcript_former.models.mil import AttentionMIL

        model = AttentionMIL(
            output_dim=args.num_pathways,
            backbone_name=args.backbone,
            pretrained=args.pretrained,
        )
    elif args.model == "transmil":
        from spatial_transcript_former.models.mil import TransMIL

        model = TransMIL(
            output_dim=args.num_pathways,
            backbone_name=args.backbone,
            pretrained=args.pretrained,
        )
    else:
        raise ValueError(f"Unknown model: {args.model}")

    model.weak_supervision = getattr(args, "weak_supervision", False)
    model = model.to(device)

    if args.compile:
        print(f"Compiling model (backend='{args.compile_backend}')...")
        try:
            model = torch.compile(model, backend=args.compile_backend)
        except Exception as e:
            print(f"Compilation failed: {e}. Using eager mode.")

    return model


def setup_criterion(args):
    """Create loss function from CLI args."""
    if args.loss == "pcc":
        return PCCLoss()
    elif args.loss == "mse_pcc":
        return CompositeLoss(alpha=args.pcc_weight)
    elif args.loss == "poisson":
        return nn.PoissonNLLLoss(log_input=True)
    elif args.loss == "logcosh":
        print("Using HuberLoss as proxy for LogCosh")
        return nn.HuberLoss()
    else:
        return MaskedMSELoss()
