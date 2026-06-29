import os
import torch
import torch.nn as nn
from spatial_transcript_former.models import (
    HE2RNA,
    ViT_ST,
    SpatialTranscriptFormer,
    SpatialTransformerRegressor,
    LinearProbe,
    MLPProbe,
)
from spatial_transcript_former.training.losses import (
    CCCLoss,
    CLIPAlignmentLoss,
    CompositeLoss,
    MaskedHuberLoss,
    MaskedMSELoss,
    PCCLoss,
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
        if getattr(args, "precomputed", False):
            from spatial_transcript_former.models.backbones import get_backbone

            _, feature_dim = get_backbone(args.backbone, pretrained=False)
            model = LinearProbe(
                input_dim=feature_dim,
                num_pathways=args.num_pathways,
            )
        else:
            model = HE2RNA(
                num_pathways=args.num_pathways,
                backbone=args.backbone,
                pretrained=args.pretrained,
            )
    elif args.model == "vit_st":
        if getattr(args, "precomputed", False):
            from spatial_transcript_former.models.backbones import get_backbone

            _, feature_dim = get_backbone(args.backbone, pretrained=False)
            model = MLPProbe(
                input_dim=feature_dim,
                num_pathways=args.num_pathways,
            )
        else:
            model = ViT_ST(
                num_pathways=args.num_pathways,
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
    elif args.model == "spatial_transformer":
        # No-pathway-token spatial baseline. Precomputed-feature only: it mixes
        # patch features with a transformer, then regresses pathways per spot.
        if not getattr(args, "precomputed", False):
            raise ValueError(
                "spatial_transformer baseline requires --precomputed features."
            )
        from spatial_transcript_former.models.backbones import get_backbone

        _, feature_dim = get_backbone(args.backbone, pretrained=False)
        model = SpatialTransformerRegressor(
            input_dim=feature_dim,
            num_pathways=args.num_pathways,
            token_dim=args.token_dim,
            n_heads=args.n_heads,
            n_layers=args.n_layers,
            use_spatial_pe=args.use_spatial_pe,
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
    clip_w = getattr(args, "clip_weight", 0.0)
    clip_t = getattr(args, "clip_temp", 0.07)

    if args.loss == "pcc":
        return PCCLoss()
    elif args.loss == "ccc":
        return CCCLoss()
    elif args.loss == "mse_pcc":
        return CompositeLoss(alpha=args.pcc_weight)
    elif args.loss == "mse_ccc":
        return CompositeLoss(alpha=args.pcc_weight, pcc_type="ccc")
    elif args.loss == "mse_ccc_clip":
        # CLIP term here is the batch-discriminative regulariser in
        # pathway-output space (see CLIPAlignmentLoss docstring). Available
        # for opt-in experiments; not part of the current default track.
        return CompositeLoss(
            alpha=args.pcc_weight,
            pcc_type="ccc",
            clip_weight=clip_w or 0.5,
            clip_temperature=clip_t,
        )
    elif args.loss == "mse_huber":
        return CompositeLoss(alpha=args.pcc_weight, mse_type="huber", pcc_type="ccc")
    else:
        return MaskedMSELoss()
