import torch
import torch.nn as nn
import torchvision.models as models
try:
    import timm
except ImportError:
    timm = None

class ConvStem(nn.Module):
    """
    ConvStem, from Original CTransPath implementation
    """
    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=768, norm_layer=None, flatten=True):
        super().__init__()

        assert patch_size == 4
        
        # CTransPath Stem: 3 -> 24 -> 48 -> 96 (if embed_dim=96)
        # Based on search: 2 stages of stride 2
        l1_embed_dim = embed_dim // 4
        l2_embed_dim = embed_dim // 2

        self.blob1 = nn.Sequential(
            nn.Conv2d(in_chans, l1_embed_dim, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(l1_embed_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(l1_embed_dim, l1_embed_dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(l1_embed_dim),
            nn.ReLU(inplace=True),
        )

        self.blob2 = nn.Sequential(
            nn.Conv2d(l1_embed_dim, l2_embed_dim, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(l2_embed_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(l2_embed_dim, embed_dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),
        )

        self.flatten = flatten

    def forward(self, x):
        x = self.blob1(x)
        x = self.blob2(x)
        
        # Swin Transformer in timm expects (B, H, W, C)
        x = x.permute(0, 2, 3, 1)
        return x

def get_backbone(name, pretrained=True, num_classes=None):
    """
    Creates a backbone model and returns both the model and its feature dimension.
    If num_classes is provided, the model will have a classification/regression head.
    Otherwise, it returns the feature extractor (head replaced by Identity).
    """
    feature_dim = None
    model = None

    if name == 'resnet50':
        if pretrained:
            weights = models.ResNet50_Weights.DEFAULT
            model = models.resnet50(weights=weights)
        else:
            model = models.resnet50(weights=None)
            
        feature_dim = model.fc.in_features
        if num_classes is not None:
            model.fc = nn.Linear(feature_dim, num_classes)
        else:
            model.fc = nn.Identity()
            
    elif name == 'ctranspath':
        if timm is None:
            raise ImportError("timm is required for ctranspath")
        
        # CTransPath: Swin-T with ConvStem
        model = timm.create_model('swin_tiny_patch4_window7_224', pretrained=False, num_classes=num_classes if num_classes else 0)
        
        # Replace Patch Embed with ConvStem
        model.patch_embed = ConvStem(embed_dim=96, norm_layer=None, flatten=True)
        feature_dim = model.num_features
        
        if pretrained:
            try:
                from huggingface_hub import hf_hub_download
                from safetensors.torch import load_file
                print(f"Downloading CTransPath weights (safetensors) from Hugging Face...")
                weights_path = hf_hub_download(repo_id="1aurent/swin_tiny_patch4_window7_224.CTransPath", filename="model.safetensors")
                state_dict = load_file(weights_path)
                
                # CTransPath (timm) vs transformers key mapping
                # The 1aurent mirror structure requires index shifting for downsample layers
                new_state_dict = {}
                for k, v in state_dict.items():
                    nk = k.replace('swin.', '').replace('encoder.', '')
                    nk = nk.replace('embeddings.patch_embeddings.', 'patch_embed.')
                    
                    # Shift downsample layers: ckpt.layers.i.downsample -> timm.layers.i+1.downsample
                    if 'downsample' in nk and 'layers.' in nk:
                        try:
                            parts = nk.split('.')
                            layer_idx = int(parts[1])
                            parts[1] = str(layer_idx + 1)
                            nk = '.'.join(parts)
                        except (ValueError, IndexError):
                            pass
                    
                    new_state_dict[nk] = v
                
                msg = model.load_state_dict(new_state_dict, strict=False)
                print(f"CTransPath weights loaded. Missing: {len(msg.missing_keys)}, Unexpected: {len(msg.unexpected_keys)}")
            except Exception as e:
                print(f"Warning: Could not load pretrained CTransPath weights: {e}")
                print("Proceeding with random initialization.")

    elif name == 'uni':
        # Dropped as per user request (no access)
        raise NotImplementedError("UNI backbone is currently disabled (requires gated access).")

    elif name == 'gigapath':
        if timm is None:
            raise ImportError("timm is required for GigaPath")
            
        # GigaPath: ViT-Giant-Patch14-224 (approx)
        # Note: GigaPath uses specific config. 'prov-gigapath/prov-gigapath'
        try:
            model = timm.create_model("hf_hub:prov-gigapath/prov-gigapath", pretrained=pretrained)
            feature_dim = model.num_features
            if num_classes is not None:
                 model.head = nn.Linear(feature_dim, num_classes)
        except Exception as e:
             print(f"Error loading GigaPath from HF: {e}")
             # Fallback?
             raise e

    elif name == 'hibou-b' or name == 'hibou-l':
        if timm is None:
            raise ImportError("timm is required for Hibou")
        
        try:
            from huggingface_hub import hf_hub_download
            repo_id = "histai/hibou-b" if name == 'hibou-b' else "histai/hibou-l"
            arch = "vit_base_patch16_224" if name == 'hibou-b' else "vit_large_patch14_224" # Adjust arch if needed
            
            model = timm.create_model(arch, pretrained=False)
            
            if pretrained:
                # Download weights manually
                weights_path = hf_hub_download(repo_id=repo_id, filename="pytorch_model.bin")
                state_dict = torch.load(weights_path, map_location='cpu')
                if 'model' in state_dict:
                    state_dict = state_dict['model']
                model.load_state_dict(state_dict, strict=False)
            
            feature_dim = model.num_features
            if num_classes is not None:
                model.head = nn.Linear(feature_dim, num_classes)
            else:
                model.head = nn.Identity()
        except Exception as e:
            print(f"Error loading {name}: {e}")
            if "403" in str(e):
                print(f"Note: {repo_id} might be gated. Please request access on HF Hub.")
            raise e

    elif name == 'phikon':
        if timm is None:
            raise ImportError("timm is required for Phikon")
        
        try:
            from huggingface_hub import hf_hub_download
            model = timm.create_model("vit_base_patch16_224", pretrained=False)
            
            if pretrained:
                weights_path = hf_hub_download(repo_id="owkin/phikon", filename="pytorch_model.bin")
                state_dict = torch.load(weights_path, map_location='cpu')
                if 'model' in state_dict:
                    state_dict = state_dict['model']
                elif 'state_dict' in state_dict:
                    state_dict = state_dict['state_dict']
                
                new_state_dict = {}
                for k, v in state_dict.items():
                    name_clean = k.replace('backbone.', '').replace('encoder.', '')
                    new_state_dict[name_clean] = v
                
                msg = model.load_state_dict(new_state_dict, strict=False)
                print(f"Phikon weights loaded. Missing keys: {len(msg.missing_keys)}, Unexpected: {len(msg.unexpected_keys)}")
                
            feature_dim = model.num_features
            if num_classes is not None:
                model.head = nn.Linear(feature_dim, num_classes)
            else:
                model.head = nn.Identity()
        except Exception as e:
            print(f"Error loading Phikon: {e}")
            raise e

    elif name == 'plip':
        if timm is None:
            raise ImportError("timm is required for PLIP")
        
        try:
            from huggingface_hub import hf_hub_download
            model = timm.create_model("vit_base_patch16_224", pretrained=False)
            
            if pretrained:
                weights_path = hf_hub_download(repo_id="vinid/plip", filename="pytorch_model.bin")
                state_dict = torch.load(weights_path, map_location='cpu')
                
                new_state_dict = {}
                for k, v in state_dict.items():
                    if k.startswith('visual_model.'):
                        new_state_dict[k.replace('visual_model.', '')] = v
                
                if not new_state_dict:
                    for k, v in state_dict.items():
                        new_state_dict[k.replace('module.visual.', '').replace('visual.', '')] = v

                msg = model.load_state_dict(new_state_dict, strict=False)
                print(f"PLIP vision weights loaded. Missing keys: {len(msg.missing_keys)}, Unexpected: {len(msg.unexpected_keys)}")
            
            feature_dim = model.num_features
            if num_classes is not None:
                model.head = nn.Linear(feature_dim, num_classes)
            else:
                model.head = nn.Identity()
        except Exception as e:
            print(f"Error loading PLIP: {e}. Note: PLIP might require `pip install transformers` for native loading.")
            raise e

    elif name == 'vit_b_16':
        if pretrained:
            weights = models.ViT_B_16_Weights.DEFAULT
            model = models.vit_b_16(weights=weights)
        else:
            model = models.vit_b_16(weights=None)
            
        feature_dim = model.heads.head.in_features
        if num_classes is not None:
            model.heads.head = nn.Linear(feature_dim, num_classes)
        else:
            model.heads.head = nn.Identity()
            
    elif name.startswith('vit_') and timm is not None:
        model = timm.create_model(name, pretrained=pretrained, num_classes=num_classes if num_classes else 0)
        feature_dim = model.num_features
        
    else:
        # Generic timm/torchvision fallback
        if timm is not None:
            try:
                model = timm.create_model(name, pretrained=pretrained, num_classes=num_classes if num_classes else 0)
                feature_dim = model.num_features
            except Exception:
                pass
        
        if model is None:
            try:
                # Try torchvision
                model_fn = getattr(models, name)
                model = model_fn(pretrained=pretrained)
                if hasattr(model, 'fc'):
                    feature_dim = model.fc.in_features
                    if num_classes is not None: model.fc = nn.Linear(feature_dim, num_classes)
                    else: model.fc = nn.Identity()
                elif hasattr(model, 'heads'):
                    feature_dim = model.heads.head.in_features
                    if num_classes is not None: model.heads.head = nn.Linear(feature_dim, num_classes)
                    else: model.heads.head = nn.Identity()
            except Exception as e:
                raise NotImplementedError(f"Backbone {name} not implemented or found: {e}")

    return model, feature_dim
