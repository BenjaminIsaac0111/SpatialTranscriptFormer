import torch
from spatial_transcript_former.models.backbones import get_backbone

def test_ctranspath_loading():
    print("Testing CTransPath weight loading...")
    try:
        model, dim = get_backbone('ctranspath', pretrained=True)
        print(f"Successfully initialized CTransPath. Feature dim: {dim}")
        
        msg = model.load_state_dict(new_state_dict, strict=False)
        print(f"Missing keys: {len(msg.missing_keys)}")
        print(f"Sample missing: {msg.missing_keys[:5]}")
        print(f"Unexpected keys: {len(msg.unexpected_keys)}")
        print(f"Sample unexpected: {msg.unexpected_keys[:5]}")
        
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_ctranspath_loading()
