import sys
from pathlib import Path
import torch
# Add project root to path (use __file__ to get absolute path)
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
from arc.models.tiny_lm import TinyLM, TinyLMConfig

def load_model(ckpt_path: str):
    '''
    loads the model from the checkpoint path
    '''
    ckpt = torch.load(ckpt_path, map_location='cpu')
    cfg = TinyLMConfig(**ckpt['cfg'])
    model = TinyLM(cfg)
    model.load_state_dict(ckpt['model'])
    model.eval()
    return model

if __name__ == "__main__":
    # Use absolute path based on project root
    ckpt_path = project_root / "trained_models" / "final(set 1).pt"
    model = load_model(str(ckpt_path))
    x = torch.randint(0, model.cfg.vocab_size, (1, 16))
    print("vocab size =", model.cfg.vocab_size)
    with torch.no_grad():
        logits, = model(x)
    print("OK: logits shape =", logits.shape) #should return (1, 16, vocab_size)
    print("OK: model loaded successfully")