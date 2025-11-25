import sys
from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from arc.grids.core import from_list
from arc.eval.solve_task import generate_and_score
from arc.models.tiny_lm import TinyLM, TinyLMConfig
from arc.io.loader import iter_tasks
from arc.viz.viz_task_attempt import visualize_results

def load_trained_model(model_name: str = "final(set 1).pt"):
    model_path = project_root / "trained_models" / model_name
    if not model_path.exists():
        return None
    ckpt = torch.load(model_path, map_location='cpu')
    cfg = TinyLMConfig(**ckpt['cfg'])
    model = TinyLM(cfg)
    model.load_state_dict(ckpt['model'])
    model.eval()
    return model

if __name__ == "__main__":
    model = load_trained_model("final(set 1).pt")
    #load a random training pair from actual data
    # Get first training pair from actual data
    task, _ = next(iter_tasks("training"))
    train_pair = task['train'][0]
    x = from_list(train_pair['input'])
    y = from_list(train_pair['output'])
    print("x = ", train_pair['input'])
    print("y = ", train_pair['output'])
    g_pred, score = generate_and_score(model, x, y, mode="row", max_new=512)
    print("g_pred = ", g_pred)
    print(f"Score: {score:.4f}")
    
    #if you want, visualize the results
    # visualize_results(x, y, g_pred, score)
