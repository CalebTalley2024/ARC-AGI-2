import torch
import sys
from pathlib import Path
# Add project root to path (use __file__ to get absolute path)
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from arc.serialize import pack_example, deserialize_grid, SEP, EOS
from arc.eval.scorer import mean_logp_output
from arc.models.infer import greedy_generate  # your existing inference

def generate_and_score(model, x_grid, y_grid, device = "cpu", mode: str = "row", max_new: int = 2048):
    '''
    Generate and score a single task.
    
    Args:
        model: The model to use for generation.
        x_grid: The input grid.
        y_grid: The output grid.
        device: The device to use for generation.
        mode: The mode to use for generation.
        max_new: The maximum number of new tokens to generate.
    '''
    # 1) pack example
    seq = pack_example(x_grid, y_grid, mode=mode) #serializes the example
    inp = torch.tensor(seq[:-1], dtype=torch.long).unsqueeze(0).to(device)

    # 2) greedy generate continuation
    out = greedy_generate(model, inp, max_new_tokens=max_new, eos_id=EOS)  # (1, T')

    # 3) compute mean log-prob on output segment
    #score can be negative
    # the larger the negative score, the worse the performanc (eg. -0.4 is better than -1.6)
    score = mean_logp_output(model, out.clone(), sep_token_id=SEP)

    # 4) extract output tokens between 2nd SEP and EOS
    ids = out.squeeze(0).tolist()

    # deserialize the output tokens back into a grid
    g_pred = deserialize_grid(ids, mode=mode)
    return g_pred, float(score)

