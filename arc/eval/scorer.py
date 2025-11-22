# arc/scorer.py
from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn.functional as F


# returns per-token log-probs (T,) given input ids and model
@torch.no_grad()
def token_logprobs(model, input_ids: torch.LongTensor) -> torch.Tensor:
    # model should return logits (B,T,V)
    logits = model(input_ids)[0]
    logp = F.log_softmax(logits, dim=-1)
    # shift for next-token prediction
    tgt = input_ids[:, 1:]  # (B, T-1)
    logp_next = logp[:, :-1, :].gather(-1, tgt.unsqueeze(-1)).squeeze(-1)  # (B, T-1)
    # pad a zero at start to align with input positions
    z = torch.zeros((input_ids.size(0), 1), device=input_ids.device)
    return torch.cat([z, logp_next], dim=1).squeeze(0)

# compute mean log-prob over the output segment (after the big SEP between X and Y)

def mean_logp_output(model, input_ids: torch.LongTensor, sep_token_id: int) -> float:
    lp = token_logprobs(model, input_ids)
    ids = input_ids.squeeze(0).tolist()
    # locate the SEP that separates X and Y (the *second* SEP in our scheme)
    sep_count = 0
    idx = 0
    for i, t in enumerate(ids):
        if t == sep_token_id:
            sep_count += 1
            if sep_count == 2:
                idx = i + 1
                break
    # average from idx until EOS
    end = ids.index(2)  # EOS id
    return float(lp[idx:end].mean().item())    # average from idx until EOS