# models/infer.py
from __future__ import annotations
from typing import List
import torch


@torch.no_grad()
def greedy_generate(model, input_ids: torch.LongTensor, max_new_tokens: int, eos_id: int) -> torch.LongTensor:
    """
    Greedy generate a sequence from the model.

    Args:
        model: The model to use for generation.
        input_ids: The input ids to use for generation.
        max_new_tokens: The maximum number of new tokens to generate.
        eos_id: The end of sequence token id.
    """
    device = next(model.parameters()).device
    cur = input_ids.to(device)
    for _ in range(max_new_tokens):
        logits, = model(cur)
        next_id = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        cur = torch.cat([cur, next_id], dim=1)
        if int(next_id.item()) == eos_id:
            break
    return cur


@torch.no_grad()
def sample_generate(
    model,
    input_ids: torch.LongTensor,
    max_new_tokens: int,
    eos_id: int,
    temperature: float = 1.0,
    top_p: float = 0.95,
    seed: int = 0,
) -> torch.LongTensor:
    g = torch.Generator(device=next(model.parameters()).device).manual_seed(seed)
    cur = input_ids
    for _ in range(max_new_tokens):
        (logits,) = model(cur)
        logits = logits[:, -1, :] / max(1e-6, temperature)
        probs = torch.softmax(logits, dim=-1)
        # nucleus sampling
        sorted_probs, sorted_idx = torch.sort(probs, descending=True)
        cumsum = torch.cumsum(sorted_probs, dim=-1)
        mask = cumsum > top_p
        mask[..., 1:] = mask[..., :-1].clone()
        mask[..., 0] = False
        sorted_probs[mask] = 0
        sorted_probs.div_(sorted_probs.sum(dim=-1, keepdim=True))
        next_id = torch.multinomial(sorted_probs, num_samples=1, generator=g)
        next_id = sorted_idx.gather(-1, next_id)
        cur = torch.cat([cur, next_id], dim=1)
        if int(next_id.item()) == eos_id:
            break
    return cur
