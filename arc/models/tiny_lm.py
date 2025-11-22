from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn

from arc.utils.constants import MODEL_CONFIG


@dataclass
class TinyLMConfig:
    vocab_size: int = MODEL_CONFIG['vocab_size']        # How many different tokens exist (from serialization)
    d_model: int = MODEL_CONFIG['d_model']              # Size of the hidden representation (embedding dimension)
    n_layers: int = MODEL_CONFIG['n_layers']            # How many transformer blocks to stack
    n_heads: int = MODEL_CONFIG['n_heads']              # Number of attention heads in each block
    d_ff: int = MODEL_CONFIG['d_ff']                    # Size of the feedforward network (usually 4 × d_model)
    p_drop: float = MODEL_CONFIG['p_drop']              # Dropout probability (randomly turns off 10% of connections during training)
    max_len: int = MODEL_CONFIG['max_len']              # Maximum sequence length the model can handle
    
#  Model size estimates: d_model n_layers n_heads d_ff Params
# ~20M 448 8 8 1792 20.2M
# ~30M 512 10 8 2048 32.5M
# ~50M 640 10 8 2560 50.5M

    
class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: TinyLMConfig):
        super().__init__()
        self.n_heads = cfg.n_heads              # Number of attention heads (8)
        self.d_head = cfg.d_model // cfg.n_heads # Size per head (448/8 = 56)
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model)  # Creates Q, K, V matrices
        self.proj = nn.Linear(cfg.d_model, cfg.d_model)      # Output projection
        self.drop = nn.Dropout(cfg.p_drop)                   # Dropout layer
        self.register_buffer('mask', torch.tril(torch.ones(cfg.max_len, cfg.max_len)).view(1, 1, cfg.max_len, cfg.max_len))                     # Causal mask (prevents looking ahead)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B,T,C)
        B, T, C = x.shape # Batch, Time/Sequence length, Channel = Dimensions
        qkv = self.qkv(x)  # (B,T,3C)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2)
        k = k.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2)
        v = v.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))
        att = att.softmax(dim=-1)
        y = att @ v  # (B,h,T,d)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.drop(self.proj(y))
        return y

class Block(nn.Module):
    def __init__(self, cfg: TinyLMConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.ff = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_ff),
            nn.GELU(),
            nn.Dropout(cfg.p_drop),
            nn.Linear(cfg.d_ff, cfg.d_model),
            nn.Dropout(cfg.p_drop),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x

class TinyLM(nn.Module):
    def __init__(self, cfg: TinyLMConfig):
        super().__init__()
        self.cfg = cfg
        self.tok = nn.Embedding(cfg.vocab_size, cfg.d_model)    # Token embeddings
        self.pos = nn.Embedding(cfg.max_len, cfg.d_model)       # Position embeddings
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])  # 8 blocks
        self.ln_f = nn.LayerNorm(cfg.d_model)                   # Final normalization
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)  # Output layer
    def forward(self, idx: torch.Tensor) -> torch.Tensor:  # idx: (B, T) = batch of token sequences
        B, T = idx.shape
        pos = torch.arange(0, T, device=idx.device).unsqueeze(0)  # Create position indices [0,1,2,...]
        x = self.tok(idx) + self.pos(pos)   # Token embeddings + position embeddings
        for blk in self.blocks:              # Pass through all 8 blocks
            x = blk(x)
        x = self.ln_f(x)                     # Final normalization
        return self.head(x)                   # Convert to vocabulary probabilities (returns tuple)