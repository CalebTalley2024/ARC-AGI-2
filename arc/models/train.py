# models/train.py
from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from arc.io import load_tasks  # implement JSON loader
from arc.models.tiny_lm import TinyLM, TinyLMConfig
from arc.serialize import VOCAB_SIZE, pack_example


class ArcPairsDataset(Dataset):
    def __init__(self, tasks, mode='row', max_len=1024):
        self.examples = []
        for t in tasks:
            for ex in t.train:
                seq = pack_example(ex.x, ex.y, mode)
                if len(seq) <= max_len:
                    self.examples.append(seq)
        random.shuffle(self.examples)
        self.max_len = max_len

    def __len__(self): return len(self.examples)

    def __getitem__(self, idx):
        seq = self.examples[idx]
        x = torch.tensor(seq[:-1], dtype=torch.long)
        y = torch.tensor(seq[1:], dtype=torch.long)
        return x, y

class Collate:
    def __init__(self, pad_id=0):
        self.pad = pad_id
    def __call__(self, batch):
        xs, ys = zip(*batch)
        T = max(t.size(0) for t in xs)
        bx = torch.full((len(xs), T), self.pad, dtype=torch.long)
        by = torch.full((len(xs), T), self.pad, dtype=torch.long)
        for i,(x,y) in enumerate(zip(xs, ys)):
            bx[i,:x.size(0)] = x
            by[i,:y.size(0)] = y
        return bx, by

# def train(..., bs: int = 8, grad_accum_steps: int = 4): for 4096
def train(model_dir: str, data_path: str, steps: int = 100_000, bs: int = 32, lr: float = 3e-4, d_model: int = 448):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("*" * 20)
    print(f"Using device: {device}")
    print("*" * 20)
    tasks = load_tasks(data_path)  # returns List[Task]
    ds = ArcPairsDataset(tasks, mode='row', max_len=1024)
    dl = DataLoader(ds, batch_size=bs, shuffle=True, drop_last=True, collate_fn=Collate())
    cfg = TinyLMConfig(vocab_size=VOCAB_SIZE, d_model=d_model)
    model = TinyLM(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.01)
    scaler = torch.cuda.amp.GradScaler(enabled=(device=='cuda'))
    loss_fn = nn.CrossEntropyLoss(ignore_index=0)

    # Best model tracking
    best_loss = float('inf')
    
    it = iter(dl)
    pbar = tqdm(range(steps), total=steps)
    for step in pbar:
        try:
            x, y = next(it)
        except StopIteration:
            it = iter(dl); x,y = next(it)
        x,y = x.to(device), y.to(device)
        with torch.cuda.amp.autocast(enabled=(device=='cuda')):
            logits, = model(x)
            loss = loss_fn(logits.view(-1, logits.size(-1)), y.view(-1))
        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
        pbar.set_description(f"loss={loss.item():.3f}")
        
        # Save best model if current loss is better
        if loss.item() < best_loss:
            best_loss = loss.item()
            out = Path(model_dir); out.mkdir(parents=True, exist_ok=True)
            torch.save({'model': model.state_dict(), 'cfg': cfg.__dict__, 'loss': best_loss}, out/"best.pt")
            pbar.set_description(f"loss={loss.item():.3f} ★NEW BEST★")
        
        if (step+1) % 1000 == 0:
            out = Path(model_dir); out.mkdir(parents=True, exist_ok=True)
            torch.save({'model': model.state_dict(), 'cfg': cfg.__dict__}, out/f"ckpt_{step+1}.pt")
    # final save
    out = Path(model_dir); out.mkdir(parents=True, exist_ok=True)
    torch.save({'model': model.state_dict(), 'cfg': cfg.__dict__}, out/"final.pt")