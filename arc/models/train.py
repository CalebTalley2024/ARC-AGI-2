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

from arc.io.loader import load_task, load_tasks  # implement JSON loader
from arc.models.tiny_lm import TinyLM, TinyLMConfig
from arc.serialize.task_tokenizer import pack_example
from arc.utils.constants import (MODEL_CONFIG, PAD, TRAINING_CONFIG,
                                 VOCAB_SIZE, get_model_config,
                                 get_training_config)


class ArcPairsDataset(Dataset):
    def __init__(self, tasks, mode=None, max_len=None):
        self.examples = []
        # Use constants with fallback to function parameters
        mode = mode or TRAINING_CONFIG['serialization_mode']
        max_len = max_len or TRAINING_CONFIG['max_sequence_length']
        
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
    def __init__(self, pad_id=None):
        self.pad = pad_id or TRAINING_CONFIG['pad_token_id']
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
def train(model_dir: str, data_path: str, 
          steps: int | None = None, bs: int | None = None, lr: float | None = None, 
          d_model: int | None = None, training_profile: str | None = None, model_size: str | None = None):
    """
    Train TinyLM model with centralized configuration management.
    
    Args:
        model_dir: Directory to save model checkpoints
        data_path: Path to training data
        steps: Number of training steps (uses TRAINING_CONFIG default if None)
        bs: Batch size (uses TRAINING_CONFIG default if None)
        lr: Learning rate (uses TRAINING_CONFIG default if None)
        d_model: Model dimension (uses MODEL_CONFIG default if None)
        training_profile: Use predefined training profile ('debug', 'small_gpu', etc.)
        model_size: Use predefined model size ('tiny', 'small', 'medium', 'large')
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("*" * 20)
    print(f"Using device: {device}")
    print("*" * 20)
    
    # Get configurations
    if training_profile:
        train_config = get_training_config(training_profile)
    else:
        train_config = TRAINING_CONFIG.copy()
    
    if model_size:
        model_config = get_model_config(model_size)
    else:
        model_config = MODEL_CONFIG.copy()
    
    # Override with function parameters if provided
    steps = steps or train_config['steps']
    bs = bs or train_config['batch_size']
    lr = lr or train_config['learning_rate']
    grad_accum_steps = train_config['grad_accumulation_steps']
    if d_model:
        model_config['d_model'] = d_model
    
    effective_batch_size = bs * grad_accum_steps
    print(f"Training config: steps={steps}, batch_size={bs}, lr={lr}")
    print(f"Gradient accumulation: {grad_accum_steps} steps, effective_batch_size={effective_batch_size}")
    print(f"Model config: d_model={model_config['d_model']}, n_layers={model_config['n_layers']}")
    
    # Load data and create dataset
    tasks = load_tasks(data_path)  # returns List[Task]
    ds = ArcPairsDataset(tasks, 
                        mode=train_config['serialization_mode'], 
                        max_len=train_config['max_sequence_length'])
    dl = DataLoader(ds, batch_size=bs, shuffle=True, drop_last=True, collate_fn=Collate())
    
    # Create model
    cfg = TinyLMConfig(vocab_size=VOCAB_SIZE, **{k: v for k, v in model_config.items() if k != 'vocab_size'})
    model = TinyLM(cfg).to(device)
    
    # Optimizer with config values
    opt = torch.optim.AdamW(model.parameters(), lr=lr, 
                           betas=train_config['betas'], 
                           weight_decay=train_config['weight_decay'])
    scaler = torch.cuda.amp.GradScaler(enabled=(device=='cuda' and train_config['use_amp']))
    loss_fn = nn.CrossEntropyLoss(ignore_index=train_config['ignore_index'])

    # Best model tracking
    best_loss = float('inf')
    
    it = iter(dl)
    pbar = tqdm(range(steps), total=steps)
    for step in pbar:
        # Gradient accumulation loop
        total_loss = 0.0
        opt.zero_grad(set_to_none=True)
        
        for accum_step in range(grad_accum_steps):
            try:
                x, y = next(it)
            except StopIteration:
                it = iter(dl); x,y = next(it)
            x,y = x.to(device), y.to(device)
            
            with torch.cuda.amp.autocast(enabled=(device=='cuda' and train_config['use_amp'])):
                logits, = model(x)
                loss = loss_fn(logits.view(-1, logits.size(-1)), y.view(-1))
                # Scale loss by accumulation steps for proper averaging
                loss = loss / grad_accum_steps
            
            scaler.scale(loss).backward()
            total_loss += loss.item()
        
        # Update parameters after accumulating gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), train_config['grad_clip_norm'])
        scaler.step(opt)
        scaler.update()
        
        # Use total_loss for logging (already averaged)
        pbar.set_description(f"loss={total_loss:.3f}")
        
        # Save best model if current loss is better
        if total_loss < best_loss:
            best_loss = total_loss
            out = Path(model_dir); out.mkdir(parents=True, exist_ok=True)
            torch.save({'model': model.state_dict(), 'cfg': cfg.__dict__, 'loss': best_loss}, out/"best.pt")
            pbar.set_description(f"loss={total_loss:.3f} ★NEW BEST★")
        
        if (step+1) % train_config['save_every'] == 0:
            out = Path(model_dir); out.mkdir(parents=True, exist_ok=True)
            torch.save({'model': model.state_dict(), 'cfg': cfg.__dict__}, out/f"ckpt_{step+1}.pt")
    
    # final save
    out = Path(model_dir); out.mkdir(parents=True, exist_ok=True)
    torch.save({'model': model.state_dict(), 'cfg': cfg.__dict__}, out/"final.pt")