# models/train.py
from __future__ import annotations

import json
import math
import random
import shutil
from datetime import datetime
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


def setup_drive_backup(drive_backup_path: str) -> bool:
    """Setup Google Drive backup directory and check accessibility."""
    try:
        # Check if Google Drive is mounted (Colab environment)
        drive_path = Path(drive_backup_path)
        if not drive_path.parent.exists():
            print("Google Drive not mounted or path not accessible. Skipping Drive backup.")
            return False
        
        # Create backup directories
        drive_path.mkdir(parents=True, exist_ok=True)
        (drive_path / "best_checkpoints").mkdir(exist_ok=True)
        (drive_path / "backup_history").mkdir(exist_ok=True)
        
        print(f"Google Drive backup setup at: {drive_path}")
        return True
    except Exception as e:
        print(f"Failed to setup Google Drive backup: {e}")
        return False


def backup_checkpoint_to_drive(checkpoint_path: Path, drive_backup_path: str) -> bool:
    """Backup a checkpoint to Google Drive."""
    try:
        if not checkpoint_path.exists():
            return False
        
        drive_path = Path(drive_backup_path)
        
        # Current backup location
        current_backup = drive_path / "best_checkpoints" / "best.pt"
        
        # Timestamped backup in history
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        history_backup = drive_path / "backup_history" / f"best_{timestamp}.pt"
        
        # Copy to both locations
        shutil.copy2(checkpoint_path, current_backup)
        shutil.copy2(checkpoint_path, history_backup)
        
        file_size_mb = checkpoint_path.stat().st_size / (1024 * 1024)
        print(f"Checkpoint backed up to Drive ({file_size_mb:.1f} MB)")
        return True
        
    except Exception as e:
        print(f"Failed to backup to Drive: {e}")
        return False


class ArcPairsDataset(Dataset):
    def __init__(self, tasks, mode=None, max_len=None):
        self.examples = []
        # Use constants with fallback to function parameters
        mode = mode or TRAINING_CONFIG['serialization_mode']
        max_len = max_len or TRAINING_CONFIG['max_sequence_length']
        
        for task_dict in tasks:
            # Access 'train' key from task dictionary
            for example in task_dict['train']:
                # Convert lists to Grid objects
                from arc.grids.core import Grid
                input_grid = Grid(np.array(example['input'], dtype=np.int8))
                output_grid = Grid(np.array(example['output'], dtype=np.int8))
                
                # Convert to token sequence
                seq = pack_example(input_grid, output_grid, mode)
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
          d_model: int | None = None, training_profile: str | None = None, model_size: str | None = None,
          resume_from_checkpoint: bool = True, enable_drive_backup: bool = True, 
          drive_backup_path: str = "/content/drive/MyDrive/ARC_AGI_Checkpoints",
          enable_lr_scheduling: bool = True, patience: int = 1000, lr_reduction_factor: float = 0.5):
    """
    Train TinyLM model with centralized configuration management and Google Drive backup.
    
    Args:
        model_dir: Directory to save model checkpoints
        data_path: Path to training data
        steps: Number of training steps (uses TRAINING_CONFIG default if None)
        bs: Batch size (uses TRAINING_CONFIG default if None)
        lr: Learning rate (uses TRAINING_CONFIG default if None)
        d_model: Model dimension (uses MODEL_CONFIG default if None)
        training_profile: Use predefined training profile ('debug', 'small_gpu', etc.)
        model_size: Use predefined model size ('tiny', 'small', 'medium', 'large')
        resume_from_checkpoint: Whether to resume from existing best.pt checkpoint if available
        enable_drive_backup: Whether to automatically backup checkpoints to Google Drive
        drive_backup_path: Google Drive path for checkpoint backups
        enable_lr_scheduling: Whether to enable adaptive learning rate scheduling
        patience: Number of steps to wait before reducing LR when loss plateaus
        lr_reduction_factor: Factor to multiply LR by when reducing (e.g., 0.5 = half the LR)
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("*" * 20)
    print(f"Using device: {device}")
    print("*" * 20)
    
    # Setup Google Drive backup if enabled
    drive_backup_enabled = False
    if enable_drive_backup:
        drive_backup_enabled = setup_drive_backup(drive_backup_path)
        if drive_backup_enabled:
            print("✓ Google Drive automatic backup enabled")
        else:
            print("⚠ Google Drive backup disabled - training will continue without cloud backup")
    
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
    
    # Get LR scheduling parameters from config with function parameter overrides
    if enable_lr_scheduling is True:  # Only use config if user didn't override
        enable_lr_scheduling = train_config.get('lr_scheduling_enabled', True)
    patience = train_config.get('lr_patience', patience)
    lr_reduction_factor = train_config.get('lr_reduction_factor', lr_reduction_factor)
    
    if d_model:
        model_config['d_model'] = d_model
    
    effective_batch_size = bs * grad_accum_steps
    print(f"Training config: steps={steps}, batch_size={bs}, lr={lr}")
    print(f"Gradient accumulation: {grad_accum_steps} steps, effective_batch_size={effective_batch_size}")
    print(f"Model config: d_model={model_config['d_model']}, n_layers={model_config['n_layers']}")
    
    if enable_lr_scheduling:
        print(f"LR Scheduling: enabled (patience={patience}, reduction_factor={lr_reduction_factor})")
    else:
        print("LR Scheduling: disabled")
    
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
    scaler = torch.amp.GradScaler(device, enabled=(device=='cuda' and train_config['use_amp']))
    loss_fn = nn.CrossEntropyLoss(ignore_index=train_config['ignore_index'])

    # Learning rate scheduling setup
    current_lr = lr
    steps_without_improvement = 0
    last_improvement_step = 0
    min_lr = lr * train_config.get('min_lr_factor', 0.01)  # Don't reduce LR below min factor
    lr_reductions = 0
    max_lr_reductions = train_config.get('max_lr_reductions', 5)
    
    # Best model tracking and checkpoint resuming
    best_loss = float('inf')
    start_step = 0
    
    # Check for existing checkpoint to resume from
    checkpoint_path = Path(model_dir) / "best.pt"
    if resume_from_checkpoint and checkpoint_path.exists():
        print(f"Found existing checkpoint: {checkpoint_path}")
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device)
            
            # Load model state
            model.load_state_dict(checkpoint['model'])
            print(f"Loaded model state from checkpoint")
            
            # Load best loss if available
            if 'loss' in checkpoint:
                best_loss = checkpoint['loss']
                print(f"Resuming with best loss: {best_loss:.4f}")
            
            # Load optimizer state if available
            if 'optimizer' in checkpoint:
                opt.load_state_dict(checkpoint['optimizer'])
                print(f"Loaded optimizer state from checkpoint")
            
            # Load scaler state if available
            if 'scaler' in checkpoint:
                scaler.load_state_dict(checkpoint['scaler'])
                print(f"Loaded scaler state from checkpoint")
            
            # Load training step if available
            if 'step' in checkpoint:
                start_step = checkpoint['step'] + 1
                print(f"Resuming from step: {start_step}")
            
            # Load learning rate scheduling state if available
            if 'lr_scheduling' in checkpoint:
                lr_sched = checkpoint['lr_scheduling']
                current_lr = lr_sched.get('current_lr', lr)
                steps_without_improvement = lr_sched.get('steps_without_improvement', 0)
                last_improvement_step = lr_sched.get('last_improvement_step', 0)
                lr_reductions = lr_sched.get('lr_reductions', 0)
                
                # Update optimizer learning rate
                for param_group in opt.param_groups:
                    param_group['lr'] = current_lr
                print(f"Resumed LR scheduling: current_lr={current_lr:.6f}, reductions={lr_reductions}")
            
            print("Successfully resumed from checkpoint!")
            
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            print("Starting training from scratch...")
            best_loss = float('inf')
            start_step = 0
    else:
        if resume_from_checkpoint:
            print(f"No checkpoint found at {checkpoint_path}, starting from scratch")
        else:
            print("Resume disabled, starting from scratch")
    
    it = iter(dl)
    # Adjust range to account for resumed training
    remaining_steps = steps - start_step
    pbar = tqdm(range(remaining_steps), total=remaining_steps, initial=0)
    pbar.set_description(f"Starting from step {start_step}")
    
    for step_idx in pbar:
        step = start_step + step_idx  # Actual step number
        # Gradient accumulation loop
        total_loss = 0.0
        opt.zero_grad(set_to_none=True)
        
        for accum_step in range(grad_accum_steps):
            try:
                x, y = next(it)
            except StopIteration:
                it = iter(dl); x,y = next(it)
            x,y = x.to(device), y.to(device)
            
            with torch.amp.autocast(device, enabled=(device=='cuda' and train_config['use_amp'])):
                logits = model(x)
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
            last_improvement_step = step
            steps_without_improvement = 0
            
            out = Path(model_dir); out.mkdir(parents=True, exist_ok=True)
            # Save comprehensive checkpoint for resuming
            checkpoint_dict = {
                'model': model.state_dict(),
                'cfg': cfg.__dict__,
                'loss': best_loss,
                'step': step,
                'optimizer': opt.state_dict(),
                'scaler': scaler.state_dict(),
                'lr': current_lr,  # Save current LR instead of original
                'training_config': train_config,
                'model_config': model_config,
                'lr_scheduling': {  # Save LR scheduling state
                    'current_lr': current_lr,
                    'steps_without_improvement': steps_without_improvement,
                    'last_improvement_step': last_improvement_step,
                    'lr_reductions': lr_reductions
                }
            }
            
            # Save local checkpoint
            best_checkpoint_path = out / "best.pt"
            torch.save(checkpoint_dict, best_checkpoint_path)
            
            # Automatically backup to Google Drive if enabled
            if drive_backup_enabled:
                backup_checkpoint_to_drive(best_checkpoint_path, drive_backup_path)
            
            pbar.set_description(f"loss={total_loss:.3f} ★NEW BEST★ (step {step}) lr={current_lr:.6f}")
        else:
            # Update steps without improvement for LR scheduling
            steps_without_improvement = step - last_improvement_step
        
        # Learning rate scheduling - reduce LR if no improvement for patience steps
        if (enable_lr_scheduling and 
            steps_without_improvement >= patience and 
            lr_reductions < max_lr_reductions and 
            current_lr > min_lr):
            
            # Reduce learning rate
            new_lr = current_lr * lr_reduction_factor
            current_lr = max(new_lr, min_lr)  # Don't go below minimum LR
            
            # Update optimizer learning rate
            for param_group in opt.param_groups:
                param_group['lr'] = current_lr
            
            lr_reductions += 1
            steps_without_improvement = 0  # Reset counter after LR reduction
            
            print(f"\n🔄 LR REDUCED: {new_lr:.6f} → {current_lr:.6f} (reduction #{lr_reductions}/{max_lr_reductions})")
            pbar.set_description(f"loss={total_loss:.3f} 📉LR REDUCED📉 lr={current_lr:.6f}")
        
        if (step+1) % train_config['save_every'] == 0:
            out = Path(model_dir); out.mkdir(parents=True, exist_ok=True)
            # Save regular checkpoint with full state
            checkpoint_dict = {
                'model': model.state_dict(),
                'cfg': cfg.__dict__,
                'loss': total_loss,
                'step': step,
                'optimizer': opt.state_dict(),
                'scaler': scaler.state_dict(),
                'lr': current_lr,  # Save current LR
                'training_config': train_config,
                'model_config': model_config,
                'lr_scheduling': {  # Save LR scheduling state
                    'current_lr': current_lr,
                    'steps_without_improvement': steps_without_improvement,
                    'last_improvement_step': last_improvement_step,
                    'lr_reductions': lr_reductions
                }
            }
            torch.save(checkpoint_dict, out/f"ckpt_{step+1}.pt")
    
    # final save
    out = Path(model_dir); out.mkdir(parents=True, exist_ok=True)
    final_checkpoint = {
        'model': model.state_dict(),
        'cfg': cfg.__dict__,
        'loss': total_loss,
        'step': step,
        'optimizer': opt.state_dict(),
        'scaler': scaler.state_dict(),
        'lr': current_lr,  # Save final LR
        'training_config': train_config,
        'model_config': model_config,
        'training_completed': True,
        'lr_scheduling': {  # Save final LR scheduling state
            'current_lr': current_lr,
            'steps_without_improvement': steps_without_improvement,
            'last_improvement_step': last_improvement_step,
            'lr_reductions': lr_reductions
        }
    }
    final_checkpoint_path = out / "final.pt"
    torch.save(final_checkpoint, final_checkpoint_path)
    
    # Final backup to Google Drive if enabled
    if drive_backup_enabled:
        print("Performing final backup to Google Drive...")
        backup_checkpoint_to_drive(out / "best.pt", drive_backup_path)
        
        # Also backup final checkpoint to history
        try:
            drive_path = Path(drive_backup_path)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            final_backup = drive_path / "backup_history" / f"final_{timestamp}.pt"
            shutil.copy2(final_checkpoint_path, final_backup)
            print(f"✓ Final checkpoint backed up: {final_backup}")
        except Exception as e:
            print(f"⚠ Failed to backup final checkpoint: {e}")
    
    print("Training completed!")
    if drive_backup_enabled:
        print(f"✓ All checkpoints backed up to: {drive_backup_path}")