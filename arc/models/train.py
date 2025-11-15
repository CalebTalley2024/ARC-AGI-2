# models/train.py
"""
Main training script for TinyLM model on ARC-AGI tasks.

This module provides the training loop and dataset implementation,
with augmentation and backup functionality imported from separate modules.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from arc.grids.core import Grid
from arc.io.loader import load_tasks
from arc.models.augmentation import (apply_augmentation_to_example,
                                     is_augmentation_enabled,
                                     sample_augmentation_type)
from arc.models.backup import (backup_checkpoint_to_drive,
                               backup_final_checkpoint, setup_drive_backup)
from arc.models.tiny_lm import TinyLM, TinyLMConfig
from arc.serialize.task_tokenizer import pack_example
from arc.utils.constants import (AUGMENTATION_CONFIG, MODEL_CONFIG,
                                 TRAINING_CONFIG, VOCAB_SIZE, get_model_config,
                                 get_training_config)


class ArcPairsDataset(Dataset):
    """
    Dataset for ARC task pairs with optional data augmentation.
    
    Loads task pairs and applies augmentation on-the-fly during training.
    """

    def __init__(self, tasks, mode=None, max_len=None, augmentation_config=None):
        """
        Initialize ARC pairs dataset.

        Args:
            tasks: List of task dictionaries
            mode: Serialization mode ('row' or 'col')
            max_len: Maximum sequence length
            augmentation_config: Dictionary specifying augmentation strategy.
                If None, uses AUGMENTATION_CONFIG from constants.
        """
        self.examples = []
        # Use constants with fallback to function parameters
        mode = mode or TRAINING_CONFIG["serialization_mode"]
        max_len = max_len or TRAINING_CONFIG["max_sequence_length"]

        # Use provided augmentation config or default from constants
        self.augmentation_config = augmentation_config or AUGMENTATION_CONFIG

        # Check if augmentation is enabled
        self.use_augmentation = is_augmentation_enabled(self.augmentation_config)

        for task_dict in tasks:
            # Access 'train' key from task dictionary
            for example in task_dict["train"]:
                # Convert lists to Grid objects
                input_grid = Grid(np.array(example["input"], dtype=np.int8))
                output_grid = Grid(np.array(example["output"], dtype=np.int8))

                # Store original grids for augmentation
                if self.use_augmentation:
                    # Store original grids and generate augmented versions on-the-fly
                    self.examples.append(
                        {
                            "input": input_grid,
                            "output": output_grid,
                            "mode": mode,
                            "is_augmented": False,
                        }
                    )
                else:
                    # Convert to token sequence directly
                    seq = pack_example(input_grid, output_grid, mode)
                    if len(seq) <= max_len:
                        self.examples.append({"sequence": seq})

        random.shuffle(self.examples)
        self.max_len = max_len
        self.mode = mode

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        example = self.examples[idx]

        if self.use_augmentation and "input" in example:
            # Sample augmentation type based on probability distribution
            aug_type = sample_augmentation_type(self.augmentation_config)

            input_grid = example["input"]
            output_grid = example["output"]

            # Apply augmentation if not identity
            if aug_type != "identity":
                input_grid, output_grid = apply_augmentation_to_example(
                    input_grid, output_grid, aug_type
                )

            # Convert augmented grids to sequence
            seq = pack_example(input_grid, output_grid, self.mode)

            # Truncate if needed
            if len(seq) > self.max_len:
                seq = seq[: self.max_len]
        else:
            seq = example["sequence"]

        x = torch.tensor(seq[:-1], dtype=torch.long)
        y = torch.tensor(seq[1:], dtype=torch.long)
        return x, y

class Collate:
    """Collate function for DataLoader with padding."""

    def __init__(self, pad_id=None):
        self.pad = pad_id or TRAINING_CONFIG["pad_token_id"]

    def __call__(self, batch):
        xs, ys = zip(*batch)
        T = max(t.size(0) for t in xs)
        bx = torch.full((len(xs), T), self.pad, dtype=torch.long)
        by = torch.full((len(xs), T), self.pad, dtype=torch.long)
        for i, (x, y) in enumerate(zip(xs, ys)):
            bx[i, : x.size(0)] = x
            by[i, : y.size(0)] = y
        return bx, by

def train(
    model_dir: str,
    data_path: str,
    steps: int | None = None,
    bs: int | None = None,
    lr: float | None = None,
    d_model: int | None = None,
    training_profile: str | None = None,
    model_size: str | None = None,
    resume_from_checkpoint: bool = True,
    enable_drive_backup: bool = True,
    drive_backup_path: str = "/content/drive/MyDrive/ARC_AGI_Checkpoints",
    augmentation_config: dict | None = None,
):
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
        resume_from_checkpoint: Whether to resume from existing best.pt checkpoint
        enable_drive_backup: Whether to backup checkpoints to Google Drive
        drive_backup_path: Google Drive path for checkpoint backups
        augmentation_config: Dictionary specifying augmentation strategy.
            If None, uses AUGMENTATION_CONFIG from constants.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("*" * 20)
    print(f"Using device: {device}")
    print("*" * 20)
    
    # Setup Google Drive backup if enabled
    drive_backup_enabled = False
    if enable_drive_backup:
        drive_backup_enabled = setup_drive_backup(drive_backup_path)
        if drive_backup_enabled:
            print("Google Drive automatic backup enabled")
        else:
            print(
                "Google Drive backup disabled - training will continue without cloud backup"
            )

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
    steps = steps or train_config["steps"]
    bs = bs or train_config["batch_size"]
    lr = lr or train_config["learning_rate"]
    grad_accum_steps = train_config["grad_accumulation_steps"]
    if d_model:
        model_config["d_model"] = d_model

    effective_batch_size = bs * grad_accum_steps
    print(f"Training config: steps={steps}, batch_size={bs}, lr={lr}")
    print(
        f"Gradient accumulation: {grad_accum_steps} steps, effective_batch_size={effective_batch_size}"
    )
    print(
        f"Model config: d_model={model_config['d_model']}, n_layers={model_config['n_layers']}"
    )

    # Use provided augmentation config or default from constants
    aug_config = augmentation_config or AUGMENTATION_CONFIG

    # Check if augmentation is enabled
    use_augmentation = is_augmentation_enabled(aug_config)

    # Display augmentation settings
    if use_augmentation:
        print(f"Data Augmentation: ENABLED")
        print(f"  Configuration:")
        print(f"    random: {aug_config.get('random', 0)}")
        print(f"    None (identity): {aug_config.get('None', 0)}")
        specific = aug_config.get("specific", {})
        if any(prob > 0 for prob in specific.values()):
            print(f"    specific:")
            for aug_type, prob in specific.items():
                if prob > 0:
                    print(f"      {aug_type}: {prob}")
    else:
        print(f"Data Augmentation: DISABLED")
    
    # Load data and create dataset
    tasks = load_tasks(data_path)
    ds = ArcPairsDataset(
        tasks,
        mode=train_config["serialization_mode"],
        max_len=train_config["max_sequence_length"],
        augmentation_config=aug_config,
    )
    dl = DataLoader(
        ds, batch_size=bs, shuffle=True, drop_last=True, collate_fn=Collate()
    )

    # Create model
    cfg = TinyLMConfig(
        vocab_size=VOCAB_SIZE,
        **{k: v for k, v in model_config.items() if k != "vocab_size"},
    )
    model = TinyLM(cfg).to(device)

    # Optimizer with config values
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        betas=train_config["betas"],
        weight_decay=train_config["weight_decay"],
    )
    scaler = torch.amp.GradScaler(
        device, enabled=(device == "cuda" and train_config["use_amp"])
    )
    loss_fn = nn.CrossEntropyLoss(ignore_index=train_config["ignore_index"])

    # Best model tracking and checkpoint resuming
    best_loss = float("inf")
    start_step = 0

    # Check for existing checkpoint to resume from
    checkpoint_path = Path(model_dir) / "best.pt"
    if resume_from_checkpoint and checkpoint_path.exists():
        print(f"Found existing checkpoint: {checkpoint_path}")
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device)

            # Load model state
            model.load_state_dict(checkpoint["model"])
            print(f"Loaded model state from checkpoint")

            # Load best loss if available
            if "loss" in checkpoint:
                best_loss = checkpoint["loss"]
                print(f"Resuming with best loss: {best_loss:.4f}")

            # Load optimizer state if available
            if "optimizer" in checkpoint:
                opt.load_state_dict(checkpoint["optimizer"])
                print(f"Loaded optimizer state from checkpoint")

            # Load scaler state if available
            if "scaler" in checkpoint:
                scaler.load_state_dict(checkpoint["scaler"])
                print(f"Loaded scaler state from checkpoint")

            # Load training step if available
            if "step" in checkpoint:
                start_step = checkpoint["step"] + 1
                print(f"Resuming from step: {start_step}")

            print("Successfully resumed from checkpoint!")

        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            print("Starting training from scratch...")
            best_loss = float("inf")
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
                it = iter(dl)
                x, y = next(it)
            x, y = x.to(device), y.to(device)

            with torch.amp.autocast(
                device, enabled=(device == "cuda" and train_config["use_amp"])
            ):
                logits = model(x)
                loss = loss_fn(logits.view(-1, logits.size(-1)), y.view(-1))
                # Scale loss by accumulation steps for proper averaging
                loss = loss / grad_accum_steps

            scaler.scale(loss).backward()
            total_loss += loss.item()

        # Update parameters after accumulating gradients
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), train_config["grad_clip_norm"]
        )
        scaler.step(opt)
        scaler.update()

        # Use total_loss for logging (already averaged)
        pbar.set_description(f"loss={total_loss:.3f}")

        # Save best model if current loss is better
        if total_loss < best_loss:
            best_loss = total_loss
            out = Path(model_dir)
            out.mkdir(parents=True, exist_ok=True)
            # Save comprehensive checkpoint for resuming
            checkpoint_dict = {
                "model": model.state_dict(),
                "cfg": cfg.__dict__,
                "loss": best_loss,
                "step": step,
                "optimizer": opt.state_dict(),
                "scaler": scaler.state_dict(),
                "lr": lr,
                "training_config": train_config,
                "model_config": model_config,
            }

            # Save local checkpoint
            best_checkpoint_path = out / "best.pt"
            torch.save(checkpoint_dict, best_checkpoint_path)

            # Automatically backup to Google Drive if enabled
            if drive_backup_enabled:
                backup_checkpoint_to_drive(best_checkpoint_path, drive_backup_path)

            pbar.set_description(f"loss={total_loss:.3f} ★NEW BEST★ (step {step})")

        if (step + 1) % train_config["save_every"] == 0:
            out = Path(model_dir)
            out.mkdir(parents=True, exist_ok=True)
            # Save regular checkpoint with full state
            checkpoint_dict = {
                "model": model.state_dict(),
                "cfg": cfg.__dict__,
                "loss": total_loss,
                "step": step,
                "optimizer": opt.state_dict(),
                "scaler": scaler.state_dict(),
                "lr": lr,
                "training_config": train_config,
                "model_config": model_config,
            }
            torch.save(checkpoint_dict, out / f"ckpt_{step+1}.pt")
    
    # Final save
    out = Path(model_dir)
    out.mkdir(parents=True, exist_ok=True)
    final_checkpoint = {
        "model": model.state_dict(),
        "cfg": cfg.__dict__,
        "loss": total_loss,
        "step": step,
        "optimizer": opt.state_dict(),
        "scaler": scaler.state_dict(),
        "lr": lr,
        "training_config": train_config,
        "model_config": model_config,
        "training_completed": True,
    }
    final_checkpoint_path = out / "final.pt"
    torch.save(final_checkpoint, final_checkpoint_path)

    # Final backup to Google Drive if enabled
    if drive_backup_enabled:
        print("Performing final backup to Google Drive...")
        backup_checkpoint_to_drive(out / "best.pt", drive_backup_path)
        backup_final_checkpoint(final_checkpoint_path, drive_backup_path)

    print("Training completed!")
    if drive_backup_enabled:
        print(f"All checkpoints backed up to: {drive_backup_path}")