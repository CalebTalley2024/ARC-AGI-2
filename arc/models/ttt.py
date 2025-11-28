# models/ttt.py
"""
Test-Time Training (TTT) logic for the TinyLM model.

Contains utility functions to prepare a single test example and the main
function to perform adaptation and prediction.
"""

from __future__ import annotations

from random import random

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Any, Tuple

from arc.grids.core import Grid
# Import from your existing codebase
from arc.models.train import ArcPairsDataset, Collate
from arc.utils.constants import TRAINING_CONFIG


'''
# Configuration for TTT
ttt_config = {
    "augmentation_multiplier": 5,  # Generate 5 augmented copies per example
    "random": 0.5,  # 50% chance of random aug
    "specific": {"rotate": 0.1, "flip": 0.1}
}

# Initialize the trainer
ttt = TestTimeTrainer(
    model=model,
    learning_rate=5e-5,  # Usually lower than pre-training LR
    steps=20,  # 10-50 steps is usually sufficient
    augmentation_config=ttt_config
)
'''

import random
import copy


def prepare_ttt_dataset(eval_task):
    """
    Splits the training examples of a task into a TTT-Train set and a
    Pseudo-Test set (validation) for Test-Time Training.

    The eval task should have more than one example.
    """
    new_task = {}
    train_examples = copy.deepcopy(eval_task['train'])
    random.shuffle(train_examples)
    new_task['test'] = [train_examples[0]]
    new_task['train'] = train_examples[1:]

    return new_task

class TestTimeTrainer:
    """
    Handles temporary fine-tuning of a model on a specific task's examples.
    """

    def __init__(
            self,
            model: nn.Module,
            learning_rate: float = 1e-4,
            steps: int = 10,
            batch_size: int = 4,
            augmentation_config: Dict = None
    ):
        self.model = model
        self.device = next(model.parameters()).device
        self.lr = learning_rate
        self.steps = steps
        self.batch_size = batch_size
        self.augmentation_config = augmentation_config

        # Loss function (same as training)
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=TRAINING_CONFIG["ignore_index"])

    def cache_weights(self) -> Dict[str, torch.Tensor]:
        """Saves a copy of the model weights to CPU memory."""
        return {k: v.cpu().clone() for k, v in self.model.state_dict().items()}

    def restore_weights(self, state_dict: Dict[str, torch.Tensor]):
        """Restores the model weights from the cache."""
        self.model.load_state_dict(state_dict)

    def train_on_task(self, task: Dict[str, Any]):
        """
        Performs Test-Time Training on the 'train' pairs of a single task.

        Args:
            task: A single task dictionary containing 'train' and 'test' keys.
        """
        # 1. Create a dataset just for this task
        ttt_tasks = prepare_ttt_dataset(task)
        dataset = ArcPairsDataset(
            tasks=ttt_tasks,
            mode=TRAINING_CONFIG["serialization_mode"],
            max_len=TRAINING_CONFIG["max_sequence_length"],
            augmentation_config=self.augmentation_config
        )

        # If the task has too few examples, we might not need a dataloader,
        # but using one ensures consistency with your main training loop.
        dl = DataLoader(
            dataset,
            batch_size=min(len(dataset), self.batch_size),
            shuffle=True,
            collate_fn=Collate()
        )

        # 2. Setup Optimizer (Reset every time TTT is called)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)

        # 3. Training Loop
        self.model.train()

        for _ in range(self.steps):
            for x, y in dl:
                x, y = x.to(self.device), y.to(self.device)

                optimizer.zero_grad()

                # Forward pass
                logits = self.model(x)

                # Calculate loss
                loss = self.loss_fn(logits.view(-1, logits.size(-1)), y.view(-1))

                # Backward pass
                loss.backward()
                optimizer.step()

        # Switch back to eval mode for the actual inference
        self.model.eval()