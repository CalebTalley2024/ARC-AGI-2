# models/augmentation.py
"""
Data augmentation utilities for ARC-AGI training.

This module provides functions to apply various transformations to grid pairs:
- Geometric transformations (rotations, flips, transposes)
- Color permutations
- Random combinations of transformations
"""

from __future__ import annotations

import random
from typing import Tuple

import numpy as np

from arc.grids.core import Grid
from arc.grids.views import (D4, ViewSpec, apply_view_grid,
                             generate_palette_permutations, identity_cmap)


def get_random_augmentation(
    palette: set[int] | None = None,
    augmentation_type: str = "random",
    seed: int = 42,
    is_square: bool = True,
) -> ViewSpec:
    """
    Get a random augmentation ViewSpec based on the specified type.

    Args:
        palette: Set of colors used in the task (for color augmentation)
        augmentation_type: Type of augmentation to apply:
            - 'random': Randomly choose between geometric and color augmentations
            - 'geometric': Apply only geometric transformations (rotation, flip, transpose)
            - 'color': Apply only color permutations
            - 'identity': No augmentation (identity transform)
            - 'rotation': Random rotation (90, 180, 270)
            - 'flip': Random flip (horizontal or vertical)
            - 'transpose': Transpose variations
        seed: Random seed for reproducibility
        is_square: Whether the grid is square (height == width). Required for
            transpose, rotation 90, and rotation 270 transformations.

    Returns:
        ViewSpec with the selected augmentation
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    # Default palette if not provided
    if palette is None:
        palette = set(range(10))

    # Choose augmentation based on type
    if augmentation_type == "identity":
        geom = "id"
        color_map = identity_cmap()

    elif augmentation_type == "geometric":
        # Randomly select a geometric transformation
        # For non-square grids, exclude transformations that change dimensions
        if is_square:
            geom = random.choice(D4)
        else:
            # Only use transformations that preserve dimensions (180° rotation and flips)
            geom = random.choice(["rot180", "flip_h", "flip_v", "id"])
        color_map = identity_cmap()

    elif augmentation_type == "rotation":
        # Only rotation transformations
        # For non-square grids, only 180° rotation is allowed
        if is_square:
            geom = random.choice(["rot90", "rot180", "rot270"])
        else:
            geom = "rot180"  # Only 180° rotation preserves dimensions
        color_map = identity_cmap()

    elif augmentation_type == "flip":
        # Only flip transformations (always safe for any grid dimensions)
        geom = random.choice(["flip_h", "flip_v"])
        color_map = identity_cmap()

    elif augmentation_type == "transpose":
        # Transpose variations - only for square grids
        if is_square:
            geom = random.choice(["transpose", "transpose_flip", "flip_transpose"])
        else:
            # Fall back to flip if not square
            geom = random.choice(["flip_h", "flip_v"])
        color_map = identity_cmap()

    elif augmentation_type == "color":
        # Only color permutations
        geom = "id"
        color_maps = generate_palette_permutations(palette, max_count=8)
        color_map = random.choice(color_maps) if len(color_maps) > 1 else identity_cmap()

    elif augmentation_type == "random":
        # Random geometric transformation
        # For non-square grids, exclude transformations that change dimensions
        if is_square:
            geom = random.choice(D4)
        else:
            # Only use transformations that preserve dimensions
            geom = random.choice(["rot180", "flip_h", "flip_v", "id"])
        color_map = identity_cmap()

        # 50% chance to also apply color permutation
        if random.random() < 0.5:
            color_maps = generate_palette_permutations(palette, max_count=8)
            if len(color_maps) > 1:
                color_map = random.choice(color_maps)

    else:
        raise ValueError(f"Unknown augmentation type: {augmentation_type}")

    return ViewSpec(geom=geom, color_map=color_map, serialization="row")


def apply_augmentation_to_example(
    input_grid: Grid,
    output_grid: Grid,
    augmentation_type: str = "random",
) -> Tuple[Grid, Grid]:
    """
    Apply augmentation to a single input-output pair.

    Args:
        input_grid: Input grid
        output_grid: Output grid
        augmentation_type: Type of augmentation to apply

    Returns:
        Tuple of (augmented_input, augmented_output)
    """
    # Extract palette from both grids
    palette = set(np.unique(input_grid.a)) | set(np.unique(output_grid.a))

    # Check if both grids are square (required for certain transformations)
    # Both input and output must be square for dimension-changing transformations
    input_is_square = input_grid.a.shape[0] == input_grid.a.shape[1]
    output_is_square = output_grid.a.shape[0] == output_grid.a.shape[1]
    is_square = input_is_square and output_is_square

    # Get random augmentation (with square constraint)
    view_spec = get_random_augmentation(palette, augmentation_type, is_square=is_square)

    # Apply same augmentation to both input and output
    aug_input = apply_view_grid(input_grid, view_spec)
    aug_output = apply_view_grid(output_grid, view_spec)

    return aug_input, aug_output


def sample_augmentation_type(augmentation_config: dict) -> str:
    """
    Sample augmentation type based on probability distribution from config.

    Args:
        augmentation_config: Dictionary specifying augmentation strategy:
            {
                'random': float,      # Probability of random augmentation
                'None': float,        # Probability of no augmentation (identity)
                'specific': {         # Specific augmentation types
                    'geometric': float,
                    'color': float,
                    'rotation': float,
                    'flip': float,
                    'transpose': float,
                }
            }

    Returns:
        Selected augmentation type as string
    """
    # Build probability distribution
    choices = []
    probs = []

    # Add 'random' option
    random_prob = augmentation_config.get("random", 0)
    if random_prob > 0:
        choices.append("random")
        probs.append(random_prob)

    # Add 'None' (identity) option
    none_prob = augmentation_config.get("None", 0)
    if none_prob > 0:
        choices.append("identity")
        probs.append(none_prob)

    # Add specific augmentation types
    specific = augmentation_config.get("specific", {})
    for aug_type, prob in specific.items():
        if prob > 0:
            choices.append(aug_type)
            probs.append(prob)

    # If no choices, return identity
    if not choices:
        return "identity"

    # Normalize probabilities
    total = sum(probs)
    if total > 0:
        probs = [p / total for p in probs]
    else:
        return "identity"

    # Sample from distribution
    return np.random.choice(choices, p=probs)


def is_augmentation_enabled(augmentation_config: dict) -> bool:
    """
    Check if augmentation is enabled in the configuration.

    Args:
        augmentation_config: Augmentation configuration dictionary

    Returns:
        True if any augmentation is enabled, False otherwise
    """
    return (
        augmentation_config.get("random", 0) > 0
        or any(prob > 0 for prob in augmentation_config.get("specific", {}).values())
    )
