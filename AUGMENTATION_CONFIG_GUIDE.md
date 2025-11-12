# Data Augmentation Configuration Guide

## Overview

The training pipeline now uses a dictionary-based augmentation configuration system that allows flexible control over augmentation strategies and probabilities.

## Configuration Structure

The augmentation configuration is defined in `arc/utils/constants.py` as `AUGMENTATION_CONFIG`:

```python
AUGMENTATION_CONFIG = {
    'random': 0.0,      # Probability of random augmentation (mix of all types)
    'None': 1.0,        # Probability of no augmentation (identity)
    'specific': {       # Specific augmentation types with individual probabilities
        'geometric': 0.0,   # All geometric transforms
        'color': 0.0,       # Color permutations
        'rotation': 0.0,    # Rotation only
        'flip': 0.0,        # Flip only
        'transpose': 0.0,   # Transpose only
    }
}
```

## Augmentation Types

### Top-Level Options

1. **`'random'`**: Random mix of geometric and color augmentations
   - Randomly selects from all D4 geometric transformations
   - May also apply color permutations

2. **`'None'`**: No augmentation (identity transform)
   - Keeps the original grid unchanged
   - Useful for controlling the proportion of non-augmented data

### Specific Augmentation Types

3. **`'geometric'`**: All geometric transformations
   - Includes rotations, flips, and transposes
   - Preserves spatial structure

4. **`'color'`**: Color permutations only
   - Randomly permutes the color palette
   - Preserves spatial structure

5. **`'rotation'`**: Rotation transformations only
   - 90°, 180°, 270° rotations

6. **`'flip'`**: Flip transformations only
   - Horizontal and vertical flips

7. **`'transpose'`**: Transpose variations only
   - Transpose, transpose+flip combinations

## Usage Examples

### Example 1: No Augmentation (Default)
```python
AUGMENTATION_CONFIG = {
    'random': 0.0,
    'None': 1.0,
    'specific': {
        'geometric': 0.0,
        'color': 0.0,
        'rotation': 0.0,
        'flip': 0.0,
        'transpose': 0.0,
    }
}
```
**Result**: 100% of examples use original data, no augmentation applied.

### Example 2: 50% Random Augmentation
```python
AUGMENTATION_CONFIG = {
    'random': 0.5,
    'None': 0.5,
    'specific': {
        'geometric': 0.0,
        'color': 0.0,
        'rotation': 0.0,
        'flip': 0.0,
        'transpose': 0.0,
    }
}
```
**Result**: 50% of examples get random augmentation, 50% remain unchanged.

### Example 3: Mix of Specific Augmentations
```python
AUGMENTATION_CONFIG = {
    'random': 0.0,
    'None': 0.3,
    'specific': {
        'geometric': 0.3,
        'color': 0.4,
        'rotation': 0.0,
        'flip': 0.0,
        'transpose': 0.0,
    }
}
```
**Result**: 
- 30% original data
- 30% geometric augmentation
- 40% color augmentation

### Example 4: Only Geometric Augmentations
```python
AUGMENTATION_CONFIG = {
    'random': 0.0,
    'None': 0.2,
    'specific': {
        'geometric': 0.8,
        'color': 0.0,
        'rotation': 0.0,
        'flip': 0.0,
        'transpose': 0.0,
    }
}
```
**Result**: 80% geometric augmentation, 20% original data.

### Example 5: Balanced Mix
```python
AUGMENTATION_CONFIG = {
    'random': 0.2,
    'None': 0.2,
    'specific': {
        'geometric': 0.2,
        'color': 0.2,
        'rotation': 0.1,
        'flip': 0.1,
        'transpose': 0.0,
    }
}
```
**Result**: Balanced distribution across multiple augmentation strategies.

## How to Use

### Method 1: Update Constants (Recommended)

Edit `arc/utils/constants.py` and modify `AUGMENTATION_CONFIG`:

```python
# In arc/utils/constants.py
AUGMENTATION_CONFIG = {
    'random': 0.3,
    'None': 0.4,
    'specific': {
        'geometric': 0.3,
        'color': 0.0,
        'rotation': 0.0,
        'flip': 0.0,
        'transpose': 0.0,
    }
}
```

Then run training normally:
```python
from arc.models.train import train

train(
    model_dir="models/tinylm_checkpoints",
    data_path="data/processed/dev_tasks.json",
    model_size='tiny',
    training_profile='small_gpu',
    # augmentation_config will use AUGMENTATION_CONFIG from constants
)
```

### Method 2: Pass Custom Config to Training

Override at training time:

```python
from arc.models.train import train

custom_aug_config = {
    'random': 0.5,
    'None': 0.5,
    'specific': {
        'geometric': 0.0,
        'color': 0.0,
        'rotation': 0.0,
        'flip': 0.0,
        'transpose': 0.0,
    }
}

train(
    model_dir="models/tinylm_checkpoints",
    data_path="data/processed/dev_tasks.json",
    model_size='tiny',
    training_profile='small_gpu',
    augmentation_config=custom_aug_config
)
```

## Important Notes

1. **Probability Sum**: While not strictly enforced, probabilities should ideally sum to 1.0 for proper interpretation. The implementation normalizes the probabilities internally.

2. **On-the-fly Augmentation**: Augmentations are applied during training, not pre-computed, which saves memory.

3. **Semantic Preservation**: All augmentations preserve the logical structure and solution of ARC tasks.

4. **Performance**: Setting all probabilities to 0 or using `{'random': 0.0, 'None': 1.0, ...}` disables augmentation entirely for maximum training speed.

## Training Output

When training starts, you'll see augmentation status:

```
Data Augmentation: ENABLED
  Configuration:
    random: 0.3
    None (identity): 0.4
    specific:
      geometric: 0.3
```

or

```
Data Augmentation: DISABLED
```

## Troubleshooting

- **No augmentation applied**: Check that at least one probability is > 0
- **Too much augmentation**: Increase the `'None'` probability
- **Inconsistent results**: Ensure probabilities are properly normalized
- **Memory issues**: Augmentation shouldn't cause memory problems as it's on-the-fly, but reduce batch size if needed
