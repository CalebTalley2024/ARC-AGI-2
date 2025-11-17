# Data Augmentation Guide for ARC-AGI Training

## Table of Contents
1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [Configuration Structure](#configuration-structure)
4. [Augmentation Modes](#augmentation-modes)
5. [Augmentation Types](#augmentation-types)
6. [Usage Examples](#usage-examples)
7. [Performance Considerations](#performance-considerations)
8. [Troubleshooting](#troubleshooting)

---

## Overview

The ARC-AGI training pipeline supports flexible data augmentation through a dictionary-based configuration system. Augmentation helps improve model generalization by exposing it to transformed versions of training examples.

### Key Features
- ✅ Multiple augmentation types (geometric, color, combinations)
- ✅ Flexible probability-based configuration
- ✅ Two modes: on-the-fly (memory efficient) or pre-generated (faster training)
- ✅ Centralized configuration in `arc/utils/constants.py`

---

## Quick Start

### Recommended Configuration (Most Users)

For GPUs with 8-16GB memory, use on-the-fly augmentation:

```python
# In arc/utils/constants.py
AUGMENTATION_CONFIG = {
    'random': 0.3,      # 30% random mix of augmentations
    'None': 0.4,        # 40% no augmentation (original data)
    'specific': {
        'geometric': 0.1,   # 10% geometric transforms
        'color': 0.2,       # 20% color permutations
        'rotation': 0.0,
        'flip': 0.0,
        'transpose': 0.0,
    },
    'augmentation_multiplier': 1,  # On-the-fly mode
}
```

---

## Configuration Structure

The augmentation configuration is defined in `arc/utils/constants.py`:

```python
AUGMENTATION_CONFIG = {
    'random': float,           # Probability of random augmentation
    'None': float,             # Probability of no augmentation (identity)
    'specific': {              # Specific augmentation types
        'geometric': float,    # All geometric transforms
        'color': float,        # Color permutations only
        'rotation': float,     # Rotation only
        'flip': float,         # Flip only
        'transpose': float,    # Transpose only
    },
    'augmentation_multiplier': int,  # 1=on-the-fly, >1=pre-generate
}
```

### Probability Rules
- All probabilities should sum to **1.0** for proper sampling
- Individual probabilities can be 0.0 (disabled) to 1.0 (always)
- Examples will be randomly sampled according to these probabilities

---

## Augmentation Modes

### Mode 1: On-the-fly Augmentation (`multiplier = 1`)

**How it works:**
- Augmentations are applied dynamically during batch loading
- Dataset size remains constant
- Different augmentations every epoch
- More memory efficient

**When to use:**
- ✅ Limited GPU memory (< 16GB)
- ✅ Long training runs (benefits from variation)
- ✅ Need maximum diversity
- ✅ Exploratory experiments

**Example:**
```python
AUGMENTATION_CONFIG = {
    'random': 0.5,
    'None': 0.5,
    'specific': {'geometric': 0.0, 'color': 0.0, ...},
    'augmentation_multiplier': 1,  # On-the-fly
}
# Result: 400 examples → 400 examples (size unchanged)
# Each epoch sees different augmentations
```

**Performance:**
- **Memory:** Baseline (stores only original grids)
- **Initialization:** Fast (~1 second for 400 examples)
- **Batch loading:** Slightly slower (augmentation computed on-demand)
- **Diversity:** High (varies each epoch)

### Mode 2: Pre-generated Augmentation (`multiplier > 1`)

**How it works:**
- Creates multiple augmented copies during initialization
- Augmented sequences stored as pre-computed tokens
- Same augmentations every epoch
- Uses more memory

**When to use:**
- ✅ Sufficient GPU memory (> 16GB free)
- ✅ Short training runs
- ✅ Need reproducible results
- ✅ Want faster batch loading

**Example:**
```python
AUGMENTATION_CONFIG = {
    'random': 0.0,
    'None': 0.3,  # 30% keep original
    'specific': {'geometric': 0.7, 'color': 0.0, ...},
    'augmentation_multiplier': 3,  # Pre-generate 3x
}
# Result: 400 examples → 1200 examples (3x increase)
# Each epoch uses same augmented copies
```

**Performance:**
- **Memory:** ~3x baseline (stores all augmented sequences)
- **Initialization:** Slower (~5-10 seconds for 400 examples)
- **Batch loading:** Faster (no computation needed)
- **Diversity:** Lower (fixed augmentations per epoch)

### Comparison Table

| Feature | On-the-fly (multiplier=1) | Pre-generated (multiplier>1) |
|---------|---------------------------|------------------------------|
| **Dataset size** | Unchanged | Multiplied |
| **Memory usage** | Baseline | ~multiplier × baseline |
| **Init time** | Fast | Slower |
| **Batch speed** | Slower | Faster |
| **Diversity** | High (varies each epoch) | Lower (fixed each epoch) |
| **Best for** | Memory-limited GPUs | GPUs with spare memory |

---

## Augmentation Types

### 1. Random (`'random'`)

Applies a random mix of geometric transformations and color permutations.

**What it does:**
- Randomly selects from all D4 geometric transformations (rotations, flips, transposes)
- 50% chance to also apply color permutation
- Maximum variety

**Example:**
```python
'random': 0.5  # 50% of examples get random augmentation
```

**Use when:**
- You want maximum augmentation diversity
- Task benefits from both spatial and color invariance
- Exploring what augmentations help

### 2. None (Identity) (`'None'`)

No augmentation - keeps original data unchanged.

**What it does:**
- Returns original input/output grids
- No transformation applied

**Example:**
```python
'None': 0.4  # 40% of examples remain original
```

**Use when:**
- Balancing augmented vs original data
- Want to prevent overfitting to augmentations
- Ensuring model can handle original distribution

### 3. Geometric (`'geometric'`)

All geometric transformations (rotations, flips, transposes).

**What it does:**
- Rotations: 90°, 180°, 270°
- Flips: horizontal, vertical
- Transposes: regular, with flips
- Preserves colors

**Example:**
```python
'specific': {
    'geometric': 0.3  # 30% geometric augmentation
}
```

**Use when:**
- Task has spatial symmetries
- Want rotation/flip invariance
- Colors matter for the task

**Note:** Transpose operations only work on square grids (height == width).

### 4. Color (`'color'`)

Color palette permutations only.

**What it does:**
- Randomly permutes the color palette
- Preserves spatial structure
- Example: swap all red↔blue pixels

**Example:**
```python
'specific': {
    'color': 0.2  # 20% color augmentation
}
```

**Use when:**
- Task is color-invariant
- Want to learn color-independent patterns
- Spatial relationships are key

### 5. Rotation (`'rotation'`)

Rotation transformations only.

**What it does:**
- 90°, 180°, 270° rotations
- Preserves colors and aspect ratio

**Example:**
```python
'specific': {
    'rotation': 0.15  # 15% rotation augmentation
}
```

### 6. Flip (`'flip'`)

Flip transformations only.

**What it does:**
- Horizontal flip (left↔right)
- Vertical flip (top↔bottom)

**Example:**
```python
'specific': {
    'flip': 0.15  # 15% flip augmentation
}
```

### 7. Transpose (`'transpose'`)

Transpose variations only (square grids only).

**What it does:**
- Regular transpose
- Transpose with flips

**Example:**
```python
'specific': {
    'transpose': 0.1  # 10% transpose augmentation
}
```

**Note:** Only applies to square grids where height == width.

---

## Usage Examples

### Example 1: No Augmentation
```python
AUGMENTATION_CONFIG = {
    'random': 0.0,
    'None': 1.0,  # 100% original data
    'specific': {
        'geometric': 0.0,
        'color': 0.0,
        'rotation': 0.0,
        'flip': 0.0,
        'transpose': 0.0,
    },
    'augmentation_multiplier': 1,
}
```
**Result:** Training uses only original data, no augmentation.

### Example 2: Balanced Random (Recommended Starting Point)
```python
AUGMENTATION_CONFIG = {
    'random': 0.3,   # 30% random mix
    'None': 0.4,     # 40% original
    'specific': {
        'geometric': 0.1,  # 10% geometric
        'color': 0.2,      # 20% color
        'rotation': 0.0,
        'flip': 0.0,
        'transpose': 0.0,
    },
    'augmentation_multiplier': 1,  # On-the-fly
}
```
**Result:** Balanced augmentation with on-the-fly mode, memory efficient.

### Example 3: Geometric Focus (On-the-fly)
```python
AUGMENTATION_CONFIG = {
    'random': 0.0,
    'None': 0.3,  # 30% original
    'specific': {
        'geometric': 0.4,  # 40% geometric
        'rotation': 0.15,  # 15% rotation
        'flip': 0.15,      # 15% flip
        'color': 0.0,
        'transpose': 0.0,
    },
    'augmentation_multiplier': 1,  # On-the-fly
}
```
**Result:** Heavy geometric augmentation, good for spatially symmetric tasks.

### Example 4: Dataset Expansion (Pre-generated)
```python
AUGMENTATION_CONFIG = {
    'random': 0.0,
    'None': 0.2,  # 20% keep original
    'specific': {
        'geometric': 0.5,  # 50% geometric
        'rotation': 0.15,  # 15% rotation
        'flip': 0.15,      # 15% flip
        'color': 0.0,
        'transpose': 0.0,
    },
    'augmentation_multiplier': 3,  # Triple dataset size
}
```
**Result:** 
- Original: 400 examples
- Final: 1200 examples (3x)
- Memory: ~3x higher
- Use only with >16GB free GPU memory

### Example 5: Color-Invariant Training
```python
AUGMENTATION_CONFIG = {
    'random': 0.0,
    'None': 0.3,  # 30% original
    'specific': {
        'geometric': 0.0,
        'color': 0.7,  # 70% color augmentation
        'rotation': 0.0,
        'flip': 0.0,
        'transpose': 0.0,
    },
    'augmentation_multiplier': 1,  # On-the-fly
}
```
**Result:** Focus on learning color-independent patterns.

### Example 6: Conservative (Memory-Constrained)
```python
AUGMENTATION_CONFIG = {
    'random': 0.0,
    'None': 0.6,  # 60% original
    'specific': {
        'geometric': 0.2,  # 20% geometric (cheap)
        'rotation': 0.1,   # 10% rotation
        'flip': 0.1,       # 10% flip
        'color': 0.0,      # 0% (expensive, avoid)
        'transpose': 0.0,
    },
    'augmentation_multiplier': 1,  # On-the-fly only
}
```
**Result:** Minimal augmentation, safe for GPUs with limited memory.

---

## Performance Considerations

### Memory Usage

**On-the-fly (multiplier=1):**
- Memory: Baseline (stores Grid objects)
- 400 examples: ~12-15 MB
- Scales linearly with dataset size

**Pre-generated (multiplier=N):**
- Memory: ~N × baseline (stores tokenized sequences)
- 400 examples, multiplier=3: ~36-45 MB
- Consider GPU memory limits

### Speed Trade-offs

**On-the-fly:**
- ⚡ Fast initialization (~1 second)
- 🐌 Slower batch loading (augmentation computed per batch)
- Best for: Long training, exploratory work

**Pre-generated:**
- 🐌 Slower initialization (~5-10 seconds for 400 examples × 3)
- ⚡ Fast batch loading (no computation)
- Best for: Short training, production runs

### GPU Memory Guidelines

| Available GPU Memory | Recommended Mode | Multiplier |
|---------------------|------------------|------------|
| < 8GB free | On-the-fly only | 1 |
| 8-16GB free | On-the-fly preferred | 1 |
| > 16GB free | Either mode works | 1-3 |

### Training Output

**With augmentation enabled:**
```
Data Augmentation: ENABLED
  Mode: On-the-fly (dynamic per epoch)
  Configuration:
    random: 0.3
    None (identity): 0.4
    specific:
      geometric: 0.1
      color: 0.2
Dataset size: 400 examples
```

**With pre-generated augmentation:**
```
Data Augmentation: ENABLED
  Mode: Pre-generated (multiplies dataset)
  Dataset multiplier: 3x
  Configuration:
    random: 0.0
    None (identity): 0.2
    specific:
      geometric: 0.8
Dataset size: 1200 examples
  (Original: ~400 examples × 3 multiplier)
```

---

## Troubleshooting

### Issue 1: Out of Memory Errors

**Symptom:** CUDA out of memory during training

**Solutions:**
1. Use on-the-fly mode (`multiplier=1`)
2. Reduce augmentation probabilities
3. Disable expensive augmentations (color)
4. Reduce batch size

**Example fix:**
```python
AUGMENTATION_CONFIG = {
    'random': 0.0,
    'None': 0.7,  # More original data
    'specific': {
        'geometric': 0.2,  # Less augmentation
        'color': 0.0,      # Disable expensive ops
        'rotation': 0.1,
        'flip': 0.0,
        'transpose': 0.0,
    },
    'augmentation_multiplier': 1,  # Must be 1
}
```

### Issue 2: Training is Too Slow

**Symptom:** Slow batch loading, low GPU utilization

**Solutions:**
1. Switch to pre-generated mode (if memory allows)
2. Reduce augmentation complexity
3. Increase batch size
4. Use simpler augmentation types

**Example fix:**
```python
AUGMENTATION_CONFIG = {
    'random': 0.0,
    'None': 0.3,
    'specific': {
        'geometric': 0.7,  # Simple geometric only
        'color': 0.0,      # Disable color (slow)
        'rotation': 0.0,
        'flip': 0.0,
        'transpose': 0.0,
    },
    'augmentation_multiplier': 2,  # Pre-generate if memory allows
}
```

### Issue 3: Model Not Improving

**Symptom:** Loss plateaus early, poor validation performance

**Possible causes:**
1. Too much augmentation (model can't learn)
2. Wrong augmentation types for task
3. Need more original data

**Solutions:**
```python
# Try reducing augmentation
AUGMENTATION_CONFIG = {
    'random': 0.2,   # Reduce from 0.5
    'None': 0.6,     # More original data
    'specific': {
        'geometric': 0.1,
        'color': 0.1,
        'rotation': 0.0,
        'flip': 0.0,
        'transpose': 0.0,
    },
    'augmentation_multiplier': 1,
}
```

### Issue 4: Non-Square Grid Errors

**Symptom:** Errors related to transpose operations

**Cause:** Transpose requires square grids (height == width)

**Solution:** Avoid transpose augmentation or check grid dimensions
```python
AUGMENTATION_CONFIG = {
    'random': 0.3,
    'None': 0.4,
    'specific': {
        'geometric': 0.2,  # OK (handles non-square)
        'color': 0.1,      # OK
        'rotation': 0.0,   # 180° OK, 90°/270° need square
        'flip': 0.0,       # OK (works on any shape)
        'transpose': 0.0,  # ❌ Set to 0 for non-square
    },
    'augmentation_multiplier': 1,
}
```

---

## Implementation Details

### Code Locations

1. **Configuration:** `arc/utils/constants.py`
   - `AUGMENTATION_CONFIG` dictionary

2. **Dataset:** `arc/models/train.py`
   - `ArcPairsDataset` class handles both modes

3. **Augmentation logic:** `arc/models/augmentation.py`
   - `apply_augmentation_to_example()` - applies transforms
   - `sample_augmentation_type()` - samples based on probabilities
   - `is_augmentation_enabled()` - checks if enabled

4. **Geometric transforms:** `arc/grids/views.py`
   - D4 transformations (rotations, flips, transposes)
   - Color permutations

### How It Works

**On-the-fly mode:**
1. Dataset initialization stores Grid objects
2. During batch loading (`__getitem__`):
   - Sample augmentation type based on probabilities
   - Apply augmentation to grids
   - Tokenize and return

**Pre-generated mode:**
1. Dataset initialization:
   - For each example, create N copies (N=multiplier)
   - For each copy, sample and apply augmentation
   - Tokenize and store all copies
2. During batch loading:
   - Simply return pre-computed sequences

---

## Best Practices

### 1. Start Conservative
Begin with moderate augmentation and increase gradually:
```python
# Phase 1: Start here
AUGMENTATION_CONFIG = {
    'random': 0.2,
    'None': 0.5,
    'specific': {'geometric': 0.2, 'color': 0.1, ...},
    'augmentation_multiplier': 1,
}

# Phase 2: If Phase 1 works well, increase
AUGMENTATION_CONFIG = {
    'random': 0.3,
    'None': 0.4,
    'specific': {'geometric': 0.2, 'color': 0.1, ...},
    'augmentation_multiplier': 1,
}
```

### 2. Monitor Training
Watch for:
- Loss convergence (should improve steadily)
- GPU memory usage (should be stable)
- Batch loading speed (should be reasonable)

### 3. Match Augmentation to Task
- **Spatially symmetric tasks:** Focus on geometric
- **Color-invariant tasks:** Use color augmentation
- **General tasks:** Mix of random and specific types

### 4. Use On-the-fly for Exploration
When experimenting with different configs, use `multiplier=1` to save memory.

### 5. Pre-generate for Production
Once you've found good settings, consider `multiplier>1` for faster training.

---

## Summary

**Key Takeaways:**
- ✅ Use `AUGMENTATION_CONFIG` in `constants.py` for centralized control
- ✅ Start with on-the-fly (`multiplier=1`) for memory efficiency
- ✅ Balance augmentation probabilities (should sum to 1.0)
- ✅ Monitor GPU memory and adjust accordingly
- ✅ Match augmentation types to your task requirements

**Quick Decision Guide:**
- **Limited memory?** → Use `multiplier=1`
- **Want speed?** → Use `multiplier>1` (if memory allows)
- **Exploring?** → Use `multiplier=1` with moderate probabilities
- **Production?** → Use `multiplier>1` with proven config

For your current settings and implementation details, see [CURRENT_AUGMENTATION_IMPLEMENTATION.md](CURRENT_AUGMENTATION_IMPLEMENTATION.md).
