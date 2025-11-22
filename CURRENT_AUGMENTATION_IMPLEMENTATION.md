# Current Augmentation Implementation

## Current Settings (As of Latest Configuration)

### Active Configuration

**Location:** `arc/utils/constants.py` (lines 192-204)

```python
AUGMENTATION_CONFIG = {
    'random': 0.3,      # 30% probability of random augmentation
    'None': 0.4,        # 40% probability of no augmentation (identity)
    'specific': {
        'geometric': 0.1,   # 10% all geometric transforms
        'color': 0.2,       # 20% color permutations
        'rotation': 0.0,    # 0% rotation only (disabled)
        'flip': 0.0,        # 0% flip only (disabled)
        'transpose': 0.0,   # 0% transpose only (disabled)
    },
    'augmentation_multiplier': 1,  # On-the-fly mode
}
```

### What This Configuration Does

**Augmentation Distribution:**
- 40% of training examples: Original (no augmentation)
- 30% of training examples: Random mix (geometric + possible color)
- 10% of training examples: Geometric transforms only
- 20% of training examples: Color permutations only

**Mode:** On-the-fly (multiplier=1)
- Dataset size: Unchanged (400 examples remain 400)
- Memory usage: Baseline
- Diversity: High (different augmentations each epoch)

### Expected Training Output

When you run training, you should see:

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

---

## Implementation Architecture

### 1. Configuration Layer (`arc/utils/constants.py`)

**Purpose:** Centralized augmentation settings

**Code:**
```python
AUGMENTATION_CONFIG = {
    'random': 0.3,
    'None': 0.4,
    'specific': {
        'geometric': 0.1,
        'color': 0.2,
        'rotation': 0.0,
        'flip': 0.0,
        'transpose': 0.0,
    },
    'augmentation_multiplier': 1,
}
```

**Key Features:**
- Dictionary-based configuration
- Probability distribution for sampling
- Multiplier controls on-the-fly vs pre-generated mode

### 2. Dataset Layer (`arc/models/train.py`)

**Class:** `ArcPairsDataset`

**Initialization Logic:**
```python
def __init__(self, tasks, mode=None, max_len=None, augmentation_config=None):
    # Use provided config or default from constants
    self.augmentation_config = augmentation_config or AUGMENTATION_CONFIG
    
    # Check if augmentation is enabled
    self.use_augmentation = is_augmentation_enabled(self.augmentation_config)
    
    # Get multiplier (1 = on-the-fly, >1 = pre-generate)
    self.augmentation_multiplier = self.augmentation_config.get('augmentation_multiplier', 1)
    
    for task_dict in tasks:
        for example in task_dict["train"]:
            input_grid = Grid(np.array(example["input"], dtype=np.int8))
            output_grid = Grid(np.array(example["output"], dtype=np.int8))
            
            if self.use_augmentation:
                if self.augmentation_multiplier > 1:
                    # PRE-GENERATED MODE: Create N copies with augmentation
                    for copy_idx in range(self.augmentation_multiplier):
                        aug_type = sample_augmentation_type(self.augmentation_config)
                        if aug_type != "identity":
                            aug_input, aug_output = apply_augmentation_to_example(
                                input_grid, output_grid, aug_type
                            )
                        else:
                            aug_input, aug_output = input_grid, output_grid
                        seq = pack_example(aug_input, aug_output, mode)
                        if len(seq) <= max_len:
                            self.examples.append({
                                "sequence": seq,
                                "is_augmented": (aug_type != "identity"),
                                "augmentation_type": aug_type
                            })
                else:
                    # ON-THE-FLY MODE: Store original grids (YOUR CURRENT MODE)
                    self.examples.append({
                        "input": input_grid,
                        "output": output_grid,
                        "mode": mode,
                        "is_augmented": False,
                    })
            else:
                # No augmentation: store tokenized sequence
                seq = pack_example(input_grid, output_grid, mode)
                if len(seq) <= max_len:
                    self.examples.append({"sequence": seq})
```

**Batch Loading Logic (`__getitem__`):**
```python
def __getitem__(self, idx):
    example = self.examples[idx]
    
    # ON-THE-FLY MODE (your current setting with multiplier=1)
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
        
        # Convert to token sequence
        seq = pack_example(input_grid, output_grid, self.mode)
        if len(seq) > self.max_len:
            seq = seq[:self.max_len]
    else:
        # PRE-GENERATED MODE or no augmentation
        seq = example["sequence"]
    
    # Return tensors for training
    x = torch.tensor(seq[:-1], dtype=torch.long)
    y = torch.tensor(seq[1:], dtype=torch.long)
    return x, y
```

### 3. Augmentation Logic (`arc/models/augmentation.py`)

**Key Functions:**

**a) `is_augmentation_enabled(augmentation_config)`**
```python
def is_augmentation_enabled(augmentation_config: dict) -> bool:
    """Check if any augmentation is enabled."""
    return (
        augmentation_config.get("random", 0) > 0
        or any(prob > 0 for prob in augmentation_config.get("specific", {}).values())
    )
```
**Result for your config:** `True` (enabled)

**b) `sample_augmentation_type(augmentation_config)`**
```python
def sample_augmentation_type(augmentation_config: dict) -> str:
    """Sample augmentation type based on probability distribution."""
    choices = []
    probs = []
    
    # Build probability distribution
    if augmentation_config.get("random", 0) > 0:
        choices.append("random")
        probs.append(0.3)  # Your setting
    
    if augmentation_config.get("None", 0) > 0:
        choices.append("identity")
        probs.append(0.4)  # Your setting
    
    # Add specific types
    specific = augmentation_config.get("specific", {})
    if specific.get("geometric", 0) > 0:
        choices.append("geometric")
        probs.append(0.1)  # Your setting
    
    if specific.get("color", 0) > 0:
        choices.append("color")
        probs.append(0.2)  # Your setting
    
    # Normalize and sample
    total = sum(probs)  # Should be 1.0
    probs = [p / total for p in probs]
    return np.random.choice(choices, p=probs)
```
**Result for your config:** Returns one of:
- `"random"` (30% chance)
- `"identity"` (40% chance)
- `"geometric"` (10% chance)
- `"color"` (20% chance)

**c) `apply_augmentation_to_example(input_grid, output_grid, augmentation_type)`**
```python
def apply_augmentation_to_example(
    input_grid: Grid,
    output_grid: Grid,
    augmentation_type: str = "random",
) -> Tuple[Grid, Grid]:
    """Apply augmentation to input-output pair."""
    # Extract palette
    palette = set(np.unique(input_grid.a)) | set(np.unique(output_grid.a))
    
    # Check if square (needed for some transforms)
    is_square = (input_grid.a.shape[0] == input_grid.a.shape[1] and 
                 output_grid.a.shape[0] == output_grid.a.shape[1])
    
    # Get augmentation ViewSpec
    view_spec = get_random_augmentation(palette, augmentation_type, is_square=is_square)
    
    # Apply same augmentation to both input and output
    aug_input = apply_view_grid(input_grid, view_spec)
    aug_output = apply_view_grid(output_grid, view_spec)
    
    return aug_input, aug_output
```

### 4. Training Function (`arc/models/train.py`)

**Augmentation Display Code:**
```python
def train(..., augmentation_config: dict | None = None):
    # Use provided config or default
    aug_config = augmentation_config or AUGMENTATION_CONFIG
    
    # Check if enabled
    use_augmentation = is_augmentation_enabled(aug_config)
    augmentation_multiplier = aug_config.get('augmentation_multiplier', 1)
    
    # Display settings
    if use_augmentation:
        print(f"Data Augmentation: ENABLED")
        mode_str = 'Pre-generated (multiplies dataset)' if augmentation_multiplier > 1 else 'On-the-fly (dynamic per epoch)'
        print(f"  Mode: {mode_str}")
        if augmentation_multiplier > 1:
            print(f"  Dataset multiplier: {augmentation_multiplier}x")
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
    
    # Create dataset with augmentation config
    ds = ArcPairsDataset(
        tasks,
        mode=train_config["serialization_mode"],
        max_len=train_config["max_sequence_length"],
        augmentation_config=aug_config,
    )
    
    # Display dataset size
    print(f"Dataset size: {len(ds)} examples")
    if use_augmentation and augmentation_multiplier > 1:
        original_size = len(ds) // augmentation_multiplier
        print(f"  (Original: ~{original_size} examples × {augmentation_multiplier} multiplier)")
```

---

## Current Training Behavior

### What Happens During Training

**Initialization Phase:**
1. Load 400 training examples from JSON files
2. Convert each to Grid objects (input, output)
3. Store Grid objects in dataset (no pre-computation)
4. Dataset size: 400 examples
5. Memory usage: ~12-15 MB (baseline)
6. Time: ~1 second

**Training Loop (Each Batch):**
1. DataLoader requests batch of examples
2. For each example in batch:
   - `__getitem__` is called
   - Sample augmentation type: `sample_augmentation_type(config)`
     - 40% chance: "identity" → no change
     - 30% chance: "random" → random geometric + maybe color
     - 10% chance: "geometric" → random rotation/flip/transpose
     - 20% chance: "color" → color permutation
   - If not "identity", apply augmentation to grids
   - Tokenize augmented grids: `pack_example(input, output)`
   - Return token tensors (x, y)
3. Batch is ready for forward pass

**Per Epoch Behavior:**
- Each example may get different augmentation each epoch
- Over 400 examples, expected distribution:
  - ~160 examples: Original (no augmentation)
  - ~120 examples: Random augmentation
  - ~40 examples: Geometric only
  - ~80 examples: Color only
- This distribution varies randomly each epoch

### Performance Characteristics

**Memory:**
- GPU memory: Baseline + model + batch
- No extra memory for augmented copies
- Safe for GPUs with limited memory

**Speed:**
- Initialization: Fast (~1 second)
- Batch loading: Slightly slower (computes augmentation on-demand)
- Overall: Good balance

**Diversity:**
- High (different augmentations each epoch)
- Model sees varied transformations
- Better generalization potential

---

## How to Modify Settings

### Change Augmentation Probabilities

**Location:** `arc/utils/constants.py`, line 192

**Example 1: Increase augmentation intensity**
```python
AUGMENTATION_CONFIG = {
    'random': 0.4,      # Increase from 0.3 to 0.4
    'None': 0.3,        # Decrease from 0.4 to 0.3 (less original)
    'specific': {
        'geometric': 0.2,   # Increase from 0.1 to 0.2
        'color': 0.1,       # Decrease from 0.2 to 0.1
        'rotation': 0.0,
        'flip': 0.0,
        'transpose': 0.0,
    },
    'augmentation_multiplier': 1,
}
```

**Example 2: Disable color augmentation**
```python
AUGMENTATION_CONFIG = {
    'random': 0.3,
    'None': 0.5,        # Increase to compensate
    'specific': {
        'geometric': 0.2,   # Increase to compensate
        'color': 0.0,       # DISABLED
        'rotation': 0.0,
        'flip': 0.0,
        'transpose': 0.0,
    },
    'augmentation_multiplier': 1,
}
```

**Example 3: More specific geometric transforms**
```python
AUGMENTATION_CONFIG = {
    'random': 0.2,      # Reduce random
    'None': 0.4,
    'specific': {
        'geometric': 0.0,   # Disable generic geometric
        'color': 0.1,
        'rotation': 0.15,   # Enable specific: rotation
        'flip': 0.15,       # Enable specific: flip
        'transpose': 0.0,
    },
    'augmentation_multiplier': 1,
}
```

### Switch to Pre-generated Mode

**Warning:** Only do this if you have >16GB free GPU memory!

```python
AUGMENTATION_CONFIG = {
    'random': 0.0,
    'None': 0.3,        # 30% keep original
    'specific': {
        'geometric': 0.5,   # 50% geometric
        'color': 0.0,
        'rotation': 0.1,
        'flip': 0.1,
        'transpose': 0.0,
    },
    'augmentation_multiplier': 3,  # 3x dataset size
}
```

**Effect:**
- Dataset size: 400 → 1200 examples
- Memory: ~3x increase (~36-45 MB for dataset)
- Initialization: Slower (~5-10 seconds)
- Training: Faster batch loading

### Disable Augmentation Completely

```python
AUGMENTATION_CONFIG = {
    'random': 0.0,
    'None': 1.0,        # 100% original
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

**Training output will show:**
```
Data Augmentation: DISABLED
Dataset size: 400 examples
```

---

## Monitoring and Validation

### Check if Settings Are Applied

**1. Training output check:**
Look for this at training start:
```
Data Augmentation: ENABLED
  Mode: On-the-fly (dynamic per epoch)
  Configuration:
    random: 0.3
    None (identity): 0.4
    specific:
      geometric: 0.1
      color: 0.2
```

**2. Dataset size check:**
Should show:
```
Dataset size: 400 examples
```
(Or 1200 if multiplier=3)

**3. Memory usage:**
Monitor GPU memory:
- On-the-fly (multiplier=1): Baseline + model
- Pre-generated (multiplier=3): ~3x dataset memory increase

### Common Issues

**Issue 1: Probabilities don't sum to 1.0**
```python
# BAD: Sum = 0.8 (too low)
AUGMENTATION_CONFIG = {
    'random': 0.3,
    'None': 0.3,
    'specific': {'geometric': 0.2, ...},  # Sum = 0.8
}

# GOOD: Sum = 1.0
AUGMENTATION_CONFIG = {
    'random': 0.3,
    'None': 0.4,
    'specific': {'geometric': 0.2, 'color': 0.1, ...},  # Sum = 1.0
}
```

**Issue 2: Out of memory with pre-generated mode**
```python
# If OOM with multiplier > 1, switch back:
'augmentation_multiplier': 1,  # Back to on-the-fly
```

**Issue 3: Training too slow**
```python
# Reduce expensive augmentations:
AUGMENTATION_CONFIG = {
    'random': 0.3,
    'None': 0.5,
    'specific': {
        'geometric': 0.2,
        'color': 0.0,  # Disable (expensive)
        'rotation': 0.0,
        'flip': 0.0,
        'transpose': 0.0,
    },
    'augmentation_multiplier': 1,
}
```

---

## Summary

### Your Current Configuration

✅ **Mode:** On-the-fly augmentation (multiplier=1)  
✅ **Dataset size:** 400 examples (unchanged)  
✅ **Augmentation mix:** 40% original, 30% random, 10% geometric, 20% color  
✅ **Memory usage:** Baseline (memory efficient)  
✅ **Diversity:** High (varies each epoch)  

### Quick Modification Guide

**To increase augmentation:**
- Decrease `'None'` probability
- Increase `'random'` or specific type probabilities

**To decrease augmentation:**
- Increase `'None'` probability
- Decrease other probabilities

**To change mode:**
- Set `'augmentation_multiplier': 3` for pre-generated (if memory allows)
- Keep `'augmentation_multiplier': 1` for on-the-fly (recommended)

**To disable augmentation:**
- Set `'None': 1.0` and all others to `0.0`

### Files to Check

1. **Configuration:** `arc/utils/constants.py` (lines 192-204)
2. **Dataset logic:** `arc/models/train.py` (`ArcPairsDataset` class)
3. **Augmentation logic:** `arc/models/augmentation.py`
4. **Training display:** `arc/models/train.py` (`train()` function)

For general information about augmentation, see [AUGMENTATION_GUIDE.md](AUGMENTATION_GUIDE.md).
