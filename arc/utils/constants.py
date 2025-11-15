"""
Centralized constants for ARC-AGI training and configuration.

This file contains all model hyperparameters, training constants, and tokenization
constants to enable easy configuration management and experimentation.
"""


# ============================================================================
# DIEDRAL CONSTANTS
# ============================================================================

D4 = ['id','rot90','rot180','rot270','flip_h','flip_v','transpose','transpose_flip','flip_transpose']


# ============================================================================
# TOKENIZATION CONSTANTS
# ============================================================================

# Special tokens
PAD = 0     # Padding token
BOS = 1     # Beginning of sequence
EOS = 2     # End of sequence  
SEP = 3     # Separator token

# Token base offsets for vocabulary construction
TOK_W_BASE = 4          # Width tokens: W=1..30 → tokens 4..33
TOK_H_BASE = 34         # Height tokens: H=1..30 → tokens 34..63  
TOK_C_BASE = 64         # Color inventory: C=0..9 → tokens 64..73
TOK_PIXEL_BASE = 74     # Pixel tokens: PIXEL_0..9 → tokens 74..83

# Calculated vocabulary size
VOCAB_SIZE = TOK_PIXEL_BASE + 10  # = 84 total tokens

# Grid constraints
MAX_GRID_SIZE = 30      # Maximum width/height for ARC grids
NUM_COLORS = 10         # Number of possible colors (0-9)


# ============================================================================
# MODEL ARCHITECTURE CONSTANTS
# ============================================================================

# TinyLM Model Configuration (default: small size, ~20.3M parameters)
MODEL_CONFIG = {
    'vocab_size': VOCAB_SIZE,
    'd_model': 448,         # Hidden dimension size
    'n_layers': 8,          # Number of transformer blocks
    'n_heads': 8,           # Number of attention heads
    'd_ff': 1792,           # Feedforward dimension (4 * d_model)
    'p_drop': 0.1,          # Dropout probability
    'max_len': 2048,        # Maximum sequence length
}

# Alternative model sizes for experimentation
MODEL_CONFIGS = {
    'tiny': {    # ~3.7M parameters, ~14 MB, fits on most GPUs
        'vocab_size': VOCAB_SIZE,
        'd_model': 256,
        'n_layers': 4,
        'n_heads': 4,
        'd_ff': 1024,
        'p_drop': 0.1,
        'max_len': 2048,
    },
    'small': {   # ~20.3M parameters, ~77 MB, good balance for training
        'vocab_size': VOCAB_SIZE,
        'd_model': 448,
        'n_layers': 8,
        'n_heads': 8,
        'd_ff': 1792,
        'p_drop': 0.1,
        'max_len': 2048,
    },
    'medium': {  # ~32.6M parameters, ~124 MB, requires decent GPU memory
        'vocab_size': VOCAB_SIZE,
        'd_model': 512,
        'n_layers': 10,
        'n_heads': 8,
        'd_ff': 2048,
        'p_drop': 0.1,
        'max_len': 2048,
    },
    'large': {   # ~51.9M parameters, ~198 MB, requires high-end GPU
        'vocab_size': VOCAB_SIZE,
        'd_model': 640,
        'n_layers': 10,
        'n_heads': 8,
        'd_ff': 2560,
        'p_drop': 0.1,
        'max_len': 4096,
    }
}


# ============================================================================
# TRAINING CONSTANTS
# ============================================================================

# Training hyperparameters
TRAINING_CONFIG = {
    # Core training parameters
    'steps': 100_000,           # Total training steps
    'batch_size': 32,           # Batch size
    'learning_rate': 3e-4,      # Learning rate
    # 'max_sequence_length': 2048, # Max sequence length for dataset filtering (matches default model max_len)
    
    # Optimizer parameters
    'betas': (0.9, 0.95),       # Adam beta parameters
    'weight_decay': 0.01,       # Weight decay for regularization
    'grad_clip_norm': 1.0,      # Gradient clipping norm
    
    # Training configuration
    'serialization_mode': 'row', # Grid serialization mode ('row' or 'col')
    'pad_token_id': PAD,        # Padding token ID
    'ignore_index': PAD,        # Index to ignore in loss calculation
    
    # Checkpointing
    'save_every': 1000,         # Save checkpoint every N steps
    'eval_every': 1000,         # Evaluate every N steps (if implemented)
    
    # Mixed precision training
    'use_amp': True,            # Use automatic mixed precision
    'grad_accumulation_steps': 4, # Gradient accumulation steps
}

# GPU memory optimized configs
TRAINING_CONFIGS = {
    'debug': {   # For quick testing and debugging (minimal GPU requirements)
        **TRAINING_CONFIG,
        'steps': 100,
        'batch_size': 4,
        'grad_accumulation_steps': 2,  # Effective batch size = 8
        'save_every': 50,
        'max_sequence_length': 1024,  # Reduced for quick debugging
    },
    'small_gpu': {  # For 4-8GB GPU memory (GTX 1060, RTX 2060, etc.)
        **TRAINING_CONFIG,
        'batch_size': 8,
        'grad_accumulation_steps': 4,  # Effective batch size = 32
        'max_sequence_length': 2048,   # Matches tiny/small/medium model max_len
    },
    'medium_gpu': { # For 8-16GB GPU memory (RTX 3070, RTX 4060 Ti, etc.)
        **TRAINING_CONFIG,
        'batch_size': 8,
        'grad_accumulation_steps': 4,  # Effective batch size = 32
        'max_sequence_length': 2048,   # Reduced for memory efficiency
    },
    'large_gpu': {  # For 16GB+ GPU memory (RTX 3080, RTX 4080, A100, etc.)
        **TRAINING_CONFIG,
        'batch_size': 12,
        'grad_accumulation_steps': 3,  # Effective batch size = 36, close to 32
        'max_sequence_length': 4096,   # Full length for large GPUs
    },
}


# ============================================================================
# DATA PROCESSING CONSTANTS
# ============================================================================

# Dataset configuration
DATA_CONFIG = {
    'train_file': 'data/raw/arc/training.txt',
    'eval_file': 'data/raw/arc/evaluation.txt',
    'processed_dir': 'data/processed/',
    'shuffle_dataset': True,
    'drop_last_batch': True,
}


# ============================================================================
# DATA AUGMENTATION CONFIGURATION
# ============================================================================

# Augmentation configuration
# Dictionary mapping augmentation strategies to their probabilities
# Supported modes:
#   - 'random': Random mix of geometric and color augmentations (set probability for this mode)
#   - 'None': No augmentation (identity transform) (set probability for no augmentation)
#   - 'specific': Dictionary of specific augmentation types with individual probabilities
#       * 'geometric': All geometric transforms (rotation, flip, transpose)
#       * 'color': Color permutations only
#       * 'rotation': Rotation transforms only (90, 180, 270)
#       * 'flip': Flip transforms only (horizontal, vertical)
#       * 'transpose': Transpose variations only
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

# Example configurations:
# 1. No augmentation (default):
#    {'random': 0.0, 'None': 1.0, 'specific': {...all 0.0}}
#
# 2. 50% random augmentation, 50% no augmentation:
#    {'random': 0.5, 'None': 0.5, 'specific': {...all 0.0}}
#
# 3. Mix of specific augmentations:
#    {'random': 0.0, 'None': 0.3, 'specific': {'geometric': 0.3, 'color': 0.4, ...}}
#
# 4. Only geometric augmentations:
#    {'random': 0.0, 'None': 0.2, 'specific': {'geometric': 0.8, 'color': 0.0, ...}}
#
# Note: Probabilities across all modes should sum to 1.0 for proper sampling

# Sequence length analysis thresholds
SEQUENCE_ANALYSIS = {
    'token_length_bins': [512, 1024, 2048, 4096],
    'memory_estimate_batch_sizes': [4, 8, 16, 32, 64],
    'max_examples_to_analyze': 1000,
}


# ============================================================================
# EVALUATION CONSTANTS  
# ============================================================================

EVAL_CONFIG = {
    'batch_size': 16,
    'max_new_tokens': 1024,
    'temperature': 0.7,
    'top_k': 50,
    'top_p': 0.9,
    'num_beams': 1,
    'do_sample': False,
}


# ============================================================================
# PATHS AND DIRECTORIES
# ============================================================================

PATHS = {
    'models_dir': 'models/',
    'checkpoints_dir': 'checkpoints/',
    'logs_dir': 'logs/',
    'results_dir': 'results/',
    'data_dir': 'data/',
}


# ============================================================================
# DEVICE AND PERFORMANCE CONSTANTS
# ============================================================================

DEVICE_CONFIG = {
    'cuda_if_available': True,
    'mixed_precision': True,
    'compile_model': False,  # PyTorch 2.0 model compilation
    'dataloader_num_workers': 4,
    'pin_memory': True,
}


# ============================================================================
# HELPER FUNCTIONS FOR TOKEN CONVERSION
# ============================================================================

def tok_w(w: int) -> int:
    """Convert width to token ID."""
    assert 1 <= w <= MAX_GRID_SIZE, f"Width {w} must be between 1 and {MAX_GRID_SIZE}"
    return TOK_W_BASE + (w - 1)

def tok_h(h: int) -> int:
    """Convert height to token ID."""
    assert 1 <= h <= MAX_GRID_SIZE, f"Height {h} must be between 1 and {MAX_GRID_SIZE}"
    return TOK_H_BASE + (h - 1)

def tok_c(c: int) -> int:
    """Convert color to color inventory token ID."""
    assert 0 <= c <= NUM_COLORS - 1, f"Color {c} must be between 0 and {NUM_COLORS - 1}"
    return TOK_C_BASE + c

def tok_px(c: int) -> int:
    """Convert color to pixel token ID."""
    assert 0 <= c <= NUM_COLORS - 1, f"Color {c} must be between 0 and {NUM_COLORS - 1}"
    return TOK_PIXEL_BASE + c


# ============================================================================
# CONFIGURATION SELECTION HELPERS
# ============================================================================

def get_model_config(size: str = 'small') -> dict:
    """Get model configuration by size."""
    if size in MODEL_CONFIGS:
        return MODEL_CONFIGS[size].copy()
    else:
        available = list(MODEL_CONFIGS.keys())
        raise ValueError(f"Model size '{size}' not found. Available: {available}")

def get_training_config(profile: str = 'medium_gpu') -> dict:
    """Get training configuration by GPU profile."""
    if profile in TRAINING_CONFIGS:
        return TRAINING_CONFIGS[profile].copy()
    else:
        available = list(TRAINING_CONFIGS.keys())
        raise ValueError(f"Training profile '{profile}' not found. Available: {available}")

def get_matched_configs(model_size: str = 'small', training_profile: str = 'medium_gpu') -> tuple:
    """
    Get matched model and training configs, ensuring max_sequence_length matches model max_len.
    
    Returns:
        tuple: (model_config, training_config) with aligned sequence lengths
    """
    model_config = get_model_config(model_size)
    training_config = get_training_config(training_profile)
    
    # Ensure training max_sequence_length matches model max_len
    training_config['max_sequence_length'] = model_config['max_len']
    
    return model_config, training_config

def estimate_model_parameters(config: dict) -> int:
    """Estimate number of parameters for a model configuration."""
    d_model = config['d_model']
    n_layers = config['n_layers']
    vocab_size = config['vocab_size']
    max_len = config['max_len']
    d_ff = config['d_ff']
    
    # Embeddings
    embedding_params = vocab_size * d_model + max_len * d_model
    
    # Transformer blocks (attention + feedforward)
    attention_params_per_layer = 4 * d_model * d_model  # Q, K, V, proj
    ff_params_per_layer = 2 * d_model * d_ff  # up and down projections
    layer_norm_params_per_layer = 2 * d_model  # 2 layer norms per block
    
    transformer_params = n_layers * (attention_params_per_layer + ff_params_per_layer + layer_norm_params_per_layer)
    
    # Output head
    output_params = d_model * vocab_size
    
    # Final layer norm
    final_ln_params = d_model
    
    total_params = embedding_params + transformer_params + output_params + final_ln_params
    return total_params