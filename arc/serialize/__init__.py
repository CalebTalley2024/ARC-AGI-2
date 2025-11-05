"""
Serialization module

Provides tokenization and encoding utilities for converting grids to/from token sequences.
"""

# Import constants from centralized location
from ..utils.constants import (BOS, EOS, PAD, SEP, VOCAB_SIZE, tok_c, tok_h,
                               tok_px, tok_w)
from .task_tokenizer import deserialize_grid, pack_example, serialize_grid

__all__ = [
    'serialize_grid', 'deserialize_grid', 'pack_example',
    'PAD', 'BOS', 'EOS', 'SEP', 'VOCAB_SIZE',
    'tok_w', 'tok_h', 'tok_c', 'tok_px'
]
