"""
Step 6: Serialization (Tokenization)

This module implements reversible mapping between (meta, grid) ↔ token sequence.

Goal: Convert ARC tasks into token sequences that can be fed to language models,
      and decode model outputs back into grids.

Key concepts:
- Special tokens: Markers for structure ([BOS], [SEP], [EOS], etc.)
- Metadata tokens: Task properties ([WIDTH]=W, [HEIGHT]=H, etc.)
- Grid tokens: Color ids (0-9) serialized in row-major or column-major order
- Reversibility: Must be able to decode tokens back to exact grid

Token vocabulary structure:
Special tokens:
  [BOS]    - Beginning of sequence
  [SEP]    - Separator between grids/pairs
  [EOS]    - End of sequence
  [PAD]    - Padding token
  [WIDTH]  - Width metadata marker
  [HEIGHT] - Height metadata marker
  [COLORS] - Number of colors metadata marker (optional)
  [N_TRAIN] - Number of training pairs

Grid tokens:
  0-9      - Color ids (direct mapping)

Example tokenization format:
[BOS] [WIDTH]=W [HEIGHT]=H [N_TRAIN]=K
  ( [SEP] <in_grid_row_major_tokens> [SEP] <out_grid_row_major_tokens> ) × K
[SEP] <test_in_row_major_tokens> [EOS]

Concrete example for a 3×3 grid task with 1 train pair:
[BOS] [N_TRAIN]=1
  [SEP] [WIDTH]=3 [HEIGHT]=3 0 1 2 3 4 5 6 7 8  # train input
  [SEP] [WIDTH]=3 [HEIGHT]=3 1 1 1 4 4 4 7 7 7  # train output
  [SEP] [WIDTH]=3 [HEIGHT]=3 2 2 2 5 5 5 8 8 8  # test input
[EOS]

Alternative format (more compact):
[BOS] [N_TRAIN]=1
  [SEP] 3 3 0 1 2 3 4 5 6 7 8      # train input (W H then cells)
  [SEP] 3 3 1 1 1 4 4 4 7 7 7      # train output
  [SEP] 3 3 2 2 2 5 5 5 8 8 8      # test input
[EOS]

Functions to implement:

1. build_vocab() -> dict
   # Create token vocabulary
   # Returns: {token_str: token_id, ...}
   # Include:
   #   - Special tokens: [BOS], [SEP], [EOS], [PAD]
   #   - Metadata tokens: [WIDTH], [HEIGHT], [N_TRAIN], etc.
   #   - Color tokens: '0'-'9'
   #   - Number tokens: '1'-'30' for dimensions (optional)

2. encode_grid(grid: Grid, serialization: str = 'row') -> list[int]
   # Convert grid to token sequence
   # Args:
   #   grid: Grid object to encode
   #   serialization: 'row' (row-major) or 'col' (column-major)
   # Returns: list of token ids
   # Steps:
   #   a. Get shape (H, W)
   #   b. Add width and height tokens
   #   c. Serialize grid cells:
   #      - row-major: iterate rows then columns
   #      - col-major: iterate columns then rows
   #   d. Convert color ids to tokens
   #   e. Return token list

3. decode_grid(tokens: list[int], vocab: dict) -> Grid
   # Convert token sequence back to grid
   # Args:
   #   tokens: list of token ids
   #   vocab: vocabulary dict (or inverse vocab)
   # Returns: Grid object
   # Steps:
   #   a. Parse width and height from tokens
   #   b. Extract color tokens (skip special/meta tokens)
   #   c. Reshape into 2D array based on serialization order
   #   d. Validate shape matches W×H
   #   e. Return Grid.from_list()

4. encode_task(task: dict, view_spec: ViewSpec) -> list[int]
   # Encode entire task (train pairs + test input) into tokens
   # Args:
   #   task: dict with 'train' and 'test' keys
   #   view_spec: ViewSpec to determine serialization order
   # Returns: list of token ids
   # Steps:
   #   a. Start with [BOS]
   #   b. Add [N_TRAIN]=len(train_pairs)
   #   c. For each train pair:
   #      - Add [SEP]
   #      - Encode input grid
   #      - Add [SEP]
   #      - Encode output grid
   #   d. Add [SEP]
   #   e. Encode test input grid
   #   f. Add [EOS]
   #   g. Return full token sequence

5. decode_output(tokens: list[int], expected_shape: tuple) -> Grid
   # Decode model output tokens into grid
   # Args:
   #   tokens: predicted token sequence
   #   expected_shape: (H, W) for validation
   # Returns: Grid object
   # Steps:
   #   a. Strip special tokens ([SEP], [EOS], etc.)
   #   b. Parse dimensions if present
   #   c. Extract color tokens
   #   d. Reshape to 2D grid
   #   e. Validate against expected_shape
   #   f. Return Grid

6. measure_sequence_length(task: dict) -> dict
   # Compute token sequence statistics for a task
   # Returns: {
   #   'total_tokens': int,
   #   'train_tokens': int,
   #   'test_tokens': int,
   #   'max_grid_tokens': int,
   #   'avg_grid_tokens': float
   # }
   # Used to understand context length requirements

Serialization strategies:

Row-major (default):
  # Read left-to-right, top-to-bottom
  # for row in grid:
  #     for cell in row:
  #         emit cell

Column-major:
  # Read top-to-bottom, left-to-right
  # for col in range(W):
  #     for row in range(H):
  #         emit grid[row][col]

Design decisions:
- Use explicit [WIDTH] and [HEIGHT] tokens for clarity
- Or use implicit format: first two numbers are W, H
- Include [N_TRAIN] to help model understand task structure
- Use [SEP] to clearly delineate grids
- Keep color tokens as raw integers (0-9) for simplicity

Typical sequence lengths:
- Small task (3×3 grids, 2 train pairs): ~100 tokens
- Medium task (10×10 grids, 3 train pairs): ~500 tokens
- Large task (30×30 grids, 4 train pairs): ~4000 tokens
- Record these statistics in README

Tests to write (tests/test_tokens.py):
- test_encode_decode_grid: Round-trip on random grids (3×3, 5×5, 10×10, 30×30)
- test_row_vs_col: Verify different serialization orders
- test_encode_task: Full task encoding includes all components
- test_decode_output: Model output tokens decode correctly
- test_special_tokens: Special tokens handled correctly
- test_sequence_lengths: Measure and validate typical lengths

Common pitfalls:
- Off-by-one errors in reshape (row vs col major)
- Forgetting to handle [SEP] tokens in decoder
- Shape mismatch between encoded and decoded grids
- Not handling variable-size grids in same batch
- Padding issues when batching sequences
- Token vocabulary conflicts (ensure unique ids)

Optimization tips:
- Cache encoded grids for each ViewSpec
- Use efficient numpy operations for serialization
- Consider using byte-pair encoding (BPE) for longer sequences (future)
- Batch encode multiple tasks together for efficiency
"""

# TODO: Import numpy as np
# TODO: Import typing (List, Dict, Tuple)
# TODO: Import arc.grids.core (Grid)
# TODO: Import arc.grids.views (ViewSpec)

# TODO: Define special token constants
# SPECIAL_TOKENS = {
#     'BOS': '[BOS]',
#     'SEP': '[SEP]',
#     'EOS': '[EOS]',
#     'PAD': '[PAD]',
#     'WIDTH': '[WIDTH]',
#     'HEIGHT': '[HEIGHT]',
#     'N_TRAIN': '[N_TRAIN]',
#     'COLORS': '[COLORS]',
# }

# TODO: Implement build_vocab() -> Dict[str, int]
#   - Create mapping from token strings to ids
#   - Start with special tokens (ids 0-10)
#   - Add color tokens '0'-'9' (ids 11-20)
#   - Add dimension tokens '1'-'30' (ids 21-50) (optional)
#   - Return vocab dict

# TODO: Implement build_inverse_vocab(vocab: dict) -> Dict[int, str]
#   - Invert vocab dict for decoding
#   - Return {id: token_str}

# TODO: Implement serialize_grid_cells(grid: Grid, serialization: str) -> List[int]
#   - If serialization == 'row':
#     - Flatten grid.a in row-major order: grid.a.flatten()
#   - If serialization == 'col':
#     - Flatten in column-major order: grid.a.T.flatten()
#   - Return list of color ids

# TODO: Implement encode_grid(grid: Grid, vocab: dict, serialization: str = 'row') -> List[int]
#   - Get H, W from grid.shape
#   - Create token list: []
#   - Add width token: vocab['[WIDTH]'], W
#   - Add height token: vocab['[HEIGHT]'], H
#   - Serialize cells: serialize_grid_cells(grid, serialization)
#   - Convert cells to tokens using vocab
#   - Return token list

# TODO: Implement decode_grid(tokens: List[int], inv_vocab: dict, serialization: str = 'row') -> Grid
#   - Parse tokens to find width and height
#   - Extract color tokens (skip special tokens)
#   - Convert token ids to color ids using inv_vocab
#   - Reshape to (H, W) based on serialization order
#   - Return Grid.from_list()

# TODO: Implement encode_task(task: dict, vocab: dict, view_spec: ViewSpec) -> List[int]
#   - Start with [BOS] token
#   - Add [N_TRAIN] metadata
#   - For each train pair:
#     - Add [SEP]
#     - Encode input grid
#     - Add [SEP]
#     - Encode output grid
#   - Add [SEP]
#   - Encode test input
#   - Add [EOS]
#   - Return full token sequence

# TODO: Implement decode_output(tokens: List[int], inv_vocab: dict, expected_shape: Tuple[int,int]) -> Grid
#   - Strip special tokens
#   - Parse dimensions if present
#   - Extract color tokens
#   - Reshape to expected_shape
#   - Validate shape
#   - Return Grid

# TODO: Implement measure_sequence_length(task: dict, vocab: dict) -> dict
#   - Encode task with identity view
#   - Count total tokens
#   - Count tokens per grid
#   - Return statistics dict

# TODO: Implement get_sequence_stats_for_dataset(tasks: list) -> dict
#   - Measure sequence lengths for all tasks
#   - Compute min, max, mean, median, p95
#   - Return summary statistics
#   - Record in README.md
