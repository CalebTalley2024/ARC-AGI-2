# TODO: Import numpy as np
import numpy as np
# TODO: Import typing (List, Dict, Tuple)
from typing import List, Dict, Tuple
# TODO: Import arc.grids.core (Grid)
import arc.grids.core as Grid
import arc.grids.views as ViewSpec

# TODO: Define special token constants
SPECIAL_TOKENS = {
    'BOS': '[BOS]',
    'SEP': '[SEP]',
    'EOS': '[EOS]',
    'PAD': '[PAD]',
    'WIDTH': '[WIDTH]',
    'HEIGHT': '[HEIGHT]',
    'N_TRAIN': '[N_TRAIN]',
    'COLORS': '[COLORS]',
}

# TODO: Implement build_vocab() -> Dict[str, int]
#   - Create mapping from token strings to ids
#   - Start with special tokens (ids 0-10)
#   - Add color tokens '0'-'9' (ids 11-20)
#   - Add dimension tokens '1'-'30' (ids 21-50) (optional)
#   - Return vocab dict
def build_vocab() -> Dict[str, int]:
    vocab = {}
    # Add special tokens
    for i, token in enumerate(SPECIAL_TOKENS.values()):
        vocab[token] = i
    # Add color tokens '0'-'9'
    for i in range(10):
        vocab[str(i)] = len(vocab)
    return vocab

# TODO: Implement build_inverse_vocab(vocab: dict) -> Dict[int, str]
#   - Invert vocab dict for decoding
#   - Return {id: token_str}
def build_inverse_vocab(vocab: Dict[str, int]) -> Dict[int, str]:
    inv_vocab = {v: k for k, v in vocab.items()}
    return inv_vocab

# TODO: Implement serialize_grid_cells(grid: Grid, serialization: str) -> List[int]
#   - If serialization == 'row':
#     - Flatten grid.a in row-major order: grid.a.flatten()
#   - If serialization == 'col':
#     - Flatten in column-major order: grid.a.T.flatten()
#   - Return list of color ids
def serialize_grid_cells(grid: Grid, serialization: str) -> List[int]:
    if serialization == 'row':
        return grid.a.flatten().tolist()
    elif serialization == 'col':
        return grid.a.T.flatten().tolist()
    else:
        raise ValueError(f"Unknown serialization: {serialization}")

# TODO: Implement encode_grid(grid: Grid, vocab: dict, serialization: str = 'row') -> List[int]
#   - Get H, W from grid.shape
#   - Create token list: []
#   - Add width token: vocab['[WIDTH]'], W
#   - Add height token: vocab['[HEIGHT]'], H
#   - Serialize cells: serialize_grid_cells(grid, serialization)
#   - Convert cells to tokens using vocab
#   - Return token list
def encode_grid(grid: Grid, vocab: Dict[str, int], serialization: str = 'row') -> List[int]:
    H, W = grid.shape
    tokens = []
    tokens.append(vocab[SPECIAL_TOKENS['WIDTH']])
    tokens.append(W)
    tokens.append(vocab[SPECIAL_TOKENS['HEIGHT']])
    tokens.append(H)
    cell_ids = serialize_grid_cells(grid, serialization)
    for cell_id in cell_ids:
        tokens.append(vocab[str(cell_id)])
    return tokens

# TODO: Implement decode_grid(tokens: List[int], inv_vocab: dict, serialization: str = 'row') -> Grid
#   - Parse tokens to find width and height
#   - Extract color tokens (skip special tokens)
#   - Convert token ids to color ids using inv_vocab
#   - Reshape to (H, W) based on serialization order
#   - Return Grid.from_list()
def decode_grid(tokens: List[int], inv_vocab: Dict[int, str], serialization: str = 'row') -> Grid:
    # Parse width and height
    W = None
    H = None
    color_tokens = []
    i = 0
    while i < len(tokens):
        token_str = inv_vocab[tokens[i]]
        if token_str == SPECIAL_TOKENS['WIDTH']:
            W = tokens[i + 1]
            i += 2
        elif token_str == SPECIAL_TOKENS['HEIGHT']:
            H = tokens[i + 1]
            i += 2
        elif token_str in SPECIAL_TOKENS.values():
            i += 1
        else:
            color_tokens.append(int(token_str))
            i += 1
    if W is None or H is None:
        raise ValueError("Width or Height not found in tokens")
    if len(color_tokens) != W * H:
        raise ValueError("Number of color tokens does not match dimensions")
    # Reshape based on serialization
    if serialization == 'row':
        array = np.array(color_tokens).reshape((H, W))
    elif serialization == 'col':
        array = np.array(color_tokens).reshape((W, H)).T
    else:
        raise ValueError(f"Unknown serialization: {serialization}")
    return Grid.from_list(array.tolist())

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
def encode_task(task: dict, vocab: Dict[str, int], view_spec: ViewSpec) -> List[int]:
    tokens = []
    tokens.append(vocab[SPECIAL_TOKENS['BOS']])
    n_train = len(task['train'])
    tokens.append(vocab[SPECIAL_TOKENS['N_TRAIN']])
    tokens.append(n_train)
    for train_pair in task['train']:
        tokens.append(vocab[SPECIAL_TOKENS['SEP']])
        in_grid = train_pair['input']
        out_grid = train_pair['output']
        tokens.extend(encode_grid(in_grid, vocab, view_spec.serialization))
        tokens.append(vocab[SPECIAL_TOKENS['SEP']])
        tokens.extend(encode_grid(out_grid, vocab, view_spec.serialization))
    tokens.append(vocab[SPECIAL_TOKENS['SEP']])
    test_in_grid = task['test']['input']
    tokens.extend(encode_grid(test_in_grid, vocab, view_spec.serialization))
    tokens.append(vocab[SPECIAL_TOKENS['EOS']])
    return tokens

# TODO: Implement decode_output(tokens: List[int], inv_vocab: dict, expected_shape: Tuple[int,int]) -> Grid
#   - Strip special tokens
#   - Parse dimensions if present
#   - Extract color tokens
#   - Reshape to expected_shape
#   - Validate shape
#   - Return Grid
def decode_output(tokens: List[int], inv_vocab: Dict[int, str], expected_shape: Tuple[int,int]) -> Grid:
    color_tokens = []
    for token in tokens:
        token_str = inv_vocab[token]
        if token_str in SPECIAL_TOKENS.values():
            continue
        color_tokens.append(int(token_str))
    H, W = expected_shape
    if len(color_tokens) != W * H:
        raise ValueError("Number of color tokens does not match expected shape")
    array = np.array(color_tokens).reshape((H, W))
    return Grid.from_list(array.tolist())

# TODO: Implement measure_sequence_length(task: dict, vocab: dict) -> dict
#   - Encode task with identity view
#   - Count total tokens
#   - Count tokens per grid
#   - Return statistics dict
def measure_sequence_length(task: dict, vocab: Dict[str, int], view_spec: ViewSpec) -> dict:
    tokens = encode_task(task, vocab, view_spec)
    total_tokens = len(tokens)
    train_tokens = 0
    max_grid_tokens = 0
    grid_token_counts = []
    for train_pair in task['train']:
        in_tokens = encode_grid(train_pair['input'], vocab, view_spec.serialization)
        out_tokens = encode_grid(train_pair['output'], vocab, view_spec.serialization)
        pair_token_count = len(in_tokens) + len(out_tokens) + 2  # +2 for SEP tokens
        train_tokens += pair_token_count
        grid_token_counts.append(len(in_tokens))
        grid_token_counts.append(len(out_tokens))
        max_grid_tokens = max(max_grid_tokens, len(in_tokens), len(out_tokens))
    test_in_tokens = encode_grid(task['test']['input'], vocab, view_spec.serialization)
    train_tokens += len(test_in_tokens) + 1  # +1 for SEP token
    grid_token_counts.append(len(test_in_tokens))
    max_grid_tokens = max(max_grid_tokens, len(test_in_tokens))
    avg_grid_tokens = sum(grid_token_counts) / len(grid_token_counts) if grid_token_counts else 0
    return {
        'total_tokens': total_tokens,
        'train_tokens': train_tokens,
        'test_tokens': total_tokens - train_tokens,
        'max_grid_tokens': max_grid_tokens,
        'avg_grid_tokens': avg_grid_tokens
    }

# TODO: Implement get_sequence_stats_for_dataset(tasks: list) -> dict
#   - Measure sequence lengths for all tasks
#   - Compute min, max, mean, median, p95
#   - Return summary statistics
#   - Record in README.md
def get_sequence_stats_for_dataset(tasks: list, vocab: Dict[str, int], view_spec: ViewSpec) -> dict:
    lengths = []
    for task in tasks:
        stats = measure_sequence_length(task, vocab, view_spec)
        lengths.append(stats['total_tokens'])
    lengths.sort()
    n = len(lengths)
    mean_length = sum(lengths) / n if n > 0 else 0
    if n > 0:
        if n % 2 == 1:
            median_length = lengths[n // 2]
        else:
            median_length = (lengths[n // 2 - 1] + lengths[n // 2]) / 2
    else:
        median_length = 0
    p95_length = lengths[int(n * 0.95)] if n > 0 else 0
    return {
        'min': lengths[0] if n > 0 else 0,
        'max': lengths[-1] if n > 0 else 0,
        'mean': mean_length,
        'median': median_length,
        'p95': p95_length
    }