"""
Step 6: Tests for Tokenization/Serialization

This module tests token encoding and decoding.

Test categories:
1. Vocabulary construction
2. Grid encoding/decoding
3. Task encoding
4. Serialization orders (row vs col major)
5. Sequence length measurement

Tests to implement:

1. test_build_vocab()
   # Build vocabulary
   # Assert all special tokens present
   # Assert color tokens 0-9 present
   # Assert no duplicate ids

2. test_encode_decode_small_grid()
   # Create 3×3 grid
   # Encode to tokens
   # Decode back to grid
   # Assert equal to original

3. test_encode_decode_large_grid()
   # Create 30×30 grid
   # Encode and decode
   # Assert equal to original

4. test_row_major_serialization()
   # Create known grid
   # Encode with row-major
   # Assert tokens in correct order

5. test_col_major_serialization()
   # Create known grid
   # Encode with col-major
   # Assert tokens in correct order

6. test_encode_task()
   # Create task with train and test
   # Encode to tokens
   # Assert structure: [BOS] ... [N_TRAIN] ... [SEP] ... [EOS]

7. test_decode_output()
   # Create token sequence
   # Decode to grid
   # Assert correct shape and values

8. test_special_tokens_handling()
   # Encode grid with special tokens in sequence
   # Decode
   # Assert special tokens handled correctly

9. test_sequence_length_small()
   # Measure sequence length for small task
   # Assert reasonable length (~100 tokens)

10. test_sequence_length_large()
    # Measure sequence length for large task
    # Assert within expected range (~4000 tokens)

11. test_variable_grid_sizes()
    # Encode task with different sized grids
    # Decode each
    # Assert all correct

12. test_metadata_tokens()
    # Encode grid
    # Assert [WIDTH] and [HEIGHT] tokens present
    # Assert values correct

Property-based tests:

13. test_encode_decode_property(random_grid)
    # For any valid grid
    # decode(encode(grid)) == grid

14. test_serialization_reversible(random_grid, random_serialization)
    # For any grid and serialization order
    # Encoding is reversible
"""

# TODO: Import pytest
# TODO: Import numpy as np
# TODO: Import hypothesis (given, strategies)
# TODO: Import arc.grids.core (Grid, from_list, to_list)
# TODO: Import arc.serialize.tokens (build_vocab, encode_grid, decode_grid, etc.)
import pytest
import numpy as np
from hypothesis import given, strategies as st
from arc.grids.core import Grid, from_list
from arc.serialize.tokens import (
    build_vocab,
    encode_grid,
    decode_grid,
    encode_task,
    decode_output,
    get_sequence_stats_for_dataset,
    measure_sequence_length,
    SPECIAL_TOKENS,
)
import arc.grids.views as ViewSpec

# TODO: Implement test_build_vocab()
def test_build_vocab():
    vocab = build_vocab()
    # Assert all special tokens are present
    for token in SPECIAL_TOKENS.values():
        assert token in vocab
    # Assert color tokens 0-9 are present
    for color in range(10):
        assert str(color) in vocab
    
# TODO: Implement test_encode_decode_small_grid()
def test_encode_decode_small_grid():
    grid = from_list([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
    vocab = build_vocab()
    tokens = encode_grid(grid, vocab)
    inv_vocab = {value: key for key, value in vocab.items()}
    decoded_grid = decode_grid(tokens, inv_vocab)
    
    assert decoded_grid.to_list() == grid.to_list()

# TODO: Implement test_encode_decode_large_grid()
def test_encode_decode_large_grid():
    vocab = build_vocab()
    inv_vocab = {value: key for key, value in vocab.items()}
    grid = from_list(np.random.randint(0, 9, (30, 30)).tolist())
    tokens = encode_grid(grid, vocab)
    decoded_grid = decode_grid(tokens, inv_vocab)
    assert decoded_grid.to_list() == grid.to_list()

# TODO: Implement test_row_major_serialization()
def test_row_major_serialization():
    vocab = build_vocab()
    grid = from_list([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
    tokens = encode_grid(grid, vocab, serialization='row')
    assert tokens == [vocab['[WIDTH]'], 3, vocab['[HEIGHT]'], 3, vocab['0'], vocab['1'], vocab['2'], vocab['3'], vocab['4'], vocab['5'], vocab['6'], vocab['7'], vocab['8']]

# TODO: Implement test_col_major_serialization()
def test_col_major_serialization():
    vocab = build_vocab()
    grid = from_list([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
    tokens = encode_grid(grid, vocab, serialization='col')
    assert tokens == [vocab['[WIDTH]'], 3, vocab['[HEIGHT]'], 3,vocab['0'], vocab['3'], vocab['6'], vocab['1'], vocab['4'], vocab['7'], vocab['2'], vocab['5'], vocab['8']]

# TODO: Implement test_encode_task()
def test_encode_task():
    task = {
        'train': [
            {'input': from_list([[0, 1], [2, 3]]), 'output': from_list([[1, 0], [3, 2]])}
        ],
        'test': {'input': from_list([[4, 5], [6, 7]])}
    }
    vocab = build_vocab()
    view_spec = ViewSpec.ViewSpec(geom='id', color_map=tuple(range(10)), serialization='row')
    tokens = encode_task(task, vocab, view_spec)
    print(tokens)
    assert tokens[0] == vocab[SPECIAL_TOKENS['BOS']]
    assert vocab[SPECIAL_TOKENS['N_TRAIN']] in tokens
    assert vocab[SPECIAL_TOKENS['SEP']] in tokens
    assert tokens[-1] == vocab[SPECIAL_TOKENS['EOS']]
    
# TODO: Implement test_decode_output()
def test_decode_output():
    vocab = build_vocab()
    inv_vocab = {value: key for key, value in vocab.items()}
    tokens = [vocab[SPECIAL_TOKENS['BOS']], vocab['0'], vocab['1'], vocab['2'], vocab[SPECIAL_TOKENS['EOS']]]
    grid = decode_output(tokens, inv_vocab, (1, 3))
    expected_grid = from_list([[0, 1, 2]])
    assert grid.to_list() == expected_grid.to_list()
    
# TODO: Implement test_special_tokens_handling()
def test_special_tokens_handling():
    grid = from_list([[0, 1], [2, 3]])
    vocab = build_vocab()
    inv_vocab = {value: key for key, value in vocab.items()}
    tokens = [vocab[SPECIAL_TOKENS['BOS']]] + encode_grid(grid, vocab) + [vocab[SPECIAL_TOKENS['EOS']]]
    decoded_grid = decode_grid(tokens[1:-1], inv_vocab)  # Exclude special tokens for decoding
    assert decoded_grid.to_list() == grid.to_list()
    
# TODO: Implement test_sequence_length_small()
def test_sequence_length_small():
    task = {
        'train': [
            {'input': from_list([[0, 1], [2, 3]]), 'output': from_list([[1, 0], [3, 2]])}
        ],
        'test': {'input': from_list([[4, 5], [6, 7]])}
    }
    vocab = build_vocab()
    viewSpec = ViewSpec.ViewSpec(geom='id', color_map=tuple(range(10)), serialization='row')
    length = measure_sequence_length(task, vocab, viewSpec)
    assert length['total_tokens'] < 200  # Reasonable length for small task
    
# TODO: Implement test_sequence_length_large()
def test_sequence_length_large():
    task = {
        'train': [
            {'input': from_list(np.random.randint(0, 9, (30, 30)).tolist()), 'output': from_list(np.random.randint(0, 9, (30, 30)).tolist())}
        ],
        'test': {'input': from_list(np.random.randint(0, 9, (30, 30)).tolist()) }
    }
    vocab = build_vocab()
    viewSpec = ViewSpec.ViewSpec(geom='id', color_map=tuple(range(10)), serialization='row')
    length = measure_sequence_length(task, vocab, viewSpec)
    assert length['total_tokens'] < 3000  # Reasonable length for large task

# TODO: Implement test_variable_grid_sizes()
def test_variable_grid_sizes():
    sizes = [(2, 2), (5, 5), (10, 10), (15, 15)]
    vocab = build_vocab()
    inv_vocab = {value: key for key, value in vocab.items()}
    for H, W in sizes:
        grid = from_list(np.random.randint(0, 9, (H, W)).tolist())
        tokens = encode_grid(grid, vocab)
        decoded_grid = decode_grid(tokens, inv_vocab)
        assert decoded_grid.to_list() == grid.to_list()
        
# TODO: Implement test_metadata_tokens()
def test_metadata_tokens():
    grid = from_list([[0, 1], [2, 3]])
    vocab = build_vocab()
    tokens = encode_grid(grid, vocab)
    assert vocab[SPECIAL_TOKENS['WIDTH']] in tokens
    assert vocab[SPECIAL_TOKENS['HEIGHT']] in tokens
    width_index = tokens.index(vocab[SPECIAL_TOKENS['WIDTH']]) + 1
    height_index = tokens.index(vocab[SPECIAL_TOKENS['HEIGHT']]) + 1
    assert tokens[width_index] == 2
    assert tokens[height_index] == 2

# TODO: Implement property-based tests
def test_property_based_tests():
    vocab = build_vocab()
    inv_vocab = {value: key for key, value in vocab.items()}
    for _ in range(100):  # Run 100 tests
        grid = from_list(np.random.randint(0, 9, (4, 4)).tolist())
        tokens = encode_grid(grid, vocab)
        decoded_grid = decode_grid(tokens, inv_vocab)
        assert decoded_grid.to_list() == grid.to_list()

# TODO: Implement test_serialization_reversible()
def test_serialization_reversible():
    grid = from_list([[0, 1], [2, 3]])
    vocab = build_vocab()
    inv_vocab = {value: key for key, value in vocab.items()}
    # Test row-major serialization
    tokens_row = encode_grid(grid, vocab, serialization='row')
    decoded_row = decode_grid(tokens_row, inv_vocab, serialization='row')
    assert decoded_row.to_list() == grid.to_list()
    # Test col-major serialization
    tokens_col = encode_grid(grid, vocab, serialization='col')
    decoded_col = decode_grid(tokens_col, inv_vocab, serialization='col')
    assert decoded_col.to_list() == grid.to_list()
