import numpy as np

import arc.grids.views as ViewSpec
from arc.grids.core import from_list
from arc.serialize.tokens import (
    SPECIAL_TOKENS,
    build_vocab,
    decode_grid,
    decode_output,
    encode_grid,
    encode_task,
    measure_sequence_length,
)


def test_build_vocab():
    vocab = build_vocab()
    # Assert all special tokens are present
    for token in SPECIAL_TOKENS.values():
        assert token in vocab
    # Assert color tokens 0-9 are present
    for color in range(10):
        assert str(color) in vocab


def test_encode_decode_small_grid():
    grid = from_list([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
    vocab = build_vocab()
    tokens = encode_grid(grid, vocab)
    inv_vocab = {value: key for key, value in vocab.items()}
    decoded_grid = decode_grid(tokens, inv_vocab)

    assert decoded_grid.to_list() == grid.to_list()


def test_encode_decode_large_grid():
    vocab = build_vocab()
    inv_vocab = {value: key for key, value in vocab.items()}
    grid = from_list(np.random.randint(0, 9, (30, 30)).tolist())
    tokens = encode_grid(grid, vocab)
    decoded_grid = decode_grid(tokens, inv_vocab)
    assert decoded_grid.to_list() == grid.to_list()


def test_row_major_serialization():
    vocab = build_vocab()
    grid = from_list([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
    tokens = encode_grid(grid, vocab, serialization="row")
    assert tokens == [
        vocab["[WIDTH]"],
        3,
        vocab["[HEIGHT]"],
        3,
        vocab["0"],
        vocab["1"],
        vocab["2"],
        vocab["3"],
        vocab["4"],
        vocab["5"],
        vocab["6"],
        vocab["7"],
        vocab["8"],
    ]


def test_col_major_serialization():
    vocab = build_vocab()
    grid = from_list([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
    tokens = encode_grid(grid, vocab, serialization="col")
    assert tokens == [
        vocab["[WIDTH]"],
        3,
        vocab["[HEIGHT]"],
        3,
        vocab["0"],
        vocab["3"],
        vocab["6"],
        vocab["1"],
        vocab["4"],
        vocab["7"],
        vocab["2"],
        vocab["5"],
        vocab["8"],
    ]


def test_encode_task():
    task = {
        "train": [{"input": from_list([[0, 1], [2, 3]]), "output": from_list([[1, 0], [3, 2]])}],
        "test": {"input": from_list([[4, 5], [6, 7]])},
    }
    vocab = build_vocab()
    view_spec = ViewSpec.ViewSpec(geom="id", color_map=tuple(range(10)), serialization="row")
    tokens = encode_task(task, vocab, view_spec)
    print(tokens)
    assert tokens[0] == vocab[SPECIAL_TOKENS["BOS"]]
    assert vocab[SPECIAL_TOKENS["N_TRAIN"]] in tokens
    assert vocab[SPECIAL_TOKENS["SEP"]] in tokens
    assert tokens[-1] == vocab[SPECIAL_TOKENS["EOS"]]


def test_decode_output():
    vocab = build_vocab()
    inv_vocab = {value: key for key, value in vocab.items()}
    tokens = [vocab[SPECIAL_TOKENS["BOS"]], vocab["0"], vocab["1"], vocab["2"], vocab[SPECIAL_TOKENS["EOS"]]]
    grid = decode_output(tokens, inv_vocab, (1, 3))
    expected_grid = from_list([[0, 1, 2]])
    assert grid.to_list() == expected_grid.to_list()


def test_special_tokens_handling():
    grid = from_list([[0, 1], [2, 3]])
    vocab = build_vocab()
    inv_vocab = {value: key for key, value in vocab.items()}
    tokens = [vocab[SPECIAL_TOKENS["BOS"]]] + encode_grid(grid, vocab) + [vocab[SPECIAL_TOKENS["EOS"]]]
    decoded_grid = decode_grid(tokens[1:-1], inv_vocab)  # Exclude special tokens for decoding
    assert decoded_grid.to_list() == grid.to_list()


def test_sequence_length_small():
    task = {
        "train": [{"input": from_list([[0, 1], [2, 3]]), "output": from_list([[1, 0], [3, 2]])}],
        "test": {"input": from_list([[4, 5], [6, 7]])},
    }
    vocab = build_vocab()
    viewSpec = ViewSpec.ViewSpec(geom="id", color_map=tuple(range(10)), serialization="row")
    length = measure_sequence_length(task, vocab, viewSpec)
    assert length["total_tokens"] < 200  # Reasonable length for small task


def test_sequence_length_large():
    task = {
        "train": [
            {
                "input": from_list(np.random.randint(0, 9, (30, 30)).tolist()),
                "output": from_list(np.random.randint(0, 9, (30, 30)).tolist()),
            }
        ],
        "test": {"input": from_list(np.random.randint(0, 9, (30, 30)).tolist())},
    }
    vocab = build_vocab()
    viewSpec = ViewSpec.ViewSpec(geom="id", color_map=tuple(range(10)), serialization="row")
    length = measure_sequence_length(task, vocab, viewSpec)
    assert length["total_tokens"] < 3000  # Reasonable length for large task


def test_variable_grid_sizes():
    sizes = [(2, 2), (5, 5), (10, 10), (15, 15)]
    vocab = build_vocab()
    inv_vocab = {value: key for key, value in vocab.items()}
    for H, W in sizes:
        grid = from_list(np.random.randint(0, 9, (H, W)).tolist())
        tokens = encode_grid(grid, vocab)
        decoded_grid = decode_grid(tokens, inv_vocab)
        assert decoded_grid.to_list() == grid.to_list()


def test_metadata_tokens():
    grid = from_list([[0, 1], [2, 3]])
    vocab = build_vocab()
    tokens = encode_grid(grid, vocab)
    assert vocab[SPECIAL_TOKENS["WIDTH"]] in tokens
    assert vocab[SPECIAL_TOKENS["HEIGHT"]] in tokens
    width_index = tokens.index(vocab[SPECIAL_TOKENS["WIDTH"]]) + 1
    height_index = tokens.index(vocab[SPECIAL_TOKENS["HEIGHT"]]) + 1
    assert tokens[width_index] == 2
    assert tokens[height_index] == 2


def test_property_based_tests():
    vocab = build_vocab()
    inv_vocab = {value: key for key, value in vocab.items()}
    for _ in range(100):  # Run 100 tests
        grid = from_list(np.random.randint(0, 9, (4, 4)).tolist())
        tokens = encode_grid(grid, vocab)
        decoded_grid = decode_grid(tokens, inv_vocab)
        assert decoded_grid.to_list() == grid.to_list()


def test_serialization_reversible():
    grid = from_list([[0, 1], [2, 3]])
    vocab = build_vocab()
    inv_vocab = {value: key for key, value in vocab.items()}
    # Test row-major serialization
    tokens_row = encode_grid(grid, vocab, serialization="row")
    decoded_row = decode_grid(tokens_row, inv_vocab, serialization="row")
    assert decoded_row.to_list() == grid.to_list()
    # Test col-major serialization
    tokens_col = encode_grid(grid, vocab, serialization="col")
    decoded_col = decode_grid(tokens_col, inv_vocab, serialization="col")
    assert decoded_col.to_list() == grid.to_list()
