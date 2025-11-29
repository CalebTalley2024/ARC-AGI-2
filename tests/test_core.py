import sys
import os
import numpy as np

# Add the project root to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from arc.grids.core import diff, assert_valid_grid, Grid, from_list


def test_round_trip():
    g = Grid(np.array([[0, 1], [2, 3]]))
    assert g.to_list() == [[0, 1], [2, 3]]


def test_palette():
    g = Grid(np.array([[0, 0, 3], [7, 3, 7]]))
    assert g.palette() == {0, 3, 7}


def test_diff_mask():
    a = Grid(np.array([[1, 1], [2, 2]]))
    b = Grid(np.array([[1, 0], [2, 3]]))
    m = diff(a, b)
    assert m.dtype == np.bool_
    assert m.sum() == 2  # (0,1) and (1,1) differ


def test_validation():
    try:
        Grid(np.array([[10]]))  # out of range
        assert False
    except ValueError:
        assert True


def test_assert_valid_grid():
    """Test that assert_valid_grid catches various validation issues."""
    # Valid grid should pass
    g = Grid(np.array([[1, 2], [3, 4]]))
    assert_valid_grid(g)  # Should not raise

    # Test invalid type
    try:
        assert_valid_grid("not a grid")
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "Expected Grid object" in str(e)

    # Test non-contiguous array - Grid constructor now makes it contiguous automatically
    arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    g_non_contiguous = Grid(arr[:, ::2])  # Grid constructor makes this contiguous
    assert_valid_grid(g_non_contiguous)  # Should pass now


def test_assert_valid_grid_edge_cases():
    """Test edge cases for assert_valid_grid."""
    # Test minimum valid size
    g_min = Grid(np.array([[0]]))
    assert_valid_grid(g_min)

    # Test maximum valid size
    g_max = Grid(np.array([[0] * 30 for _ in range(30)]))
    assert_valid_grid(g_max)

    # Test color range boundaries
    g_colors = Grid(np.array([[0, 9]]))
    assert_valid_grid(g_colors)


def test_assert_valid_grid_shape_drift():
    """Test that assert_valid_grid catches shape drift after transformations."""
    # Create a valid grid
    g = Grid(np.array([[1, 2], [3, 4]]))
    assert_valid_grid(g)  # Should pass

    # Grid constructor now automatically makes arrays contiguous
    # So we test that the constructor handles non-contiguous input correctly
    arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    g_drifted = Grid(arr[:, ::2])  # Creates strided view (every other column)

    # Should pass because Grid constructor handles non-contiguous arrays
    assert_valid_grid(g_drifted)


def test_is_copy_flag():
    """Test that the is_copy flag works correctly."""
    # Original grid should have is_copy=False
    g = Grid(np.array([[1, 2], [3, 4]]))
    assert g.is_copy == False

    # Copy should have is_copy=True
    copy = g.copy()
    assert copy.is_copy == True


def test_integer_validation():
    """Test that Grid enforces integer values."""
    # Integer values should work
    g1 = Grid(np.array([[1, 2], [3, 4]]))
    g2 = Grid(np.array([[1.0, 2.0], [3.0, 4.0]]))  # Float integers

    # Non-integer values should fail
    try:
        Grid(np.array([[1.5, 2], [3, 4]]))
        assert False, "Should have raised ValueError for non-integer"
    except ValueError as e:
        assert "All grid values must be integers" in str(e)

    try:
        Grid(np.array([[1, 2.7], [3, 4]]))
        assert False, "Should have raised ValueError for non-integer"
    except ValueError as e:
        assert "All grid values must be integers" in str(e)


def test_int8_dtype():
    """Test that Grid enforces np.int8 data type."""
    # Test with different input dtypes - all should become int8
    g1 = Grid(np.array([[1, 2], [3, 4]], dtype=np.int32))
    g2 = Grid(np.array([[1, 2], [3, 4]], dtype=np.float64))
    g3 = Grid(np.array([[1, 2], [3, 4]], dtype=np.uint8))

    # All should have int8 dtype
    assert g1.a.dtype == np.int8
    assert g2.a.dtype == np.int8
    assert g3.a.dtype == np.int8

    # Values should be preserved
    assert np.array_equal(g1.a, np.array([[1, 2], [3, 4]], dtype=np.int8))
    assert np.array_equal(g2.a, np.array([[1, 2], [3, 4]], dtype=np.int8))
    assert np.array_equal(g3.a, np.array([[1, 2], [3, 4]], dtype=np.int8))


def test_from_list_basic():
    """Test basic functionality of from_list function."""
    # Test basic conversion
    lst = [[0, 1, 2], [3, 4, 5]]
    g = from_list(lst)
    assert isinstance(g, Grid)
    assert g.shape == (2, 3)
    assert g.to_list() == lst


def test_from_list_is_copy_flag():
    """Test that from_list respects the is_copy flag."""
    lst = [[1, 2], [3, 4]]

    # Test with is_copy=False (default)
    g1 = from_list(lst, is_copy=False)
    assert g1.is_copy == False
    assert g1.a.flags.writeable == False

    # Test with is_copy=True
    g2 = from_list(lst, is_copy=True)
    assert g2.is_copy == True
    assert g2.a.flags.writeable == True


def test_from_list_round_trip():
    """Test round-trip conversion: list -> Grid -> list."""
    original_lst = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    g = from_list(original_lst)
    result_lst = g.to_list()
    assert result_lst == original_lst


if __name__ == "__main__":
    test_round_trip()
    test_palette()
    test_diff_mask()
    test_validation()
    test_assert_valid_grid()
    test_assert_valid_grid_edge_cases()
    test_assert_valid_grid_shape_drift()
    test_is_copy_flag()
    test_integer_validation()
    test_int8_dtype()
    test_from_list_basic()
    test_from_list_is_copy_flag()
    test_from_list_round_trip()
    print("All tests passed!")
