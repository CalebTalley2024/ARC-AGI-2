import numpy as np
from hypothesis import given
from hypothesis import strategies as st

from arc.grids.core import Grid
from arc.grids.views import (
    D4,
    ViewSpec,
    apply_color_map,
    apply_view_grid,
    apply_view_task,
    generate_data_driven_permutations,
    generate_palette_permutations,
    geom_apply,
    geom_inverse,
    identity_cmap,
    invert_color_map,
    invert_view_grid,
)


# Geometry identity
def test_geometry_identity():
    """Apply identity transform - grid should remain unchanged."""
    grid = Grid(np.array([[1, 2, 3], [4, 5, 6]]))
    result = geom_apply(grid.a, 'id')
    np.testing.assert_array_equal(result, grid.a)


# Geometry rot90
def test_geometry_rot90():
    """Apply 90-degree rotation."""
    grid = Grid(np.array([[1, 2], [3, 4]]))
    result = geom_apply(grid.a, 'rot90')
    expected = np.array([[2, 4], [1, 3]])
    np.testing.assert_array_equal(result, expected)


# Geometry rot180
def test_geometry_rot180():
    """Apply 180-degree rotation."""
    grid = Grid(np.array([[1, 2], [3, 4]]))
    result = geom_apply(grid.a, 'rot180')
    expected = np.array([[4, 3], [2, 1]])
    np.testing.assert_array_equal(result, expected)


# Geometry flip_h
def test_geometry_flip_h():
    """Apply horizontal flip."""
    grid = Grid(np.array([[1, 2, 3], [4, 5, 6]]))
    result = geom_apply(grid.a, 'flip_h')
    expected = np.array([[3, 2, 1], [6, 5, 4]])
    np.testing.assert_array_equal(result, expected)


# Geometry flip_v
def test_geometry_flip_v():
    """Apply vertical flip."""
    grid = Grid(np.array([[1, 2, 3], [4, 5, 6]]))
    result = geom_apply(grid.a, 'flip_v')
    expected = np.array([[4, 5, 6], [1, 2, 3]])
    np.testing.assert_array_equal(result, expected)


# Geometry transpose
def test_geometry_transpose():
    """Apply transpose - rows become columns."""
    grid = Grid(np.array([[1, 2, 3], [4, 5, 6]]))
    result = geom_apply(grid.a, 'transpose')
    expected = np.array([[1, 4], [2, 5], [3, 6]])
    np.testing.assert_array_equal(result, expected)


# Geometry inverse
def test_geometry_inverse():
    """For each geometry transform, apply then invert - should return to original."""
    grid = Grid(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))

    for geom_name in D4:
        transformed = geom_apply(grid.a, geom_name)
        inv_name = geom_inverse(geom_name)
        restored = geom_apply(transformed, inv_name)
        np.testing.assert_array_equal(restored, grid.a,
                                      err_msg=f"Failed for {geom_name}")


# Color map identity
def test_color_map_identity():
    """Apply identity color map - grid should remain unchanged."""
    grid = Grid(np.array([[0, 1, 2], [3, 4, 5]]))
    cmap = identity_cmap()
    result = apply_color_map(grid.a, cmap)
    np.testing.assert_array_equal(result, grid.a)


# Color map swap
def test_color_map_swap():
    """Swap two colors."""
    grid = Grid(np.array([[0, 1, 2], [1, 0, 2]]))
    # Swap colors 0 and 1
    cmap = (1, 0, 2, 3, 4, 5, 6, 7, 8, 9)
    result = apply_color_map(grid.a, cmap)
    expected = np.array([[1, 0, 2], [0, 1, 2]])
    np.testing.assert_array_equal(result, expected)


# Color map inverse
def test_color_map_inverse():
    """Apply color map then invert - should return to original."""
    grid = Grid(np.array([[0, 1, 2, 3], [4, 5, 6, 7]]))
    cmap = (1, 0, 3, 2, 5, 4, 7, 6, 9, 8)  # Swap pairs

    transformed = apply_color_map(grid.a, cmap)
    inv_cmap = invert_color_map(cmap)
    restored = apply_color_map(transformed, inv_cmap)
    np.testing.assert_array_equal(restored, grid.a)


#   View inverse grid
def test_view_inverse_grid():
    """Apply view then invert - should return to original grid."""
    grid = Grid(np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]]))
    spec = ViewSpec(geom='rot90', color_map=(1, 0, 2, 3, 4, 5, 6, 7, 8, 9), serialization='row')

    transformed = apply_view_grid(grid, spec)
    restored = invert_view_grid(transformed, spec)
    np.testing.assert_array_equal(restored.a, grid.a)


# View inverse task
def test_view_inverse_task():
    """Apply view to task - verify train pairs still match."""
    task = {
        "train": [
            {"input": Grid(np.array([[0, 1], [2, 3]])), "output": Grid(np.array([[1, 0], [3, 2]]))},
            {"input": Grid(np.array([[4, 5], [6, 7]])), "output": Grid(np.array([[5, 4], [7, 6]]))}
        ],
        "test": [Grid(np.array([[8, 9], [0, 1]]))]
    }

    spec = ViewSpec(geom='flip_h', color_map=identity_cmap(), serialization='row')
    transformed_task = apply_view_task(task, spec)

    # Verify structure
    assert len(transformed_task["train"]) == 2
    assert len(transformed_task["test"]) == 1

    # Verify train pairs are transformed
    for pair in transformed_task["train"]:
        assert isinstance(pair["input"], Grid)
        assert isinstance(pair["output"], Grid)


# Color safety
def test_color_safety():
    """Apply various views - all values should remain in 0-9 range."""
    grid = Grid(np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]]))

    test_specs = [
        ViewSpec(geom='rot90', color_map=identity_cmap(), serialization='row'),
        ViewSpec(geom='flip_h', color_map=(9, 8, 7, 6, 5, 4, 3, 2, 1, 0), serialization='row'),
        ViewSpec(geom='transpose', color_map=(1, 0, 2, 3, 4, 5, 6, 7, 8, 9), serialization='col'),
    ]

    for spec in test_specs:
        transformed = apply_view_grid(grid, spec)
        assert transformed.a.min() >= 0, f"Min value out of range for {spec}"
        assert transformed.a.max() <= 9, f"Max value out of range for {spec}"


# Shape handling
def test_shape_handling():
    """Apply rot90 to (H,W) grid - result should be (W,H)."""
    grid = Grid(np.array([[1, 2, 3], [4, 5, 6]]))  # Shape: (2, 3)

    # rot90 should swap dimensions
    result = geom_apply(grid.a, 'rot90')
    assert result.shape == (3, 2), f"Expected (3, 2), got {result.shape}"

    # transpose should also swap dimensions
    result = geom_apply(grid.a, 'transpose')
    assert result.shape == (3, 2), f"Expected (3, 2), got {result.shape}"

    # flip_h should preserve dimensions
    result = geom_apply(grid.a, 'flip_h')
    assert result.shape == (2, 3), f"Expected (2, 3), got {result.shape}"


# Test 15: Bijection
def test_bijection():
    """For color_map, verify it's a valid permutation."""
    cmap = (1, 0, 3, 2, 5, 4, 7, 6, 9, 8)

    # Assert set(cmap) == {0,1,2,3,4,5,6,7,8,9}
    assert set(cmap) == set(range(10)), "Color map is not a valid permutation"

    # Assert invert(invert(cmap)) == cmap
    inv_cmap = invert_color_map(cmap)
    double_inv = invert_color_map(inv_cmap)
    assert double_inv == cmap, "Double inversion should return original"


# Hypothesis strategies for property-based testing
@st.composite
def valid_grid_strategy(draw):
    """Generate random valid grids."""
    height = draw(st.integers(min_value=1, max_value=30))
    width = draw(st.integers(min_value=1, max_value=30))
    arr = draw(st.lists(
        st.lists(st.integers(min_value=0, max_value=9), min_size=width, max_size=width),
        min_size=height, max_size=height
    ))
    return Grid(np.array(arr))


@st.composite
def valid_viewspec_strategy(draw):
    """Generate random valid ViewSpecs."""
    geom = draw(st.sampled_from(D4))

    # Generate a valid permutation of 0-9
    perm = list(range(10))
    draw(st.randoms()).shuffle(perm)
    color_map = tuple(perm)

    serialization = draw(st.sampled_from(['row', 'col']))

    return ViewSpec(geom=geom, color_map=color_map, serialization=serialization)


# Property-based test for view inverse
@given(grid=valid_grid_strategy(), spec=valid_viewspec_strategy())
def test_view_inverse_property(grid, spec):
    """For any grid and viewspec, invert_view_grid(apply_view_grid(g, spec), spec) == g."""
    transformed = apply_view_grid(grid, spec)
    restored = invert_view_grid(transformed, spec)
    np.testing.assert_array_equal(restored.a, grid.a)


#Property-based test for task consistency
@given(spec=valid_viewspec_strategy())
def test_task_consistency_property(spec):
    """For any task and viewspec, after transformation, train pairs still match."""
    task = {
        "train": [
            {"input": Grid(np.array([[0, 1], [2, 3]])), "output": Grid(np.array([[1, 0], [3, 2]]))},
        ],
        "test": [Grid(np.array([[8, 9], [0, 1]]))]
    }

    transformed_task = apply_view_task(task, spec)

    # Verify structure is preserved
    assert len(transformed_task["train"]) == len(task["train"])
    assert len(transformed_task["test"]) == len(task["test"])

    # Verify all grids are valid
    for pair in transformed_task["train"]:
        assert isinstance(pair["input"], Grid)
        assert isinstance(pair["output"], Grid)


# Property-based test for color range
@given(grid=valid_grid_strategy(), spec=valid_viewspec_strategy())
def test_color_range_property(grid, spec):
    """For any grid and viewspec, after transformation, all values in 0-9."""
    transformed = apply_view_grid(grid, spec)
    assert transformed.a.min() >= 0
    assert transformed.a.max() <= 9


# Test palette permutation generation
def test_generate_palette_permutations_small():
    """Test permutation generation for small palette."""
    palette = {0, 1, 2}
    perms = generate_palette_permutations(palette, max_count=8)

    # Should include identity
    assert identity_cmap() in perms

    # All should be valid permutations
    for perm in perms:
        assert len(perm) == 10
        assert set(perm) == set(range(10))

    # Should have multiple permutations for palette with 3 colors
    assert len(perms) > 1


def test_generate_palette_permutations_large():
    """Test permutation generation for larger palette."""
    palette = {0, 1, 2, 3, 4, 5}
    perms = generate_palette_permutations(palette, max_count=8)

    # Should include identity
    assert identity_cmap() in perms

    # Should respect max_count
    assert len(perms) <= 8

    # All should be valid
    for perm in perms:
        assert set(perm) == set(range(10))


def test_generate_data_driven_permutations():
    """Test data-driven permutation generation."""
    # Create simple training pairs where colors swap
    train_pairs = [
        {
            "input": Grid(np.array([[0, 1], [1, 0]])),
            "output": Grid(np.array([[1, 0], [0, 1]]))
        }
    ]
    palette = {0, 1}

    perms = generate_data_driven_permutations(train_pairs, palette, max_count=5)

    # Should include identity
    assert identity_cmap() in perms

    # All should be valid
    for perm in perms:
        assert len(perm) == 10
        assert set(perm) == set(range(10))
