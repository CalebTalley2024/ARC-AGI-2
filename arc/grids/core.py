"""
Step 4: Core Grid Type + Safe Operations

This module defines the fundamental Grid type and basic operations.

Key concepts:
- Grid: A 2D array (H×W) where each cell is a color id (0-9)
- Palette: Set of unique color ids present in a grid
- Shape: (H, W) - height and width
- Dtype: np.int8 or np.uint8 for memory efficiency

Design goals:
- Small, strict, fast operations
- Immutability by convention (copy when transforming)
- Deterministic behavior
- Early validation to catch bugs

Functions to implement:
1. from_list(lst: list[list[int]]) -> Grid
   # Convert Python list to Grid with validation
   # - Check 2D structure
   # - Validate shape: 0 < H,W <= 30
   # - Validate values: 0 <= value <= 9
   # - Ensure contiguous memory layout
   # - Return Grid object

2. to_list(g: Grid) -> list[list[int]]
   # Convert Grid back to Python list
   # - Simple .tolist() call
   # - Used for JSON serialization

3. palette(g: Grid) -> set[int]
   # Extract unique colors present in grid
   # - Use np.unique() for efficiency
   # - Return as set for fast membership tests
   # - O(H*W) operation

4. assert_same_shape(a: Grid, b: Grid) -> None
   # Validate two grids have identical shapes
   # - Compare (H, W) tuples
   # - Raise ValueError if mismatch
   # - Used before diff/comparison operations

5. diff(a: Grid, b: Grid) -> np.ndarray
   # Create boolean mask of differing cells
   # - First check same shape
   # - Return (a.a != b.a) boolean array
   # - Used for verification and visualization

Grid class structure:
@dataclass(frozen=True)
class Grid:
    # Light wrapper to enforce invariants
    # a: GridArray (np.ndarray, shape (H,W), dtype int8/uint8)
    
    # @property shape -> tuple[int, int]
    # Return (H, W) from underlying array
    
    # def copy() -> Grid
    # Return new Grid with copied array
    # Prevents accidental mutation

Safety behaviors:
- Never mutate inputs in place
- Always return new Grid objects
- Use np.ascontiguousarray after transforms
- Validate values stay in 0-9 range
- Explicit .copy() when modifying

Tests to write (tests/test_core.py):
- test_round_trip: from_list -> to_list identity
- test_palette: correct unique colors extracted
- test_diff_mask: boolean mask has correct True positions
- test_validation: reject invalid inputs (out of range, wrong shape)
- test_shape_validation: reject grids > 30x30 or empty
"""

# TODO: Import numpy as np
# TODO: Import dataclasses (dataclass)
# TODO: Define GridArray type alias = np.ndarray

# TODO: Implement Grid dataclass
#   - frozen=True for immutability
#   - field: a (GridArray)
#   - property: shape
#   - method: copy()

# TODO: Implement from_list(lst, dtype=np.uint8) -> Grid
#   - Convert to numpy array
#   - Validate 2D (ndim == 2)
#   - Validate shape (0 < H,W <= 30)
#   - Validate values (0 <= v <= 9)
#   - Ensure contiguous memory
#   - Return Grid

# TODO: Implement to_list(g: Grid) -> list[list[int]]
#   - Return g.a.tolist()

# TODO: Implement palette(g: Grid) -> set[int]
#   - Use np.unique(g.a)
#   - Convert to set and return

# TODO: Implement assert_same_shape(a: Grid, b: Grid) -> None
#   - Compare a.shape vs b.shape
#   - Raise ValueError with descriptive message if different

# TODO: Implement diff(a: Grid, b: Grid) -> np.ndarray
#   - Call assert_same_shape first
#   - Return boolean array (a.a != b.a)

# Common pitfalls to avoid:
# - Row/column confusion: always use (row, col) = (y, x) indexing
# - Dtype overflow: uint8 wraps at 255, prefer int8 with range checks
# - In-place edits: always work on copies
# - Non-contiguous arrays: call np.ascontiguousarray() after transforms
# - Shape drift: validate shapes after operations
