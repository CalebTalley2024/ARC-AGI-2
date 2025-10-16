from dataclasses import dataclass
import numpy as np

# @caleb: i'm using np.int8 instead of np.uint8 in case there is signinficance to adding/subtracting palettes with results outside of 0 to 9
GridArray = np.ndarray  # (H, W), dtype int8 or uint8

class Grid:
    """Light wrapper to enforce invariants."""
    #treat a as read-only. use copy when changing 
    #attributes in dataclass format
    # create Grid (from_list function from Week 0)
    def __init__(self, a: GridArray, is_copy: bool = False):
        """Initialize Grid using from_list validation logic."""
        # validate grid
        if a.ndim != 2:
            raise ValueError("Grid must be 2D")
        if a.shape[0] == 0 or a.shape[1] == 0 or a.shape[0] > 30 or a.shape[1] > 30:
            raise ValueError("Invalid grid shape; ARC uses <= 30x30 and > 0")
        if a.min() < 0 or a.max() > 9:
            raise ValueError("Color ids must be in between 0 to 9")
        # Check that all values are integers  (using modulo to check end is 0 or a decimal)
        if not np.all(np.equal(np.mod(a, 1), 0)):
            raise ValueError("All grid values must be integers")
        # Convert to int8 and ensure contiguous memory
        a = np.ascontiguousarray(a, dtype=np.int8)
        # Set attributes manually
        self.a = a
        self.is_copy = is_copy
        # Make array immutable unless it's a copy
        self.a.flags.writeable = self.is_copy

    @property
    def shape(self) -> tuple[int, int]:
        return self.a.shape

    def copy(self) -> "Grid":
        '''
        *treat* loaded grids as read-only and copy when transforming. 
        This prevents accidental in-place edits during search or scoring.
        Note: not effected by the frozen=True b/c this is a new Grid
        '''
        return Grid(self.a.copy(), is_copy=True)


    def palette(self) -> set[int]:
        """
        returns the unique colors present in the grid
        """
        return set(np.unique(self.a).tolist())

    def to_list(self) -> list[list[int]]:
        return self.a.tolist()


    #representation method for debugging
    def __repr__(self) -> str:
        """
        representation method for debugging
        """
        return f"Grid(shape={self.shape}, is_copy={self.is_copy})"

def from_list(lst: list[list[int]], dtype=np.int8,  is_copy = False) -> Grid:
    '''
    converts list of lists of integers to a Grid object
    '''
    a = np.array(lst, dtype=dtype, copy=True)
    #all validation happens in the Grid constructor, so no need to validate here
    return Grid(a, is_copy)

def assert_same_shape(a: Grid, b: Grid) -> None:
    """
    checks if the shapes of two grids are the same
    """
    if a.shape != b.shape:
        raise ValueError(f"Shape mismatch: {a.shape} vs {b.shape}")

def diff(a: Grid, b: Grid) -> np.ndarray:
    """Return boolean mask where cells differ."""
    assert_same_shape(a, b)
    return (a.a != b.a)

def assert_valid_grid(g: Grid) -> None:
    """
    Assert that a Grid object is valid according to ARC constraints.
    
    This helper catches hidden shape drift and other validation issues
    that might occur after transformations, views, or other operations.
    
    Args:
        g: Grid object to validate
        
    Raises:
        ValueError: If grid violates ARC constraints
    """
    if not isinstance(g, Grid):
        raise ValueError(f"Expected Grid object, got {type(g)}")
    
    if g.a.ndim != 2:
        raise ValueError(f"Grid must be 2D, got {g.a.ndim}D")
    
    if g.a.shape[0] == 0 or g.a.shape[1] == 0:
        raise ValueError(f"Grid cannot have zero dimensions, got shape {g.a.shape}")
    
    if g.a.shape[0] > 30 or g.a.shape[1] > 30:
        raise ValueError(f"Grid exceeds ARC size limit (30x30), got shape {g.a.shape}")
    
    if g.a.min() < 0 or g.a.max() > 9:
        raise ValueError(f"Grid colors must be in range [0, 9], got range [{g.a.min()}, {g.a.max()}]")
    
    if not g.a.flags.c_contiguous:
        raise ValueError("Grid array is not contiguous in memory - use np.ascontiguousarray()")