"""
Step 5: Implement Views (Φ) and Inverse Views

This module implements three families of views for data augmentation:
1. Geometry views (D₄ dihedral group - 8 symmetries)
2. Color permutation views (bijective relabeling of colors 0-9)
3. Serialization views (row-major vs column-major)

Every view must have a correct inverse for mapping model outputs back.

Key concepts:
- ViewSpec: Compact, hashable description of a view transformation
- D₄ group: 8 geometric symmetries (rotations + reflections)
- Color permutation: Bijective mapping of color ids
- Serialization: Order of reading grid cells (row/col major)

ViewSpec structure:
@dataclass(frozen=True)
class ViewSpec:
    # geom: str - geometry transform name
    #   Options: 'id', 'rot90', 'rot180', 'rot270',
    #            'flip_h', 'flip_v', 'transpose', 'transpose_flip'
    # color_map: Tuple[int, ...] - length-10 permutation of {0..9}
    #   Example: (0,1,2,3,4,5,6,7,8,9) is identity
    #   Example: (1,0,2,3,4,5,6,7,8,9) swaps colors 0 and 1
    # serialization: str - 'row' or 'col'
    #   Controls how grid is linearized into tokens

Geometry transforms (D₄ operations):
1. geom_apply(arr: np.ndarray, name: str) -> np.ndarray
   # Apply geometric transformation to array
   # - 'id': identity (no change)
   # - 'rot90': rotate 90° counterclockwise
   # - 'rot180': rotate 180°
   # - 'rot270': rotate 270° (= 90° clockwise)
   # - 'flip_h': horizontal flip (left-right)
   # - 'flip_v': vertical flip (top-bottom)
   # - 'transpose': swap rows and columns
   # - 'transpose_flip': transpose then flip horizontally
   # Use np.rot90(arr, k) and np.flip(arr, axis)

2. geom_inverse(name: str) -> str
   # Return inverse operation name
   # - Most are self-inverse (flip_h, flip_v, transpose, rot180)
   # - rot90 ↔ rot270
   # - 'id' ↔ 'id'

Color permutation operations:
3. apply_color_map(arr: np.ndarray, cmap: Tuple[int,...]) -> np.ndarray
   # Apply color permutation using lookup table
   # - cmap is length-10 tuple (permutation of 0..9)
   # - Convert to numpy array for vectorized indexing
   # - Return cmap_array[arr] (fast gather operation)
   # Example: if cmap=(1,0,2,3,4,5,6,7,8,9), then 0→1, 1→0

4. invert_color_map(cmap: Tuple[int,...]) -> Tuple[int,...]
   # Compute inverse permutation
   # - Create inverse array: inv[cmap[i]] = i
   # - Return as tuple
   # Example: if cmap=(1,0,2,3,4,5,6,7,8,9), inverse is same

View application functions:
5. apply_view_grid(g: Grid, spec: ViewSpec) -> Grid
   # Apply view transformation to a single grid
   # Order: geometry first, then colors
   # Steps:
   #   a. Apply geometry: arr = geom_apply(g.a, spec.geom)
   #   b. Apply colors: arr = apply_color_map(arr, spec.color_map)
   #   c. Ensure contiguous: arr = np.ascontiguousarray(arr)
   #   d. Return Grid(arr)

6. invert_view_grid(g: Grid, spec: ViewSpec) -> Grid
   # Invert view transformation
   # Order: colors first, then geometry (reverse of apply)
   # Steps:
   #   a. Invert colors: inv_cmap = invert_color_map(spec.color_map)
   #                     arr = apply_color_map(g.a, inv_cmap)
   #   b. Invert geometry: arr = geom_apply(arr, geom_inverse(spec.geom))
   #   c. Ensure contiguous: arr = np.ascontiguousarray(arr)
   #   d. Return Grid(arr)

Task-level view operations:
7. apply_view_task(task: dict, spec: ViewSpec) -> dict
   # Apply view to entire task (all train pairs + test inputs)
   # task structure: {"train": [{"input": Grid, "output": Grid}, ...],
   #                  "test": [Grid, ...]}
   # Transform both input AND output for train pairs
   # Transform only input for test pairs
   # Return new task dict with transformed grids

8. invert_view_answer(ans_grid: Grid, spec: ViewSpec) -> Grid
   # Invert predicted output back to original frame
   # Used at test time to map model prediction back
   # Simply calls invert_view_grid

Helper functions for generating color maps:
9. identity_cmap() -> Tuple[int,...]
   # Return (0,1,2,3,4,5,6,7,8,9)

10. generate_palette_permutations(palette: set[int], max_count: int) -> list[Tuple[int,...]]
    # Generate smart color permutations for a given palette
    # Strategy A: Identity + small cycles
    #   - Always include identity
    #   - Add k-cycles on palette colors only
    #   - Example: palette {0,3,7} → try (0→3→7→0) rotation
    # Strategy B: Data-driven assignment (recommended)
    #   - Build co-occurrence matrix C[in_color][out_color] from train pairs
    #   - Use Hungarian algorithm (scipy.optimize.linear_sum_assignment)
    #   - Get top-1 assignment, then add nearby permutations (swap pairs)
    #   - Keep unmapped colors as identity
    # Strategy C: Random jitter
    #   - Add 1-2 random palette permutations (seeded)
    # Budget: aim for 3-8 color permutations total

View generation strategy:
- Geometry set: Use 4-6 transforms ['id','rot90','rot180','rot270','flip_h','flip_v']
- Color permutations: 3-8 smart permutations per task (palette-aware)
- Serialization: Both 'row' and 'col'
- Total views per task: geometry × color_perm × serialization ≈ 40-80

Caching strategy:
- Precompute transformed inputs for each ViewSpec once per task
- Store in RAM with string key: f"{geom}|{''.join(map(str,color_map))}|{serialization}"
- Reuse for PoE (Product of Experts) or DFS search

Property tests to implement (tests/test_views.py):
- test_view_inverse_grid: apply then invert returns original
  # For 100 random (grid, viewspec) pairs
  # assert invert_view_grid(apply_view_grid(g, spec), spec) == g

- test_task_consistency: transformed train pairs still match
  # After transforming, input→output mapping preserved

- test_color_safety: values stay in 0-9 range after transforms
  # assert transformed.a.min() >= 0 and transformed.a.max() <= 9

- test_shape_handling: rotations/transpose swap dimensions correctly
  # rot90 on (H,W) → (W,H)
  # Verify tokenizer handles new shape

- test_bijection: color_map is valid permutation
  # set(cmap) == {0,1,2,3,4,5,6,7,8,9}
  # invert_color_map(invert_color_map(cmap)) == cmap

Safety checks:
- After any transform, assert values in 0..9
- Ensure arrays are contiguous (np.ascontiguousarray)
- Handle shape changes from transpose/rot90 (H×W ↔ W×H)
- Verify inverse correctness with property tests

Common pitfalls:
- Order matters: apply does geom→colors, inverse must do colors⁻¹→geom⁻¹
- Transpose changes shape: (H,W) → (W,H), tokenizer must handle this
- Non-contiguous arrays: always call np.ascontiguousarray after transforms
- Color map must be complete: all 10 colors mapped (bijection)
- Self-inverse ops: flip_h, flip_v, transpose, rot180 are their own inverses
"""

from dataclasses import dataclass
from typing import Tuple

import numpy as np

from arc.grids.core import Grid

D4 = ['id','rot90','rot180','rot270','flip_h','flip_v','transpose','transpose_flip','flip_transpose']

# frozen for hashability (config spec)
@dataclass(frozen=True)
class ViewSpec:
    geom: str
    color_map: Tuple[int,...]
    serialization: str


def geom_apply(a: np.ndarray, name: str) -> np.ndarray:
   """
   apply a geometry operation to a grid.
   """
   if name == 'id':
      return a
   elif name == 'rot90':
      return np.rot90(a, 1)
   elif name == 'rot180':
      return np.rot90(a, 2)
   elif name == 'rot270':
      return np.rot90(a, 3)
   elif name == 'flip_h':
      return np.flip(a, axis=1)
   elif name == 'flip_v':
      return np.flip(a, axis=0)
   elif name == 'transpose':
      return a.T
   elif name == 'transpose_flip':
      return np.flip(a.T, axis=0)
   elif name == 'flip_transpose':
      return np.flip(a, axis=0).T
   else:
      raise ValueError(f"Unknown geometry operation: {name}")

# TODO: double check
def geom_inverse(name: str) -> str:
   """
   get the inverse of a geometry operation.
   """
   inv = {'id':'id', 'rot90':'rot270', 'rot180':'rot180', 'rot270':'rot90', 'flip_h':'flip_h', 'flip_v':'flip_v', 'transpose':'transpose', 'transpose_flip':'flip_transpose', 'flip_transpose':'transpose_flip'}
   return inv[name]

def apply_color_map(a: np.ndarray, cmap: Tuple[int,...]) -> np.ndarray:
   """
   apply a color map to a grid.
   """
   lut = np.array(cmap)
   return lut[a]

def invert_color_map(cmap: Tuple[int,...]) -> Tuple[int,...]:
   """
   invert a color map.
   """
   inv = [0]*10
   for i in range(10):
      inv[cmap[i]] = i
   return tuple(inv)

def apply_view_grid(g: Grid, spec: ViewSpec) -> Grid:
   """
   apply a view spec to a grid.
   """
   arr = geom_apply(g.a, spec.geom)
   arr = apply_color_map(arr, spec.color_map)
   arr = np.ascontiguousarray(arr)
   return Grid(arr)

def invert_view_grid(g: Grid, spec: ViewSpec) -> Grid:
   """
   invert a view spec from a grid.
   """
   inv_cmap = invert_color_map(spec.color_map)
   arr = apply_color_map(g.a, inv_cmap)
   arr = geom_apply(arr, geom_inverse(spec.geom))
   arr = np.ascontiguousarray(arr)
   return Grid(arr)

def apply_view_task(task: dict, spec: ViewSpec) -> dict:
   """
   apply a view spec to a task. returns a dict containing train and test views for the task
   """
   new_task = {"train":[], "test":[]}
   for pair in task["train"]:
      new_task["train"].append({"input":apply_view_grid(pair["input"], spec), "output":apply_view_grid(pair["output"], spec)})
   for grid in task["test"]:
      new_task["test"].append(apply_view_grid(grid, spec))
   return new_task

def invert_view_answer(ans_grid: Grid, spec: ViewSpec) -> Grid:
   """
   invert a view spec to a grid.
   """
   return invert_view_grid(ans_grid, spec)

def identity_cmap() -> Tuple[int,...]:
   """
   return cmap identity
   """
   return tuple(range(10))

def generate_palette_permutations(palette: set[int], max_count: int = 8, seed: int = 42) -> list[Tuple[int,...]]:
   """
   Generate smart color permutations for a given palette.

   Combines three strategies:
   - Strategy A: Identity + small k-cycles on palette colors
   - Strategy B: Random permutations on palette (seeded for reproducibility)
   - Strategy C: Swap pairs within palette

   Args:
       palette: Set of colors actually used in the task (subset of 0-9)
       max_count: Maximum number of permutations to return
       seed: Random seed for reproducibility

   Returns:
       List of color map tuples (length 10), at most max_count items
   """
   import random
   from itertools import permutations as iter_perms

   result = []
   pal_list = sorted(palette)

   # Always include identity
   result.append(identity_cmap())

   # strategy A: generate k-cycles on palette colors
   # for small palettes (≤4 colors), try all permutations
   if len(pal_list) <= 4 and len(pal_list) > 1:
       for perm in iter_perms(pal_list):
           if len(result) >= max_count:
               break
           # build full color map with this permutation on palette
           cmap = list(range(10))
           for i, color in enumerate(pal_list):
               cmap[color] = perm[i]
           result.append(tuple(cmap))

   # strategy B: for larger palettes, generate rotations (k-cycles)
   elif len(pal_list) > 1:
       # rotate palette colors: [a,b,c,d] -> [b,c,d,a], [c,d,a,b], etc.
       for shift in range(1, min(len(pal_list), max_count - len(result))):
           cmap = list(range(10))
           rotated = pal_list[shift:] + pal_list[:shift]
           for i, color in enumerate(pal_list):
               cmap[color] = rotated[i]
           result.append(tuple(cmap))

   # strategy C: random jitter - add 1-2 random permutations (seeded)
   if len(pal_list) > 1 and len(result) < max_count:
       rng = random.Random(seed)
       for _ in range(min(2, max_count - len(result))):
           cmap = list(range(10))
           shuffled = pal_list.copy()
           rng.shuffle(shuffled)
           for i, color in enumerate(pal_list):
               cmap[color] = shuffled[i]
           # avoid duplicates
           cmap_tuple = tuple(cmap)
           if cmap_tuple not in result:
               result.append(cmap_tuple)

   return result[:max_count]

# TODO: integrate with steps 6 and 7
def generate_data_driven_permutations(train_pairs: list[dict], palette: set[int], max_count: int = 5) -> list[Tuple[int,...]]:
   """
   generate color permutations based on input->output color co-occurrence in training pairs.

   uses Hungarian algorithm (linear_sum_assignment) to find optimal color mappings
   based on how colors align between inputs and outputs.

   Args:
       train_pairs: List of {"input": Grid, "output": Grid} dicts
       palette: Set of colors used in the task
       max_count: Maximum permutations to generate

   Returns:
       List of color map tuples based on training data patterns
   """
   try:
       from scipy.optimize import linear_sum_assignment
   except ImportError:
       # Fallback if scipy not available
       return [identity_cmap()]

   # Build co-occurrence matrix C[in_color][out_color]
   # Count how often input color i appears at same position as output color j
   C = np.zeros((10, 10), dtype=int)

   for pair in train_pairs:
       inp = pair["input"].a
       out = pair["output"].a

       # Only count if shapes match
       if inp.shape == out.shape:
           for in_color in range(10):
               for out_color in range(10):
                   # Count positions where input has in_color and output has out_color
                   C[in_color, out_color] += np.sum((inp == in_color) & (out == out_color))

   result = []
   pal_list = sorted(palette)

   # Always include identity
   result.append(identity_cmap())

   if len(pal_list) <= 1:
       return result

   # Extract submatrix for palette colors only
   C_pal = np.zeros((len(pal_list), len(pal_list)), dtype=int)
   for i, in_c in enumerate(pal_list):
       for j, out_c in enumerate(pal_list):
           C_pal[i, j] = C[in_c, out_c]

   # Solve max-weight assignment (negate for min-cost algorithm)
   row_ind, col_ind = linear_sum_assignment(-C_pal)

   # Build color map from assignment
   cmap = list(range(10))
   for i, j in zip(row_ind, col_ind):
       cmap[pal_list[i]] = pal_list[j]

   cmap_tuple = tuple(cmap)
   if cmap_tuple != identity_cmap():
       result.append(cmap_tuple)

   # Generate nearby permutations by swapping pairs
   if len(result) < max_count and len(pal_list) >= 2:
       for i in range(len(pal_list) - 1):
           if len(result) >= max_count:
               break
           # Swap two adjacent colors in the assignment
           cmap_swap = list(cmap)
           c1, c2 = pal_list[i], pal_list[i + 1]
           cmap_swap[c1], cmap_swap[c2] = cmap_swap[c2], cmap_swap[c1]
           swap_tuple = tuple(cmap_swap)
           if swap_tuple not in result:
               result.append(swap_tuple)

   return result[:max_count]
