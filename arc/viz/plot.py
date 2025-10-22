"""
Step 7: Visualization & Tiny CLIs

This module provides visualization utilities for ARC grids and tasks.

Goal: Create clear, color-coded visualizations of grids for debugging and analysis.

Key concepts:
- ARC color palette: 10 distinct colors (0-9)
- Grid rendering: Each cell as a colored square
- Pair visualization: Input and output side-by-side
- Diff visualization: Highlight mismatched cells

ARC official color palette (0-9):
  0: Black (#000000)
  1: Blue (#0074D9)
  2: Red (#FF4136)
  3: Green (#2ECC40)
  4: Yellow (#FFDC00)
  5: Grey (#AAAAAA)
  6: Magenta (#F012BE)
  7: Orange (#FF851B)
  8: Light Blue (#7FDBFF)
  9: Brown (#870C25)

Functions to implement:

1. get_color_palette() -> dict
   # Return mapping from color id to RGB hex code
   # Returns: {0: '#000000', 1: '#0074D9', ...}
   # Use official ARC colors for consistency

2. save_grid_png(grid: Grid, path: str, cell_size: int = 20) -> None
   # Render single grid to PNG file
   # Args:
   #   grid: Grid object to visualize
   #   path: Output file path (e.g., 'output/grid.png')
   #   cell_size: Size of each cell in pixels
   # Steps:
   #   a. Create image with size (W*cell_size, H*cell_size)
   #   b. For each cell, draw colored rectangle
   #   c. Add grid lines (optional, thin black lines)
   #   d. Save to path using PIL/matplotlib
   # Libraries: Use matplotlib.pyplot or PIL (Pillow)

3. save_pair_png(in_grid: Grid, out_grid: Grid, path: str, cell_size: int = 20) -> None
   # Render input-output pair side-by-side
   # Args:
   #   in_grid: Input grid
   #   out_grid: Output grid
   #   path: Output file path
   #   cell_size: Size of each cell in pixels
   # Steps:
   #   a. Create figure with 1 row, 2 columns
   #   b. Plot in_grid on left with title "Input"
   #   c. Plot out_grid on right with title "Output"
   #   d. Add arrow or label between them
   #   e. Save to path

4. save_task_png(task: dict, path: str, cell_size: int = 20) -> None
   # Render entire task (all train pairs + test input)
   # Args:
   #   task: Task dict with 'train' and 'test' keys
   #   path: Output file path
   #   cell_size: Size of each cell
   # Layout:
   #   Row 1: Train pair 1 (input | output)
   #   Row 2: Train pair 2 (input | output)
   #   ...
   #   Row N+1: Test input | ?
   # Steps:
   #   a. Calculate grid layout (rows = n_train + 1, cols = 2)
   #   b. Plot each train pair
   #   c. Plot test input with placeholder for output
   #   d. Add labels ("Train 1", "Train 2", "Test")
   #   e. Save to path

5. save_diff_png(pred_grid: Grid, true_grid: Grid, path: str, cell_size: int = 20) -> None
   # Visualize differences between predicted and true grids
   # Args:
   #   pred_grid: Predicted output
   #   true_grid: Ground truth output
   #   path: Output file path
   #   cell_size: Size of each cell
   # Steps:
   #   a. Compute diff mask: diff(pred_grid, true_grid)
   #   b. Create 3-panel figure:
   #      - Left: predicted grid
   #      - Middle: true grid
   #      - Right: diff overlay (red for mismatches)
   #   c. Add accuracy metric: f"{correct}/{total} cells correct"
   #   d. Save to path

6. plot_grid_inline(grid: Grid, ax=None, cell_size: int = 20) -> None
   # Plot grid to matplotlib axis (for notebooks)
   # Args:
   #   grid: Grid to plot
   #   ax: Matplotlib axis (create if None)
   #   cell_size: Size of each cell
   # Steps:
   #   a. Create or use provided axis
   #   b. Use imshow with custom colormap
   #   c. Add grid lines
   #   d. Remove axis ticks
   #   e. Return axis for further customization

7. create_color_legend(path: str) -> None
   # Create legend showing color id to color mapping
   # Args:
   #   path: Output file path
   # Steps:
   #   a. Create 1×10 or 2×5 grid showing each color
   #   b. Label each cell with color id (0-9)
   #   c. Save to path
   # Useful for documentation and presentations

Helper functions:

8. grid_to_rgb_array(grid: Grid) -> np.ndarray
   # Convert grid to RGB array for visualization
   # Args:
   #   grid: Grid object
   # Returns: np.ndarray of shape (H, W, 3) with RGB values
   # Steps:
   #   a. Get color palette
   #   b. Map each cell to RGB tuple
   #   c. Return as numpy array

9. add_grid_lines(ax, grid: Grid, color: str = 'black', linewidth: float = 0.5) -> None
   # Add grid lines to matplotlib axis
   # Args:
   #   ax: Matplotlib axis
   #   grid: Grid object (for dimensions)
   #   color: Line color
   #   linewidth: Line width
   # Steps:
   #   a. Draw horizontal lines between rows
   #   b. Draw vertical lines between columns

CLI script to implement (scripts/render_task.py):

Usage: python scripts/render_task.py --task-id 123 --out-dir out/vis/

Script structure:
  # Parse arguments: task_id, out_dir, split (train/eval)
  # Load task using arc.io.loader.load_task()
  # Create output directory if not exists
  # Render task using save_task_png()
  # Print confirmation message

Example output structure:
  out/vis/
    task_123_full.png       # Full task visualization
    task_123_train_0.png    # First train pair
    task_123_train_1.png    # Second train pair
    task_123_test.png       # Test input

Deliverables:
- arc/viz/plot.py with all visualization functions
- scripts/render_task.py CLI script
- PNGs for 3 random tasks in out/vis/ (git-ignored if large)
- Color legend PNG for documentation

Tests to write (tests/test_viz.py):
- test_save_grid_png: Create PNG, verify file exists and has correct size
- test_save_pair_png: Create pair PNG, verify layout
- test_save_diff_png: Create diff PNG, verify highlights
- test_color_palette: Verify all 10 colors defined
- test_grid_to_rgb: Verify RGB array shape and values

Common pitfalls:
- Color palette mismatch with official ARC colors
- Image dimensions don't match grid dimensions
- Grid lines obscure cell colors (use thin lines)
- Large grids create huge images (limit cell_size)
- Forgetting to create output directories
- Not handling grids of different sizes in same task

Optimization tips:
- Cache color palette as constant
- Use vectorized operations for RGB conversion
- Limit image resolution for large grids (max 1000×1000 pixels)
- Use PNG compression for smaller files
- Consider SVG for vector graphics (scalable)

Best practices:
- Always show input and output together for context
- Use consistent color palette across all visualizations
- Add titles and labels for clarity
- Include grid dimensions in titles
- Save at reasonable resolution (not too large)
- Use descriptive filenames with task_id
"""

# TODO: Import numpy as np
# TODO: Import matplotlib.pyplot as plt
# TODO: Import PIL (Image, ImageDraw)
# TODO: Import pathlib (Path)
# TODO: Import arc.grids.core (Grid, diff)

# TODO: Define ARC_COLORS constant
# ARC_COLORS = {
#     0: '#000000',  # Black
#     1: '#0074D9',  # Blue
#     2: '#FF4136',  # Red
#     3: '#2ECC40',  # Green
#     4: '#FFDC00',  # Yellow
#     5: '#AAAAAA',  # Grey
#     6: '#F012BE',  # Magenta
#     7: '#FF851B',  # Orange
#     8: '#7FDBFF',  # Light Blue
#     9: '#870C25',  # Brown
# }

# TODO: Implement get_color_palette() -> dict
#   - Return ARC_COLORS constant

# TODO: Implement grid_to_rgb_array(grid: Grid) -> np.ndarray
#   - Get color palette
#   - Create (H, W, 3) array
#   - Map each cell to RGB values
#   - Return array

# TODO: Implement save_grid_png(grid: Grid, path: str, cell_size: int = 20) -> None
#   - Convert grid to RGB array
#   - Create image with PIL or matplotlib
#   - Draw colored cells
#   - Add grid lines (optional)
#   - Save to path
#   - Create parent directories if needed

# TODO: Implement save_pair_png(in_grid: Grid, out_grid: Grid, path: str, cell_size: int = 20) -> None
#   - Create figure with 2 subplots (1 row, 2 cols)
#   - Plot input on left
#   - Plot output on right
#   - Add titles "Input" and "Output"
#   - Add arrow or separator
#   - Save to path

# TODO: Implement save_task_png(task: dict, path: str, cell_size: int = 20) -> None
#   - Count train pairs
#   - Create figure with (n_train + 1) rows, 2 cols
#   - Plot each train pair in a row
#   - Plot test input in last row
#   - Add labels for each row
#   - Save to path

# TODO: Implement save_diff_png(pred_grid: Grid, true_grid: Grid, path: str, cell_size: int = 20) -> None
#   - Compute diff mask using diff()
#   - Create 3-panel figure
#   - Plot predicted, true, and diff overlay
#   - Highlight mismatches in red
#   - Add accuracy text
#   - Save to path

# TODO: Implement plot_grid_inline(grid: Grid, ax=None, cell_size: int = 20) -> axis
#   - Create or use provided axis
#   - Convert grid to RGB array
#   - Use ax.imshow()
#   - Add grid lines
#   - Remove ticks
#   - Return axis

# TODO: Implement create_color_legend(path: str) -> None
#   - Create figure with 1×10 or 2×5 layout
#   - Show each color with its id
#   - Add labels
#   - Save to path

# TODO: Implement add_grid_lines(ax, grid: Grid, color: str = 'black', linewidth: float = 0.5) -> None
#   - Get H, W from grid.shape
#   - Draw horizontal lines at y = 0.5, 1.5, ..., H-0.5
#   - Draw vertical lines at x = 0.5, 1.5, ..., W-0.5
#   - Use ax.hlines() and ax.vlines()
