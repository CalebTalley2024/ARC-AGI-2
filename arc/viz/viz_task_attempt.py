import numpy as np
import matplotlib.pyplot as plt

# ARC color palette
COLOR_MAP = [
    '#000000',  # 0: black
    '#0074D9',  # 1: blue
    '#FF4136',  # 2: red
    '#2ECC40',  # 3: green
    '#FFDC00',  # 4: yellow
    '#AAAAAA',  # 5: grey
    '#F012BE',  # 6: magenta
    '#FF851B',  # 7: orange
    '#7FDBFF',  # 8: sky blue
    '#870C25'   # 9: brown
]

def plot_grid(grid, title="", ax=None):
    """Plot a single ARC grid with proper colors"""
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    
    # Convert Grid to numpy array
    if hasattr(grid, 'a'):
        grid_array = grid.a
    else:
        grid_array = np.array(grid)
    
    height, width = grid_array.shape
    
    # Create color mapping
    colored_grid = np.zeros((height, width, 3))
    for i in range(height):
        for j in range(width):
            color_hex = COLOR_MAP[grid_array[i, j]]
            # Convert hex to RGB
            r = int(color_hex[1:3], 16) / 255.0
            g = int(color_hex[3:5], 16) / 255.0
            b = int(color_hex[5:7], 16) / 255.0
            colored_grid[i, j] = [r, g, b]
    
    ax.imshow(colored_grid, interpolation='nearest')
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Add grid lines
    for i in range(height + 1):
        ax.axhline(i - 0.5, color='white', linewidth=1)
    for j in range(width + 1):
        ax.axvline(j - 0.5, color='white', linewidth=1)
    
    return ax

def visualize_results(x, y, g_pred, score):
    """Visualize input, expected output, and predicted output side by side."""
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    plot_grid(x, "Input (x)", axes[0])
    plot_grid(y, "Expected Output (y)", axes[1])
    plot_grid(g_pred, f"Predicted Output (score: {score:.4f})", axes[2])
    
    plt.suptitle("ARC Task: Input, Expected, and Predicted", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()
