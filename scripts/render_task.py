"""
Step 7: CLI Script for Rendering Tasks

This script renders ARC tasks as PNG images for visualization.

Usage:
  python scripts/render_task.py --task-id abc123 --out-dir out/vis/
  python scripts/render_task.py --task-id abc123 --split training --out-dir out/vis/
  python scripts/render_task.py --random 5 --out-dir out/vis/  # Render 5 random tasks

Arguments:
  --task-id: Task ID to render (e.g., 'abc123')
  --split: Dataset split ('training' or 'evaluation', default: 'training')
  --out-dir: Output directory for PNG files (default: 'out/vis/')
  --random: Number of random tasks to render (instead of specific task-id)
  --cell-size: Size of each cell in pixels (default: 20)

Output files:
  out/vis/
    task_abc123_full.png       # Full task visualization
    task_abc123_train_0.png    # First train pair
    task_abc123_train_1.png    # Second train pair
    task_abc123_test.png       # Test input

Implementation steps:

1. Parse command-line arguments
   # Use argparse
   # Define arguments: task-id, split, out-dir, random, cell-size

2. Load task(s)
   # If --task-id provided:
   #   - Construct path: f'data/raw/arc/{split}/{task_id}.json'
   #   - Load using arc.io.loader.load_task()
   # If --random provided:
   #   - Use arc.io.loader.iter_tasks() to get all tasks
   #   - Sample N random tasks

3. Create output directory
   # Use pathlib.Path
   # Create directory if not exists: out_dir.mkdir(parents=True, exist_ok=True)

4. Render task
   # For each task:
   #   - Render full task: arc.viz.plot.save_task_png()
   #   - Render individual train pairs: arc.viz.plot.save_pair_png()
   #   - Render test input: arc.viz.plot.save_grid_png()

5. Print confirmation
   # Print: "Rendered task abc123 to out/vis/"
   # List generated files

Example usage:
  # Render specific task
  python scripts/render_task.py --task-id 007bbfb7 --split training

  # Render 3 random tasks
  python scripts/render_task.py --random 3 --split evaluation

  # Custom output directory and cell size
  python scripts/render_task.py --task-id 007bbfb7 --out-dir images/ --cell-size 30
"""

# TODO: Import argparse
# TODO: Import pathlib (Path)
# TODO: Import random
# TODO: Import arc.io.loader (load_task, iter_tasks)
# TODO: Import arc.viz.plot (save_task_png, save_pair_png, save_grid_png)
# TODO: Import arc.grids.core (Grid, from_list)

# TODO: Implement parse_args() -> argparse.Namespace
#   - Create ArgumentParser
#   - Add arguments: task-id, split, out-dir, random, cell-size
#   - Parse and return args

# TODO: Implement main()
#   - Parse arguments
#   - If --task-id:
#     - Construct path
#     - Load task
#     - Render task
#   - If --random:
#     - Load all tasks from split
#     - Sample N random tasks
#     - Render each task
#   - Print confirmation

# TODO: Add if __name__ == "__main__": main()
