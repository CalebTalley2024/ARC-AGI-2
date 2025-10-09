import json
import glob #pattern matching
from pathlib import Path #pathlib is a library for handling file paths #Path is a class in pathlib
from arc.utils.seeding import set_seed

'''
- load_task(path) -> dict (train pairs + test pairs)
- iter_tasks(split) -> iterator
- **Validation checks:** grid fits ≤ 30×30, colors are ints, shapes consistent.
'''

def load_task(path):
    """
    Load a single ARC task JSON file.
    """
    with open(path, 'r') as f:
        return json.load(f)



#iterates over all tasks in the split path (training or evaluation)
#split is either {training, evaluation}
def iter_tasks(split):
    """
    Iterate over ARC tasks for a given split.

    Args:
        split (str): Dataset split, either 'training' or 'evaluation'.

    Yields:
        tuple: (task_dict, task_path) where task_dict is the parsed task data
               and task_path is the file path to the task.
    """
    if split not in ["training", "evaluation"]:
        raise ValueError(f"Invalid split: {split}. Expected 'training' or 'evaluation'.")
    
    #glob.glob: returns a list of paths that match the pattern
    #iterates over each path in the list
    for path in glob.glob(f'data/raw/arc/{split}/*.json'): 
        #yield: pauses function,resumes later when the generator's __next__() is called
        yield load_task(path), path



def is_grid_valid(grid):
    """

    checks if train/test input/output grid is validd
    Returns True if a grid passes validation checks, else False.

    Validation checks:
    - grid fits 03 30 30
    - all colors (i.e. values in grid) are ints
    - shapes consistent (all rows have the same length)
    """
    # Must be a non-empty list of non-empty lists
    if not isinstance(grid, list) or len(grid) == 0:
        return False
    if not all(isinstance(row, list) and len(row) > 0 for row in grid):
        return False

    n_rows, n_cols = len(grid), len(grid[0])

    # Consistent row lengths
    if not all(len(row) == n_cols for row in grid):
        return False

    # Max dimensions
    if n_rows > 30 or n_cols > 30:
        return False

    # All entries must be ints in range [0, 9]
    for row in grid:
        for value in row:
            if not isinstance(value, int) or not (0 <= value <= 9):
                return False

    return True


def _grid_shape(grid):
    """
    Returns the shape of a grid as a tuple of (rows, columns).
    """
    return (len(grid), len(grid[0]))    


def cache_index():
    """
    Build a light index of ARC tasks and write to data/processed/index.jsonl.

    Each line is a JSON object with:
    - task_id: filename stem
    - n_train_pairs: number of train pairs
    - n_test_pairs: number of test pairs    
    - train_input_shapes: list of (rows, cols) for each train input
    - train_output_shapes: list of (rows, cols) for each train output
    - test_input_shapes: list of (rows, cols) for each t    est input
    - test_output_shapes: list of (rows, cols) for each test output (may be None)
    """
    processed_dir = Path('data/processed')
    processed_dir.mkdir(parents=True, exist_ok=True)
    index_path = processed_dir / 'index.jsonl'

    with index_path.open('w') as f:
        for split in ('training', 'evaluation'):
            for task, task_path in iter_tasks(split):
                task_id = Path(task_path).stem #filename without extension
                # get train and test pairs from task
                train_pairs = task.get('train', []) # if no train, default to empty list
                test_pairs = task.get('test', [])

                # Validate entire task first - raise error if any pair is invalid
                # Check all train pairs
                for pair in train_pairs:
                    if not is_grid_valid(pair['input']) or not is_grid_valid(pair['output']):
                        raise ValueError(f"Task {task_id}: invalid train pair")
                
                # Check all test pairs
                for pair in test_pairs:
                    if not is_grid_valid(pair['input']):
                        raise ValueError(f"Task {task_id}: invalid test input") #TODO is this how we want to handle invalid grids?
                    if 'output' in pair and pair['output'] and not is_grid_valid(pair['output']):
                        raise ValueError(f"Task {task_id}: invalid test output")

                n_train_pairs = len(train_pairs)
                n_test_pairs = len(test_pairs)

                # Collect all shapes for train pairs (all validated)
                train_input_shapes = []
                train_output_shapes = []
                for pair in train_pairs:
                    train_input_shapes.append(_grid_shape(pair['input']))
                    train_output_shapes.append(_grid_shape(pair['output']))

                # Collect all shapes for test pairs (all validated)
                test_input_shapes = []
                test_output_shapes = []
                for pair in test_pairs:
                    test_input_shapes.append(_grid_shape(pair['input']))
                    if 'output' in pair and pair['output']:
                        test_output_shapes.append(_grid_shape(pair['output']))
                    else:
                        test_output_shapes.append(None)

                f.write(json.dumps({
                    'task_id': task_id,
                    'split': split, # training or evaluation
                    'n_train_pairs': n_train_pairs,
                    'n_test_pairs': n_test_pairs,
                    'train_input_shapes': train_input_shapes,
                    'train_output_shapes': train_output_shapes,
                    'test_input_shapes': test_input_shapes,
                    'test_output_shapes': test_output_shapes,
                }) + '\n')


#TODO check this
def create_split_manifests(dev_ratio=0.2, seed=42):
    """

    Create development/validation and public-evaluation split manifest files.
    
    Reads 'evaluation'task IDs from processed/index.jsonl, shuffles them, and splits into dev/eval.
    This prevents accidental data leakage during prototyping. (i.e training data being mixed with evaluation data)
    
    Args:
        dev_ratio (float): Fraction of tasks to use for development (default: 0.2)
        seed (int): Random seed for reproducible splits (default: 42)
    """
    
    set_seed(seed)
    
    # Read task IDs from index (only evaluation tasks)
    index_path = Path('data/processed/index.jsonl')
    task_ids = []
    
    with index_path.open('r') as f:
        for line in f:
            entry = json.loads(line)
            if entry['split'] == 'evaluation':
                task_ids.append(entry['task_id'])
    
    # Shuffle and split
    random.shuffle(task_ids)
    dev_count = int(len(task_ids) * dev_ratio)
    dev_tasks = task_ids[:dev_count]
    eval_tasks = task_ids[dev_count:]
    
    # Write manifest files
    processed_dir = Path('data/processed')
    dev_manifest_path = processed_dir / 'dev_tasks.json'
    eval_manifest_path = processed_dir / 'eval_tasks.json'
    
    with dev_manifest_path.open('w') as f:
        json.dump(dev_tasks, f, indent=2)
    
    with eval_manifest_path.open('w') as f:
        json.dump(eval_tasks, f, indent=2)
    
    print(f"Created dev manifest with {len(dev_tasks)} tasks: {dev_manifest_path}")
    print(f"Created eval manifest with {len(eval_tasks)} tasks: {eval_manifest_path}")


if __name__ == "__main__":
    # Generate index and split manifests
    cache_index()
    create_split_manifests()
    