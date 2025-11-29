import pytest
import json
import random  # TODO move seed to arc/util/seeding.py
from pathlib import Path
import sys
import os

# Add the project root to Python path (goes one lvl above tests folder)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from arc.io.loader import load_task, iter_tasks, is_grid_valid


def test_load_task_basic():
    """Test loading a single task and basic structure validation."""
    # Find any training task file
    training_files = list(Path("data/raw/arc/training").glob("*.json"))
    assert len(training_files) > 0, "No training files found"

    task_path = training_files[0]
    task = load_task(task_path)

    # Check basic structure
    assert isinstance(task, dict), "Task should be a dictionary"
    assert "train" in task, "Task should have 'train' key"
    assert "test" in task, "Task should have 'test' key"
    assert isinstance(task["train"], list), "Train should be a list"
    assert isinstance(task["test"], list), "Test should be a list"


def test_load_5_random_tasks():
    """Test loading 5 random tasks and validate counts, shapes, color ranges."""
    # Get all training files
    training_files = list(Path("data/raw/arc/training").glob("*.json"))
    assert len(training_files) >= 5, (
        f"Need at least 5 training files, found {len(training_files)}"
    )

    # Randomly select 5 tasks
    random.seed(42)  # For reproducible tests
    selected_file_paths = random.sample(training_files, 5)

    for i, task_path in enumerate(selected_file_paths):
        print(f"Testing task {i + 1}: {task_path.name}")
        task = load_task(task_path)

        # Test train pairs
        train_pairs = task["train"]
        assert len(train_pairs) > 0, (
            f"Task {task_path.name} should have at least 1 train pair"
        )

        for j, pair in enumerate(train_pairs):
            # Check structure
            assert "input" in pair, f"Train pair {j} missing 'input'"
            assert "output" in pair, f"Train pair {j} missing 'output'"

            # Validate grids
            assert is_grid_valid(pair["input"]), f"Train pair {j} input grid invalid"
            assert is_grid_valid(pair["output"]), f"Train pair {j} output grid invalid"

            # Check shapes
            input_shape = (len(pair["input"]), len(pair["input"][0]))
            output_shape = (len(pair["output"]), len(pair["output"][0]))

            assert input_shape[0] <= 30 and input_shape[1] <= 30, (
                f"Train input shape {input_shape} exceeds 30x30"
            )
            assert output_shape[0] <= 30 and output_shape[1] <= 30, (
                f"Train output shape {output_shape} exceeds 30x30"
            )

            # Check color ranges
            for row in pair["input"]:
                for color in row:
                    assert isinstance(color, int), f"Input color {color} is not int"
                    assert 0 <= color <= 9, f"Input color {color} not in range [0,9]"

            for row in pair["output"]:
                for color in row:
                    assert isinstance(color, int), f"Output color {color} is not int"
                    assert 0 <= color <= 9, f"Output color {color} not in range [0,9]"

        # Test test pairs
        test_pairs = task["test"]
        assert len(test_pairs) > 0, (
            f"Task {task_path.name} should have at least 1 test pair"
        )

        for j, pair in enumerate(test_pairs):
            # Check structure
            assert "input" in pair, f"Test pair {j} missing 'input'"

            # Validate input grid
            assert is_grid_valid(pair["input"]), f"Test pair {j} input grid invalid"

            # Check input shape
            input_shape = (len(pair["input"]), len(pair["input"][0]))
            assert input_shape[0] <= 30 and input_shape[1] <= 30, (
                f"Test input shape {input_shape} exceeds 30x30"
            )

            # Check input color ranges
            for row in pair["input"]:
                for color in row:
                    assert isinstance(color, int), (
                        f"Test input color {color} is not int"
                    )
                    assert 0 <= color <= 9, (
                        f"Test input color {color} not in range [0,9]"
                    )

            # Test output may or may not exist
            if "output" in pair and pair["output"]:
                assert is_grid_valid(pair["output"]), (
                    f"Test pair {j} output grid invalid"
                )
                output_shape = (len(pair["output"]), len(pair["output"][0]))
                assert output_shape[0] <= 30 and output_shape[1] <= 30, (
                    f"Test output shape {output_shape} exceeds 30x30"
                )

                for row in pair["output"]:
                    for color in row:
                        assert isinstance(color, int), (
                            f"Test output color {color} is not int"
                        )
                        assert 0 <= color <= 9, (
                            f"Test output color {color} not in range [0,9]"
                        )


def test_iter_tasks_training():
    """Test iterating over training tasks."""
    task_count = 0
    for task, task_path in iter_tasks("training"):
        assert isinstance(task, dict), "Task should be a dictionary"
        assert isinstance(task_path, str), "Task path should be a string"
        assert "train" in task, "Task should have 'train' key"
        assert "test" in task, "Task should have 'test' key"
        task_count += 1
        if task_count >= 5:  # Test first 5 tasks
            break

    assert task_count > 0, "Should find at least one training task"


def test_iter_tasks_evaluation():
    """Test iterating over evaluation tasks."""
    task_count = 0
    for task, task_path in iter_tasks("evaluation"):
        assert isinstance(task, dict), "Task should be a dictionary"
        assert isinstance(task_path, str), "Task path should be a string"
        assert "train" in task, "Task should have 'train' key"
        assert "test" in task, "Task should have 'test' key"
        task_count += 1
        if task_count >= 5:  # Test first 5 tasks
            break

    assert task_count > 0, "Should find at least one evaluation task"


def test_iter_tasks_invalid_split():
    """Test that iter_tasks raises error for invalid split."""
    with pytest.raises(ValueError, match="Invalid split"):
        list(iter_tasks("invalid_split"))


def test_is_grid_valid():
    """Test the is_grid_valid function with various cases."""
    # Valid grid
    valid_grid = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    assert is_grid_valid(valid_grid), "Valid grid should pass validation"

    # Empty grid
    empty_grid = []
    assert not is_grid_valid(empty_grid), "Empty grid should fail validation"

    # Grid with inconsistent row lengths
    inconsistent_grid = [[0, 1], [2, 3, 4]]
    assert not is_grid_valid(inconsistent_grid), (
        "Inconsistent grid should fail validation"
    )

    # Grid too large
    large_grid = [[0] * 31 for _ in range(31)]
    assert not is_grid_valid(large_grid), "Large grid should fail validation"

    # Grid with invalid colors (non-integers)
    invalid_color_grid = [[0, 1.5, 2], [3, 4, 5]]
    assert not is_grid_valid(invalid_color_grid), (
        "Grid with non-integer colors should fail validation"
    )

    # Grid with colors out of range
    out_of_range_grid = [[0, 1, 10], [3, 4, 5]]
    assert not is_grid_valid(out_of_range_grid), (
        "Grid with colors out of range should fail validation"
    )


if __name__ == "__main__":
    # Run tests
    test_load_task_basic()
    test_load_5_random_tasks()
    test_iter_tasks_training()
    test_iter_tasks_evaluation()
    test_iter_tasks_invalid_split()
    test_is_grid_valid()
    print("All tests passed!")
