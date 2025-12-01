#!/usr/bin/env python3
"""
Evaluation script for ARC-AGI baseline model.
Runs solve() on dev tasks and computes exact-match accuracy.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from arc.eval.solve_task import solve as solve
from arc.grids.core import Grid
from arc.io.loader import load_task


def grid_equal(pred_grid: Grid, gt_grid: Grid) -> bool:
    """Check if two grids are exactly equal."""
    if pred_grid.shape != gt_grid.shape:
        return False
    return (pred_grid.a == gt_grid.a).all()


def load_dev_tasks():
    """Load dev task IDs from processed data."""
    dev_tasks_file = project_root / "data" / "processed" / "dev_tasks.json"
    if not dev_tasks_file.exists():
        print(f"Warning: {dev_tasks_file} not found, using empty list")
        return []

    with open(dev_tasks_file) as f:
        return json.load(f)


def eval_model(ckpt_path: str, max_tasks: int = None, use_ttt: bool = True):
    """
    Evaluate model on dev tasks.

    Args:
        ckpt_path: Path to model checkpoint
        max_tasks: Maximum number of tasks to evaluate (None = all)
        use_ttt: Whether to use test-time training (default: True)
    """
    mode_str = "with TTT" if use_ttt else "WITHOUT TTT (fast mode)"
    dev_tasks = load_dev_tasks()

    if not dev_tasks:
        print("No dev tasks found. Please check data/processed/dev_tasks.json")
        return

    if max_tasks:
        dev_tasks = dev_tasks[:max_tasks]

    print(f"Evaluating on {len(dev_tasks)} dev tasks {mode_str}")
    print(f"Checkpoint: {ckpt_path}")
    print("=" * 60)

    total = 0
    correct = 0
    task_results = []

    for i, task_id in enumerate(dev_tasks, 1):
        print(f"\n[{i}/{len(dev_tasks)}] Processing task: {task_id}")

        try:
            # Load ground truth
            task_path = project_root / "data" / "raw" / "arc" / "evaluation" / f"{task_id}.json"
            if not task_path.exists():
                task_path = project_root / "data" / "raw" / "arc" / "training" / f"{task_id}.json"

            if not task_path.exists():
                print(f"  ✗ Task file not found: {task_id}")
                continue

            task = load_task(str(task_path))

            # Run solver
            preds = solve(task_id, ckpt_path)

            # use test outputs if available, otherwise skip
            task_correct = 0
            task_total = 0

            for pred_idx, (pred_grid, test_pair) in enumerate(zip(preds, task["test"])):
                if "output" in test_pair:
                    gt_grid = Grid(np.array(test_pair["output"], dtype=np.int8))
                    task_total += 1

                    if grid_equal(pred_grid, gt_grid):
                        task_correct += 1
                        correct += 1
                        print(f"  ✓ Test {pred_idx + 1}: CORRECT")
                    else:
                        print(f"  ✗ Test {pred_idx + 1}: INCORRECT")
                        print(f"    Predicted shape: {pred_grid.shape}, GT shape: {gt_grid.shape}")
                else:
                    print(f"  - Test {pred_idx + 1}: No ground truth available")

            total += task_total

            task_acc = task_correct / task_total if task_total > 0 else 0.0
            task_results.append(
                {"task_id": task_id, "correct": task_correct, "total": task_total, "accuracy": task_acc}
            )

            print(f"  Task accuracy: {task_correct}/{task_total} = {task_acc:.1%}")

        except Exception as e:
            print(f"  ✗ Error processing task {task_id}: {e}")
            import traceback

            traceback.print_exc()

    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)

    acc = correct / total if total > 0 else 0.0
    print(f"Overall accuracy: {correct}/{total} = {acc:.1%}")
    print(f"Tasks evaluated: {len(task_results)}")

    # Save results
    results_dir = project_root / "outputs" / "eval_results"
    results_dir.mkdir(parents=True, exist_ok=True)

    results_file = results_dir / "baseline_eval.json"
    with open(results_file, "w") as f:
        json.dump(
            {
                "checkpoint": ckpt_path,
                "overall_accuracy": acc,
                "correct": correct,
                "total": total,
                "tasks_evaluated": len(task_results),
                "task_results": task_results,
            },
            f,
            indent=2,
        )

    print(f"\nResults saved to: {results_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate ARC-AGI baseline model")
    parser.add_argument("--ckpt", required=True, help="Path to model checkpoint")
    parser.add_argument(
        "--max-tasks", type=int, default=None, help="Maximum number of tasks to evaluate (default: all)"
    )
    parser.add_argument(
        "--no-ttt", action="store_true", help="Skip test-time training for faster evaluation"
    )

    args = parser.parse_args()

    eval_model(args.ckpt, args.max_tasks, use_ttt=not args.no_ttt)
