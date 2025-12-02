"""
ARC-AGI Model Evaluation Script
================================

Comprehensive evaluation of TinyLM models on the full ARC-AGI dataset.

Features:
- Multiple evaluation strategies: Baseline, TTT, PoE, TTT+PoE
- Separate results for training and evaluation datasets
- Detailed success/failure tracking with reasons
- JSON output with structured results
- Progress tracking and time estimation

Usage:
    python evaluate_arc_model.py --help
"""

import sys
from pathlib import Path
import torch
import numpy as np
import time
import json
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import copy
import argparse

# Setup paths
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from arc.grids.core import from_list
from arc.io.loader import iter_tasks
from arc.models.tiny_lm import TinyLM, TinyLMConfig
from arc.serialize import pack_example, deserialize_grid, BOS, SEP, EOS
from arc.eval.scorer import mean_logp_output
from arc.models.infer import greedy_generate
from arc.models.ttt import TestTimeTrainer
from arc.eval.poe import poe_sum


# ============================================================================
# CORE EVALUATION FUNCTIONS
# ============================================================================


def generate_and_score_with_mode(model, x_grid, y_grid, device="cpu", mode: str = "row", max_new: int = 512):
    """Generate and score with a specific serialization mode."""
    try:
        seq = pack_example(x_grid, y_grid, mode=mode)
        inp = torch.tensor(seq[:-1], dtype=torch.long).unsqueeze(0).to(device)

        out = greedy_generate(model, inp, max_new_tokens=max_new, eos_id=EOS)
        score = mean_logp_output(model, out.clone(), sep_token_id=SEP)

        ids = out.squeeze(0).tolist()

        first_eos_idx = ids.index(EOS)

        sep_after_eos = None
        for i in range(first_eos_idx + 1, len(ids)):
            if ids[i] == SEP:
                sep_after_eos = i
                break

        if sep_after_eos is None:
            raise ValueError("No SEP token found after first EOS")

        output_tokens = [BOS] + ids[sep_after_eos + 1 :]
        g_pred = deserialize_grid(output_tokens, mode=mode)

        return g_pred, float(score), None
    except Exception as e:
        return None, None, f"{type(e).__name__}: {str(e)}"


def evaluate_baseline(model, x, y, device="cpu"):
    """Baseline: Simple greedy generation with row mode."""
    g_pred, score, error = generate_and_score_with_mode(model, x, y, device=device, mode="row", max_new=512)
    return g_pred, score, error


def evaluate_poe(model, x, y, device="cpu"):
    """PoE: Product of Experts with multiple views."""
    modes = ["row", "col", "rc_pair"]
    predictions = []
    scores = []
    errors = []

    for mode in modes:
        g_pred, score, error = generate_and_score_with_mode(model, x, y, device=device, mode="row", max_new=512)
        if error is None and g_pred is not None:
            predictions.append(g_pred)
            scores.append(score)
        else:
            errors.append(f"{mode}: {error}")

    if not predictions:
        return None, None, f"All modes failed: {'; '.join(errors)}"

    # PoE: Select prediction with best score
    best_idx = np.argmax(scores)
    poe_score = poe_sum(scores)

    return predictions[best_idx], poe_score, None


def evaluate_ttt(model, task, test_idx, device="cpu", ttt_config=None):
    """TTT: Test-Time Training on task examples."""
    if ttt_config is None:
        ttt_config = {"augmentation_multiplier": 3, "random": 0.3, "specific": {"rotate": 0.1, "flip": 0.1}}

    # Cache original weights
    original_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    try:
        # Initialize TTT trainer
        ttt = TestTimeTrainer(model=model, learning_rate=5e-5, steps=10, batch_size=2, augmentation_config=ttt_config)

        # Train on task
        ttt.train_on_task(task)

        # Evaluate with adapted model
        train_pair = task["train"][test_idx]
        x = from_list(train_pair["input"])
        y = from_list(train_pair["output"])

        g_pred, score, error = generate_and_score_with_mode(model, x, y, device=device, mode="row", max_new=512)

        return g_pred, score, error

    except Exception as e:
        return None, None, f"TTT failed: {type(e).__name__}: {str(e)}"
    finally:
        # Restore original weights
        model.load_state_dict(original_state)


def evaluate_ttt_poe(model, task, test_idx, device="cpu", ttt_config=None):
    """TTT + PoE: Combined approach."""
    if ttt_config is None:
        ttt_config = {"augmentation_multiplier": 3, "random": 0.3, "specific": {"rotate": 0.1, "flip": 0.1}}

    # Cache original weights
    original_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    try:
        # Initialize TTT trainer
        ttt = TestTimeTrainer(model=model, learning_rate=5e-5, steps=10, batch_size=2, augmentation_config=ttt_config)

        # Train on task
        ttt.train_on_task(task)

        # Evaluate with PoE
        train_pair = task["train"][test_idx]
        x = from_list(train_pair["input"])
        y = from_list(train_pair["output"])

        g_pred, score, error = evaluate_poe(model, x, y, device=device)

        return g_pred, score, error

    except Exception as e:
        return None, None, f"TTT+PoE failed: {type(e).__name__}: {str(e)}"
    finally:
        # Restore original weights
        model.load_state_dict(original_state)


# ============================================================================
# MODEL LOADING
# ============================================================================


def load_model(model_path: str, device: str = "cpu"):
    """Load a trained model."""
    model_file = Path(model_path)

    if not model_file.exists():
        raise FileNotFoundError(f"Model not found: {model_file}")

    print(f"Loading model: {model_file.name}")
    ckpt = torch.load(model_file, map_location=device)
    cfg = TinyLMConfig(**ckpt["cfg"])
    model = TinyLM(cfg)
    model.load_state_dict(ckpt["model"])
    model.eval()
    model.to(device)

    return model, ckpt.get("step", "unknown"), cfg


# ============================================================================
# EVALUATION RUNNER
# ============================================================================


def evaluate_dataset(
    model,
    model_name: str,
    dataset_name: str,
    strategies: List[str],
    device: str = "cpu",
    max_tasks: Optional[int] = None,
    ttt_config: Optional[Dict] = None,
):
    """
    Evaluate model on a dataset with specified strategies.

    Args:
        model: Loaded model
        model_name: Name for reporting
        dataset_name: 'training' or 'evaluation'
        strategies: List of strategies to test: ['baseline', 'poe', 'ttt', 'ttt_poe']
        device: Device to run on
        max_tasks: Maximum tasks to evaluate (None = all)
        ttt_config: Configuration for TTT (if used)

    Returns:
        Dictionary with detailed results
    """
    print(f"\n{'='*80}")
    print(f"EVALUATING: {model_name}")
    print(f"DATASET: {dataset_name.upper()}")
    print(f"STRATEGIES: {', '.join(strategies)}")
    print(f"{'='*80}")

    # Initialize results structure
    results = {
        "model_name": model_name,
        "dataset": dataset_name,
        "strategies_tested": strategies,
        "start_time": datetime.now().isoformat(),
        "tasks": [],
        "summary": {},
    }

    # Initialize counters for each strategy
    for strategy in strategies:
        results["summary"][strategy] = {
            "total": 0,
            "success": 0,
            "exact_match": 0,
            "shape_mismatch": 0,
            "errors": {},
        }

    start_time = time.time()
    task_count = 0

    for i, (task, task_path) in enumerate(iter_tasks(dataset_name)):
        if max_tasks and i >= max_tasks:
            break

        task_id = Path(task_path).stem
        task_count += 1

        # Progress indicator
        if task_count % 10 == 0:
            elapsed = time.time() - start_time
            rate = task_count / elapsed if elapsed > 0 else 0
            print(
                f"  Progress: {task_count} tasks | {rate:.2f} tasks/sec | " f"Elapsed: {elapsed/60:.1f}min", end="\r"
            )

        # Skip if no training examples
        if len(task["train"]) == 0:
            task_result = {
                "task_id": task_id,
                "error": "No training examples",
            }
            for strategy in strategies:
                results["summary"][strategy]["total"] += 1
                results["summary"][strategy]["errors"]["NoTrainingExamples"] = (
                    results["summary"][strategy]["errors"].get("NoTrainingExamples", 0) + 1
                )
            results["tasks"].append(task_result)
            continue

        # Get test example
        test_idx = 0
        train_pair = task["train"][test_idx]
        x = from_list(train_pair["input"])
        y = from_list(train_pair["output"])
        expected_shape = tuple(y.a.shape)

        task_result = {
            "task_id": task_id,
            "input_shape": tuple(x.shape),
            "expected_shape": expected_shape,
        }

        # Evaluate each strategy
        for strategy in strategies:
            strategy_result = {
                "success": False,
                "exact_match": False,
                "shape_match": False,
                "predicted_shape": None,
                "score": None,
                "error": None,
            }

            results["summary"][strategy]["total"] += 1

            try:
                # Run appropriate evaluation
                if strategy == "baseline":
                    g_pred, score, error = evaluate_baseline(model, x, y, device)
                elif strategy == "poe":
                    g_pred, score, error = evaluate_poe(model, x, y, device)
                elif strategy == "ttt":
                    g_pred, score, error = evaluate_ttt(model, task, test_idx, device, ttt_config)
                elif strategy == "ttt_poe":
                    g_pred, score, error = evaluate_ttt_poe(model, task, test_idx, device, ttt_config)
                else:
                    error = f"Unknown strategy: {strategy}"
                    g_pred, score = None, None

                if error:
                    strategy_result["error"] = error
                    error_type = error.split(":")[0]
                    results["summary"][strategy]["errors"][error_type] = (
                        results["summary"][strategy]["errors"].get(error_type, 0) + 1
                    )

                elif g_pred is not None:
                    pred_shape = tuple(g_pred.shape)
                    strategy_result["predicted_shape"] = pred_shape
                    strategy_result["score"] = score

                    # Check shape match
                    if pred_shape == expected_shape:
                        strategy_result["shape_match"] = True
                        strategy_result["success"] = True
                        results["summary"][strategy]["success"] += 1

                        # Check exact match
                        if np.array_equal(g_pred, y.a):
                            strategy_result["exact_match"] = True
                            results["summary"][strategy]["exact_match"] += 1
                    else:
                        strategy_result["error"] = f"Shape mismatch: {pred_shape} vs {expected_shape}"
                        results["summary"][strategy]["shape_mismatch"] += 1

            except Exception as e:
                strategy_result["error"] = f"Exception: {type(e).__name__}: {str(e)}"
                error_type = type(e).__name__
                results["summary"][strategy]["errors"][error_type] = (
                    results["summary"][strategy]["errors"].get(error_type, 0) + 1
                )

            task_result[strategy] = strategy_result

        results["tasks"].append(task_result)

    elapsed = time.time() - start_time
    results["end_time"] = datetime.now().isoformat()
    results["elapsed_seconds"] = elapsed
    results["total_tasks"] = task_count

    print(f"\n  Completed: {task_count} tasks in {elapsed:.1f}s ({task_count/elapsed:.2f} tasks/sec)")

    return results


# ============================================================================
# REPORTING
# ============================================================================
def print_results_table(results: Dict):
    """Print formatted results table."""
    print(f"\n{'='*80}")
    print(f"RESULTS: {results['model_name']} - {results['dataset'].upper()}")
    print(f"{'='*80}")

    print(f"\n{'Strategy':<15} | {'Success':<18} | {'Exact Match':<18} | {'Failed':<12}")
    print("-" * 80)

    for strategy, summary in results["summary"].items():
        total = summary["total"]
        success = summary["success"]
        exact = summary["exact_match"]
        failed = total - success

        if total > 0:
            success_str = f"{success}/{total} ({100*success/total:.1f}%)"
            exact_str = f"{exact}/{total} ({100*exact/total:.1f}%)"
        else:
            success_str = "N/A"
            exact_str = "N/A"

        print(f"{strategy:<15} | {success_str:<18} | {exact_str:<18} | {failed:<12}")

    print(f"\n⏱️  Time: {results['elapsed_seconds']:.1f}s")
    print(f"📊 Total tasks: {results['total_tasks']}")

    # Error breakdown
    print(f"\n{'='*80}")
    print("ERROR BREAKDOWN")
    print(f"{'='*80}")

    for strategy, summary in results["summary"].items():
        if summary["errors"]:
            print(f"\n{strategy.upper()}:")
            for error_type, count in sorted(summary["errors"].items(), key=lambda x: -x[1]):
                print(f"  {error_type}: {count}")


def save_results(all_results: Dict, output_path: str):
    """Save results to JSON file."""
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n✓ Results saved to: {output_path}")


# ============================================================================
# MAIN
# ============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate ARC-AGI model with multiple strategies",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Baseline only on training set (quick test)
  python evaluate_arc_model.py --model best.pt --strategies baseline --dataset training --max-tasks 100
  
  # All strategies on evaluation set
  python evaluate_arc_model.py --model best.pt --strategies baseline poe ttt ttt_poe --dataset evaluation
  
  # Full evaluation on both datasets
  python evaluate_arc_model.py --model best.pt --strategies baseline poe ttt ttt_poe --dataset both
  
  # Baseline on both datasets (fastest)
  python evaluate_arc_model.py --model best.pt --strategies baseline --dataset both
        """,
    )

    parser.add_argument("--model", type=str, required=True, help="Path to model checkpoint (.pt file)")
    parser.add_argument(
        "--strategies",
        nargs="+",
        choices=["baseline", "poe", "ttt", "ttt_poe"],
        default=["baseline"],
        help="Evaluation strategies to use",
    )
    parser.add_argument(
        "--dataset", choices=["training", "evaluation", "both"], default="both", help="Which dataset(s) to evaluate on"
    )
    parser.add_argument("--max-tasks", type=int, default=None, help="Maximum tasks per dataset (None = all)")
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"], default="cpu", help="Device to run on")
    parser.add_argument("--output", type=str, default="evaluation_results.json", help="Output JSON file path")

    args = parser.parse_args()

    # Print configuration
    print("=" * 80)
    print("ARC-AGI MODEL EVALUATION")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Model: {args.model}")
    print(f"  Strategies: {', '.join(args.strategies)}")
    print(f"  Dataset: {args.dataset}")
    print(f"  Max tasks: {args.max_tasks if args.max_tasks else 'All'}")
    print(f"  Device: {args.device}")
    print(f"  Output: {args.output}")

    # Time estimates
    if args.max_tasks:
        est_time = args.max_tasks * len(args.strategies) * 2 / 60  # ~2 sec per task per strategy
    else:
        total_tasks = 1000 + 120  # training + eval
        est_time = total_tasks * len(args.strategies) * 2 / 60

    print(f"\n⏱️  Estimated time: {est_time:.0f} minutes")

    if "ttt" in args.strategies or "ttt_poe" in args.strategies:
        print("⚠️  TTT strategies are much slower (~30s per task)")
        est_time *= 15
        print(f"   Revised estimate: {est_time:.0f} minutes ({est_time/60:.1f} hours)")

    # Load model
    print("\n" + "=" * 80)
    print("LOADING MODEL")
    print("=" * 80)
    try:
        model, step, cfg = load_model(args.model, args.device)
        print(f"✓ Model loaded successfully")
        print(f"  Step: {step}")
        print(f"  Config: d_model={cfg.d_model}, n_layers={cfg.n_layers}, n_heads={cfg.n_heads}")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return 1

    # Determine datasets to evaluate
    datasets = []
    if args.dataset == "both":
        datasets = ["training", "evaluation"]
    else:
        datasets = [args.dataset]

    # Run evaluation
    all_results = {
        "timestamp": datetime.now().isoformat(),
        "model_path": args.model,
        "model_step": step,
        "strategies": args.strategies,
        "datasets": {},
    }

    for dataset in datasets:
        results = evaluate_dataset(
            model=model,
            model_name=f"{Path(args.model).name} (step {step})",
            dataset_name=dataset,
            strategies=args.strategies,
            device=args.device,
            max_tasks=args.max_tasks,
        )

        all_results["datasets"][dataset] = results
        print_results_table(results)

    # Overall summary if multiple datasets
    if len(datasets) > 1:
        print(f"\n{'='*80}")
        print("OVERALL SUMMARY ACROSS ALL DATASETS")
        print(f"{'='*80}\n")

        print(f"{'Strategy':<15} | {'Success Rate':<20} | {'Exact Match Rate':<20}")
        print("-" * 70)

        for strategy in args.strategies:
            total_success = 0
            total_exact = 0
            total_tasks = 0

            for dataset in datasets:
                summary = all_results["datasets"][dataset]["summary"][strategy]
                total_success += summary["success"]
                total_exact += summary["exact_match"]
                total_tasks += summary["total"]

            if total_tasks > 0:
                success_rate = 100 * total_success / total_tasks
                exact_rate = 100 * total_exact / total_tasks
                success_str = f"{total_success}/{total_tasks} ({success_rate:.1f}%)"
                exact_str = f"{total_exact}/{total_tasks} ({exact_rate:.1f}%)"
            else:
                success_str = "N/A"
                exact_str = "N/A"

            print(f"{strategy:<15} | {success_str:<20} | {exact_str:<20}")

    # Save results
    save_results(all_results, args.output)

    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE")
    print("=" * 80)

    return 0


if __name__ == "__main__":
    sys.exit(main())
