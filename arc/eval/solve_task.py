import json
from pathlib import Path
from typing import Dict, Any, Tuple
import numpy as np
import torch

from arc.grids.core import Grid
from arc.io.loader import load_task
from arc.grids.views import (
    ViewSpec,
    apply_view_grid,
    invert_view_answer,
    identity_cmap,
    generate_palette_permutations,
)
from arc.eval.poe import poe_sum
from arc.eval.scorer import mean_logp_output
from arc.models.tiny_lm import TinyLM, TinyLMConfig
from arc.models.ttt import TestTimeTrainer
from arc.models.infer import greedy_generate
from arc.serialize.task_tokenizer import deserialize_grid, serialize_grid, SEP, EOS
from arc.utils.constants import BOS

from arc.serialize.task_tokenizer import pack_example


def load_task_by_id(task_id: str) -> Dict[str, Any]:
    paths = [
        Path(f"data/raw/arc/evaluation/{task_id}.json"),
    ]
    for p in paths:
        if p.exists():
            return load_task(str(p))
    raise FileNotFoundError(f"Task {task_id} not found in raw data")


def load_model(ckpt_path: str, device: str):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if "cfg" in ckpt:
        cfg = TinyLMConfig(**ckpt["cfg"])
    else:
        cfg = TinyLMConfig()

    model = TinyLM(cfg)
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()
    return model


def generate_and_score(
    model, x_grid: Grid, y_grid: Grid, device="cpu", mode: str = "row", max_new: int = 2048
) -> Tuple[Grid, float]:
    """
    Generate and score a single pair (input -> output).
    Used for selecting the best view based on training data.
    """

    prompt = pack_example(x_grid, y_grid, mode=mode) + [SEP]
    inp = torch.tensor(prompt, dtype=torch.long).unsqueeze(0).to(device)

    max_allowed_new = max(1, max_new - inp.size(1))
    eff_max_new = min(max_new, max_allowed_new)

    out = greedy_generate(model, inp, max_new_tokens=eff_max_new, eos_id=EOS)  # (1, T')

    score = mean_logp_output(model, out.clone(), sep_token_id=SEP)

    ids = out.squeeze(0).tolist()

    prompt_len = len(prompt)
    y_tokens = ids[prompt_len:]

    y_seq = [BOS] + y_tokens

    try:
        g_pred = deserialize_grid(y_seq, mode=mode)
    except Exception:
        g_pred = x_grid

    return g_pred, float(score)


def test_time_train_on_task(base_model, task_dict, device, steps=50, lr=1e-4, bs=4):
    trainer = TestTimeTrainer(model=base_model, learning_rate=lr, steps=steps, batch_size=bs)
    cached = trainer.cache_weights()
    trainer.train_on_task(task_dict)

    return base_model, cached, trainer


def build_fewshot_prompt(
    task_grids, x_test_v: Grid, best_view: ViewSpec, mode: str = "row", max_train_examples: int = 3
):
    seq = []

    train_pairs = task_grids["train"][:max_train_examples]
    for pair in train_pairs:
        x_tr = apply_view_grid(pair["input"], best_view)
        y_tr = apply_view_grid(pair["output"], best_view)

        sx = serialize_grid(x_tr, mode=mode)
        sy = serialize_grid(y_tr, mode=mode)

        seq.extend(sx)
        seq.append(SEP)
        seq.extend(sy[1:])
        seq.append(SEP)

    sx_test = serialize_grid(x_test_v, mode=mode)
    seq.extend(sx_test)
    seq.append(SEP)

    return seq


def solve(task_id: str, ckpt: str, use_ttt: bool = True):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    base_model = load_model(ckpt, device)

    task = load_task_by_id(task_id)

    # Convert raw task dict to Grid objects for manipulation
    def to_grid_dict(t):
        t_new = {"train": [], "test": []}
        for pair in t["train"]:
            t_new["train"].append(
                {
                    "input": Grid(np.array(pair["input"], dtype=np.int8)),
                    "output": Grid(np.array(pair["output"], dtype=np.int8)),
                }
            )
        for pair in t["test"]:
            p = {"input": Grid(np.array(pair["input"], dtype=np.int8))}
            if "output" in pair:
                p["output"] = Grid(np.array(pair["output"], dtype=np.int8))
            t_new["test"].append(p)
        return t_new

    task_grids = to_grid_dict(task)

    # ttt implementation
    if use_ttt:
        model, cached_weights, trainer = test_time_train_on_task(
        base_model,
        task,
        device=device,
        steps=50,
        lr=1e-4,
        bs=4,
    )
    else:
        model = base_model
        cached_weights = None
        trainer = None

    # multiple views
    views = [
        ViewSpec(geom="id", color_map=identity_cmap(), serialization="row"),
        ViewSpec(geom="rot90", color_map=identity_cmap(), serialization="row"),
        ViewSpec(geom="rot180", color_map=identity_cmap(), serialization="row"),
        # ViewSpec(geom="rot270", color_map=identity_cmap(), serialization="row"),
        # ViewSpec(geom="flip_h", color_map=identity_cmap(), serialization="row"),
        # ViewSpec(geom="flip_v", color_map=identity_cmap(), serialization="row"),
        # ViewSpec(geom="transpose", color_map=identity_cmap(), serialization="row"),
    ]

    x0 = task_grids["train"][0]["input"]
    y0 = task_grids["train"][0]["output"]

    candidates = []
    for v in views:
        x_v = apply_view_grid(x0, v)
        y_v = apply_view_grid(y0, v)

        g_pred_v, score_v = generate_and_score(model, x_v, y_v, device, mode="row")

        # Invert prediction to get back to original space
        g_pred = invert_view_answer(g_pred_v, v)
        candidates.append((g_pred, score_v))

    # candidates is list of (grid, score)
    best_grid_train, best_view_score = max(candidates, key=lambda t: t[1])

    # Find which view was best
    best_view_idx = candidates.index((best_grid_train, best_view_score))
    best_view = views[best_view_idx]

    # PoE
    all_scores = [s for _, s in candidates]
    poe_score = poe_sum(all_scores)

    print(
        f"Task {task_id}: best_view_score={best_view_score:.3f}, "
        f"poe_sum={poe_score:.3f}, best_view={best_view.geom}"
    )

    # Make predictions on TEST set using the best view
    preds = []
    for test_idx, test_pair in enumerate(task_grids["test"]):
        x_test = test_pair["input"]

        # Apply best view
        x_test_v = apply_view_grid(x_test, best_view)

        # Construct prompt: x_v + SEP (few-shot with train examples)
        prompt = build_fewshot_prompt(task_grids, x_test_v, best_view, mode="row")

        # Ensure prompt + generated tokens do not exceed model max_len
        max_len = getattr(model, "cfg", None).max_len if getattr(model, "cfg", None) is not None else 2048
        if len(prompt) > max_len:
            prompt = prompt[-max_len:]
        max_new = max(1, max_len - len(prompt))

        inp = torch.tensor(prompt, dtype=torch.long).unsqueeze(0).to(device)

        # Generate
        out = greedy_generate(model, inp, max_new_tokens=max_new, eos_id=EOS)
        ids = out.squeeze(0).tolist()

        # Extract Y part
        prompt_len = len(prompt)
        y_tokens = ids[prompt_len:]
        y_seq = [BOS] + y_tokens

        try:
            g_pred_v = deserialize_grid(y_seq, mode="row")
        except Exception as e:
            print(f"Warning: Failed to deserialize test output {test_idx}: {e}")
            print(f"  y_seq length: {len(y_seq)}, first few tokens: {y_seq[:10]}")
            g_pred_v = x_test_v  # Fallback

        # 3. Invert view
        g_pred = invert_view_answer(g_pred_v, best_view)
        preds.append(g_pred)

    # Save outputs
    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)

    # Convert grids to list of lists for JSON serialization
    pred_lists = [p.to_list() for p in preds]

    with open(out_dir / f"{task_id}.json", "w") as f:
        json.dump(
            {
                "task_id": task_id,
                "preds": pred_lists,
                "scores": {
                    "view_scores": all_scores,
                    "poe_sum": poe_score,
                    "best_view_score": best_view_score,
                    "selected_view": best_view.geom,
                },
            },
            f,
            indent=2,
        )

    # Restore weights (cleanup)
    trainer.restore_weights(cached_weights)

    return preds


def solve_fast(task_id: str, ckpt: str, use_ttt: bool = True):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    base_model = load_model(ckpt, device)

    task = load_task_by_id(task_id)

    # Convert raw task dict to Grid objects
    def to_grid_dict(t):
        t_new = {"train": [], "test": []}
        for pair in t["train"]:
            t_new["train"].append(
                {
                    "input": Grid(np.array(pair["input"], dtype=np.int8)),
                    "output": Grid(np.array(pair["output"], dtype=np.int8)),
                }
            )
        for pair in t["test"]:
            p = {"input": Grid(np.array(pair["input"], dtype=np.int8))}
            if "output" in pair:
                p["output"] = Grid(np.array(pair["output"], dtype=np.int8))
            t_new["test"].append(p)
        return t_new

    task_grids = to_grid_dict(task)

    # Test-Time Training (optional)
    if use_ttt:
        model, cached_weights, trainer = test_time_train_on_task(
            base_model,
            task,
            device=device,
            steps=1,
            lr=1e-4,
            bs=4,
        )
    else:
        model = base_model
        cached_weights = None
        trainer = None

    # Only 2 views for speed
    views = [
        ViewSpec(geom="id", color_map=identity_cmap(), serialization="row"),
        ViewSpec(geom="rot90", color_map=identity_cmap(), serialization="row"),
    ]

    x0 = task_grids["train"][0]["input"]
    y0 = task_grids["train"][0]["output"]

    candidates = []
    for v in views:
        x_v = apply_view_grid(x0, v)
        y_v = apply_view_grid(y0, v)
        g_pred_v, score_v = generate_and_score(model, x_v, y_v, device, mode="row")
        g_pred = invert_view_answer(g_pred_v, v)
        candidates.append((g_pred, score_v))

    best_grid_train, best_view_score = max(candidates, key=lambda t: t[1])
    best_view_idx = candidates.index((best_grid_train, best_view_score))
    best_view = views[best_view_idx]

    all_scores = [s for _, s in candidates]
    poe_score = poe_sum(all_scores)

    print(
        f"Task {task_id}: best_view_score={best_view_score:.3f}, "
        f"poe_sum={poe_score:.3f}, best_view={best_view.geom} (fast mode)"
    )

    # Generate predictions on test set
    preds = []
    for test_idx, test_pair in enumerate(task_grids["test"]):
        x_test = test_pair["input"]
        x_test_v = apply_view_grid(x_test, best_view)
        prompt = build_fewshot_prompt(task_grids, x_test_v, best_view, mode="row")

        max_len = getattr(model, "cfg", None).max_len if getattr(model, "cfg", None) is not None else 2048
        if len(prompt) > max_len:
            prompt = prompt[-max_len:]
        max_new = max(1, max_len - len(prompt))

        inp = torch.tensor(prompt, dtype=torch.long).unsqueeze(0).to(device)

        out = greedy_generate(model, inp, max_new_tokens=max_new, eos_id=EOS)
        ids = out.squeeze(0).tolist()

        prompt_len = len(prompt)
        y_tokens = ids[prompt_len:]
        y_seq = [BOS] + y_tokens

        try:
            g_pred_v = deserialize_grid(y_seq, mode="row")
        except Exception as e:
            print(f"Warning: Failed to deserialize test output {test_idx}: {e}")
            g_pred_v = x_test_v

        g_pred = invert_view_answer(g_pred_v, best_view)
        preds.append(g_pred)

    # Save outputs
    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)

    pred_lists = [p.to_list() for p in preds]

    with open(out_dir / f"{task_id}_fast.json", "w") as f:
        json.dump(
            {
                "task_id": task_id,
                "mode": "fast",
                "preds": pred_lists,
                "scores": {
                    "view_scores": all_scores,
                    "poe_sum": poe_score,
                    "best_view_score": best_view_score,
                    "selected_view": best_view.geom,
                },
            },
            f,
            indent=2,
        )

    if trainer is not None and cached_weights is not None:
        trainer.restore_weights(cached_weights)

    return preds
