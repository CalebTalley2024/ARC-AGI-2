# Week 0 - Setup

Responsible: seun, Xinyu Wu, Vedant Tibrewal, Caleb Talley, Weiwen Dong, Fariha Sheikh
Status: In progress

# **Week 0 — Setup & Data Hygiene (3–5 days total)**

## **0) Choose where you’ll run (0.5–1 hr)**

- **Primary GPU:** NVIDIA **RTX A4000 16 GB** (preferred) or **RTX 3060 12 GB**.
- **Secondary / overflow:** Google Colab (T4/L4 for small tests).
- **Avoid for week 0:** Long AMD ROCm debugging; you can revisit later.

**Deliverable:** A note in README: “Primary GPU = ___; Backup = ___.”

---

## **1) Create a clean repo scaffold (1–2 hrs) → Caleb (Done)**

```python
# Sample template

arc-lab/
  README.md
  LICENSE
  pyproject.toml          # packaging + tools
  requirements.txt        # pinned versions
  .pre-commit-config.yaml # black/ruff hooks
  arc/
    __init__.py
    io/                   # data loading/saving
    grids/                # grid ops & views
    serialize/            # tokenizers/encoders
    viz/                  # visualization helpers
    eval/                 # runners, scoring, timing
    utils/                # seeding, logging
    models/               # (stubs) perspective_token_predictor/, lpn/
  data/
    raw/arc/              # official ARC JSON
    processed/            # cached tensors, indices
  tests/                  # pytest + hypothesis
  notebooks/              # quick sanity checks
  scripts/                # tiny CLIs (render, stats)
```

- **Open-source (reuse):** basic Python repo patterns, pre-commit hooks, black/ruff config.
- **From scratch:** directory layout tuned to your plan.
- *Directory readme if following different structure*

**Deliverable:** Repo created, pre-commit installed and working.

---

## **2) Lock environments (2–4 hrs) (Parallel 1)**

### **2a) NVIDIA machine (PyTorch CUDA)**

- Create venv/conda, install **PyTorch (CUDA-matching)**, **numpy**, **einops**, **tqdm**, **pytest**, **hypothesis**, **matplotlib**, **Pillow**, **pyyaml**, **rich/loguru**.
- Save requirements.txt (pin versions).
- Add make env-check or a tiny script that prints: torch version, cuda availability, device name.

```python
# PyTorch-CUDA testing notebook
import torch
print("CUDA available:", torch.cuda.is_available())
print("Torch CUDA:", torch.version.cuda)
if torch.cuda.is_available():
    print("Device:", torch.cuda.get_device_name(0))
```

### **2b) Colab Starter Notebook**

- Make a **starter notebook** in notebooks/00_env_colab.ipynb that installs your requirements.txt (or minimal subset), runs import torch; print(device) and a tiny CUDA kernel test.
- requirements.txt, 00_env_colab.ipynb

**Time:** 2–4 hrs

---

## **3) Download & validate ARC and ReARC dataset (2–3 hrs) ( Parallel 1) → Done**

- Place official ARC train/eval JSON into data/raw/arc/.
- Write arc/io/loader.py:
    - load_task(path) -> dict (train pairs + test input(s))
    - iter_tasks(split) -> iterator
    - **Validation checks:** grid fits ≤ 30×30, colors are ints, shapes consistent.
- Cache a light index: processed/index.jsonl with {task_id, n_train_pairs, in_shape, out_shape}.
- Create a **split manifest** you will use for dev vs. public-eval (to avoid accidental leakage while prototyping).

**Tests:**

- tests/test_loader.py: load 5 random tasks; assert counts, shapes, color id range.

**Deliverables:**

- arc/io/loader.py, processed/index.jsonl, passing tests.

---

## **4) Define the core Grid type + safe operations (2–4 hrs)**

- Decide internal representation: np.ndarray[int8] (H×W), palette = {0 - 9}.
- Implement in arc/grids/core.py:
    - Grid.from_list(list2d), to_list()
    - palette(grid) -> set[int]
    - assert_same_shape(a,b)
    - diff(a,b) -> mask (for visualization & verifiers)
- Add **deterministic seeding** helper in arc/utils/seeding.py (seed numpy, torch, python).

**Tests:**

- Round-trip from_list → to_list.
- Palette computed correctly.

**Deliverables:** core.py, seeding.py, unit tests.

- Explanation
    
    **What are the key terms?**
    
    - **Grid**: The puzzle board. A small 2-D array (height × width) where each cell is a **color id**.
    - **Color id**: An integer label for a color, always in **0…9** in ARC.
    - **Palette**: The **set of unique color ids** that actually appear in a grid. Example: if a grid only uses colors 0, 3, and 7, its palette is {0,3,7}.
    - **Shape**: (H, W); number of rows (height) and columns (width).
    - **Dtype**: The numeric type used in memory. We’ll use np.int8 (or np.uint8) to keep it small and fast.
    - **Mask**: A boolean grid (same shape) where True marks cells of interest (e.g., differences).
    
    ---
    
    ## **Design goals (why these actions are suggested)**
    
    1. **Small, strict, fast.** ARC grids are tiny (≤ 30×30). A compact numeric array with strict checks is fastest and avoids silent bugs.
    2. **Immutability by convention.** We *treat* loaded grids as read-only and copy when transforming. This prevents accidental in-place edits during search or scoring.
    3. **Determinism.** Reproducible results matter (for Kaggle ≤12h budgeting and debugging). One seeding utility to control randomness everywhere.
    
    ---
    
    ## **Recommended representation**
    
    - Use **NumPy** arrays for the core type and convert to Torch tensors only in model code:
    
    ```python
    # arc/grids/core.py
    from dataclasses import dataclass
    import numpy as np
    
    GridArray = np.ndarray  # (H, W), dtype int8 or uint8
    
    @dataclass(frozen=True)
    class Grid:
        """Light wrapper to enforce invariants."""
        a: GridArray  # must be 2D, compact, values in between 0 to 9
    
        @property
        def shape(self) -> tuple[int, int]:
            return self.a.shape
    
        def copy(self) -> "Grid":
            return Grid(self.a.copy())
    ```
    
    **Dtype choice:**
    
    - `np.uint8` is natural (colors are 0…9).
    - `np.int8` also works (saves space, still covers 0…9).
        
        Pick one and **stick with it** across the project. If you’ll ever subtract palettes or use negative sentinels, prefer int8. Otherwise uint8 is slightly safer for “non-negative only.”
        
    
    ---
    
    ## **Functions to implement (and why)**
    
    ### **1) Constructors & round-trip**
    
    ```python
    def from_list(lst: list[list[int]], dtype=np.uint8) -> Grid:
        a = np.array(lst, dtype=dtype, copy=True)
        if a.ndim != 2:
            raise ValueError("Grid must be 2D")
        if a.shape[0] == 0 or a.shape[1] == 0 or a.shape[0] > 30 or a.shape[1] > 30: # check for the max grid size
            raise ValueError("Invalid grid shape; ARC uses <= 30x30 and > 0")
        if a.min() < 0 or a.max() > 9:
            raise ValueError("Color ids must be in between 0 to 9")
        # Ensure contiguous memory to avoid surprises
        a = np.ascontiguousarray(a)
        return Grid(a)
    
    def to_list(g: Grid) -> list[list[int]]:
        return g.a.tolist()
    ```
    
    **Why:** Strict validation early prevents hard-to-debug errors later (e.g., unexpected shapes after a transform).
    
    ---
    
    ### **2) Palette**
    
    ```python
    def palette(g: Grid) -> set[int]:
        # Unique colors present in the grid
        return set(np.unique(g.a).tolist())
    ```
    
    **Why:** Many ARC rules depend on *which* colors exist (e.g., “map color 2 → 5”). Exposing the palette cheaply (O(HW)) is useful for heuristics, color permutations, and tests.
    
    ---
    
    ### **3) Shape checks**
    
    ```python
    def assert_same_shape(a: Grid, b: Grid) -> None:
        if a.shape != b.shape:
            raise ValueError(f"Shape mismatch: {a.shape} vs {b.shape}")
    ```
    
    **Why:** Comparing different shapes is a logic bug. Fail fast.
    
    ---
    
    ### **4) Diff mask**
    
    ```python
    def diff(a: Grid, b: Grid) -> np.ndarray:
        """Return boolean mask where cells differ."""
        assert_same_shape(a, b)
        return (a.a != b.a)
    ```
    
    **Why:** You’ll constantly verify candidates against ground truth. A mask is easy to visualize and count (mask.sum()).
    
    ---
    
    ### **5) Deterministic seeding (project-wide)**
    
    ```python
    # arc/utils/seeding.py
    import os, random
    import numpy as np
    
    def set_seed(seed: int) -> None:
        os.environ["PYTHONHASHSEED"] = str(seed)
        random.seed(seed)
        np.random.seed(seed)
        try:
            import torch
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        except ImportError:
            pass
    ```
    
    **Why:** Views may sample color permutations; DFS may break ties randomly; TTT may shuffle batches. With one function call, you keep runs reproducible.
    
    ---
    
    ## **“Safe operation” behaviors (best practices)**
    
    - **Never mutate inputs in place** inside library code. If an operation changes a grid, **return a new Grid**.
    - **Contiguous arrays**: call np.ascontiguousarray after transforms. Non-contiguous strides can break some downstream libs or slow things down.
    - **Value guardrails**: after color operations, assert values remain in 0…9.
    - **Copy on write**: Python/Numpy sometimes returns views; be explicit with .copy() when you’re going to modify.
    
    ---
    
    ## **Minimal tests to write (fast and valuable)**
    
    tests/test_core.py
    
    ```python
    from arc.grids.core import from_list, to_list, palette, diff
    import numpy as np
    
    def test_round_trip():
        g = from_list([[0,1],[2,3]])
        assert to_list(g) == [[0,1],[2,3]]
    
    def test_palette():
        g = from_list([[0,0,3],[7,3,7]])
        assert palette(g) == {0,3,7}
    
    def test_diff_mask():
        a = from_list([[1,1],[2,2]])
        b = from_list([[1,0],[2,3]])
        m = diff(a,b)
        assert m.dtype == np.bool_
        assert m.sum() == 2  # (0,1) and (1,1) differ
    
    def test_validation():
        try:
            from_list([[10]])  # out of range
            assert False
        except ValueError:
            assert True
    ```
    
    ---
    
    ## **Common pitfalls (and how to avoid them)**
    
    - **Row/column confusion**: Always treat index as (row, col) = (x, y); document it in README.md.
    - **Dtype overflow**: If you ever add/subtract, uint8 can wrap (e.g., −1 becomes 255). Either cast to a wider type temporarily or prefer int8 and clamp into 0…9 with checks.
    - **In-place edits**: Accidentally changing grid.a[...] inside a function can corrupt upstream state. Work on `grid.a.copy()` when modifying.
    - **Non-contiguous arrays**: Some chains of transpose/flip produce strided arrays; ensure `np.ascontiguousarray()` before handing off to Torch.
    - **Hidden shape drift**: After a view, verify the result’s shape still respects ARC limits and matches expectations (write a tiny assert_valid_grid(g) helper).
    
    ---
    
    ## **Quick rationale recap**
    
    - **Why a class (Grid) not just np.ndarray?**
        
        A class lets you enforce invariants (2D, ≤30×30, values 0…9) and makes accidental misuse less likely. It also gives a neat place for helpers (g.shape, g.copy()).
        
    - **Why a separate seeding utility?**
        
        One call at program start makes *all* randomness predictable—essential for timing budgets and debugging Kaggle runs.
        
    - **Why strict errors early?**
        
        It’s cheaper to catch a bad grid **now** than to debug a wrong solution **later**.
        

---

## **5) Implement “views” Φ and inverse views (3–5 hrs)**

- In arc/grids/views.py implement:
    - **Geometry (D₄):** rotate90k, flip_h, flip_v, transpose.
    - **Color permutation:** permute_colors(grid, mapping: dict[int,int]).
    - **Serialization changes:** row_major(grid) -> tokens, col_major(grid) -> tokens.
- Define a **ViewSpec** dataclass:

```python
@dataclass(frozen=True)
class ViewSpec:
    geom: str          # 'rot90','rot180','flip_h','id',...
    color_map: tuple   # e.g., (0,1,2,3,4,5,6,7,8,9) -> permutation
    serialization: str # 'row','col'
```

- Add apply_view(grid, spec) -> grid and invert_view(grid, spec) -> grid.
- **Property tests** (Hypothesis): invert_view(apply_view(G, spec), spec) == G.

**Deliverables:** views.py with inverses + passing property tests.

- Explain
    
    We’ll implement three families:
    
    1. **Geometry views** (the 8 symmetries of a rectangle: the **D₄**/dihedral-group actions):
        
        `id`, `rot90`, `rot180`, `rot270`, `flip_h`, `flip_v`, `transpose`, `transpose_then_flip`
        
    2. **Color-permutation views** (bijective relabeling of color ids 0..9)
    3. **Serialization views** (how we read/linearize a grid, e.g., row-major vs column-major).
    
    **Every view must have a correct inverse so you can map model outputs back to the original frame.**
    
    ---
    
    ### **core design (what to build)**
    
    **5.1 ViewSpec: a compact, hashable description**
    
    ```python
    # arc/grids/views.py
    from dataclasses import dataclass
    from typing import Tuple
    
    @dataclass(frozen=True)
    class ViewSpec:
        geom: str                 # 'id','rot90','rot180','rot270','flip_h','flip_v','transpose','transpose_flip'
        color_map: Tuple[int, ...]# length=10 permutation over {0..9}
        serialization: str        # 'row' or 'col'
    ```
    
    - **geom** chooses a geometry transform.
    - **color_map** is a complete length-10 permutation (fast to index; no dict overhead).
    - **serialization** controls linearization order (used by the tokenizer/decoder in Step 6).
    
    > Why a tuple of length 10?
    > explicit, cache-friendly, and **always bijective. Also lets you quickly invert by inverse[color_map[i]] = i.**
    > 
    
    ---
    
    **5.2 Implement geometry transforms (and their inverses)**
    
    For a **Grid** g.a (NumPy array H×W):
    
    ```python
    import numpy as np
    
    def geom_apply(arr: np.ndarray, name: str) -> np.ndarray:
        if name == 'id':         return arr
        if name == 'rot90':      return np.rot90(arr, k=1)
        if name == 'rot180':     return np.rot90(arr, k=2)
        if name == 'rot270':     return np.rot90(arr, k=3)
        if name == 'flip_h':     return np.flip(arr, axis=1)
        if name == 'flip_v':     return np.flip(arr, axis=0)
        if name == 'transpose':  return arr.T
        if name == 'transpose_flip': return np.flip(arr.T, axis=1)
        raise ValueError(f"unknown geom: {name}")
    
    def geom_inverse(name: str) -> str:
        # All D4 ops are invertible; many are self-inverse.
        inv = {
            'id':'id',
            'rot90':'rot270',
            'rot180':'rot180',
            'rot270':'rot90',
            'flip_h':'flip_h',
            'flip_v':'flip_v',
            'transpose':'transpose',
            'transpose_flip':'transpose_flip',
        }
        return inv[name]
    ```
    
    ---
    
    **5.3 implement color permutation (and its inverse)**
    
    ```python
    def apply_color_map(arr: np.ndarray, cmap: Tuple[int, ...]) -> np.ndarray:
        # cmap is length-10; values are a permutation of 0..9
        out = cmap_np = np.array(cmap, dtype=arr.dtype)
        return cmap_np[arr]  # vectorized gather
    
    def invert_color_map(cmap: Tuple[int, ...]) -> Tuple[int, ...]:
        inv = [0]*10
        for src, dst in enumerate(cmap):
            inv[dst] = src
        return tuple(inv)
    ```
    
    ---
    
    **5.4 apply a ViewSpec to a single grid and invert it**
    
    ```python
    from arc.grids.core import Grid
    
    def apply_view_grid(g: Grid, spec: ViewSpec) -> Grid:
        arr = g.a
        # 1) geometry
        arr = geom_apply(arr, spec.geom)
        # 2) colors
        arr = apply_color_map(arr, spec.color_map)
        # ensure contiguous (good habit)
        arr = np.ascontiguousarray(arr)
        return Grid(arr)
    
    def invert_view_grid(g: Grid, spec: ViewSpec) -> Grid:
        arr = g.a
        # 1) invert colors
        inv_cmap = invert_color_map(spec.color_map)
        arr = apply_color_map(arr, inv_cmap)
        # 2) invert geometry (apply inverse geom)
        arr = geom_apply(arr, geom_inverse(spec.geom))
        arr = np.ascontiguousarray(arr)
        return Grid(arr)
    ```
    
    > **order matters:** we define **apply** as geometry → colors.
    then the **inverse** must do colors⁻¹ → geometry⁻¹ to undo in reverse.
    > 
    
    ---
    
    **5.5 apply a ViewSpec to a whole task (all train pairs + test)**
    
    ```python
    def apply_view_task(task: dict, spec: ViewSpec) -> dict:
        """
        task: {
          "train": [{"input": Grid, "output": Grid}, ...],
          "test":  [Grid, ...]
        }
        """
        out = {"train": [], "test": []}
        for p in task["train"]:
            out["train"].append({
                "input":  apply_view_grid(p["input"],  spec),
                "output": apply_view_grid(p["output"], spec),
            })
        for t in task["test"]:
            out["test"].append(apply_view_grid(t, spec))
        return out
    
    def invert_view_answer(ans_grid: Grid, spec: ViewSpec) -> Grid:
        # For final predictions only (test outputs)
        return invert_view_grid(ans_grid, spec)
    ```
    
    - **always** transform both input **and output** for training pairs; otherwise you break the mapping.
    - For test time, you only transform the **inputs**; later you invert your predicted **output** back.
    
    ---
    
    **5.6 geometry set (cheap)**
    
    - Always try the 8 D₄ ops or a subset like:
        
        ['id','rot90','rot180','rot270','flip_h','flip_v'] (6)
        
        Start with 4 or 6; expand if runtime is fine.
        
    
    ### **5.7 color permutations (smart, small)**
    
    **A) identity + small cycles**
    
    - Always include **identity** (0,1,2,3,4,5,6,7,8,9).
    - Add a few **k-cycles** on the **palette** only (colors used in the task):
        - example: for palette {0,3,7} try mappings that rotate (0→3→7→0) and its inverse.
        - generate ≤ 8–12 candidates total.
    
    **B) data-driven assignment (recommended)**
    
    Use the training pairs to build a **color co-occurrence matrix** C[in_color][out_color] across all input→output examples (count matches at same coordinates, plus looser co-occurrence if helpful). Then solve a **max-weight bipartite assignment** on the palette to get the **best** permutation(s).
    
    - If |palette| ≤ 6, you can brute-force itertools.permutations(palette) and pad identity for unused colors.
    - If larger, use **Hungarian algorithm** (scipy linear_sum_assignment) to propose the top-1 mapping; then add a few **nearby** permutations (swap one or two pairs) to make a small set (e.g., 3–5 total).
    - Keep unmapped colors as identity.
    
    **C) random jitter on palette**
    
    - Add 1–2 random palette permutations (seeded!) to avoid blind spots.
    
    > Budget tip: geometry x color_perm_count ≤ **40–80 total views**
    > 
    
    ---
    
    ### **serialization views**
    
    - **row-major** (default) and **column-major** as two options.
    - You won’t “invert” serialization for a grid (it’s only used when you **encode tokens**); but you track it in ViewSpec.serialization so your tokenizer uses the right order and your **decoder knows how to place tokens back**.
    
    ---
    
    ### **property tests (must-haves)**
    
    1. **grid round-trip under a view**
    
    ```python
    @given(random_grid(), random_viewspec())
    def test_view_inverse_grid(g, spec):
        g2 = apply_view_grid(g, spec)
        g3 = invert_view_grid(g2, spec)
        assert np.array_equal(g.a, g3.a)
    ```
    
    1. **task round-trip** (apply view to a task then verify train pairs still match)
    - For every train pair (in, out), confirm that your solver-side verifier passes: the transformed input maps to the transformed output **exactly**.
    1. **color map is bijective**
    - set(cmap) == {0..9} and invert_color_map(invert_color_map(cmap)) == cmap.
    
    ---
    
    ### **caching & performance**
    
    - **Precompute** transformed **inputs** for each ViewSpec once per task (store arrays in RAM).
    - Keep a **string id** for each spec to key caches:
        
        f"{geom}|{''.join(map(str,color_map))}|{serialization}"
        
    - If you use PoE or DFS, you’ll ask the model for log-probs under many views; caching cuts total wall-clock time.
    
    ---
    
    ### **safety & correctness checks**
    
    - After any transform, **assert** values still in **0..9**.
    - Ensure arrays are **contiguous** (np.ascontiguousarray) after transpose/flip.
    - Be careful that transpose and rot90 **swap shape** (H×W → W×H). Your tokenizer and model must accept the new shape; your inverse must bring it back exactly.
    
    ---
    
    ### **optional “bonus” views (add later)**
    
    - **Connected-component relabeling**: compute components, then reorder their labels by size/position. this can reveal object-centric invariance, but it’s not strictly color-bijective; treat as a separate augmentation family (with its own inverse bookkeeping).
    - **Cropping / patch views**: center on the main object and normalize position; requires careful inverse logic (pad back).
    
    ---
    
    ### **minimal runnable stubs (drop-in)**
    
    ```python
    # arc/grids/views.py (add to your repo)
    import numpy as np
    from dataclasses import dataclass
    from typing import Tuple
    from arc.grids.core import Grid
    
    @dataclass(frozen=True)
    class ViewSpec:
        geom: str
        color_map: Tuple[int, ...]  # length=10 permutation
        serialization: str          # 'row' | 'col'
    
    D4 = ['id','rot90','rot180','rot270','flip_h','flip_v','transpose','transpose_flip']
    
    def geom_apply(a: np.ndarray, name: str) -> np.ndarray:
        if name == 'id': return a
        if name == 'rot90': return np.rot90(a, 1)
        if name == 'rot180': return np.rot90(a, 2)
        if name == 'rot270': return np.rot90(a, 3)
        if name == 'flip_h': return np.flip(a, 1)
        if name == 'flip_v': return np.flip(a, 0)
        if name == 'transpose': return a.T
        if name == 'transpose_flip': return np.flip(a.T, 1)
        raise ValueError(name)
    
    def geom_inverse(name: str) -> str:
        inv = {'id':'id','rot90':'rot270','rot180':'rot180','rot270':'rot90',
               'flip_h':'flip_h','flip_v':'flip_v','transpose':'transpose',
               'transpose_flip':'transpose_flip'}
        return inv[name]
    
    def apply_color_map(a: np.ndarray, cmap: Tuple[int, ...]) -> np.ndarray:
        lut = np.asarray(cmap, dtype=a.dtype)  # 10 entries
        return lut[a]
    
    def invert_color_map(cmap: Tuple[int, ...]) -> Tuple[int, ...]:
        inv = [0]*10
        for src,dst in enumerate(cmap):
            inv[dst] = src
        return tuple(inv)
    
    def apply_view_grid(g: Grid, spec: ViewSpec) -> Grid:
        arr = geom_apply(g.a, spec.geom)
        arr = apply_color_map(arr, spec.color_map)
        return Grid(np.ascontiguousarray(arr))
    
    def invert_view_grid(g: Grid, spec: ViewSpec) -> Grid:
        inv_cmap = invert_color_map(spec.color_map)
        arr = apply_color_map(g.a, inv_cmap)
        arr = geom_apply(arr, geom_inverse(spec.geom))
        return Grid(np.ascontiguousarray(arr))
    ```
    
    ---
    
    ### **quick strategy for generating color_map candidates (palette-aware)**
    
    ```
    def identity_cmap() -> Tuple[int,...]:
        return tuple(range(10))
    
    def palette_only_cmap(palette: set[int], perm: Tuple[int,...]) -> Tuple[int,...]:
        # Replace only colors in palette according to 'perm' of the palette list; others stay identity.
        full = list(range(10))
        pal = sorted(palette)
        for i, c in enumerate(pal):
            full[c] = pal[perm.index(c)]  # or map via a dict built from perm
        return tuple(full)
    ```
    
    - better: build cmaps from assignment on a C[10,10] matrix you compute from training pairs (counts of input-→output color alignments). keep top-N assignments.
    
    ---
    
    **testing checklist for Step 5**
    
    - **Invertibility:** invert_view_grid(apply_view_grid(g,spec), spec) == g for 100 random specs.
    - **Task consistency:** after transforming a train pair, the mapping still holds (input→output).
    - **Color safety:** after applying any cmap, .min() >= 0, .max() <= 9.
    - **Shapes:** rot90/transpose swap dims; tokenizer still works (Step 6).
    - **Cache keys:** specs serialize into stable string ids (useful for logging & PoE).

---

## **6) Decide and implement serialization (tokenization) (4–6 hrs)**

**Goal:** A reversible mapping between (meta, grid) ↔ token sequence.

- In arc/serialize/tokens.py:
    - **Special tokens:** [BOS], [SEP], [EOS], [PAD], [WIDTH], [HEIGHT], [COLORS]...
    - **Tokenization format (example):**

```
[BOS] [WIDTH]=W [HEIGHT]=H [N_TRAIN]=K
   ( [SEP] <in_grid_row_major_tokens> [SEP] <out_grid_row_major_tokens> ) x K
[SEP] <test_in_row_major_tokens> [EOS]
```

- Implement encode_example(task, view_spec) and decode_grid(tokens) (for predicted outputs).
- Add **invertibility tests** on small grids (3×3, 5×5, 10×10).
- Measure typical sequence lengths and record in README.

**Deliverables:** tokens.py, tests showing encode→decode correctness.

---

## **7) Visualization & tiny CLIs (2–3 hrs)**

- In arc/viz/plot.py:
    - save_grid_png(grid, path), save_pair_png(in_grid, out_grid, path).
    - A color legend (0–9).
- CLI in scripts/render_task.py:

```bash
python scripts/render_task.py --task-id 123 --out-dir out/vis/
```

- **Diff visual:** highlight mismatched cells in red overlay for quick debugging.

**Deliverables:** PNGs for 3 random tasks, committed to out/vis/ (git-ignored if large).

---

## **8) Build a Kaggle-safe runner skeleton (2–3 hrs)**

- In arc/eval/runner.py:
    - solve_task(task, mode="baseline") -> list[grid] returning **two** attempts (required by ARC scoring). For week 0, just return trivial copies (e.g., input as guess, or a blank grid) to exercise the pipeline.
    - solve_all(tasks_iter, out_path="submission.json").
- Add a --time-budget flag and print per-task timings (you’ll need this later to fit ≤12h).
- Ensure no internet calls; pure file I/O.

**Deliverables:** Running end-to-end dummy submission on public tasks.

---

## **9) Reproducibility, config (1–2 hrs) (logging optional)**

- Add arc/utils/logging.py (rich/loguru) and configs/base.yaml:
    - seeds, view list, serialization mode, device, num_workers.
- Ensure every script calls set_seed(seed) and logs device info.

**Deliverables:** configs/base.yaml checked into git + a sample log.

**Deliverables:** notebook with screenshots; timing row for 5 tasks.

---

---

## **Acceptance checklist (end of Week 0)**

- Repo scaffold with working **pre-commit**
- Environment prints correct **CUDA device** (or HIP on AMD)
- ARC JSON loads; **index.jsonl** built; validators pass
- **Grid** core ops + **views** and **inverse views** pass property tests
- **Serialization** encode/decode is reversible on random grids
- **Visualizer** saves example PNGs

---