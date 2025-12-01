from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from arc.grids.core import Grid
from arc.utils.constants import (
    BOS,
    EOS,
    MAX_GRID_SIZE,
    NUM_COLORS,
    PAD,
    SEP,
    TOK_C_BASE,
    TOK_H_BASE,
    TOK_PIXEL_BASE,
    TOK_W_BASE,
    VOCAB_SIZE,
    tok_c,
    tok_h,
    tok_px,
    tok_w,
)

# ---- Serialization ----


def serialize_grid(g: Grid, mode: str = "row") -> List[int]:
    """
    Serialize a grid to a list of tokens.

    Args:
        g: The grid to serialize.
        mode: The mode to use for serialization. Note we plan on only using "row".
    """
    H, W = g.shape
    seq = [BOS, tok_w(W), tok_h(H)]
    # set of colors present, capped to 10 anyway
    cols = sorted(list(set(g.a.flatten().tolist())))
    seq.append(SEP)
    # write color inventory (optional; helpful prior)
    for c in cols:
        seq.append(tok_c(c))
    seq.append(SEP)
    # pixels
    if mode == "row":
        it = g.a.flatten()
    elif mode == "col":
        it = g.a.T.flatten()
    else:
        raise ValueError("mode must be 'row' or 'col'")
    seq.extend([tok_px(int(c)) for c in it])
    seq.append(EOS)
    return seq


def deserialize_grid(seq: List[int], mode: str = "row") -> Grid:
    """
    Deserialize a token sequence back into a Grid.

    Expected format: BOS, W_tok, H_tok, SEP, color_inventory..., SEP, pixels..., EOS

    Handles edge cases:
    - Missing EOS: reads pixels until end of sequence
    - Too few tokens: raises ValueError with descriptive message
    """
    if len(seq) < 5:
        raise ValueError(f"Sequence too short: {len(seq)} tokens, need at least 5 (BOS, W, H, SEP, SEP)")

    if seq[0] != BOS:
        raise ValueError(f"Expected BOS={BOS} at position 0, got {seq[0]}")

    W = (seq[1] - TOK_W_BASE) + 1
    H = (seq[2] - TOK_H_BASE) + 1

    if W < 1 or W > 30 or H < 1 or H > 30:
        raise ValueError(f"Invalid dimensions: W={W}, H={H} (tokens: W_tok={seq[1]}, H_tok={seq[2]})")

    # find separators
    i = 3
    if seq[i] != SEP:
        raise ValueError(f"Expected SEP={SEP} at position 3, got {seq[i]}")
    i += 1

    # consume color inventory until SEP
    while i < len(seq) and seq[i] != SEP:
        i += 1

    if i >= len(seq):
        raise ValueError("Missing second SEP after color inventory")
    i += 1  # skip second SEP

    # remaining until EOS (or end of sequence if no EOS)
    pix = []
    while i < len(seq) and seq[i] != EOS:
        pix.append(seq[i] - TOK_PIXEL_BASE)
        i += 1

    expected_pixels = H * W
    
    if len(pix) < expected_pixels:
        raise ValueError(f"Pixel count too few: got {len(pix)}, expected {expected_pixels} for {H}x{W} grid")
    elif len(pix) > expected_pixels:
        pix = pix[:expected_pixels]

    arr = np.array(pix, dtype=np.int64)
    if mode == "row":
        g = arr.reshape(H, W)
    else:
        g = arr.reshape(W, H).T
    return Grid(g)


# ---- Pair (X->Y) example packing ----


def pack_example(x: Grid, y: Grid, mode: str = "row") -> List[int]:
    # input then output separated by SEP
    sx = serialize_grid(x, mode)
    sy = serialize_grid(y, mode)
    # drop BOS from sy to avoid nested BOS/EOS; model learns structure via SEP
    seq = sx + [SEP] + sy[1:]
    return seq
