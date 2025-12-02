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
    assert seq[0] == BOS
    W = (seq[1] - TOK_W_BASE) + 1
    H = (seq[2] - TOK_H_BASE) + 1
    # find separators
    i = 3
    # print(SEP, seq[i])
    assert seq[i] == SEP; i += 1
    # consume color inventory until SEP
    while seq[i] != SEP:
        i += 1
    i += 1  # skip second SEP
    # remaining until EOS
    pix = []
    while seq[i] != EOS:
        pix.append(seq[i] - TOK_PIXEL_BASE)
        i += 1
    arr = np.array(pix, dtype=np.int64)
    if mode == "row":
        if len(arr)>=H*W:
            arr = arr[:H*W]
        if len(arr)<H*W:
            arr = np.pad(arr, (0, H*W-len(arr)), mode='constant', constant_values=0)
        g = arr.reshape(H, W)
    else:
        if len(arr)>=H*W:
            arr = arr[:H*W]
        if len(arr)<H*W:
            arr = np.pad(arr, (0, H*W-len(arr)), mode='constant', constant_values=0)
        g = arr.reshape(W, H).T
    return g


# ---- Pair (X->Y) example packing ----


def pack_example(x: Grid, y: Grid, mode: str = "row") -> List[int]:
    # input then output separated by SEP
    sx = serialize_grid(x, mode)
    sy = serialize_grid(y, mode)
    # drop BOS from sy to avoid nested BOS/EOS; model learns structure via SEP
    seq = sx + [SEP] + sy[1:]
    return seq
