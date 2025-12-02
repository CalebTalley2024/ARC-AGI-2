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
    if len(seq) < 3 or seq[0] != BOS:
        raise ValueError("Malformed sequence: missing BOS/shape tokens")
    
    W = (seq[1] - TOK_W_BASE) + 1
    H = (seq[2] - TOK_H_BASE) + 1

    def _find_token(start: int, token: int) -> int | None:
        for idx in range(start, len(seq)):
            if seq[idx] == token:
                return idx
        return None

    i = 3
    first_sep = _find_token(i, SEP)
    if first_sep is None:
        return Grid(np.zeros((H, W), dtype=np.int8))
    i = first_sep + 1

    second_sep = _find_token(i, SEP)
    if second_sep is None:
        return Grid(np.zeros((H, W), dtype=np.int8))
    i = second_sep + 1  # skip color inventory

    pix: List[int] = []
    while i < len(seq) and seq[i] != EOS:
        pix_val = seq[i] - TOK_PIXEL_BASE
        pix.append(int(np.clip(pix_val, 0, NUM_COLORS - 1)))
        i += 1

    total = H * W
    if len(pix) < total:
        pix.extend([0] * (total - len(pix)))
    elif len(pix) > total:
        pix = pix[:total]

    arr = np.array(pix, dtype=np.int8)
    if mode == "row":
        g = arr.reshape(H, W)
    elif mode == "col":
        g = arr.reshape(W, H).T
    else:
        raise ValueError("mode must be 'row' or 'col'")

    return Grid(g)


# ---- Pair (X->Y) example packing ----
def pack_example(x: Grid, y: Grid, mode: str = "row") -> List[int]:
    # input then output separated by SEP
    sx = serialize_grid(x, mode)
    sy = serialize_grid(y, mode)
    # drop BOS from sy to avoid nested BOS/EOS; model learns structure via SEP
    seq = sx + [SEP] + sy[1:]
    return seq
